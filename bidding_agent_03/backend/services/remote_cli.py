"""让 CLI 复用正在运行的 Web 后端，避免跨进程争抢 Milvus Lite。"""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import websockets


RemoteTokenCallback = Callable[[int, str, str], Awaitable[None]]


async def backend_is_available(server_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=0.8) as client:
            response = await client.get(
                f"{server_url.rstrip('/')}/api/v1/health/live"
            )
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def _websocket_url(server_url: str) -> str:
    parsed = urlsplit(server_url.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("服务端地址必须是 http:// 或 https:// 地址")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    prefix = parsed.path.rstrip("/")
    return f"{scheme}://{parsed.netloc}{prefix}/api/v1/ws/chat"


async def answer_questions_via_server(
    questions: Sequence[str],
    *,
    server_url: str,
    origin: str,
    username: str,
    password: str,
    concurrency: int = 4,
    timeout: float = 180.0,
    on_token: RemoteTokenCallback | None = None,
) -> list[dict]:
    """登录 Web 后端，在一个 WebSocket 上同时提交多个独立会话。"""
    clean_questions = [" ".join((item or "").split()) for item in questions]
    if not clean_questions or any(not item for item in clean_questions):
        raise ValueError("问题不能为空")
    if concurrency < 1:
        raise ValueError("并发数必须大于 0")
    base_url = server_url.rstrip("/")
    batch_started = time.perf_counter()

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(20.0, read=60.0),
    ) as http:
        login = await http.post(
            "/api/v1/auth/login",
            json={"username": username, "password": password},
        )
        if login.status_code != 200:
            raise RuntimeError(f"Web 后端登录失败，HTTP {login.status_code}")

        conversations = []
        for index, question in enumerate(clean_questions, 1):
            response = await http.post(
                "/api/v1/conversations",
                json={"title": f"CLI 并发 {index}：{question[:32]}"},
            )
            if response.status_code != 201:
                raise RuntimeError(
                    f"创建 CLI 会话失败，HTTP {response.status_code}"
                )
            conversations.append(response.json()["id"])

        cookie = "; ".join(
            f"{name}={value}" for name, value in http.cookies.items()
        )
        if not cookie:
            raise RuntimeError("Web 后端登录成功但未返回认证 Cookie")

        requests: list[str | None] = [None] * len(clean_questions)
        results: list[dict | None] = [None] * len(clean_questions)
        first_tokens: list[float | None] = [None] * len(clean_questions)
        started_times: list[float] = [0.0] * len(clean_questions)
        citations: list[list[dict]] = [[] for _ in clean_questions]
        stages: list[list[str]] = [[] for _ in clean_questions]

        try:
            async with websockets.connect(
                _websocket_url(base_url),
                origin=origin,
                additional_headers={"Cookie": cookie},
            ) as socket:
                request_indexes: dict[str, int] = {}

                async def send_question(index: int) -> None:
                    question = clean_questions[index]
                    conversation_id = conversations[index]
                    request_id = str(uuid4())
                    requests[index] = request_id
                    request_indexes[request_id] = index
                    started_times[index] = time.perf_counter()
                    await socket.send(json.dumps({
                        "type": "ask",
                        "request_id": request_id,
                        "conversation_id": conversation_id,
                        "client_message_id": str(uuid4()),
                        "question": question,
                        "file_ids": [],
                    }, ensure_ascii=False))

                next_index = 0
                while next_index < min(concurrency, len(clean_questions)):
                    await send_question(next_index)
                    next_index += 1
                completed = 0
                while completed < len(clean_questions):
                    raw = await asyncio.wait_for(socket.recv(), timeout=timeout)
                    event = json.loads(raw)
                    index = request_indexes.get(event.get("request_id"))
                    if index is None:
                        continue
                    event_type = event.get("type")
                    payload = event.get("payload", {})
                    if event_type == "status":
                        stages[index].append(str(payload.get("stage", "")))
                    elif event_type == "token":
                        if first_tokens[index] is None:
                            first_tokens[index] = time.perf_counter()
                        if on_token is not None:
                            await on_token(
                                index,
                                clean_questions[index],
                                str(payload.get("content", "")),
                            )
                    elif event_type == "citations":
                        citations[index] = list(payload.get("items", []))
                    elif event_type == "done":
                        finished = time.perf_counter()
                        results[index] = {
                            "answer": str(payload.get("answer", "")),
                            "citations": citations[index],
                            "usage": payload.get("usage", {}),
                            "stages": stages[index],
                            "conversation_id": conversations[index],
                            "timing": {
                                "started_offset_ms": round(
                                    (started_times[index] - batch_started) * 1000
                                ),
                                "first_token_ms": round(
                                    ((first_tokens[index] or finished)
                                     - started_times[index]) * 1000
                                ),
                                "duration_ms": round(
                                    (finished - started_times[index]) * 1000
                                ),
                            },
                        }
                        completed += 1
                        if next_index < len(clean_questions):
                            await send_question(next_index)
                            next_index += 1
                    elif event_type in {"error", "cancelled"}:
                        raise RuntimeError(
                            f"问题 {index + 1} 执行失败："
                            f"{payload.get('message', event_type)}"
                        )
        finally:
            try:
                await http.post("/api/v1/auth/logout")
            except httpx.HTTPError:
                pass

    return [item for item in results if item is not None]
