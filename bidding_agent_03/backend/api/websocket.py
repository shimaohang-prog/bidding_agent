import asyncio
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from backend.core.errors import ApiError
from backend.repositories.conversations import ConversationRepository
from backend.repositories.files import FileRepository
from backend.repositories.jobs import JobRepository
from backend.repositories.messages import MessageRepository
from backend.repositories.users import UserRepository
from backend.schemas.websocket import AskEvent, PingEvent, ResumeEvent, StopEvent, inbound_adapter
from backend.services.auth_service import AuthService
from backend.services.generation_manager import SocketConnection


router = APIRouter(tags=["websocket"])
ZERO_ID = "00000000-0000-0000-0000-000000000000"


async def _authenticate(websocket: WebSocket):
    settings = websocket.app.state.settings
    origin = (websocket.headers.get("origin") or "").rstrip("/")
    if origin not in settings.origin_set:
        await websocket.close(code=4403, reason="Origin not allowed")
        return None
    async with websocket.app.state.session_factory() as session:
        try:
            return await AuthService(UserRepository(session), settings).authenticate(websocket.cookies.get("access_token"))
        except ApiError:
            await websocket.close(code=4401, reason="Authentication required")
            return None


@router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket) -> None:
    user = await _authenticate(websocket)
    if user is None:
        return
    await websocket.accept()
    connection = SocketConnection(websocket)
    manager = websocket.app.state.generation_manager
    settings = websocket.app.state.settings
    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_json(), timeout=settings.websocket_idle_seconds)
            except TimeoutError:
                await websocket.close(code=4408, reason="Idle timeout")
                return
            try:
                event = inbound_adapter.validate_python(raw)
            except ValidationError:
                request_id = str(raw.get("request_id", ZERO_ID)) if isinstance(raw, dict) else ZERO_ID
                conversation_id = str(raw.get("conversation_id", ZERO_ID)) if isinstance(raw, dict) else ZERO_ID
                await connection.send({
                    "type": "error", "request_id": request_id, "conversation_id": conversation_id,
                    "seq": None, "payload": {"error_code": "INVALID_EVENT", "message": "消息格式无效"},
                })
                continue

            request_id = str(event.request_id)
            conversation_id = str(event.conversation_id)
            if isinstance(event, PingEvent):
                await connection.send({"type": "pong", "request_id": request_id, "conversation_id": conversation_id, "seq": None, "payload": {}})
                continue

            async with websocket.app.state.session_factory() as session:
                conversations = ConversationRepository(session)
                if await conversations.owned(user.id, conversation_id) is None:
                    await connection.send({"type": "error", "request_id": request_id, "conversation_id": conversation_id, "seq": None, "payload": {"error_code": "CONVERSATION_NOT_FOUND", "message": "会话不存在"}})
                    continue

                if isinstance(event, ResumeEvent):
                    job = await JobRepository(session).owned(user.id, request_id)
                    if job is None or job.conversation_id != conversation_id:
                        await connection.send({"type": "error", "request_id": request_id, "conversation_id": conversation_id, "seq": None, "payload": {"error_code": "JOB_NOT_FOUND", "message": "生成任务不存在"}})
                        continue
                    manager.subscribe(request_id, connection)
                    replay = await websocket.app.state.cache.replay(request_id, event.last_seq)
                    if not replay and event.last_seq < job.last_seq:
                        await connection.send({"type": "error", "request_id": request_id, "conversation_id": conversation_id, "seq": None, "payload": {"error_code": "STREAM_EXPIRED", "message": "重连事件已过期，请读取历史消息"}})
                    for item in replay:
                        await connection.send(item)
                    continue

                if isinstance(event, StopEvent):
                    stopped = await manager.stop(user.id, request_id, conversation_id)
                    await connection.send({"type": "ack", "request_id": request_id, "conversation_id": conversation_id, "seq": None, "payload": {"action": "stop", "accepted": stopped}})
                    continue

                if isinstance(event, AskEvent):
                    jobs = JobRepository(session)
                    existing = await jobs.by_request(request_id)
                    if existing:
                        if existing.user_id != user.id or existing.conversation_id != conversation_id:
                            await connection.send({"type": "error", "request_id": request_id, "conversation_id": conversation_id, "seq": None, "payload": {"error_code": "JOB_NOT_FOUND", "message": "生成任务不存在"}})
                            continue
                        manager.subscribe(request_id, connection)
                        await connection.send({"type": "ack", "request_id": request_id, "conversation_id": conversation_id, "seq": None, "payload": {"idempotent": True, "status": existing.status}})
                        for item in await websocket.app.state.cache.replay(request_id, 0):
                            await connection.send(item)
                        continue
                    if not await websocket.app.state.cache.allow_request(user.id, settings.rate_limit_per_minute):
                        await connection.send({"type": "error", "request_id": request_id, "conversation_id": conversation_id, "seq": None, "payload": {"error_code": "RATE_LIMITED", "message": "请求过于频繁"}})
                        continue
                    if not await websocket.app.state.cache.acquire_generation(user.id, settings.max_active_generations_per_user):
                        await connection.send({"type": "error", "request_id": request_id, "conversation_id": conversation_id, "seq": None, "payload": {"error_code": "TOO_MANY_GENERATIONS", "message": "并发生成数已达上限"}})
                        continue
                    file_ids = [str(item) for item in event.file_ids]
                    valid_files = True
                    file_repo = FileRepository(session)
                    for file_id in file_ids:
                        item = await file_repo.owned(user.id, file_id)
                        if item is None or item.conversation_id != conversation_id or item.status != "ready":
                            valid_files = False
                            break
                    if not valid_files:
                        await websocket.app.state.cache.release_generation(user.id)
                        await connection.send({"type": "error", "request_id": request_id, "conversation_id": conversation_id, "seq": None, "payload": {"error_code": "FILE_NOT_READY", "message": "文件不存在、无权访问或尚未处理完成"}})
                        continue
                    question = " ".join(event.question.split())
                    message = await MessageRepository(session).create(
                        user_id=user.id, conversation_id=conversation_id, role="user", content=question,
                        request_id=request_id, client_message_id=event.client_message_id,
                    )
                    job, created = await jobs.create(
                        request_id=request_id, user_id=user.id, conversation_id=conversation_id,
                        user_message_id=message.id,
                    )
                    if not created and (job.user_id != user.id or job.conversation_id != conversation_id):
                        await websocket.app.state.cache.release_generation(user.id)
                        await connection.send({"type": "error", "request_id": request_id, "conversation_id": conversation_id, "seq": None, "payload": {"error_code": "JOB_NOT_FOUND", "message": "生成任务不存在"}})
                        continue
                    await session.commit()
                    manager.subscribe(request_id, connection)
                    await connection.send({"type": "ack", "request_id": request_id, "conversation_id": conversation_id, "seq": None, "payload": {"idempotent": not created, "status": job.status}})
                    if created:
                        manager.start(
                            request_id=request_id, user_id=user.id, conversation_id=conversation_id,
                            question=question, file_ids=file_ids,
                        )
                    else:
                        # 唯一约束解决了跨 Worker 竞态，本请求不重复生成/扣费。
                        await websocket.app.state.cache.release_generation(user.id)
                        for item in await websocket.app.state.cache.replay(request_id, 0):
                            await connection.send(item)
    except WebSocketDisconnect:
        pass
    finally:
        manager.unsubscribe(connection)
