import asyncio
import time
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app import create_app
from backend.cache.store import MemoryCacheStore
from backend.core.config import Settings
from backend.core.security import hash_password
from backend.db.base import Base
from backend.repositories.users import UserRepository
from backend.services.deepseek_stream import StreamChunk
from common.retrieval_models import RetrievalPlan, RetrievalResult, SemanticQueryTask


class FakeRAG:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.concurrent_started = asyncio.Event()

    async def retrieve(self, question, *, user_id, conversation_id, file_ids, status):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await status("retrieving")
            if question == "并发问题":
                if self.active >= 2:
                    self.concurrent_started.set()
                try:
                    await asyncio.wait_for(self.concurrent_started.wait(), timeout=0.5)
                except TimeoutError:
                    pass
            else:
                await asyncio.sleep(0.04)
            await status("reranking")
            return RetrievalResult(
                plan=RetrievalPlan(semantic_queries=[SemanticQueryTask(query=question, categories=["laws"])]),
                candidates=[],
            )
        finally:
            self.active -= 1


class FakeDeepSeek:
    async def stream(self, messages):
        if any("慢速" in str(item.get("content", "")) for item in messages):
            for index in range(50):
                await asyncio.sleep(0.02)
                yield StreamChunk(content=str(index))
            return
        for value in ("基于证据", "，当前无法确认。[E1]"):
            await asyncio.sleep(0.02)
            yield StreamChunk(content=value)
        yield StreamChunk(usage={"total_tokens": 9})


async def prepare_database(path: Path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        await UserRepository(session).create("alice", hash_password("password-a"))
        await UserRepository(session).create("bob", hash_password("password-b"))
        await session.commit()
    await engine.dispose()


@pytest.fixture
def client(tmp_path):
    database = tmp_path / "api.db"
    asyncio.run(prepare_database(database))
    settings = Settings(
        environment="test", database_url=f"sqlite+aiosqlite:///{database}",
        redis_url="redis://127.0.0.1:6399/15", cookie_secure=False,
        allowed_origins="http://testserver", upload_root=tmp_path / "uploads",
        private_milvus_uri=str(tmp_path / "private.db"),
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        memory = MemoryCacheStore()
        app.state.cache = memory
        app.state.generation_manager.cache = memory
        app.state.generation_manager.rag = FakeRAG()
        app.state.generation_manager.deepseek = FakeDeepSeek()
        yield test_client, app


def login(client: TestClient, username: str = "alice"):
    response = client.post("/api/v1/auth/login", json={"username": username, "password": f"password-{username[0]}"})
    assert response.status_code == 200


def test_rest_login_conversation_crud_and_owner_isolation(client):
    http, _ = client
    login(http)
    created = http.post("/api/v1/conversations", json={"title": "会话 A"})
    assert created.status_code == 201
    conversation_id = created.json()["id"]
    assert http.get("/api/v1/conversations").json()[0]["title"] == "会话 A"
    assert http.patch(f"/api/v1/conversations/{conversation_id}", json={"title": "已改名"}).json()["title"] == "已改名"
    http.post("/api/v1/auth/logout")
    login(http, "bob")
    hidden = http.get(f"/api/v1/conversations/{conversation_id}/messages")
    assert hidden.status_code == 404 and hidden.json()["error_code"] == "CONVERSATION_NOT_FOUND"


def test_knowledge_listing_and_uploaded_file_open_are_authenticated(client):
    http, _ = client
    login(http)
    public_files = http.get("/api/v1/knowledge/enterprise/files")
    assert public_files.status_code == 200
    assert any(item["name"] == "enterprise.csv" for item in public_files.json())

    conversation_id = http.post("/api/v1/conversations", json={}).json()["id"]
    uploaded = http.post(
        "/api/v1/files",
        data={"conversation_id": conversation_id},
        files={"upload": ("说明.txt", b"private evidence", "text/plain")},
    )
    assert uploaded.status_code == 202
    file_id = uploaded.json()["id"]
    opened = http.get(f"/api/v1/files/{file_id}/content")
    assert opened.status_code == 200 and opened.content == b"private evidence"
    assert "inline" in opened.headers["content-disposition"]

    http.post("/api/v1/auth/logout")
    login(http, "bob")
    assert http.get(f"/api/v1/files/{file_id}/content").status_code == 404


def test_websocket_stream_ping_resume_idempotency(client):
    http, app = client
    login(http)
    conversation_id = http.post("/api/v1/conversations", json={}).json()["id"]
    request_id = str(uuid4())
    ask = {
        "type": "ask", "request_id": request_id, "conversation_id": conversation_id,
        "client_message_id": str(uuid4()), "question": "测试问题", "file_ids": [],
    }
    received = []
    with http.websocket_connect("/api/v1/ws/chat", headers={"origin": "http://testserver"}) as socket:
        socket.send_json(ask)
        first_ack = socket.receive_json()
        assert first_ack["type"] == "ack"
        assert first_ack["payload"]["conversation_title"] == "测试问题"
        socket.send_json({"type": "ping", "request_id": request_id, "conversation_id": conversation_id})
        while True:
            event = socket.receive_json()
            received.append(event)
            if event["type"] == "pong":
                break
        assert any(item["type"] == "pong" for item in received)
        while True:
            event = socket.receive_json()
            received.append(event)
            if event["type"] == "done":
                break
    sequenced = [item for item in received if item.get("seq") is not None]
    assert [item["seq"] for item in sequenced] == sorted(item["seq"] for item in sequenced)
    assert [item["type"] for item in sequenced] == [
        "status", "status", "status", "status", "token", "token", "citations", "done"
    ]
    assert sequenced[-1]["payload"]["final_seq"] == sequenced[-1]["seq"]
    assert http.get("/api/v1/conversations").json()[0]["title"] == "测试问题"

    with http.websocket_connect("/api/v1/ws/chat", headers={"origin": "http://testserver"}) as socket:
        socket.send_json({"type": "resume", "request_id": request_id, "conversation_id": conversation_id, "last_seq": 5})
        replay = [socket.receive_json() for _ in range(3)]
        assert [item["seq"] for item in replay] == [6, 7, 8]
        socket.send_json(ask)
        ack = socket.receive_json()
        assert ack["type"] == "ack" and ack["payload"]["idempotent"] is True
        assert ack["payload"]["conversation_title"] == "测试问题"


def test_websocket_rejects_origin(client):
    http, _ = client
    login(http)
    with pytest.raises(Exception):
        with http.websocket_connect("/api/v1/ws/chat", headers={"origin": "https://evil.example"}):
            pass


def test_websocket_stop_cancels_owned_task(client):
    http, _ = client
    login(http)
    conversation_id = http.post("/api/v1/conversations", json={"title": "停止"}).json()["id"]
    request_id = str(uuid4())
    with http.websocket_connect("/api/v1/ws/chat", headers={"origin": "http://testserver"}) as socket:
        socket.send_json({
            "type": "ask", "request_id": request_id, "conversation_id": conversation_id,
            "client_message_id": str(uuid4()), "question": "慢速生成", "file_ids": [],
        })
        assert socket.receive_json()["type"] == "ack"
        while socket.receive_json()["type"] != "token":
            pass
        socket.send_json({"type": "stop", "request_id": request_id, "conversation_id": conversation_id})
        types = set()
        while not {"ack", "cancelled"}.issubset(types):
            types.add(socket.receive_json()["type"])
        assert {"ack", "cancelled"}.issubset(types)


def test_same_user_can_generate_in_two_conversations_concurrently(client):
    http, app = client
    login(http)
    conversation_ids = [
        http.post("/api/v1/conversations", json={"title": f"并发 {index}"}).json()["id"]
        for index in (1, 2)
    ]
    request_ids = [str(uuid4()), str(uuid4())]

    with http.websocket_connect(
        "/api/v1/ws/chat", headers={"origin": "http://testserver"},
    ) as socket:
        for request_id, conversation_id in zip(
            request_ids, conversation_ids, strict=True,
        ):
            socket.send_json({
                "type": "ask", "request_id": request_id,
                "conversation_id": conversation_id,
                "client_message_id": str(uuid4()), "question": "并发问题",
                "file_ids": [],
            })
            while True:
                event = socket.receive_json()
                if event["type"] == "ack" and event["request_id"] == request_id:
                    break

        completed = set()
        while len(completed) < 2:
            event = socket.receive_json()
            if event["type"] == "done":
                completed.add(event["request_id"])

        deadline = time.monotonic() + 1.0
        while app.state.generation_manager.tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not app.state.generation_manager.tasks

    assert app.state.generation_manager.rag.max_active >= 2
