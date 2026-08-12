import asyncio
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.api.knowledge import _list_public_files, _resolve_public_file
from backend.api.websocket import _automatic_conversation_title
from backend.cache.store import MemoryCacheStore
from backend.core.config import Settings
from backend.core.errors import ApiError
from backend.core.security import create_token, decode_token, hash_password, verify_password
from backend.schemas.websocket import inbound_adapter


def test_automatic_conversation_title_uses_first_question_summary():
    assert _automatic_conversation_title("  查询   某企业的统一社会信用代码。 ") == "查询 某企业的统一社会信用代码"
    assert _automatic_conversation_title("招标文件评分办法如何计算？" * 4).endswith("…")
    assert len(_automatic_conversation_title("招标文件评分办法如何计算？" * 4)) == 29


def test_knowledge_file_listing_and_path_boundary(tmp_path):
    root = tmp_path / "laws"
    nested = root / "subcategory"
    nested.mkdir(parents=True)
    first = root / "法规一.txt"
    second = nested / "法规二.txt"
    first.write_text("第一条", encoding="utf-8")
    second.write_text("第二条", encoding="utf-8")

    listed = _list_public_files(root)
    assert [item.relative_path for item in listed] == ["subcategory/法规二.txt", "法规一.txt"]
    assert _resolve_public_file(root, "subcategory/法规二.txt") == second
    with pytest.raises(ApiError, match="知识库文件不存在"):
        _resolve_public_file(root, "../outside.txt")


def test_lite_rejects_multiple_api_workers():
    with pytest.raises(ValidationError, match="Milvus Lite"):
        Settings(milvus_mode="lite", api_workers=2)


def test_same_lite_path_serializes_first_client_start(monkeypatch, tmp_path):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import pymilvus
    from common.milvus_config import get_milvus_client

    active = 0
    max_active = 0
    counter_lock = threading.Lock()

    class FakeClient:
        def __init__(self, *, uri):
            nonlocal active, max_active
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.04)
            with counter_lock:
                active -= 1
            self.uri = uri

    monkeypatch.setattr(pymilvus, "MilvusClient", FakeClient)
    db_path = tmp_path / "same.db"
    db_path.mkdir()
    with ThreadPoolExecutor(max_workers=2) as executor:
        clients = list(executor.map(
            lambda _: get_milvus_client(db_path),
            range(2),
        ))

    assert len(clients) == 2
    assert max_active == 1


def test_real_enterprise_lite_allows_two_clients_in_one_process():
    from concurrent.futures import ThreadPoolExecutor
    from common.milvus_config import (
        COLLECTION_NAME,
        category_db_path,
        close_milvus_client,
        get_milvus_client,
    )

    db_path = category_db_path("enterprise")
    if not db_path.exists():
        pytest.skip("本地 enterprise Milvus Lite 数据库不存在")

    def inspect_collection(_: int) -> bool:
        client = get_milvus_client(db_path)
        try:
            return bool(client.has_collection(collection_name=COLLECTION_NAME))
        finally:
            close_milvus_client(client)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(inspect_collection, range(2)))

    assert results == [True, True]


def test_password_and_typed_tokens():
    settings = Settings(cookie_secure=False)
    user_id = uuid4()
    password_hash = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", password_hash)
    assert not verify_password("wrong password", password_hash)
    access = create_token(user_id=user_id, token_version=3, kind="access", settings=settings)
    assert decode_token(access, kind="access", settings=settings)["ver"] == 3
    with pytest.raises(Exception):
        decode_token(access, kind="refresh", settings=settings)


def test_websocket_protocol_rejects_unknown_fields():
    body = {
        "type": "ask", "request_id": str(uuid4()), "conversation_id": str(uuid4()),
        "question": "测试", "client_message_id": "m1", "unexpected": "secret",
    }
    with pytest.raises(ValidationError):
        inbound_adapter.validate_python(body)


@pytest.mark.asyncio
async def test_memory_cache_context_replay_cancel_and_limits():
    cache = MemoryCacheStore()
    await cache.set_context("u", "c", [{"role": "user", "content": "q"}])
    assert (await cache.get_context("u", "c"))[0]["content"] == "q"
    first = await cache.append_event("r", {"type": "token", "payload": {"content": "a"}})
    done = await cache.append_event("r", {"type": "done", "payload": {}})
    assert first["seq"] == 1
    assert done["payload"]["final_seq"] == 2
    assert [item["seq"] for item in await cache.replay("r", 1)] == [2]
    await cache.set_cancelled("r")
    assert await cache.is_cancelled("r")
    assert await cache.acquire_generation("u", 1)
    assert not await cache.acquire_generation("u", 1)
    assert await cache.acquire_generation("another-user", 1)
    await cache.release_generation("u")
    assert await cache.acquire_generation("u", 1)


@pytest.mark.asyncio
async def test_thread_bridge_keeps_event_loop_responsive():
    import anyio
    import time

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0.01)
            ticks += 1

    async with anyio.create_task_group() as group:
        group.start_soon(heartbeat)
        await anyio.to_thread.run_sync(time.sleep, 0.08)
    assert ticks == 5


@pytest.mark.asyncio
async def test_async_rag_encodes_duplicate_query_once_and_searches_categories_concurrently(monkeypatch):
    import threading
    import time
    import anyio
    from backend.services.rag_service import AsyncRAGService
    from common.retrieval_models import RetrievalPlan, SemanticQueryTask

    encode_calls = 0
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_encode(_: str):
        nonlocal encode_calls
        encode_calls += 1
        return [0.1, 0.2]

    def fake_search(*args, categories, **kwargs):
        nonlocal active, max_active
        assert kwargs["query_vector"] == [0.1, 0.2]
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        return [], []

    monkeypatch.setattr("backend.services.rag_service.encode_query", fake_encode)
    monkeypatch.setattr("backend.services.rag_service.category_vector_search_multi", fake_search)
    service = AsyncRAGService(limiter=anyio.CapacityLimiter(4), tavily=object(), private_store=object())
    plan = RetrievalPlan(semantic_queries=[
        SemanticQueryTask(query="同一问题", categories=["laws", "policy", "enterprise"]),
        SemanticQueryTask(query="同一问题", categories=["product", "tender"]),
    ])
    await service._local_hits(plan)
    assert encode_calls == 1
    assert max_active == 2
