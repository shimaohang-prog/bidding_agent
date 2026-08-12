import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

import main
from backend.services.chat_service import ChatService
from backend.services.deepseek_stream import StreamChunk
from backend.services.remote_cli import answer_questions_via_server
from backend.services.qa_pipeline import build_generation_messages
from common.retrieval_models import RetrievalPlan, RetrievalResult


class FakeHttpClient:
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


class FakeDeepSeek:
    def __init__(self):
        self.messages = None

    async def stream(self, messages):
        self.messages = messages
        yield StreamChunk(content="共享管线回答")
        yield StreamChunk(usage={"total_tokens": 12})


@pytest.mark.asyncio
async def test_cli_uses_shared_web_retrieval_and_generation_messages():
    settings = SimpleNamespace(
        milvus_thread_limit=1,
        private_milvus_uri="milvus_db/private/main.db",
        private_collection="private_documents",
        tavily_url="https://tavily.test/search",
        tavily_api_key=SecretStr("tavily-key"),
        deepseek_url="https://deepseek.test/chat",
        deepseek_api_key=SecretStr("deepseek-key"),
        answer_model="answer-model",
        answer_temperature=0.0,
    )
    http = FakeHttpClient()
    deepseek = FakeDeepSeek()
    retrieval = RetrievalResult(plan=RetrievalPlan())
    shared_context = "canonical evidence context"
    shared_citations = [{"evidence_id": "E1"}]
    retrieve = AsyncMock(
        return_value=(retrieval, shared_context, shared_citations)
    )

    with (
        patch("main.get_settings", return_value=settings),
        patch("main.create_upstream_http_client", return_value=http),
        patch("main.PrivateDocumentStore"),
        patch("main.AsyncTavilyClient"),
        patch("main.AsyncRAGService"),
        patch("main.DeepSeekStreamClient", return_value=deepseek),
        patch("main.retrieve_qa_context", retrieve),
    ):
        streamed = []

        async def on_token(content):
            streamed.append(content)

        result = await main.answer_question_async("同一个问题", on_token=on_token)

    assert result["answer"] == "共享管线回答"
    assert result["context"] == shared_context
    assert result["citations"] == shared_citations
    assert result["usage"] == {"total_tokens": 12}
    assert streamed == ["共享管线回答"]
    assert deepseek.messages == build_generation_messages(
        "同一个问题",
        shared_context,
    )
    assert retrieve.await_args.kwargs["file_ids"] == []
    assert http.closed is True


@pytest.mark.asyncio
async def test_cli_runs_multiple_questions_concurrently_and_preserves_order():
    class ConcurrentRuntime:
        active = 0
        max_active = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def answer(self, question, *, on_token=None):
            ConcurrentRuntime.active += 1
            ConcurrentRuntime.max_active = max(
                ConcurrentRuntime.max_active,
                ConcurrentRuntime.active,
            )
            try:
                if on_token:
                    await on_token(f"{question}-token")
                import asyncio
                await asyncio.sleep(0.03)
                return {"answer": question}
            finally:
                ConcurrentRuntime.active -= 1

    streamed = []

    async def on_token(index, question, content):
        streamed.append((index, question, content))

    with patch("main.AsyncQuestionRuntime", ConcurrentRuntime):
        results = await main.answer_questions_async(
            ["问题一", "问题二"], concurrency=2, on_token=on_token,
        )

    assert [item["answer"] for item in results] == ["问题一", "问题二"]
    assert ConcurrentRuntime.max_active == 2
    assert all("timing" in item for item in results)
    assert max(item["timing"]["started_offset_ms"] for item in results) < 100
    assert sorted(streamed) == [
        (0, "问题一", "问题一-token"),
        (1, "问题二", "问题二-token"),
    ]


@pytest.mark.asyncio
async def test_remote_cli_submits_two_questions_before_receiving_stream(monkeypatch):
    class Response:
        def __init__(self, status_code, body=None):
            self.status_code = status_code
            self._body = body or {}

        def json(self):
            return self._body

    class FakeHttp:
        def __init__(self, *args, **kwargs):
            self.cookies = {"access_token": "short-lived"}
            self.conversation_count = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, path, json=None):
            if path.endswith("/login"):
                return Response(200)
            if path.endswith("/conversations"):
                self.conversation_count += 1
                return Response(201, {"id": f"conversation-{self.conversation_count}"})
            return Response(204)

    class FakeSocket:
        def __init__(self):
            self.sent = []
            self.events = []

        async def send(self, raw):
            self.sent.append(json.loads(raw))

        async def recv(self):
            if not self.events:
                assert len(self.sent) == 2
                first, second = self.sent
                self.events = [
                    {"type": "token", "request_id": first["request_id"], "payload": {"content": "一"}},
                    {"type": "token", "request_id": second["request_id"], "payload": {"content": "二"}},
                    {"type": "done", "request_id": first["request_id"], "payload": {"answer": "答案一"}},
                    {"type": "done", "request_id": second["request_id"], "payload": {"answer": "答案二"}},
                ]
            return json.dumps(self.events.pop(0), ensure_ascii=False)

    socket = FakeSocket()

    class Connection:
        async def __aenter__(self):
            return socket

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    monkeypatch.setattr("backend.services.remote_cli.httpx.AsyncClient", FakeHttp)
    monkeypatch.setattr(
        "backend.services.remote_cli.websockets.connect",
        lambda *args, **kwargs: Connection(),
    )
    streamed = []

    async def on_token(index, question, content):
        streamed.append((index, question, content))

    results = await answer_questions_via_server(
        ["问题一", "问题二"],
        server_url="http://127.0.0.1:8000",
        origin="http://127.0.0.1:5173",
        username="alice",
        password="password-a",
        concurrency=2,
        on_token=on_token,
    )

    assert [item["answer"] for item in results] == ["答案一", "答案二"]
    assert streamed == [(0, "问题一", "一"), (1, "问题二", "二")]
    assert len(socket.sent) == 2


@pytest.mark.asyncio
async def test_web_zero_context_mode_never_reads_conversation_history():
    messages = MagicMock()
    messages.recent_turns = AsyncMock()
    cache = MagicMock()
    cache.get_context = AsyncMock()
    service = ChatService(
        rag=MagicMock(),
        deepseek=MagicMock(),
        messages=messages,
        cache=cache,
        context_turns=0,
    )

    history = await service._history(
        "user-id",
        "conversation-id",
        "request-id",
    )

    assert history == []
    cache.get_context.assert_not_awaited()
    messages.recent_turns.assert_not_awaited()
