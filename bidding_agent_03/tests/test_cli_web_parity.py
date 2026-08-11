from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

import main
from backend.services.chat_service import ChatService
from backend.services.deepseek_stream import StreamChunk
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
        result = await main.answer_question_async("同一个问题")

    assert result["answer"] == "共享管线回答"
    assert result["context"] == shared_context
    assert result["citations"] == shared_citations
    assert result["usage"] == {"total_tokens": 12}
    assert deepseek.messages == build_generation_messages(
        "同一个问题",
        shared_context,
    )
    assert retrieve.await_args.kwargs["file_ids"] == []
    assert http.closed is True


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
