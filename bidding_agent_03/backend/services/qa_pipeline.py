"""Shared single-turn retrieval context and generation-message pipeline."""

from collections.abc import Awaitable, Callable, Sequence

import httpx

from common.retrieval_models import RetrievalResult
from generation.answer_generator import build_messages
from generation.context_builder import build_answer_context_with_citations

from backend.services.rag_service import AsyncRAGService


StatusCallback = Callable[[str], Awaitable[None]]


def create_upstream_http_client() -> httpx.AsyncClient:
    """Create the canonical upstream client used by web and CLI runtimes."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=90, write=30, pool=10),
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
        ),
    )


async def retrieve_qa_context(
    rag: AsyncRAGService,
    question: str,
    *,
    user_id: str,
    conversation_id: str,
    file_ids: list[str],
    status: StatusCallback,
) -> tuple[RetrievalResult, str, list[dict]]:
    """Run the shared RAG path and build one canonical evidence context."""
    retrieval = await rag.retrieve(
        question,
        user_id=user_id,
        conversation_id=conversation_id,
        file_ids=file_ids,
        status=status,
    )
    context, citations = build_answer_context_with_citations(
        question,
        retrieval,
    )
    return retrieval, context, citations


def build_generation_messages(
    question: str,
    context: str,
    history: Sequence[dict[str, str]] = (),
) -> list[dict[str, str]]:
    """Build the exact message list sent by both the web app and CLI."""
    base = build_messages(question, context)
    return [base[0], *history, base[1]]
