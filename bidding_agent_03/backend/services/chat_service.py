"""可供 WebSocket/SSE/测试复用的领域事件异步迭代器。"""

import asyncio
import time
from collections.abc import AsyncIterator

from backend.cache.store import CacheStore
from backend.repositories.messages import MessageRepository
from backend.schemas.websocket import DomainEvent
from backend.services.deepseek_stream import DeepSeekStreamClient
from backend.services.rag_service import AsyncRAGService
from backend.services.qa_pipeline import (
    build_generation_messages,
    retrieve_qa_context,
)


class ChatService:
    def __init__(
        self, *, rag: AsyncRAGService, deepseek: DeepSeekStreamClient,
        messages: MessageRepository, cache: CacheStore, context_turns: int,
    ) -> None:
        self.rag = rag
        self.deepseek = deepseek
        self.messages = messages
        self.cache = cache
        self.context_turns = context_turns

    async def _history(self, user_id: str, conversation_id: str, request_id: str) -> list[dict[str, str]]:
        if self.context_turns <= 0:
            return []
        cached = await self.cache.get_context(user_id, conversation_id)
        if cached is not None:
            return cached
        rows = await self.messages.recent_turns(user_id, conversation_id, self.context_turns)
        history = [
            {"role": item.role, "content": item.content}
            for item in rows
            if item.request_id != request_id
        ]
        await self.cache.set_context(user_id, conversation_id, history)
        return history

    async def stream_answer(
        self, *, request_id: str, user_id: str, conversation_id: str,
        question: str, file_ids: list[str],
    ) -> AsyncIterator[DomainEvent]:
        started = time.perf_counter()
        yield DomainEvent(type="status", payload={"stage": "planning"})
        stages: list[str] = []

        async def status(stage: str) -> None:
            stages.append(stage)

        retrieval, context, citations = await retrieve_qa_context(
            self.rag,
            question,
            user_id=user_id,
            conversation_id=conversation_id,
            file_ids=file_ids,
            status=status,
        )
        for stage in stages:
            yield DomainEvent(type="status", payload={"stage": stage})
        yield DomainEvent(type="status", payload={"stage": "generating"})

        history = await self._history(user_id, conversation_id, request_id)
        history_window = (
            history[-self.context_turns * 2 :]
            if self.context_turns > 0
            else []
        )
        upstream_messages = build_generation_messages(
            question,
            context,
            history_window,
        )
        answer_parts: list[str] = []
        usage: dict = {}
        async for chunk in self.deepseek.stream(upstream_messages):
            if await self.cache.is_cancelled(request_id):
                raise asyncio.CancelledError
            if chunk.usage:
                usage = chunk.usage
            if chunk.content:
                answer_parts.append(chunk.content)
                yield DomainEvent(type="token", payload={"content": chunk.content})
        answer = "".join(answer_parts).strip()
        if not answer:
            raise RuntimeError("模型未返回可用答案")

        assistant = await self.messages.create(
            user_id=user_id, conversation_id=conversation_id, role="assistant",
            content=answer, request_id=request_id,
        )
        await self.messages.add_citations(assistant.id, citations)
        if self.context_turns > 0:
            new_history = [*history, {"role": "user", "content": question}, {"role": "assistant", "content": answer}]
            await self.cache.set_context(
                user_id,
                conversation_id,
                new_history[-self.context_turns * 2 :],
            )
        yield DomainEvent(type="citations", payload={"items": citations})
        latency_ms = round((time.perf_counter() - started) * 1000)
        yield DomainEvent(type="done", payload={
            "message_id": assistant.id, "answer": answer, "usage": usage,
            "latency_ms": latency_ms,
        })
