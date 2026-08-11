"""生成任务、可靠事件写入、广播与取消。"""

import asyncio
from collections import defaultdict
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.cache.store import CacheStore
from backend.core.config import Settings
from backend.repositories.jobs import JobRepository
from backend.repositories.messages import MessageRepository
from backend.services.chat_service import ChatService
from backend.services.deepseek_stream import DeepSeekStreamClient
from backend.services.rag_service import AsyncRAGService


class SocketConnection:
    def __init__(self, websocket: Any) -> None:
        self.websocket = websocket
        self.lock = asyncio.Lock()

    async def send(self, body: dict[str, Any]) -> bool:
        try:
            async with self.lock:
                await self.websocket.send_json(body)
            return True
        except Exception:
            return False


class GenerationManager:
    def __init__(
        self, *, session_factory: async_sessionmaker[AsyncSession], cache: CacheStore,
        rag: AsyncRAGService, deepseek: DeepSeekStreamClient, settings: Settings,
    ) -> None:
        self.session_factory = session_factory
        self.cache = cache
        self.rag = rag
        self.deepseek = deepseek
        self.settings = settings
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.subscribers: dict[str, set[SocketConnection]] = defaultdict(set)
        self.grace_tasks: dict[str, asyncio.Task[None]] = {}

    def subscribe(self, request_id: str, connection: SocketConnection) -> None:
        grace = self.grace_tasks.pop(request_id, None)
        if grace:
            grace.cancel()
        self.subscribers[request_id].add(connection)

    def unsubscribe(self, connection: SocketConnection) -> None:
        for request_id in list(self.subscribers):
            self.subscribers[request_id].discard(connection)
            if not self.subscribers[request_id]:
                self.subscribers.pop(request_id, None)
                task = self.tasks.get(request_id)
                if task and not task.done() and request_id not in self.grace_tasks:
                    self.grace_tasks[request_id] = asyncio.create_task(
                        self._cancel_after_grace(request_id),
                        name=f"reconnect-grace:{request_id}",
                    )

    async def _cancel_after_grace(self, request_id: str) -> None:
        try:
            await asyncio.sleep(self.settings.reconnect_grace_seconds)
            task = self.tasks.get(request_id)
            if task and not task.done() and not self.subscribers.get(request_id):
                await self.cache.set_cancelled(request_id)
                task.cancel()
        except asyncio.CancelledError:
            pass
        finally:
            self.grace_tasks.pop(request_id, None)

    async def broadcast(self, request_id: str, event: dict[str, Any]) -> None:
        dead = []
        for connection in list(self.subscribers.get(request_id, set())):
            if not await connection.send(event):
                dead.append(connection)
        for connection in dead:
            self.unsubscribe(connection)

    def start(
        self, *, request_id: str, user_id: str, conversation_id: str,
        question: str, file_ids: list[str],
    ) -> None:
        if request_id in self.tasks and not self.tasks[request_id].done():
            return
        self.tasks[request_id] = asyncio.create_task(
            self._run(
                request_id=request_id, user_id=user_id, conversation_id=conversation_id,
                question=question, file_ids=file_ids,
            ),
            name=f"generation:{request_id}",
        )

    async def _append(self, request_id: str, conversation_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = await self.cache.append_event(request_id, {
            "type": event_type, "request_id": request_id,
            "conversation_id": conversation_id, "payload": payload,
        })
        await self.broadcast(request_id, event)
        return event

    async def _run(self, *, request_id: str, user_id: str, conversation_id: str, question: str, file_ids: list[str]) -> None:
        async with self.session_factory() as session:
            jobs = JobRepository(session)
            job = await jobs.owned(user_id, request_id)
            if job is None:
                await self.cache.release_generation(user_id)
                return
            service = ChatService(
                rag=self.rag, deepseek=self.deepseek, messages=MessageRepository(session),
                cache=self.cache, context_turns=self.settings.context_turns,
            )
            try:
                async for domain_event in service.stream_answer(
                    request_id=request_id, user_id=user_id, conversation_id=conversation_id,
                    question=question, file_ids=file_ids,
                ):
                    body = await self._append(request_id, conversation_id, domain_event.type, domain_event.payload)
                    if domain_event.type == "status":
                        await jobs.set_state(job, domain_event.payload["stage"], last_seq=body["seq"])
                    elif domain_event.type == "done":
                        await jobs.set_state(
                            job, "done", last_seq=body["seq"],
                            assistant_message_id=domain_event.payload["message_id"],
                            usage=domain_event.payload.get("usage", {}),
                        )
                    await session.commit()
            except asyncio.CancelledError:
                await session.rollback()
                job = await jobs.owned(user_id, request_id)
                if job and job.status not in {"done", "cancelled", "error"}:
                    body = await self._append(request_id, conversation_id, "cancelled", {"message": "生成已停止"})
                    await jobs.set_state(job, "cancelled", last_seq=body["seq"])
                    await session.commit()
            except Exception:
                await session.rollback()
                job = await jobs.owned(user_id, request_id)
                if job and job.status not in {"done", "cancelled", "error"}:
                    body = await self._append(request_id, conversation_id, "error", {
                        "error_code": "GENERATION_FAILED", "message": "生成失败，请稍后重试",
                    })
                    await jobs.set_state(job, "error", last_seq=body["seq"], error_code="GENERATION_FAILED")
                    await session.commit()
            finally:
                await self.cache.release_generation(user_id)
                self.tasks.pop(request_id, None)
                grace = self.grace_tasks.pop(request_id, None)
                if grace:
                    grace.cancel()

    async def stop(self, user_id: str, request_id: str, conversation_id: str) -> bool:
        async with self.session_factory() as session:
            jobs = JobRepository(session)
            job = await jobs.owned(user_id, request_id)
            if job is None or job.conversation_id != conversation_id:
                return False
            if job.status in {"done", "cancelled", "error"}:
                return True
            await self.cache.set_cancelled(request_id)
            task = self.tasks.get(request_id)
            if task and not task.done():
                task.cancel()
                return True
            body = await self._append(request_id, conversation_id, "cancelled", {"message": "生成已停止"})
            await jobs.set_state(job, "cancelled", last_seq=body["seq"])
            await session.commit()
            return True
