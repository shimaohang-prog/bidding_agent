from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from backend.db.models import GenerationJob


ACTIVE_STATUSES = ("accepted", "planning", "retrieving", "reranking", "generating")


class JobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def by_request(self, request_id: str) -> GenerationJob | None:
        return await self.session.scalar(select(GenerationJob).where(GenerationJob.request_id == request_id))

    async def owned(self, user_id: str, request_id: str) -> GenerationJob | None:
        return await self.session.scalar(select(GenerationJob).where(
            GenerationJob.request_id == request_id, GenerationJob.user_id == user_id
        ))

    async def active_count(self, user_id: str) -> int:
        value = await self.session.scalar(select(func.count()).select_from(GenerationJob).where(
            GenerationJob.user_id == user_id, GenerationJob.status.in_(ACTIVE_STATUSES)
        ))
        return int(value or 0)

    async def create(self, *, request_id: str, user_id: str, conversation_id: str, user_message_id: str) -> tuple[GenerationJob, bool]:
        existing = await self.by_request(request_id)
        if existing:
            return existing, False
        item = GenerationJob(
            request_id=request_id, user_id=user_id, conversation_id=conversation_id,
            user_message_id=user_message_id, status="accepted", started_at=datetime.now(UTC),
        )
        self.session.add(item)
        try:
            await self.session.flush()
            return item, True
        except IntegrityError:
            await self.session.rollback()
            existing = await self.by_request(request_id)
            if existing is None:
                raise
            return existing, False

    async def set_state(
        self, item: GenerationJob, status: str, *, last_seq: int | None = None,
        assistant_message_id: str | None = None, usage: dict | None = None, error_code: str | None = None,
    ) -> None:
        item.status = status
        if last_seq is not None:
            item.last_seq = last_seq
        if assistant_message_id is not None:
            item.assistant_message_id = assistant_message_id
        if usage is not None:
            item.usage_json = usage
        item.error_code = error_code
        if status in {"done", "cancelled", "error"}:
            item.finished_at = datetime.now(UTC)
        await self.session.flush()
