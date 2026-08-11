from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Conversation


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def owned(self, user_id: str, conversation_id: str) -> Conversation | None:
        return await self.session.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
                Conversation.is_deleted.is_(False),
            )
        )

    async def list_owned(self, user_id: str, limit: int = 50) -> list[Conversation]:
        rows = await self.session.scalars(
            select(Conversation)
            .where(Conversation.user_id == user_id, Conversation.is_deleted.is_(False))
            .order_by(desc(Conversation.updated_at))
            .limit(limit)
        )
        return list(rows)

    async def create(self, user_id: str, title: str) -> Conversation:
        item = Conversation(user_id=user_id, title=title)
        self.session.add(item)
        await self.session.flush()
        # MySQL server defaults (created_at/updated_at) are not guaranteed to
        # be populated by flush.  Load them before FastAPI serializes the ORM
        # object; otherwise async lazy loading raises MissingGreenlet.
        await self.session.refresh(item)
        return item

    async def rename(self, user_id: str, conversation_id: str, title: str) -> Conversation | None:
        item = await self.owned(user_id, conversation_id)
        if item:
            item.title = title
            await self.session.flush()
            await self.session.refresh(item)
        return item

    async def soft_delete(self, user_id: str, conversation_id: str) -> bool:
        item = await self.owned(user_id, conversation_id)
        if not item:
            return False
        item.is_deleted = True
        await self.session.flush()
        return True
