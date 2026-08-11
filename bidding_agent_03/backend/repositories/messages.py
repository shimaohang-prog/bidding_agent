from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Message, MessageCitation


class MessageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self, *, user_id: str, conversation_id: str, role: str, content: str,
        request_id: str | None = None, client_message_id: str | None = None,
    ) -> Message:
        item = Message(
            user_id=user_id, conversation_id=conversation_id, role=role, content=content,
            request_id=request_id, client_message_id=client_message_id,
        )
        self.session.add(item)
        await self.session.flush()
        return item

    async def recent_turns(self, user_id: str, conversation_id: str, turns: int) -> list[Message]:
        rows = await self.session.scalars(
            select(Message)
            .where(Message.user_id == user_id, Message.conversation_id == conversation_id)
            .order_by(desc(Message.created_at), desc(Message.id))
            .limit(turns * 2)
        )
        return list(reversed(list(rows)))

    async def page(
        self, user_id: str, conversation_id: str, *, limit: int = 50, before_id: str | None = None,
    ) -> list[Message]:
        statement = select(Message).where(
            Message.user_id == user_id, Message.conversation_id == conversation_id
        )
        if before_id:
            pivot = await self.session.get(Message, before_id)
            if pivot is None or pivot.user_id != user_id or pivot.conversation_id != conversation_id:
                return []
            statement = statement.where(Message.created_at <= pivot.created_at, Message.id != pivot.id)
        rows = await self.session.scalars(statement.order_by(desc(Message.created_at), desc(Message.id)).limit(limit))
        return list(reversed(list(rows)))

    async def add_citations(self, message_id: str, citations: list[dict]) -> None:
        for item in citations:
            self.session.add(MessageCitation(
                message_id=message_id,
                evidence_id=item["evidence_id"], source_type=item["source_type"],
                category=item["category"], title=item.get("title", "")[:500],
                source_url=item.get("url"), source_id=item["source_id"][:500],
                metadata_json=item.get("metadata", {}),
            ))
        await self.session.flush()
