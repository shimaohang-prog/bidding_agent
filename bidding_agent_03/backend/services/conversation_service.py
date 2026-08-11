from backend.cache.store import CacheStore
from backend.core.errors import ApiError
from backend.db.models import Conversation, Message
from backend.repositories.conversations import ConversationRepository
from backend.repositories.messages import MessageRepository


class ConversationService:
    def __init__(self, conversations: ConversationRepository, messages: MessageRepository, cache: CacheStore) -> None:
        self.conversations = conversations
        self.messages = messages
        self.cache = cache

    async def require_owned(self, user_id: str, conversation_id: str) -> Conversation:
        item = await self.conversations.owned(user_id, conversation_id)
        if item is None:
            raise ApiError(404, "CONVERSATION_NOT_FOUND", "会话不存在")
        return item

    async def list_conversations(self, user_id: str) -> list[Conversation]:
        return await self.conversations.list_owned(user_id)

    async def create(self, user_id: str, title: str) -> Conversation:
        return await self.conversations.create(user_id, " ".join(title.split()))

    async def rename(self, user_id: str, conversation_id: str, title: str) -> Conversation:
        item = await self.conversations.rename(user_id, conversation_id, " ".join(title.split()))
        if item is None:
            raise ApiError(404, "CONVERSATION_NOT_FOUND", "会话不存在")
        return item

    async def delete(self, user_id: str, conversation_id: str) -> None:
        if not await self.conversations.soft_delete(user_id, conversation_id):
            raise ApiError(404, "CONVERSATION_NOT_FOUND", "会话不存在")
        await self.cache.delete_conversation(user_id, conversation_id)

    async def message_page(self, user_id: str, conversation_id: str, limit: int, before_id: str | None) -> list[Message]:
        await self.require_owned(user_id, conversation_id)
        return await self.messages.page(user_id, conversation_id, limit=limit, before_id=before_id)
