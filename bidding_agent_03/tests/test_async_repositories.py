from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.core.security import hash_password
from backend.db.base import Base
from backend.repositories.conversations import ConversationRepository
from backend.repositories.jobs import JobRepository
from backend.repositories.messages import MessageRepository
from backend.repositories.users import UserRepository


@pytest.mark.asyncio
async def test_conversation_create_refreshes_server_defaults():
    session = MagicMock(spec=AsyncSession)
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    repository = ConversationRepository(session)

    conversation = await repository.create("user-1", "测试会话")

    session.add.assert_called_once_with(conversation)
    session.flush.assert_awaited_once_with()
    session.refresh.assert_awaited_once_with(conversation)


@pytest.mark.asyncio
async def test_repository_ownership_and_idempotent_job(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        users = UserRepository(session)
        user_a = await users.create("alice", hash_password("password-a"))
        user_b = await users.create("bob", hash_password("password-b"))
        await session.commit()
        conversations = ConversationRepository(session)
        conversation = await conversations.create(user_a.id, "测试")
        await session.commit()
        assert await conversations.owned(user_a.id, conversation.id)
        assert await conversations.owned(user_b.id, conversation.id) is None
        message = await MessageRepository(session).create(
            user_id=user_a.id, conversation_id=conversation.id, role="user", content="问题"
        )
        await session.commit()
        jobs = JobRepository(session)
        first, created = await jobs.create(
            request_id="00000000-0000-0000-0000-000000000001", user_id=user_a.id,
            conversation_id=conversation.id, user_message_id=message.id,
        )
        await session.commit()
        second, created_again = await jobs.create(
            request_id=first.request_id, user_id=user_a.id,
            conversation_id=conversation.id, user_message_id=message.id,
        )
        assert created and not created_again and second.id == first.id
        assert await jobs.owned(user_b.id, first.request_id) is None
    await engine.dispose()
