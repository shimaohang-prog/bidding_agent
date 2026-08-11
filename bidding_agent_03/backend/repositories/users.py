from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import User


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def by_username(self, username: str) -> User | None:
        return await self.session.scalar(select(User).where(User.username == username))

    async def by_id(self, user_id: str) -> User | None:
        return await self.session.scalar(select(User).where(User.id == user_id, User.is_active.is_(True)))

    async def create(self, username: str, password_hash: str) -> User:
        user = User(username=username, password_hash=password_hash)
        self.session.add(user)
        await self.session.flush()
        return user

    async def revoke_tokens(self, user_id: str) -> None:
        user = await self.by_id(user_id)
        if user:
            user.token_version += 1
            await self.session.flush()
