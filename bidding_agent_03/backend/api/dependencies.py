from collections.abc import AsyncIterator

from fastapi import Cookie, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.cache.store import CacheStore
from backend.core.config import Settings
from backend.db.models import User
from backend.repositories.users import UserRepository
from backend.services.auth_service import AuthService


def settings(request: Request) -> Settings:
    return request.app.state.settings


def cache(request: Request) -> CacheStore:
    return request.app.state.cache


async def db_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def current_user(
    request: Request,
    session: AsyncSession = Depends(db_session),
    access_token: str | None = Cookie(default=None),
) -> User:
    service = AuthService(UserRepository(session), request.app.state.settings)
    return await service.authenticate(access_token)
