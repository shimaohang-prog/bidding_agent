from fastapi import APIRouter, Cookie, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import current_user, db_session
from backend.core.security import clear_auth_cookies, set_auth_cookies
from backend.db.models import User
from backend.repositories.users import UserRepository
from backend.schemas.auth import LoginRequest, UserView
from backend.services.auth_service import AuthService


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=UserView)
async def login(payload: LoginRequest, request: Request, response: Response, session: AsyncSession = Depends(db_session)) -> User:
    user, access, refresh = await AuthService(UserRepository(session), request.app.state.settings).login(payload.username, payload.password)
    set_auth_cookies(response, access, refresh, request.app.state.settings)
    return user


@router.post("/refresh", response_model=UserView)
async def refresh(
    request: Request, response: Response, session: AsyncSession = Depends(db_session),
    refresh_token: str | None = Cookie(default=None),
) -> User:
    user, access, new_refresh = await AuthService(UserRepository(session), request.app.state.settings).refresh(refresh_token)
    set_auth_cookies(response, access, new_refresh, request.app.state.settings)
    return user


@router.post("/logout", status_code=204)
async def logout(
    request: Request, response: Response, user: User = Depends(current_user),
    session: AsyncSession = Depends(db_session),
) -> Response:
    await UserRepository(session).revoke_tokens(user.id)
    clear_auth_cookies(response, request.app.state.settings)
    response.status_code = 204
    return response


@router.get("/me", response_model=UserView)
async def me(user: User = Depends(current_user)) -> User:
    return user
