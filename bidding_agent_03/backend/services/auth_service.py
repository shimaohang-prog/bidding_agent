import jwt

from backend.core.config import Settings
from backend.core.errors import ApiError
from backend.core.security import create_token, decode_token, verify_password
from backend.db.models import User
from backend.repositories.users import UserRepository


class AuthService:
    def __init__(self, users: UserRepository, settings: Settings) -> None:
        self.users = users
        self.settings = settings

    async def login(self, username: str, password: str) -> tuple[User, str, str]:
        user = await self.users.by_username(username)
        if user is None or not user.is_active or not verify_password(password, user.password_hash):
            raise ApiError(401, "INVALID_CREDENTIALS", "用户名或密码错误")
        return user, *self._tokens(user)

    async def authenticate(self, token: str | None, kind: str = "access") -> User:
        if not token:
            raise ApiError(401, "AUTH_REQUIRED", "请先登录")
        try:
            payload = decode_token(token, kind=kind, settings=self.settings)  # type: ignore[arg-type]
        except jwt.PyJWTError as exc:
            raise ApiError(401, "INVALID_TOKEN", "登录状态已失效") from exc
        user = await self.users.by_id(str(payload["sub"]))
        if user is None or user.token_version != int(payload.get("ver", -1)):
            raise ApiError(401, "INVALID_TOKEN", "登录状态已失效")
        return user

    async def refresh(self, refresh_token: str | None) -> tuple[User, str, str]:
        user = await self.authenticate(refresh_token, "refresh")
        return user, *self._tokens(user)

    def _tokens(self, user: User) -> tuple[str, str]:
        return (
            create_token(user_id=user.id, token_version=user.token_version, kind="access", settings=self.settings),
            create_token(user_id=user.id, token_version=user.token_version, kind="refresh", settings=self.settings),
        )
