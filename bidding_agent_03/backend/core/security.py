"""密码、短期访问令牌与 Cookie 策略。"""

from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import UUID

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHash, VerifyMismatchError
from fastapi import Response

from backend.core.config import Settings


_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHash):
        return False


def create_token(
    *, user_id: UUID | str, token_version: int, kind: Literal["access", "refresh"], settings: Settings
) -> str:
    now = datetime.now(UTC)
    lifetime = timedelta(minutes=settings.access_token_minutes) if kind == "access" else timedelta(days=settings.refresh_token_days)
    payload: dict[str, Any] = {
        "sub": str(user_id), "typ": kind, "ver": token_version,
        "iat": now, "exp": now + lifetime,
    }
    return jwt.encode(payload, settings.jwt_secret.get_secret_value(), algorithm="HS256")


def decode_token(token: str, *, kind: Literal["access", "refresh"], settings: Settings) -> dict[str, Any]:
    payload = jwt.decode(token, settings.jwt_secret.get_secret_value(), algorithms=["HS256"])
    if payload.get("typ") != kind:
        raise jwt.InvalidTokenError("令牌类型错误")
    UUID(str(payload["sub"]))
    return payload


def set_auth_cookies(response: Response, access: str, refresh: str, settings: Settings) -> None:
    common = {"httponly": True, "secure": settings.cookie_secure, "samesite": settings.cookie_samesite, "path": "/"}
    response.set_cookie("access_token", access, max_age=settings.access_token_minutes * 60, **common)
    response.set_cookie("refresh_token", refresh, max_age=settings.refresh_token_days * 86400, **common)


def clear_auth_cookies(response: Response, settings: Settings) -> None:
    for name in ("access_token", "refresh_token"):
        response.delete_cookie(name, path="/", secure=settings.cookie_secure, httponly=True, samesite=settings.cookie_samesite)
