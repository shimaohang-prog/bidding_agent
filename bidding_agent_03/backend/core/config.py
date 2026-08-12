"""集中、强类型配置；真实密钥只能由运行环境注入。"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "智能招投标问答机器人"
    environment: Literal["development", "test", "production"] = "development"
    api_prefix: str = "/api/v1"
    database_url: str = "mysql+asyncmy://bidding:change-me@127.0.0.1:3306/bidding_agent"
    redis_url: str = "redis://127.0.0.1:6379/0"
    jwt_secret: SecretStr = SecretStr("development-only-change-this-secret")
    access_token_minutes: int = Field(default=15, ge=5, le=120)
    refresh_token_days: int = Field(default=7, ge=1, le=30)
    cookie_secure: bool = True
    cookie_samesite: Literal["strict", "lax", "none"] = "strict"
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    max_question_chars: int = Field(default=6000, ge=100, le=20000)
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1024, le=200 * 1024 * 1024)
    max_active_generations_per_user: int = Field(default=8, ge=1, le=32)
    rate_limit_per_minute: int = Field(default=30, ge=1, le=600)
    websocket_idle_seconds: int = Field(default=90, ge=20, le=600)
    reconnect_grace_seconds: int = Field(default=120, ge=10, le=1800)
    context_turns: int = Field(default=0, ge=0, le=20)
    context_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    stream_ttl_seconds: int = Field(default=900, ge=60, le=86400)
    stream_max_events: int = Field(default=4000, ge=100, le=20000)

    milvus_mode: Literal["lite", "server"] = "lite"
    # Avoid the reserved MILVUS_URI environment variable used internally by
    # PyMilvus for a remote server connection. This path is only for uploaded
    # private documents stored in local Milvus Lite.
    private_milvus_uri: str = "milvus_db/private/main.db"
    api_workers: int = Field(default=1, ge=1, le=32)
    milvus_thread_limit: int = Field(default=4, ge=1, le=16)
    upload_root: Path = PROJECT_ROOT / "uploads"
    private_collection: str = "private_documents"
    file_worker_stream: str = "bidding:file-jobs"
    file_worker_group: str = "file-workers"

    deepseek_api_key: SecretStr = SecretStr("")
    deepseek_url: str = "https://api.deepseek.com/chat/completions"
    answer_model: str = "deepseek-v4-pro"
    answer_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    tavily_api_key: SecretStr = SecretStr("")
    tavily_url: str = "https://api.tavily.com/search"

    @field_validator("upload_root", mode="before")
    @classmethod
    def resolve_upload_root(cls, value: object) -> Path:
        path = Path(str(value)).expanduser()
        return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()

    @model_validator(mode="after")
    def validate_deployment(self) -> "Settings":
        if self.milvus_mode == "lite" and self.api_workers != 1:
            raise ValueError("Milvus Lite 只允许 API_WORKERS=1")
        if self.environment == "production":
            if len(self.jwt_secret.get_secret_value()) < 32:
                raise ValueError("生产环境 JWT_SECRET 至少 32 字符")
            if not self.cookie_secure:
                raise ValueError("生产环境必须启用 Secure Cookie")
            if "change-me" in self.database_url:
                raise ValueError("生产环境必须配置 DATABASE_URL")
        return self

    @property
    def origin_set(self) -> set[str]:
        return {item.strip().rstrip("/") for item in self.allowed_origins.split(",") if item.strip()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
