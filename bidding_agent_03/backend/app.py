"""FastAPI 应用工厂与共享资源生命周期。"""

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

import anyio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from backend.api import auth, conversations, files, health, websocket
from backend.cache.store import RedisCacheStore
from backend.core.config import Settings, get_settings
from backend.core.errors import install_error_handlers
from backend.db.session import create_engine_and_sessionmaker
from backend.services.deepseek_stream import DeepSeekStreamClient
from backend.services.generation_manager import GenerationManager
from backend.services.private_documents import PrivateDocumentStore
from backend.services.qa_pipeline import create_upstream_http_client
from backend.services.rag_service import AsyncRAGService
from backend.services.tavily_client import AsyncTavilyClient


logger = logging.getLogger("bidding_agent")


def create_app(settings_override: Settings | None = None) -> FastAPI:
    settings = settings_override or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.upload_root.mkdir(parents=True, exist_ok=True)
        engine, session_factory = create_engine_and_sessionmaker(settings.database_url)
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        http = create_upstream_http_client()
        limiter = anyio.CapacityLimiter(settings.milvus_thread_limit)
        cache = RedisCacheStore(
            redis, context_ttl=settings.context_ttl_seconds,
            stream_ttl=settings.stream_ttl_seconds,
            stream_max_events=settings.stream_max_events,
            file_stream=settings.file_worker_stream,
        )
        private_store = PrivateDocumentStore(
            settings.private_milvus_uri,
            settings.private_collection,
            limiter,
        )
        tavily = AsyncTavilyClient(
            http, url=settings.tavily_url,
            api_key=settings.tavily_api_key.get_secret_value(),
        )
        rag = AsyncRAGService(limiter=limiter, tavily=tavily, private_store=private_store)
        deepseek = DeepSeekStreamClient(
            http, url=settings.deepseek_url,
            api_key=settings.deepseek_api_key.get_secret_value(),
            model=settings.answer_model,
            temperature=settings.answer_temperature,
        )
        manager = GenerationManager(
            session_factory=session_factory, cache=cache, rag=rag,
            deepseek=deepseek, settings=settings,
        )
        app.state.settings = settings
        app.state.engine = engine
        app.state.session_factory = session_factory
        app.state.redis = redis
        app.state.cache = cache
        app.state.http = http
        app.state.private_store = private_store
        app.state.rag = rag
        app.state.deepseek = deepseek
        app.state.generation_manager = manager
        try:
            yield
        finally:
            for task in list(manager.tasks.values()):
                task.cancel()
            for task in list(manager.grace_tasks.values()):
                task.cancel()
            await http.aclose()
            await redis.aclose()
            await engine.dispose()

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(settings.origin_set),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info("request_complete request_id=%s method=%s path=%s status=%s", request_id, request.method, request.url.path, response.status_code)
        return response

    for router in (health.router, auth.router, conversations.router, files.router, websocket.router):
        app.include_router(router, prefix=settings.api_prefix)
    install_error_handlers(app)
    return app


app = create_app()
