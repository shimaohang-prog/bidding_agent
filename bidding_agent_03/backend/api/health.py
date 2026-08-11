from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text


router = APIRouter(tags=["health"])


@router.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
async def ready(request: Request):
    checks = {"mysql": False, "redis": False}
    try:
        async with request.app.state.session_factory() as session:
            await session.execute(text("SELECT 1"))
            checks["mysql"] = True
    except Exception:
        pass
    try:
        checks["redis"] = bool(await request.app.state.redis.ping())
    except Exception:
        pass
    status = 200 if all(checks.values()) else 503
    return JSONResponse(status_code=status, content={"status": "ready" if status == 200 else "not_ready", "checks": checks})
