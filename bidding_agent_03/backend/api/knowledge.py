import mimetypes
from pathlib import Path
from typing import Literal

import anyio
from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.api.dependencies import current_user
from backend.core.config import PROJECT_ROOT
from backend.core.errors import ApiError
from backend.db.models import User


KnowledgeCategory = Literal["enterprise", "tender", "product", "laws", "policy"]
PUBLIC_KNOWLEDGE_ROOTS: dict[str, Path] = {
    "enterprise": PROJECT_ROOT / "data" / "csv" / "enterprise.csv",
    "tender": PROJECT_ROOT / "data" / "csv" / "tender.csv",
    "product": PROJECT_ROOT / "data" / "csv" / "product.csv",
    "laws": PROJECT_ROOT / "data" / "laws",
    "policy": PROJECT_ROOT / "data" / "policy",
}
MAX_LISTED_FILES = 5000


class KnowledgeFileView(BaseModel):
    name: str
    relative_path: str
    size_bytes: int
    mime_type: str


router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _category_root(category: str) -> Path:
    root = PUBLIC_KNOWLEDGE_ROOTS.get(category)
    if root is None:
        raise ApiError(404, "KNOWLEDGE_CATEGORY_NOT_FOUND", "知识库不存在")
    return root.resolve()


def _list_public_files(root: Path) -> list[KnowledgeFileView]:
    if not root.exists():
        return []
    paths = [root] if root.is_file() else sorted(
        (path for path in root.rglob("*") if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)),
        key=lambda path: path.as_posix().casefold(),
    )
    items: list[KnowledgeFileView] = []
    for path in paths[:MAX_LISTED_FILES]:
        relative_path = path.name if root.is_file() else path.relative_to(root).as_posix()
        items.append(KnowledgeFileView(
            name=path.name,
            relative_path=relative_path,
            size_bytes=path.stat().st_size,
            mime_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        ))
    return items


def _resolve_public_file(root: Path, relative_path: str) -> Path:
    if root.is_file():
        if relative_path != root.name:
            raise ApiError(404, "KNOWLEDGE_FILE_NOT_FOUND", "知识库文件不存在")
        return root
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ApiError(404, "KNOWLEDGE_FILE_NOT_FOUND", "知识库文件不存在")
    return candidate


@router.get("/{category}/files", response_model=list[KnowledgeFileView])
async def list_knowledge_files(category: KnowledgeCategory, _: User = Depends(current_user)):
    root = _category_root(category)
    return await anyio.to_thread.run_sync(_list_public_files, root)


@router.get("/{category}/open")
async def open_knowledge_file(
    category: KnowledgeCategory,
    path: str = Query(min_length=1, max_length=1000),
    _: User = Depends(current_user),
):
    target = _resolve_public_file(_category_root(category), path)
    return FileResponse(
        target,
        media_type=mimetypes.guess_type(target.name)[0] or "application/octet-stream",
        filename=target.name,
        content_disposition_type="inline",
        headers={"X-Content-Type-Options": "nosniff"},
    )
