"""安全上传、可靠入队和同步删除。"""

import hashlib
import re
import secrets
from pathlib import Path

import anyio
from fastapi import UploadFile

from backend.cache.store import CacheStore
from backend.core.config import Settings
from backend.core.errors import ApiError
from backend.db.models import UploadedFile
from backend.repositories.conversations import ConversationRepository
from backend.repositories.files import FileRepository
from backend.services.private_documents import PrivateDocumentStore


ALLOWED_FILES = {
    ".pdf": {"application/pdf"},
    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/plain"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
}


class UploadService:
    def __init__(
        self, *, files: FileRepository, conversations: ConversationRepository,
        cache: CacheStore, private_store: PrivateDocumentStore, settings: Settings,
    ) -> None:
        self.files = files
        self.conversations = conversations
        self.cache = cache
        self.private_store = private_store
        self.settings = settings

    def _path(self, stored_name: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}\.[a-z0-9]+", stored_name):
            raise ApiError(400, "INVALID_FILE_PATH", "文件标识无效")
        root = self.settings.upload_root.resolve()
        path = (root / stored_name).resolve()
        if not path.is_relative_to(root):
            raise ApiError(400, "INVALID_FILE_PATH", "文件路径无效")
        return path

    @staticmethod
    def _validate_magic(extension: str, first: bytes) -> None:
        if extension == ".pdf" and not first.startswith(b"%PDF-"):
            raise ApiError(400, "FILE_CONTENT_MISMATCH", "文件内容与 PDF 类型不符")
        if extension == ".docx" and not first.startswith(b"PK"):
            raise ApiError(400, "FILE_CONTENT_MISMATCH", "文件内容与 DOCX 类型不符")

    async def upload(self, user_id: str, conversation_id: str, upload: UploadFile) -> UploadedFile:
        if await self.conversations.owned(user_id, conversation_id) is None:
            raise ApiError(404, "CONVERSATION_NOT_FOUND", "会话不存在")
        original = Path(upload.filename or "").name
        extension = Path(original).suffix.lower()
        if extension not in ALLOWED_FILES or (upload.content_type or "") not in ALLOWED_FILES[extension]:
            raise ApiError(415, "UNSUPPORTED_FILE_TYPE", "仅支持 PDF、TXT、Markdown 和 DOCX")
        stored_name = f"{secrets.token_hex(16)}{extension}"
        path = self._path(stored_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        first = b""
        try:
            async with await anyio.open_file(path, "xb") as handle:
                while chunk := await upload.read(1024 * 1024):
                    if not first:
                        first = chunk[:16]
                    size += len(chunk)
                    if size > self.settings.max_upload_bytes:
                        raise ApiError(413, "FILE_TOO_LARGE", "上传文件超过大小限制")
                    digest.update(chunk)
                    await handle.write(chunk)
            self._validate_magic(extension, first)
            sha256 = digest.hexdigest()
            duplicate = await self.files.by_hash(user_id, conversation_id, sha256)
            if duplicate:
                await anyio.to_thread.run_sync(path.unlink, True)
                return duplicate
            item = await self.files.create(
                user_id=user_id, conversation_id=conversation_id, original_name=original[:255],
                stored_name=stored_name, mime_type=upload.content_type or "application/octet-stream",
                size_bytes=size, sha256=sha256, status="queued",
            )
            await self.cache.enqueue_file({"file_id": item.id})
            return item
        except Exception:
            if path.exists():
                await anyio.to_thread.run_sync(path.unlink, True)
            raise
        finally:
            await upload.close()

    async def delete(self, user_id: str, file_id: str) -> None:
        item = await self.files.owned(user_id, file_id)
        if item is None:
            raise ApiError(404, "FILE_NOT_FOUND", "文件不存在")
        await self.private_store.delete_file(user_id, item.conversation_id, item.id)
        path = self._path(item.stored_name)
        if path.exists():
            await anyio.to_thread.run_sync(path.unlink, True)
        await self.cache.delete_conversation(user_id, item.conversation_id)
        await self.files.delete(item)
