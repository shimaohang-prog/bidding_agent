"""运行方式：python -m backend.workers.file_worker。"""

import asyncio
import io
import json
import socket
from pathlib import Path

from docx import Document
from pypdf import PdfReader
from redis.asyncio import Redis

from backend.core.config import get_settings
from backend.db.models import UploadedFile
from backend.db.session import create_engine_and_sessionmaker
from backend.repositories.files import FileRepository
from backend.services.private_documents import PrivateDocumentStore


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("TEXT_ENCODING_ERROR")
    if suffix == ".pdf":
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix == ".docx":
        return "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    raise ValueError("UNSUPPORTED_FILE_TYPE")


def split_chunks(text: str, size: int = 1200, overlap: int = 120) -> list[str]:
    clean = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not clean:
        return []
    return [clean[start : start + size] for start in range(0, len(clean), max(1, size - overlap))]


async def run() -> None:
    settings = get_settings()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    engine, factory = create_engine_and_sessionmaker(settings.database_url)
    import anyio
    store = PrivateDocumentStore(
        settings.private_milvus_uri,
        settings.private_collection,
        anyio.CapacityLimiter(1),
    )
    try:
        try:
            await redis.xgroup_create(settings.file_worker_stream, settings.file_worker_group, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        consumer = f"{socket.gethostname()}-{id(asyncio.current_task())}"
        while True:
            # 先接管崩溃 Worker 留下的超时 pending 消息，再读取新消息。
            claimed = await redis.xautoclaim(
                settings.file_worker_stream, settings.file_worker_group, consumer,
                min_idle_time=60_000, start_id="0-0", count=1,
            )
            rows = [(settings.file_worker_stream, claimed[1])] if claimed[1] else await redis.xreadgroup(
                settings.file_worker_group, consumer, {settings.file_worker_stream: ">"}, count=1, block=5000
            )
            if not rows:
                continue
            for _, entries in rows:
                for stream_id, fields in entries:
                    payload = json.loads(fields["payload"])
                    async with factory() as session:
                        files = FileRepository(session)
                        item = await session.get(UploadedFile, payload["file_id"])
                        if item is None:
                            await redis.xack(settings.file_worker_stream, settings.file_worker_group, stream_id)
                            continue
                        try:
                            await files.set_status(item.id, "processing")
                            await session.commit()
                            path = (settings.upload_root / item.stored_name).resolve()
                            if not path.is_relative_to(settings.upload_root.resolve()):
                                raise ValueError("INVALID_FILE_PATH")
                            chunks = split_chunks(extract_text(path))
                            if not chunks:
                                raise ValueError("EMPTY_DOCUMENT")
                            count = await asyncio.to_thread(
                                store.index_chunks, user_id=item.user_id, conversation_id=item.conversation_id,
                                file_id=item.id, original_name=item.original_name, chunks=chunks,
                            )
                            await files.set_status(item.id, "ready", chunk_count=count)
                            await session.commit()
                            await redis.xack(settings.file_worker_stream, settings.file_worker_group, stream_id)
                        except Exception as exc:
                            await session.rollback()
                            item = await session.get(UploadedFile, payload["file_id"])
                            if item:
                                code = str(exc) if str(exc).isupper() else "FILE_PROCESSING_FAILED"
                                await files.set_status(item.id, "failed", error_code=code[:64])
                                await session.commit()
                            await redis.xack(settings.file_worker_stream, settings.file_worker_group, stream_id)
    finally:
        await redis.aclose()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run())
