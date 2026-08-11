from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import UploadedFile


class FileRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def owned(self, user_id: str, file_id: str) -> UploadedFile | None:
        return await self.session.scalar(
            select(UploadedFile).where(UploadedFile.id == file_id, UploadedFile.user_id == user_id)
        )

    async def list_owned(self, user_id: str, conversation_id: str) -> list[UploadedFile]:
        rows = await self.session.scalars(
            select(UploadedFile).where(
                UploadedFile.user_id == user_id,
                UploadedFile.conversation_id == conversation_id,
            ).order_by(UploadedFile.created_at)
        )
        return list(rows)

    async def by_hash(self, user_id: str, conversation_id: str, sha256: str) -> UploadedFile | None:
        return await self.session.scalar(select(UploadedFile).where(
            UploadedFile.user_id == user_id,
            UploadedFile.conversation_id == conversation_id,
            UploadedFile.sha256 == sha256,
        ))

    async def create(self, **values: object) -> UploadedFile:
        item = UploadedFile(**values)
        self.session.add(item)
        await self.session.flush()
        return item

    async def set_status(self, file_id: str, status: str, *, chunk_count: int = 0, error_code: str | None = None) -> None:
        item = await self.session.get(UploadedFile, file_id)
        if item:
            item.status = status
            item.chunk_count = chunk_count
            item.error_code = error_code
            await self.session.flush()

    async def delete(self, item: UploadedFile) -> None:
        await self.session.delete(item)
        await self.session.flush()
