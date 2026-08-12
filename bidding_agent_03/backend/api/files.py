from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import current_user, db_session
from backend.core.errors import ApiError
from backend.db.models import User
from backend.repositories.conversations import ConversationRepository
from backend.repositories.files import FileRepository
from backend.schemas.files import FileView
from backend.services.upload_service import UploadService


router = APIRouter(prefix="/files", tags=["files"])


def service(request: Request, session: AsyncSession) -> UploadService:
    return UploadService(
        files=FileRepository(session), conversations=ConversationRepository(session),
        cache=request.app.state.cache, private_store=request.app.state.private_store,
        settings=request.app.state.settings,
    )


@router.post("", response_model=FileView, status_code=202)
async def upload_file(
    request: Request, conversation_id: str = Form(...), upload: UploadFile = File(...),
    user: User = Depends(current_user), session: AsyncSession = Depends(db_session),
):
    return await service(request, session).upload(user.id, conversation_id, upload)


@router.get("", response_model=list[FileView])
async def list_files(
    conversation_id: str, user: User = Depends(current_user), session: AsyncSession = Depends(db_session),
):
    if await ConversationRepository(session).owned(user.id, conversation_id) is None:
        raise ApiError(404, "CONVERSATION_NOT_FOUND", "会话不存在")
    return await FileRepository(session).list_owned(user.id, conversation_id)


@router.get("/{file_id}/content")
async def open_uploaded_file(
    file_id: str, request: Request,
    user: User = Depends(current_user), session: AsyncSession = Depends(db_session),
):
    item = await FileRepository(session).owned(user.id, file_id)
    if item is None:
        raise ApiError(404, "FILE_NOT_FOUND", "文件不存在")
    path = service(request, session)._path(item.stored_name)
    if not path.is_file():
        raise ApiError(404, "FILE_NOT_FOUND", "文件不存在")
    return FileResponse(
        path,
        media_type=item.mime_type,
        filename=item.original_name,
        content_disposition_type="inline",
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.get("/{file_id}", response_model=FileView)
async def file_status(file_id: str, user: User = Depends(current_user), session: AsyncSession = Depends(db_session)):
    item = await FileRepository(session).owned(user.id, file_id)
    if item is None:
        raise ApiError(404, "FILE_NOT_FOUND", "文件不存在")
    return item


@router.delete("/{file_id}", status_code=204)
async def delete_file(file_id: str, request: Request, user: User = Depends(current_user), session: AsyncSession = Depends(db_session)):
    await service(request, session).delete(user.id, file_id)
    return Response(status_code=204)
