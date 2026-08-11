from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import current_user, db_session
from backend.db.models import User
from backend.repositories.conversations import ConversationRepository
from backend.repositories.messages import MessageRepository
from backend.schemas.conversation import ConversationCreate, ConversationRename, ConversationView, MessageView
from backend.services.conversation_service import ConversationService


router = APIRouter(prefix="/conversations", tags=["conversations"])


def service(request: Request, session: AsyncSession) -> ConversationService:
    return ConversationService(ConversationRepository(session), MessageRepository(session), request.app.state.cache)


@router.get("", response_model=list[ConversationView])
async def list_conversations(request: Request, user: User = Depends(current_user), session: AsyncSession = Depends(db_session)):
    return await service(request, session).list_conversations(user.id)


@router.post("", response_model=ConversationView, status_code=201)
async def create_conversation(payload: ConversationCreate, request: Request, user: User = Depends(current_user), session: AsyncSession = Depends(db_session)):
    return await service(request, session).create(user.id, payload.title)


@router.patch("/{conversation_id}", response_model=ConversationView)
async def rename_conversation(conversation_id: str, payload: ConversationRename, request: Request, user: User = Depends(current_user), session: AsyncSession = Depends(db_session)):
    return await service(request, session).rename(user.id, conversation_id, payload.title)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, request: Request, user: User = Depends(current_user), session: AsyncSession = Depends(db_session)):
    await service(request, session).delete(user.id, conversation_id)
    return Response(status_code=204)


@router.get("/{conversation_id}/messages", response_model=list[MessageView])
async def messages(
    conversation_id: str, request: Request, user: User = Depends(current_user),
    session: AsyncSession = Depends(db_session), limit: int = Query(default=50, ge=1, le=100),
    before_id: str | None = None,
):
    return await service(request, session).message_page(user.id, conversation_id, limit, before_id)
