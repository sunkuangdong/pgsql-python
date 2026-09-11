from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.conversations.schema import (
    ConversationMessagesResponse,
    SemanticSearchRequest,
    SemanticSearchResult,
    UserConversationsResponse,
)
from app.conversations.service import ConversationsService
from app.database import get_session
from app.embeddings.service import EmbeddingService


router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
)


SessionDep = Annotated[
    AsyncSession,
    Depends(get_session),
]


def get_conversations_service(
    request: Request,
    session: SessionDep,
) -> ConversationsService:
    embedding_service: EmbeddingService = (
        request.app.state.embedding_service
    )

    return ConversationsService(
        session=session,
        embedding_service=embedding_service,
    )


ServiceDep = Annotated[
    ConversationsService,
    Depends(get_conversations_service),
]


@router.get(
    "/users/{user_id}",
    response_model=UserConversationsResponse,
)
async def find_by_user(
    user_id: int,
    service: ServiceDep,
):
    return await service.find_conversations_by_user_id(user_id)


@router.get(
    "/{conversation_id}/messages",
    response_model=ConversationMessagesResponse,
)
async def find_messages(
    conversation_id: int,
    service: ServiceDep,
):
    return await service.find_messages_by_conversation_id(
        conversation_id
    )


@router.post(
    "/{conversation_id}/search",
    response_model=list[SemanticSearchResult],
)
async def search_messages(
    conversation_id: int,
    body: SemanticSearchRequest,
    service: ServiceDep,
):
    return await service.search_similar_messages(
        conversation_id=conversation_id,
        search_text=body.query,
        limit=body.limit,
    )
