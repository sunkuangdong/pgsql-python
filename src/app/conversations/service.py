from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.conversations.model import Conversation
from app.conversations.schema import (
    ConversationMessagesResponse,
    ConversationResponse,
    MessageResponse,
    SemanticSearchResult,
    UserConversationsResponse,
)
from app.embeddings.service import EmbeddingService
from app.messages.model import Message
from app.users.model import User


class ConversationsService:
    def __init__(
        self,
        session: AsyncSession,
        embedding_service: EmbeddingService,
    ):
        self.session = session
        self.embedding_service = embedding_service

    async def find_conversations_by_user_id(
        self,
        user_id: int,
    ) -> UserConversationsResponse:
        """查询用户及其会话列表"""

        user = await self.session.get(User, user_id)

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User #{user_id} not found",
            )

        statement = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.created_at.desc())
        )

        result = await self.session.scalars(statement)
        conversations = list(result.all())

        return UserConversationsResponse(
            id=user.id,
            name=user.name,
            created_at=user.created_at,
            conversations=[
                ConversationResponse.model_validate(conversation)
                for conversation in conversations
            ],
        )

    async def find_messages_by_conversation_id(
        self,
        conversation_id: int,
    ) -> ConversationMessagesResponse:
        """查询会话及其消息列表"""

        conversation = await self.session.get(
            Conversation,
            conversation_id,
        )

        if conversation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation #{conversation_id} not found",
            )

        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )

        result = await self.session.scalars(statement)
        messages = list(result.all())

        return ConversationMessagesResponse(
            id=conversation.id,
            user_id=conversation.user_id,
            title=conversation.title,
            created_at=conversation.created_at,
            messages=[
                MessageResponse.model_validate(message)
                for message in messages
            ],
        )

    async def search_similar_messages(
        self,
        conversation_id: int,
        search_text: str,
        limit: int = 5,
    ) -> list[SemanticSearchResult]:
        """在指定会话中进行语义搜索"""

        conversation = await self.session.get(
            Conversation,
            conversation_id,
        )

        if conversation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation #{conversation_id} not found",
            )

        vector = await self.embedding_service.embed_query(search_text)

        distance = Message.embedding.cosine_distance(vector)

        statement = (
            select(
                Message.id,
                Message.conversation_id,
                Message.role,
                Message.content,
                Message.created_at,
                (1 - distance).label("similarity"),
            )
            .where(
                Message.conversation_id == conversation_id,
                Message.embedding.is_not(None),
            )
            .order_by(distance)
            .limit(limit)
        )

        result = await self.session.execute(statement)
        rows = result.mappings().all()

        return [
            SemanticSearchResult.model_validate(dict(row))
            for row in rows
        ]
