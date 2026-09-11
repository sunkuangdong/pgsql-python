from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.messages.model import MessageRole


class ConversationResponse(BaseModel):
    """会话基本信息"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    title: str | None
    created_at: datetime


class MessageResponse(BaseModel):
    """消息基本信息，不返回 embedding"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: int
    role: MessageRole
    content: str
    created_at: datetime


class UserConversationsResponse(BaseModel):
    """用户以及他的会话列表"""

    id: int
    name: str
    created_at: datetime
    conversations: list[ConversationResponse]


class ConversationMessagesResponse(BaseModel):
    """会话以及它的消息列表"""

    id: int
    user_id: int
    title: str | None
    created_at: datetime
    messages: list[MessageResponse]


class SemanticSearchRequest(BaseModel):
    """语义搜索请求参数"""

    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)


class SemanticSearchResult(BaseModel):
    """单条语义搜索结果"""

    id: int
    conversation_id: int
    role: MessageRole
    content: str
    created_at: datetime
    similarity: float