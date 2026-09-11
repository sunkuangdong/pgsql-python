import asyncio

from app.database import SessionLocal, engine
from app.embeddings.service import EmbeddingService
from app.users.model import User
from app.conversations.model import Conversation
from app.messages.model import Message, MessageRole


SEED_MESSAGES = [
    (
        MessageRole.USER,
        "PostgreSQL 支持哪些数据类型？",
    ),
    (
        MessageRole.ASSISTANT,
        "PostgreSQL 支持整数、文本、JSON、数组，以及 pgvector 提供的向量类型。",
    ),
    (
        MessageRole.USER,
        "如何做向量相似度搜索？",
    ),
    (
        MessageRole.ASSISTANT,
        "可以使用 pgvector 的余弦距离运算符，并通过 HNSW 索引加速检索。",
    ),
]


async def main():
    embedding_service = EmbeddingService()

    try:
        embedded_messages = []

        for role, content in SEED_MESSAGES:
            print(f"正在生成向量：{content}")

            vector = await embedding_service.embed_query(content)

            embedded_messages.append(
                (role, content, vector)
            )

        async with SessionLocal() as session:
            async with session.begin():
                user = User(name="张三")
                session.add(user)
                await session.flush()

                conversation = Conversation(
                    user_id=user.id,
                    title="PostgreSQL 学习",
                )
                session.add(conversation)
                await session.flush()

                for role, content, vector in embedded_messages:
                    message = Message(
                        conversation_id=conversation.id,
                        role=role,
                        content=content,
                        embedding=vector,
                    )
                    session.add(message)

            print()
            print("测试数据创建成功")
            print("USER_ID =", user.id)
            print("CONVERSATION_ID =", conversation.id)

    finally:
        await embedding_service.close()
        await engine.dispose()


asyncio.run(main())