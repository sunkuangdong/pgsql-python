from openai import AsyncOpenAI

from app.config import settings


class EmbeddingService:
    def __init__(self):
        if settings.openai_api_key is None:
            raise RuntimeError("没有配置 OPENAI_API_KEY")

        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.openai_base_url,
        )

    async def embed_query(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=settings.embedding_model,
            input=text,
            dimensions=settings.embedding_dimensions,
            encoding_format="float",
        )

        return response.data[0].embedding