from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.conversations.router import router
from app.database import engine
from app.embeddings.service import EmbeddingService


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_service = EmbeddingService()
    app.state.embedding_service = embedding_service

    try:
        yield
    finally:
        await embedding_service.close()
        await engine.dispose()


app = FastAPI(
    title="PostgreSQL API",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/")
async def root():
    return {"message": "服务启动成功"}