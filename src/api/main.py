from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel

from ..llm import generate_cerebras, generate_groq
from ..rag import RAGService


class DocumentResponse(BaseModel):
    content: str
    metadata: dict[str, object]


class RetrievalResponse(BaseModel):
    query: str
    documents: list[DocumentResponse]


class GenerateRequest(BaseModel):
    prompt: str


class GenerateResponse(BaseModel):
    response: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.rag_service = RAGService()
    yield


app = FastAPI(
    title="MultiHop RAG API",
    version="0.1.0",
    lifespan=lifespan,
)


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


@app.get("/query", response_model=RetrievalResponse, tags=["rag"])
def query(
    q: str,
    rag: Annotated[RAGService, Depends(get_rag_service)],
) -> RetrievalResponse:
    documents = rag.retrieve(q)
    return RetrievalResponse(
        query=q,
        documents=[
            DocumentResponse(content=doc.page_content, metadata=doc.metadata)
            for doc in documents
        ],
    )


@app.post("/generate", response_model=GenerateResponse, tags=["rag"])
def generate_response(
    request: GenerateRequest,
) -> GenerateResponse:
    return GenerateResponse(response=generate_groq(request.prompt))


@app.post("/generate/cerebras", response_model=GenerateResponse, tags=["rag"])
def generate_cerebras_response(request: GenerateRequest) -> GenerateResponse:
    return GenerateResponse(response=generate_cerebras(request.prompt))
