from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import AwareDatetime, BaseModel

from ..llm import generate_cerebras, generate_groq
from ..rag import MetadataFilters, RAGService


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


def get_metadata_filters(
    title: str | None = None,
    title_text: str | None = None,
    author: list[str] | None = Query(default=None),
    source: list[str] | None = Query(default=None),
    published_from: AwareDatetime | None = None,
    published_to: AwareDatetime | None = None,
    url: str | None = None,
    url_prefix: str | None = None,
) -> MetadataFilters:
    try:
        return MetadataFilters(
            title=title,
            title_text=title_text,
            authors=tuple(author or ()),
            sources=tuple(source or ()),
            published_from=published_from,
            published_to=published_to,
            url=url,
            url_prefix=url_prefix,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/query", response_model=RetrievalResponse, tags=["rag"])
def query(
    q: str,
    rag: Annotated[RAGService, Depends(get_rag_service)],
    metadata_filters: Annotated[MetadataFilters, Depends(get_metadata_filters)],
) -> RetrievalResponse:
    documents = rag.retrieve(q, metadata_filters)
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
