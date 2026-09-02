import json
from dataclasses import replace
from datetime import datetime
from functools import cache

from langchain_core.documents import Document

from src.rag.config import Settings
from src.rag.retrieval import MetadataFilters
from src.rag.service import RAGService


@cache
def get_rag_service() -> RAGService:
    return RAGService()


def parse_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def build_filters(args: dict[str, object]) -> MetadataFilters:
    return MetadataFilters(
        title=args.get("title"),
        title_text=args.get("title_text"),
        authors=tuple(args.get("authors", [])),
        sources=tuple(args.get("sources", [])),
        published_from=parse_datetime(args.get("published_from")),
        published_to=parse_datetime(args.get("published_to")),
        url=args.get("url"),
        url_prefix=args.get("url_prefix"),
    )


@cache
def known_sources() -> tuple[str, ...]:
    records = json.loads(Settings().corpus_path.read_text(encoding="utf-8"))
    return tuple(sorted({record["source"] for record in records}))


def sources_named_in(query: str) -> tuple[str, ...]:
    query = query.casefold()
    return tuple(source for source in known_sources() if source.casefold() in query)


def unique_documents(documents: list[Document]) -> list[Document]:
    unique = {}
    for document in documents:
        key = (document.metadata.get("doc_id"), document.page_content)
        unique[key] = document
    return list(unique.values())


def retrieve_documents(
    rag: RAGService,
    query: str,
    filters: MetadataFilters,
) -> list[Document]:
    if len(filters.sources) < 2:
        return unique_documents(rag.retrieve(query, filters))

    documents = []
    for source in filters.sources:
        source_filter = replace(filters, sources=(source,))
        documents.extend(rag.retrieve(query, source_filter))
    return unique_documents(documents)
