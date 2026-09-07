import json
import re
from dataclasses import dataclass
from datetime import date
from functools import cache
from time import perf_counter

from langchain_core.documents import Document

from src.observability.tracing import chunk_metadata, enabled, observation
from src.rag.config import Settings
from src.rag.reranker import DocumentReranker
from src.rag.retrieval import MetadataFilters
from src.rag.service import RAGService


@cache
def get_rag_service() -> RAGService:
    return RAGService()


def parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value
    ):
        raise ValueError("Publication bounds must use YYYY-MM-DD")
    return date.fromisoformat(value)


def build_filters(args: dict[str, object]) -> MetadataFilters:
    source = args.get("source")
    if source is not None and (not isinstance(source, str) or not source.strip()):
        raise ValueError("source must be a nonblank string or null")
    authors = args.get("authors", [])
    if not isinstance(authors, list) or any(not isinstance(a, str) for a in authors):
        raise ValueError("authors must be a list of strings")
    for key in ("url", "url_prefix"):
        if args.get(key) is not None and not isinstance(args[key], str):
            raise ValueError(f"{key} must be a string or null")
    return MetadataFilters(
        authors=tuple(authors),
        sources=(source.strip(),) if source else (),
        published_from=parse_date(args.get("published_from")),
        published_to=parse_date(args.get("published_to")),
        url=args.get("url"),
        url_prefix=args.get("url_prefix"),
    )


@dataclass(frozen=True, slots=True)
class RetrievalBranch:
    retrieval_query: str
    filters: MetadataFilters


def build_branches(args: dict[str, object]) -> list[RetrievalBranch]:
    from src.agent.tool import BRANCH_PROPERTIES

    if not isinstance(args, dict) or set(args) != {"branches"}:
        raise ValueError("Expected a branches object")
    branches = args["branches"]
    if not isinstance(branches, list) or not 1 <= len(branches) <= 6:
        raise ValueError("branches must contain between 1 and 6 branches")
    result = []
    for branch in branches:
        if not isinstance(branch, dict) or set(branch) - set(BRANCH_PROPERTIES):
            raise ValueError("Invalid retrieval branch fields")
        query = branch.get("retrieval_query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Each branch requires a nonblank retrieval_query")
        result.append(RetrievalBranch(query.strip(), build_filters(branch)))
    return result


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
    branches: list[RetrievalBranch],
) -> list[Document]:
    vectors = []
    timings = []
    with observation("embedding", embedding_model=rag.settings.embedding_model) as span:
        started = perf_counter()
        for branch in branches:
            branch_started = perf_counter()
            vectors.append(rag.embeddings.embed_query(branch.retrieval_query))
            timings.append((perf_counter() - branch_started) * 1000)
        span.update(
            embedding_latency_ms=(perf_counter() - started) * 1000,
            branch_latency_ms=timings,
            branch_count=len(branches),
        )

    documents = {}
    results = []
    with observation(
        "retrieval",
        top_k=rag.settings.candidate_k,
        embedding_model=rag.settings.embedding_model,
    ) as span:
        started = perf_counter()
        for branch_id, (branch, vector) in enumerate(
            zip(branches, vectors, strict=True)
        ):
            branch_started = perf_counter()
            found = rag.search(vector, branch.filters)
            results.append(
                {
                    "branch_id": branch_id,
                    "vector_search_latency_ms": (perf_counter() - branch_started)
                    * 1000,
                    "chunks": chunk_metadata(found),
                }
            )
            for document in found:
                key = (document.metadata.get("doc_id"), document.page_content)
                if key not in documents:
                    document = document.model_copy(deep=True)
                    document.metadata["branch_ids"] = []
                    documents[key] = document
                memberships = documents[key].metadata["branch_ids"]
                if branch_id not in memberships:
                    memberships.append(branch_id)
        span.update(
            vector_search_latency_ms=(perf_counter() - started) * 1000,
            branches=results,
            chunks=chunk_metadata(list(documents.values())),
        )
        if enabled("chunk_text"):
            span.payload(output=[d.page_content for d in documents.values()])
    return list(documents.values())


@cache
def get_reranker() -> DocumentReranker:
    settings = Settings()
    return DocumentReranker(settings.reranker_model)
