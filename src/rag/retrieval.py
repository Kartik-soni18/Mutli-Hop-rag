from dataclasses import dataclass
from datetime import datetime

from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    MatchAny,
    MatchPrefix,
    MatchText,
    MatchValue,
    Range,
)


@dataclass(frozen=True, slots=True)
class MetadataFilters:
    title: str | None = None
    title_text: str | None = None
    authors: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    published_from: datetime | None = None
    published_to: datetime | None = None
    url: str | None = None
    url_prefix: str | None = None

    def __post_init__(self) -> None:
        if (
            self.published_from is not None
            and self.published_to is not None
            and self.published_from > self.published_to
        ):
            raise ValueError("published_from cannot be later than published_to")


def _match_one_or_any(
    key: str,
    values: tuple[str, ...] | tuple[int, ...],
) -> FieldCondition:
    if len(values) == 1:
        return FieldCondition(key=key, match=MatchValue(value=values[0]))
    return FieldCondition(key=key, match=MatchAny(any=list(values)))


def build_metadata_filter(metadata: MetadataFilters | None) -> Filter | None:
    if metadata is None:
        return None

    conditions = []
    if metadata.title is not None:
        conditions.append(
            FieldCondition(
                key="metadata.title",
                match=MatchValue(value=metadata.title),
            )
        )
    if metadata.title_text is not None:
        conditions.append(
            FieldCondition(
                key="metadata.title",
                match=MatchText(text=metadata.title_text),
            )
        )
    if metadata.authors:
        conditions.append(_match_one_or_any("metadata.author", metadata.authors))
    if metadata.sources:
        conditions.append(_match_one_or_any("metadata.source", metadata.sources))
    if metadata.published_from is not None or metadata.published_to is not None:
        conditions.append(
            FieldCondition(
                key="metadata.published_at",
                range=DatetimeRange(
                    gte=metadata.published_from,
                    lte=metadata.published_to,
                ),
            )
        )
    if metadata.url is not None:
        conditions.append(
            FieldCondition(
                key="metadata.url",
                match=MatchValue(value=metadata.url),
            )
        )
    if metadata.url_prefix is not None:
        conditions.append(
            FieldCondition(
                key="metadata.url",
                match=MatchPrefix(prefix=metadata.url_prefix),
            )
        )

    return Filter(must=conditions) if conditions else None


def create_retriever(
    vectorstore: QdrantVectorStore,
    top_k: int,
    metadata_filters: MetadataFilters | None = None,
) -> BaseRetriever:
    search_kwargs: dict[str, object] = {"k": top_k}

    qdrant_filter = build_metadata_filter(metadata_filters)
    if qdrant_filter is not None:
        search_kwargs["filter"] = qdrant_filter

    return vectorstore.as_retriever(search_kwargs=search_kwargs)
