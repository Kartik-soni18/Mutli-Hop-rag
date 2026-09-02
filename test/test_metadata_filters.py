import unittest
from datetime import datetime, timezone

from qdrant_client.models import (
    DatetimeRange,
    MatchAny,
    MatchPrefix,
    MatchText,
    MatchValue,
    Range,
)

from src.rag.retrieval import MetadataFilters, build_metadata_filter


class MetadataFilterTests(unittest.TestCase):
    def test_empty_filters_do_not_add_a_qdrant_filter(self) -> None:
        self.assertIsNone(build_metadata_filter(MetadataFilters()))

    def test_exact_and_in_filters_use_the_expected_match_types(self) -> None:
        result = build_metadata_filter(
            MetadataFilters(
                doc_ids=(4, 8),
                title="Exact title",
                authors=("Single author",),
                sources=("Reuters", "BBC"),
                url="https://example.com/article",
            )
        )

        self.assertIsNotNone(result)
        conditions = {condition.key: condition for condition in result.must}
        self.assertIsInstance(conditions["metadata.doc_id"].match, MatchAny)
        self.assertIsInstance(conditions["metadata.title"].match, MatchValue)
        self.assertIsInstance(conditions["metadata.author"].match, MatchValue)
        self.assertIsInstance(conditions["metadata.source"].match, MatchAny)
        self.assertIsInstance(conditions["metadata.url"].match, MatchValue)

    def test_text_prefix_and_range_filters_use_typed_conditions(self) -> None:
        published_from = datetime(2023, 10, 1, tzinfo=timezone.utc)
        published_to = datetime(2023, 11, 1, tzinfo=timezone.utc)
        result = build_metadata_filter(
            MetadataFilters(
                doc_id_min=10,
                doc_id_max=20,
                title_text="climate change",
                published_from=published_from,
                published_to=published_to,
                url_prefix="https://example.com/news/",
            )
        )

        self.assertIsNotNone(result)
        by_key = {}
        for condition in result.must:
            by_key.setdefault(condition.key, []).append(condition)

        self.assertIsInstance(by_key["metadata.doc_id"][0].range, Range)
        self.assertIsInstance(by_key["metadata.title"][0].match, MatchText)
        self.assertIsInstance(
            by_key["metadata.published_at"][0].range,
            DatetimeRange,
        )
        self.assertIsInstance(by_key["metadata.url"][0].match, MatchPrefix)

    def test_reversed_id_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "doc_id_min"):
            MetadataFilters(doc_id_min=20, doc_id_max=10)

    def test_reversed_date_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "published_from"):
            MetadataFilters(
                published_from=datetime(2023, 11, 1, tzinfo=timezone.utc),
                published_to=datetime(2023, 10, 1, tzinfo=timezone.utc),
            )


if __name__ == "__main__":
    unittest.main()
