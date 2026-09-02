"""Calculate case-insensitive exact match and update the README summary."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = Path(__file__).with_name("test_output.csv")
DEFAULT_README = PROJECT_ROOT / "README.MD"
START_MARKER = "<!-- exact-match-results:start -->"
END_MARKER = "<!-- exact-match-results:end -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update the README with exact-match metrics from a result CSV."
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    return parser.parse_args()


def normalize(value: str | None) -> str:
    """Normalize casing while keeping exact-match evaluation intentionally strict."""
    return (value or "").strip().upper()


def load_rows(results_path: Path) -> list[dict[str, str]]:
    with results_path.open(encoding="utf-8", newline="") as results_file:
        reader = csv.DictReader(results_file)
        required_columns = {"correct_answer", "answer_received"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required CSV columns: {missing}")
        return list(reader)


def build_summary(rows: list[dict[str, str]]) -> str:
    total = len(rows)
    matches = sum(
        normalize(row["correct_answer"]) == normalize(row["answer_received"])
        for row in rows
    )
    mismatches = total - matches
    blank_answers = sum(not normalize(row["answer_received"]) for row in rows)
    score = (matches / total * 100) if total else 0.0

    expected_answers = Counter(normalize(row["correct_answer"]) for row in rows)
    most_common_answer, most_common_count = (
        expected_answers.most_common(1)[0] if expected_answers else ("N/A", 0)
    )
    majority_baseline = (most_common_count / total * 100) if total else 0.0
    baseline_gap = score - majority_baseline

    return "\n".join(
        [
            START_MARKER,
            "### Latest exact-match result",
            "",
            f"- **Exact match (EM): {score:.1f}% ({matches}/{total})**",
            f"- Non-matching answers: {mismatches}",
            f"- Blank model answers: {blank_answers}",
            f"- Most common reference answer: `{most_common_answer}` "
            f"({most_common_count}/{total} questions)",
            f"- Majority-answer baseline: {majority_baseline:.1f}% "
            f"(EM is {abs(baseline_gap):.1f} percentage points "
            f"{'above' if baseline_gap >= 0 else 'below'} it)",
            "",
            "This is strict: surrounding whitespace is trimmed and both answers are",
            "converted to uppercase. Punctuation, different Unicode characters, and",
            "additional explanation still count as a mismatch, so the score highlights",
            "both answer correctness and instruction-following.",
            END_MARKER,
        ]
    )


def update_readme(readme_path: Path, summary: str) -> None:
    readme = readme_path.read_text(encoding="utf-8")
    if START_MARKER not in readme or END_MARKER not in readme:
        raise ValueError(
            f"{readme_path} must contain the exact-match result markers"
        )

    prefix, remainder = readme.split(START_MARKER, maxsplit=1)
    _, suffix = remainder.split(END_MARKER, maxsplit=1)
    readme_path.write_text(f"{prefix}{summary}{suffix}", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results)
    summary = build_summary(rows)
    update_readme(args.readme, summary)
    print(summary)


if __name__ == "__main__":
    main()
