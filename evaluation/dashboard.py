"""Offline score calculation and README/SVG rendering; no model dependencies."""

import csv
import html
import re
import unicodedata
from pathlib import Path

START = "<!-- evaluation-dashboard:start -->"
END = "<!-- evaluation-dashboard:end -->"
FIELDS = [
    "run_id",
    "timestamp",
    "commit",
    "seed",
    "model",
    "matches",
    "errors",
    "total",
]


def normalize(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(
        "-" if unicodedata.category(char) == "Pd" else char for char in value
    )
    value = re.sub(r"\s+", " ", value).strip()
    while value and unicodedata.category(value[-1]).startswith("P"):
        value = value[:-1].rstrip()
    return value


def score(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["question_no", "expected", "received"]:
            raise ValueError("Unexpected results.csv columns")
        rows = list(reader)
    if len(rows) != 100 or len({row["question_no"] for row in rows}) != 100:
        raise ValueError("A completed evaluation needs 100 distinct question rows")
    errors = sum(row["received"] == "ERROR" for row in rows)
    matches = sum(
        row["received"] != "ERROR"
        and normalize(row["expected"]) == normalize(row["received"])
        for row in rows
    )
    return matches, errors, len(rows)


def render_svg(history):
    latest = history[-1] if history else None
    number = f"{latest['matches']} / 100" if latest else "— / 100"
    subtitle = "NORMALIZED EXACT MATCH" if latest else "AWAITING FIRST EVALUATION"
    detail = (
        f"COMMIT {latest['commit'][:8]} · {latest['timestamp']}"
        if latest
        else "LOCAL EXECUTION · TWO LLM CALLS PER QUESTION · SIX CONTEXT CHUNKS"
    )
    provider_label = html.escape(latest["model"] if latest else "inclusionai/ling-3.0-flash")
    parts = [
        """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="560" viewBox="0 0 1200 560" role="img" aria-labelledby="title desc">
<title id="title">MultiHop-RAG evaluation progress</title>
<desc id="desc">Latest normalized exact matches out of 100 and historical evaluations.</desc>
<defs><linearGradient id="bg" x2="1" y2="1"><stop stop-color="#081122"/><stop offset="1" stop-color="#16102d"/></linearGradient><linearGradient id="line"><stop stop-color="#39e7e0"/><stop offset="1" stop-color="#a78bfa"/></linearGradient></defs>
<rect width="1200" height="560" rx="24" fill="url(#bg)"/>
<g font-family="Inter,Arial,sans-serif">
<text x="48" y="54" fill="#65ece6" font-size="14" letter-spacing="4">MULTIHOP / OBSERVATORY</text>
<text x="48" y="134" fill="#f4f7ff" font-size="60" font-weight="700">""",
        html.escape(number),
        "</text>",
        f'<text x="48" y="170" fill="#a6b2cd" font-size="13" letter-spacing="2">{subtitle}</text>',
        f'<text x="1148" y="58" fill="#a6b2cd" font-size="13" text-anchor="end">AI CREDITS / {provider_label}</text>',
        f'<text x="48" y="204" fill="#8997b5" font-size="12">{html.escape(detail)}</text>',
    ]
    for value in (0, 25, 50, 75, 100):
        y = 458 - value * 2
        parts.append(
            f'<line x1="78" y1="{y}" x2="1148" y2="{y}" stroke="#25304a"/><text x="60" y="{y + 4}" text-anchor="end" font-size="11" fill="#8997b5">{value}</text>'
        )
    if history:
        points = [
            (78 + i * 1070 / max(1, len(history) - 1), 458 - int(row["matches"]) * 2)
            for i, row in enumerate(history)
        ]
        parts.append(
            '<polyline points="'
            + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            + '" fill="none" stroke="url(#line)" stroke-width="3"/>'
        )
        for (x, y), row in zip(points, history):
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y}" r="4" fill="#65ece6"><title>{html.escape(row["commit"][:8])}: {row["matches"]}/100</title></circle>'
            )
        parts.append(
            f'<text x="78" y="484" fill="#8997b5" font-size="11">{html.escape(history[0]["commit"][:8])}</text><text x="1148" y="484" text-anchor="end" fill="#8997b5" font-size="11">{html.escape(history[-1]["commit"][:8])}</text>'
        )
    else:
        parts.append(
            '<text x="600" y="357" text-anchor="middle" fill="#a6b2cd" font-size="20">Your first evaluation starts the timeline.</text>'
        )
    parts.append(
        '<text x="48" y="530" fill="#8997b5" font-size="12">100 fresh random questions per run · Different samples affect score comparability</text></g></svg>'
    )
    return "".join(parts)


def update_dashboard(root: Path, run=None):
    history_path = root / "evaluation/history.csv"
    history = []
    if history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
    if run:
        if run["status"] != "complete":
            raise ValueError("Cannot publish incomplete evaluation")
        matches, errors, total = score(root / "results.csv")
        row = {
            key: run[key] for key in FIELDS if key not in ("matches", "errors", "total")
        }
        row.update(matches=matches, errors=errors, total=total)
        history = [old for old in history if old["run_id"] != row["run_id"]]
        history.append(row)
        history.sort(key=lambda row: row["timestamp"])
    readme = root / "README.MD"
    text = readme.read_text()
    if (
        text.count(START) != 1
        or text.count(END) != 1
        or text.index(START) >= text.index(END)
    ):
        raise ValueError("Expected one ordered pair of README dashboard markers")
    section = "\n\n![Evaluation progress](assets/evaluation.svg)\n\n"
    if history:
        last = history[-1]
        section += f"**Latest: {last['matches']}/100 normalized exact matches** · Errors: {last['errors']} · Updated: {last['timestamp']}\n\n"
        section += "| Evaluated commit | Matches | Errors | UTC timestamp |\n| :-- | --: | --: | :-- |\n"
        for row in reversed(history[-10:]):
            section += f"| `{row['commit'][:8]}` | {row['matches']}/100 | {row['errors']} | {row['timestamp']} |\n"
        section += "\n[Latest answers](results.csv) · [Full history](evaluation/history.csv)\n\n"
    else:
        section += "**Awaiting first evaluation.** No paid evaluation has populated this dashboard yet.\n\n"
    before, tail = text.split(START, 1)
    _, after = tail.split(END, 1)
    readme.write_text(before + START + section + END + after)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(history)
    (root / "assets").mkdir(exist_ok=True)
    (root / "assets/evaluation.svg").write_text(render_svg(history))
