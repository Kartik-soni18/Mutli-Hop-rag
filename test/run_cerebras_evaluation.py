"""Run a 30-question MultiHop-RAG evaluation through the Groq API route."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "MultiHopRAG.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "test_output.csv"
DEFAULT_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")
SAMPLE_SIZE = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample 30 MultiHop-RAG questions, retrieve context, and "
            "send each prompt to the API's Groq endpoint."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--seed",
        type=int,
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def request_json(
    url: str,
    *,
    payload: dict[str, str] | None = None,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"

    for attempt in range(1, retries + 1):
        request = Request(
            url,
            data=body,
            headers=headers,
            method="POST" if body else "GET",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Request failed after {retries} attempts: {url}") from exc
            time.sleep(2 ** (attempt - 1))

    raise AssertionError("unreachable")


def build_prompt(question: str, documents: list[dict[str, Any]]) -> str:
    context = "\n\n".join(
        f"Document {number}:\n{document['content']}"
        for number, document in enumerate(documents, start=1)
    )
    return (
        "Answer the question using only the supplied context. Return only the "
        "answer, without an explanation.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def get_groq_answer(
    base_url: str,
    question: str,
    *,
    timeout: float,
    retries: int,
) -> str:
    base_url = base_url.rstrip("/")
    query_url = f"{base_url}/query?{urlencode({'q': question})}"
    retrieval = request_json(query_url, timeout=timeout, retries=retries)
    prompt = build_prompt(question, retrieval["documents"])
    generation = request_json(
        f"{base_url}/generate",
        payload={"prompt": prompt},
        timeout=timeout,
        retries=retries,
    )
    return str(generation["response"]).strip()


def main() -> None:
    args = parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if len(dataset) < SAMPLE_SIZE:
        raise ValueError(
            f"Dataset has {len(dataset)} questions; at least {SAMPLE_SIZE} are required."
        )

    rng = random.Random(args.seed) if args.seed is not None else random.SystemRandom()
    sampled_indices = rng.sample(range(len(dataset)), SAMPLE_SIZE)
    rows: list[dict[str, str | int]] = []

    for position, dataset_index in enumerate(sampled_indices, start=1):
        record = dataset[dataset_index]
        print(f"[{position:02d}/{SAMPLE_SIZE}] Question {dataset_index + 1}", flush=True)
        answer = get_groq_answer(
            args.base_url,
            record["query"],
            timeout=args.timeout,
            retries=args.retries,
        )
        rows.append(
            {
                "question_id": dataset_index + 1,
                "correct_answer": record["answer"],
                "answer_received": answer,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(f"{args.output.suffix}.tmp")
    with temporary_output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["question_id", "correct_answer", "answer_received"],
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary_output.replace(args.output)
    print(f"Wrote {len(rows)} results to {args.output}")


if __name__ == "__main__":
    main()
