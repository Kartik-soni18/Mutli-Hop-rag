"""Paid evaluation worker. Invoked by scripts/evaluate-and-push.sh."""

import argparse
import csv
import json
import random
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    from src.agent.llm import run_agent
    from src.agent.retrieval import get_rag_service
    from src.llm.factory import create_llm

    records = json.loads((ROOT / "data/MultiHopRAG.json").read_text())
    sample = random.Random(args.seed).sample(list(enumerate(records, 1)), 100)
    client = create_llm()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": output.name,
        "commit": args.commit,
        "seed": args.seed,
        "timestamp": datetime.now(UTC).isoformat(),
        "model": client.config.model,
        "status": "running",
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2))
    errors = []
    try:
        with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["question_no", "expected", "received"]
            )
            writer.writeheader()
            for position, (number, record) in enumerate(sample, 1):
                print(f"[{position}/100] Dataset question {number}", flush=True)
                try:
                    received = run_agent(record["query"]).answer
                except Exception as exc:
                    received = "ERROR"
                    # Do not retain provider response bodies or credentials.
                    errors.append(
                        {
                            "question_no": number,
                            "type": type(exc).__name__,
                            "status_code": getattr(exc, "status_code", None),
                        }
                    )
                writer.writerow(
                    {
                        "question_no": number,
                        "expected": record["answer"],
                        "received": received,
                    }
                )
                stream.flush()
                (output / "errors.json").write_text(json.dumps(errors, indent=2))
                if errors and errors[-1]["status_code"] in (401, 403, 402):
                    raise RuntimeError(
                        "Provider authentication or billing failed; stopping."
                    )
        if len(errors) == 100:
            raise RuntimeError(
                "All questions failed; leaving the previous dashboard intact."
            )
        metadata["status"] = "complete"
        metadata["errors"] = len(errors)
        (output / "run.json").write_text(json.dumps(metadata, indent=2))
    finally:
        client.close()
        if get_rag_service.cache_info().currsize:
            get_rag_service().vectorstore.client.close()


if __name__ == "__main__":
    main()
