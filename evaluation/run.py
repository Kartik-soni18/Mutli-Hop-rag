import csv
import json
from pathlib import Path

from src.agent.llm import run_groq_agent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "MultiHopRAG.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "evaluation.csv"
LIMIT = 4
FIELDNAMES = [
    "question_id",
    "query",
    "rag_query",
    "retrieved_context",
    "final_answer",
    "original_answer",
]


def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))[:LIMIT]

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for question_id, record in enumerate(dataset, start=1):
            print(f"[{question_id:02d}/{len(dataset)}] {record['query']}", flush=True)
            result = run_groq_agent(record["query"])
            writer.writerow(
                {
                    "question_id": question_id,
                    "query": record["query"],
                    "rag_query": result.retrieval_query,
                    "retrieved_context": json.dumps(result.context, default=str),
                    "final_answer": result.answer,
                    "original_answer": record["answer"],
                }
            )
            output_file.flush()

    print(f"Wrote {len(dataset)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
