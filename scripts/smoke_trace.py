
import json
from dataclasses import asdict
from pathlib import Path

from src.agent.llm import run_agent
from src.agent.retrieval import get_rag_service, get_reranker
from src.llm.client import create_llm
from src.observability.langfuse_adapter import flush, get_backend


def main():
    rag = get_rag_service()
    get_reranker()
    record = json.loads(Path("data/MultiHopRAG.json").read_text())[0]
    backend = get_backend()
    if backend is None:
        raise RuntimeError("Langfuse configuration unavailable for smoke verification")
    if not backend.client.auth_check():
        raise RuntimeError("Langfuse authentication failed")
    client = create_llm()
    try:
        result = run_agent(record["query"], session_id="single-query-validation")
        report = asdict(result)
        report.pop("context")
        report.pop("branches")
        report["expected_answer"] = record["answer"]
        report["trace_url"] = backend.client.get_trace_url(trace_id=result.trace_id)
        destination = Path("results/smoke_trace.json")
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        flush()
        client.close()
        rag.vectorstore.client.close()


if __name__ == "__main__":
    main()
