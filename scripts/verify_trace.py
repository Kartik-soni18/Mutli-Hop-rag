"""Read and validate the saved smoke trace; never invokes an LLM."""

import json
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

from src.observability.langfuse_adapter import get_backend


def main():
    load_dotenv(".env")
    path = Path("results/smoke_trace.json")
    report = json.loads(path.read_text())
    backend = get_backend()
    if backend is None:
        raise RuntimeError("Langfuse unavailable")
    trace = backend.client.api.trace.get(report["trace_id"])
    data = trace.model_dump(mode="json")
    observations = data["observations"]
    assert len(observations) == 6, "Expected one root and five children"
    by_name = {o["name"]: o for o in observations}
    assert set(by_name) == {
        "rag_request",
        "llm_tool_call",
        "embedding",
        "retrieval",
        "reranking",
        "llm_final_answer",
    }
    root = by_name["rag_request"]
    assert all(
        o["parentObservationId"] == root["id"] for o in observations if o is not root
    )
    costs = []
    for call in report["llm_calls"]:
        generation = by_name["llm_" + call["purpose"]]
        assert generation["type"] == "GENERATION"
        cost = Decimal(str(generation["costDetails"]["total"]))
        assert cost == Decimal(call["cost_usd"])
        costs.append(cost)
        assert generation["usageDetails"]["input"] == call["input_tokens"]
        assert generation["usageDetails"]["output"] == call["output_tokens"]
    total = sum(costs, Decimal(0))
    assert Decimal(str(data["totalCost"])) == total
    assert Decimal(str(data["metadata"]["total_llm_cost"])) == total
    assert data["metadata"]["cost_complete"] is True
    assert all(o["input"] is None and o["output"] is None for o in observations), (
        "Smoke validation expects all content logging disabled"
    )
    report.update(cloud_verified=True, total_llm_cost_usd=str(total))
    path.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "cloud_verified": True,
                "observations": len(observations),
                "total_cost_usd": str(total),
                "trace_url": report["trace_url"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
