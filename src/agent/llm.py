import json
from dataclasses import dataclass, field, replace
from time import perf_counter

from src.agent.retrieval import (
    RetrievalBranch,
    build_branches,
    get_rag_service,
    get_reranker,
    retrieve_documents,
    sources_named_in,
)
from src.agent.tool import RAG_TOOL
from src.llm.client import create_llm
from src.observability.tracing import (
    chunk_metadata,
    enabled,
    observation,
    request_trace,
)


@dataclass(frozen=True, slots=True)
class AgentResult:
    branches: list[RetrievalBranch]
    context: list[dict[str, object]]
    answer: str
    timings: dict[str, float]
    request_id: str | None = None
    trace_id: str | None = None
    llm_calls: list[dict] = field(default_factory=list)


def run_agent(query: str, *, request_id=None, session_id=None) -> AgentResult:
    with request_trace(request_id, session_id) as trace:
        result = _run_agent(query)
        return replace(
            result,
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            llm_calls=trace.llm_calls,
        )


def _run_agent(query: str) -> AgentResult:
    total_started = perf_counter()
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")

    client = create_llm()
    messages = [
        {
            "role": "system",
            "content": (
                "Plan independent retrieval branches for the distinct evidence needed "
                "to answer the question. Return 1 to 6 branches, each with "
                "its own focused retrieval_query and applicable filters. Create "
                "separate "
                "branches for named publishers; each branch has at most one source. "
                "Use source-free branches when no publisher is specified. Preserve "
                "entities and facts, do not answer, and do not invent metadata. "
                "Publication bounds must be YYYY-MM-DD dates only, with both endpoints "
                "inclusive. For one day set both bounds to that date; "
                "omit unknown bounds. "
                f"Detected publishers: {json.dumps(sources_named_in(query))}."
            ),
        },
        {"role": "user", "content": query},
    ]

    planner_started = perf_counter()
    plan = client.complete(
        messages=messages,
        purpose="tool_call",
        tools=[RAG_TOOL],
        max_tokens=2048,
    )
    planner_seconds = perf_counter() - planner_started

    if len(plan.tool_calls) != 1 or plan.tool_calls[0]["name"] != "retrieve_documents":
        raise ValueError("Planner must return one retrieve_documents tool call")
    tool_arguments = plan.tool_calls[0]["args"]
    branches = build_branches(tool_arguments)

    retrieval_started = perf_counter()
    unranked_documents = retrieve_documents(get_rag_service(), branches)
    retrieval_seconds = perf_counter() - retrieval_started

    reranking_started = perf_counter()
    with observation(
        "reranking",
        reranker_model=get_rag_service().settings.reranker_model,
        input_chunk_count=len(unranked_documents),
        scored_pairs=[],
        top_n=min(get_rag_service().settings.rerank_k, 6),
        chunks_before=chunk_metadata(unranked_documents),
    ) as span:
        reranker = get_reranker()
        reranked_documents = reranker.rerank_branches(
            [branch.retrieval_query for branch in branches],
            unranked_documents,
        )
        span.update(
            chunks_after=chunk_metadata(reranked_documents),
            reranking_latency_ms=(perf_counter() - reranking_started) * 1000,
        )
        if enabled("chunk_text"):
            span.payload(output=[d.page_content for d in reranked_documents])
    reranking_seconds = perf_counter() - reranking_started

    context = [
        {
            "content": document.page_content,
            "metadata": {
                k: v
                for k, v in document.metadata.items()
                if k != "published_date_ordinal"
            },
        }
        for document in reranked_documents
    ]

    answer_started = perf_counter()
    answer_response = client.complete(
        messages=[
            message
            for message in [
                {
                    "role": "system",
                    "content": (
                        "Answer using only the supplied documents. Return only the "
                        "shortest exact answer without explanation. For comparisons, "
                        "check the evidence for every required part. If any necessary "
                        "fact is missing, ambiguous, or unsupported, answer exactly "
                        "Insufficient Information. Never treat missing evidence or "
                        "silence in an excerpt as proof of Yes or No. Answer Yes only "
                        "when the evidence supports the comparison, and No only when "
                        "the evidence explicitly establishes a contradiction. Do not "
                        "fill evidence gaps using assumptions or outside knowledge."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\n"
                        f"Documents: {json.dumps(context, default=str)}"
                    ),
                },
            ]
        ],
        purpose="final_answer",
        context_ids=[d.metadata.get("_id") for d in reranked_documents],
        max_tokens=512,
    )
    answer = (answer_response.content or "").strip()
    if not answer:
        raise ValueError("Answer generation returned empty content")
    answer_seconds = perf_counter() - answer_started
    total_seconds = perf_counter() - total_started

    return AgentResult(
        branches=branches,
        context=context,
        answer=answer,
        timings={
            "planner_seconds": planner_seconds,
            "retrieval_seconds": retrieval_seconds,
            "reranking_seconds": reranking_seconds,
            "answer_generation_seconds": answer_seconds,
            "total_seconds": total_seconds,
        },
    )


def generate(query: str) -> str:
    return run_agent(query).answer
