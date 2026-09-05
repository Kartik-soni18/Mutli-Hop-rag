import json
from dataclasses import dataclass
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
from src.llm.factory import create_llm
from src.llm.types import CompletionOptions, LLMError, LLMMessage


@dataclass(frozen=True, slots=True)
class AgentResult:
    branches: list[RetrievalBranch]
    context: list[dict[str, object]]
    answer: str
    timings: dict[str, float]


def run_agent(query: str) -> AgentResult:
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
        messages=[LLMMessage(**message) for message in messages],
        tools=[RAG_TOOL],
        tool_choice="required",
        options=CompletionOptions(
            temperature=0,
            max_completion_tokens=2048,
            reasoning_effort="low",
            reasoning_format="hidden",
        ),
    ).message
    planner_seconds = perf_counter() - planner_started

    if len(plan.tool_calls) != 1 or plan.tool_calls[0].name != "retrieve_documents":
        raise LLMError("Planner must return one retrieve_documents tool call")
    tool_arguments = plan.tool_calls[0].arguments
    branches = build_branches(tool_arguments)

    retrieval_started = perf_counter()
    unranked_documents = retrieve_documents(get_rag_service(), branches)
    retrieval_seconds = perf_counter() - retrieval_started

    reranking_started = perf_counter()
    reranker = get_reranker()
    reranked_documents = reranker.rerank_branches(
        [branch.retrieval_query for branch in branches],
        unranked_documents,
    )
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
            LLMMessage(**message)
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
        options=CompletionOptions(
            temperature=0,
            max_completion_tokens=512,
            reasoning_effort="low",
            reasoning_format="hidden",
        ),
    )
    answer = (answer_response.message.content or "").strip()
    if not answer:
        raise LLMError("Answer generation returned empty content")
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


# Preserve older scripts while routing through the configured provider.
run_groq_agent = run_agent
generate_groq = generate
