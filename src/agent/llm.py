import json
import os
from dataclasses import dataclass, replace
from time import perf_counter

from groq import Groq

from src.agent.retrieval import (
    build_filters,
    get_rag_service,
    get_reranker,
    retrieve_documents,
    sources_named_in,
)
from src.agent.tool import RAG_TOOL
from src.rag.retrieval import MetadataFilters


@dataclass(frozen=True, slots=True)
class AgentResult:
    retrieval_query: str
    filters: MetadataFilters
    context: list[dict[str, object]]
    answer: str
    timings: dict[str, float]


def run_groq_agent(query: str) -> AgentResult:
    total_started = perf_counter()
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    messages = [
        {
            "role": "system",
            "content": (
                "Select metadata filters and rewrite the user's question into one "
                "concise semantic-search query. Return the rewritten query in the "
                "retrieval_query tool argument. Preserve all entities and facts "
                "needed to retrieve the supporting evidence; do not answer the "
                "question and do not invent metadata."
            ),
        },
        {"role": "user", "content": query},
    ]

    planner_started = perf_counter()
    plan = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[RAG_TOOL],
        tool_choice="required",
        temperature=0,
        max_completion_tokens=512,
        reasoning_effort="low",
        reasoning_format="hidden",
    ).choices[0].message
    planner_seconds = perf_counter() - planner_started

    tool_call = plan.tool_calls[0]
    tool_arguments = json.loads(tool_call.function.arguments)
    retrieval_query = str(tool_arguments["retrieval_query"]).strip()
    if not retrieval_query:
        retrieval_query = query

    filters = build_filters(tool_arguments)
    named_sources = sources_named_in(query)
    if named_sources:
        filters = replace(filters, sources=named_sources)

    retrieval_started = perf_counter()
    unranked_documents = retrieve_documents(
        get_rag_service(), retrieval_query, filters
    )
    retrieval_seconds = perf_counter() - retrieval_started

    reranking_started = perf_counter()
    reranker = get_reranker()
    reranked_documents = reranker.rerank(
        retrieval_query,
        unranked_documents,
    )
    reranking_seconds = perf_counter() - reranking_started

    context = [
        {"content": document.page_content, "metadata": document.metadata}
        for document in reranked_documents
    ]

    answer_started = perf_counter()
    answer = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied documents. Return only the "
                    "shortest exact answer without explanation. And if the documents "
                    "are not relevant answer Insufficient Information"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Documents: {json.dumps(context, default=str)}"
                ),
            },
        ],
        temperature=0,
        max_completion_tokens=512,
        reasoning_effort="low",
        reasoning_format="hidden",
    ).choices[0].message.content.strip()
    answer_seconds = perf_counter() - answer_started
    total_seconds = perf_counter() - total_started

    return AgentResult(
        retrieval_query=retrieval_query,
        filters=filters,
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


def generate_groq(query: str) -> str:
    return run_groq_agent(query).answer
