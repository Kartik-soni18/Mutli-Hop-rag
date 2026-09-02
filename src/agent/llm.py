import json
import os
from dataclasses import dataclass, replace

from groq import Groq

from src.agent.retrieval import (
    build_filters,
    get_rag_service,
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


def run_groq_agent(query: str) -> AgentResult:
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    messages = [
        {
            "role": "system",
            "content": (
                "Select metadata filters, then answer from the retrieved documents. "
                "The original user question is always the semantic search query. "
                "Do not invent metadata. After retrieval, return only the shortest "
                "exact answer."
            ),
        },
        {"role": "user", "content": query},
    ]

    plan = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[RAG_TOOL],
        tool_choice="required",
        temperature=0,
        max_completion_tokens=256,
    ).choices[0].message
    tool_call = plan.tool_calls[0]
    filters = build_filters(json.loads(tool_call.function.arguments))
    named_sources = sources_named_in(query)
    if named_sources:
        filters = replace(filters, sources=named_sources)
    documents = retrieve_documents(get_rag_service(), query, filters)
    context = [
        {"content": document.page_content, "metadata": document.metadata}
        for document in documents
    ]

    answer = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied documents. Return only the "
                    "shortest exact answer without explanation."
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

    return AgentResult(
        retrieval_query=query,
        filters=filters,
        context=context,
        answer=answer,
    )


def generate_groq(query: str) -> str:
    return run_groq_agent(query).answer
