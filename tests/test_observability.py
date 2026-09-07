from decimal import Decimal
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.observability import tracing as t
from src.observability.pricing import price


class Handle:
    trace_id = "test-trace"

    def __init__(self, backend, name, parent):
        self.backend, self.name, self.parent = backend, name, parent
        self.fields = {}
        self.ended = False

    def update(self, **kwargs):
        if self.backend.fail == "update":
            raise RuntimeError("telemetry unavailable")
        self.fields.update(kwargs)

    def update_trace(self, **kwargs):
        self.update(**kwargs)

    def end(self):
        if self.backend.fail == "end":
            raise RuntimeError("telemetry unavailable")
        self.ended = True


class Backend:
    def __init__(self, fail=None):
        self.fail, self.handles = fail, []

    def start(self, name, kind, parent=None):
        if self.fail == "start" or (self.fail == "child" and parent):
            raise RuntimeError("telemetry unavailable")
        handle = Handle(self, name, parent)
        self.handles.append(handle)
        return handle


@pytest.fixture
def flow(monkeypatch):
    from src.agent import llm as agent
    from src.agent import retrieval
    from src.llm.client import AICreditsLLM, Config
    from src.rag.config import Settings
    from src.rag.reranker import DocumentReranker

    calls = []

    class Model:
        def bind_tools(self, tools, **kwargs):
            return self

        def invoke(self, messages, **kwargs):
            calls.append(messages)
            return AIMessage(
                content="" if len(calls) == 1 else "Answer",
                tool_calls=[
                    {
                        "name": "retrieve_documents",
                        "args": {"branches": [{"retrieval_query": "sensitive query"}]},
                        "id": "call-1",
                    }
                ]
                if len(calls) == 1
                else [],
                response_metadata={
                    "model_name": "ling-3.0-flash",
                    "finish_reason": "stop",
                    "token_usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "total_tokens": 110,
                    },
                },
            )

    client = object.__new__(AICreditsLLM)
    client.config = Config("inclusionai/ling-3.0-flash", "unused")
    client.model = Model()
    rag = SimpleNamespace(
        settings=Settings(),
        embeddings=SimpleNamespace(embed_query=lambda q: [1.0]),
        search=lambda v, f: [
            Document(
                page_content="sensitive document",
                metadata={"doc_id": 1, "_id": "chunk-1", "similarity_score": 0.9},
            )
        ],
    )
    reranker = object.__new__(DocumentReranker)
    reranker.model = SimpleNamespace(predict=lambda pairs: [0.8] * len(pairs))
    monkeypatch.setattr(agent, "create_llm", lambda: client)
    monkeypatch.setattr(agent, "get_rag_service", lambda: rag)
    monkeypatch.setattr(agent, "get_reranker", lambda: reranker)
    monkeypatch.setattr(agent, "sources_named_in", lambda q: ())
    monkeypatch.setattr(retrieval, "get_rag_service", lambda: rag)
    for name in ["PROMPTS", "CHUNK_TEXT", "RESPONSES"]:
        monkeypatch.delenv("LANGFUSE_LOG_" + name, raising=False)
    return agent, calls


@pytest.mark.parametrize("failure", [None, "init", "start", "child", "update", "end"])
def test_flow_survives_telemetry_failures(monkeypatch, flow, failure):
    agent, calls = flow
    backend = Backend(failure)

    def get_backend():
        if failure == "init":
            raise RuntimeError("init failed")
        return backend

    monkeypatch.setattr(t, "get_backend", get_backend)
    result = agent.run_agent("private question", request_id="req", session_id="session")
    assert result.answer == "Answer"
    assert len(calls) == 2
    assert len(result.llm_calls) == 2
    assert t._current.get() is None
    if failure is None:
        assert [h.name for h in backend.handles] == [
            "rag_request",
            "llm_tool_call",
            "embedding",
            "retrieval",
            "reranking",
            "llm_final_answer",
        ]
        root = backend.handles[0]
        assert all(h.parent is root for h in backend.handles[1:])
        assert all(h.ended for h in backend.handles)
        assert Decimal(root.fields["metadata"]["total_llm_cost"]) == Decimal(
            "0.00000546"
        )
        serialized = repr([h.fields for h in backend.handles])
        for secret in [
            "private question",
            "sensitive query",
            "sensitive document",
            "Answer",
        ]:
            assert secret not in serialized
        assert "chunk-1" in serialized


@pytest.mark.parametrize("failure", ["start", "update", "end"])
def test_application_exception_preserved(monkeypatch, failure):
    monkeypatch.setattr(t, "get_backend", lambda: Backend(failure))
    original = ValueError("private error")
    with pytest.raises(ValueError) as caught:
        with t.request_trace():
            with t.observation("retrieval"):
                raise original
    assert caught.value is original
    assert t._current.get() is None


def test_pricing():
    usage = dict(input_tokens=1000, output_tokens=100, cached_tokens=200)
    result = price(usage, "org/model", "model")
    assert Decimal(result["cost_usd"]) == Decimal("0.00002394")
    result = price(usage, "org/model", "model", {"amount": 0, "currency": "USD"})
    assert result["cost_source"] == "provider"
    assert result["cost_usd"] == "0"
    assert price({}, "model", "model")["cost_usd"] is None
    assert price(usage, "model", "other")["cost_usd"] is None
    assert (
        price(usage, "model", "model", {"amount": 10, "currency": "INR"})["cost_source"]
        == "estimated"
    )
    assert price(dict(input_tokens=10, output_tokens=5), "m", "m")[
        "cached_tokens_assumed_zero"
    ]
    assert (
        price(dict(input_tokens=10, output_tokens=5, cached_tokens=20), "m", "m")[
            "cost_usd"
        ]
        is None
    )


def test_nested_request_isolation(monkeypatch):
    monkeypatch.setattr(t, "get_backend", lambda: Backend())
    with t.request_trace("outer") as outer:
        with t.request_trace("inner") as inner:
            t.record_llm({"cost_usd": "1"})
        assert outer.costs == []
        assert inner.costs == ["1"]
        t.record_llm({"cost_usd": "2"})
    assert outer.costs == ["2"]


def test_flush_failure(monkeypatch):
    from src.observability import langfuse_adapter as adapter

    class Broken:
        def flush(self):
            raise RuntimeError("failed")

    monkeypatch.setattr(adapter, "get_backend", lambda: Broken())
    adapter.flush()


@pytest.mark.parametrize("chunks", [False, True])
def test_prompt_privacy(monkeypatch, flow, chunks):
    agent, _ = flow
    backend = Backend()
    monkeypatch.setattr(t, "get_backend", lambda: backend)
    monkeypatch.setenv("LANGFUSE_LOG_PROMPTS", "true")
    monkeypatch.setenv("LANGFUSE_LOG_CHUNK_TEXT", str(chunks).lower())
    agent.run_agent("question")
    assert "input" in backend.handles[1].fields
    assert ("input" in backend.handles[-1].fields) == chunks


def test_response_adapter():
    from src.llm.client import AICreditsChat

    model = AICreditsChat(api_key="dummy", model="model", max_retries=0)
    result = model._create_chat_result(
        {
            "model": "actual",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "x"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost_usd": 0.2,
            },
            "secret_extra": "DO NOT EXPORT",
        }
    )
    assert result.llm_output["provider_billing"] == {"amount": 0.2, "currency": "USD"}
    assert "DO NOT EXPORT" not in repr(result)
    model.root_client.close()


def test_installed_langfuse_sdk(monkeypatch):
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from src.observability.langfuse_adapter import LangfuseBackend

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    backend = object.__new__(LangfuseBackend)
    backend.client = Langfuse(
        public_key="pk-local-sdk-test",
        secret_key="sk-local",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    monkeypatch.setattr(t, "get_backend", lambda: backend)
    with t.request_trace("sdk-request", "sdk-session") as request:
        with t.observation("llm_tool_call", "generation") as obs:
            obs.payload(
                model="test-model",
                usage_details={"input": 2, "output": 1},
                cost_details={"total": 0.01},
            )
            t.record_llm({"cost_usd": ".01"})
        with t.observation("llm_final_answer", "generation"):
            t.record_llm({"cost_usd": ".02"})
    backend.flush()
    spans = exporter.get_finished_spans()
    root = next(s for s in spans if s.name == "rag_request")
    assert request.trace_id == format(root.context.trace_id, "032x")
    assert len(spans) == 3
    assert all(s.parent.span_id == root.context.span_id for s in spans if s is not root)
    assert root.attributes["session.id"] == "sdk-session"
    assert "0.03" in repr(root.attributes)


def test_missing_sdk_methods_cannot_break_request(monkeypatch):
    monkeypatch.setattr(
        t, "get_backend", lambda: SimpleNamespace(start=lambda *args: object())
    )
    with t.request_trace():
        with t.observation("retrieval"):
            pass


def test_backend_initialization_failure(monkeypatch):
    from src.observability import langfuse_adapter as adapter

    for name in ["PUBLIC_KEY", "SECRET_KEY", "BASE_URL"]:
        monkeypatch.setenv("LANGFUSE_" + name, "dummy")
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")

    def broken():
        raise RuntimeError("unavailable")

    monkeypatch.setattr(adapter, "LangfuseBackend", broken)
    adapter.get_backend.cache_clear()
    assert adapter.get_backend() is None
    adapter.get_backend.cache_clear()


def test_scored_search_keeps_filter_and_ids():
    from src.rag.config import Settings
    from src.rag.retrieval import MetadataFilters
    from src.rag.service import RAGService

    captured = {}

    def search(vector, **kwargs):
        captured.update(kwargs)
        return [
            (Document(page_content="x", metadata={"_id": "point", "doc_id": 3}), 0.75)
        ]

    rag = object.__new__(RAGService)
    rag.settings = Settings()
    rag.vectorstore = SimpleNamespace(similarity_search_with_score_by_vector=search)
    docs = rag.search([1.0], MetadataFilters(sources=("The Verge",)))
    assert captured["k"] == 10
    assert captured["filter"].must[0].match.value == "The Verge"
    assert docs[0].metadata == {"_id": "point", "doc_id": 3, "similarity_score": 0.75}
