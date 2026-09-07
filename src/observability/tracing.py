"""Request-local, best-effort observations; no business operation is retried here."""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from decimal import Decimal
from time import perf_counter
from uuid import uuid4

from .langfuse_adapter import get_backend

_current = ContextVar("rag_request", default=None)
_active = ContextVar("rag_observation", default=None)


def safe(call, *args, **kwargs):
    try:
        return call(*args, **kwargs)
    except Exception:
        return None


def safe_method(target, name, *args, **kwargs):
    """Resolve SDK methods inside the failure boundary too."""
    return safe(lambda: getattr(target, name)(*args, **kwargs))


def enabled(kind):
    return os.getenv(f"LANGFUSE_LOG_{kind.upper()}", "false").lower() == "true"


@dataclass
class Observation:
    handle: object = None
    metadata: dict = field(default_factory=dict)

    def update(self, **metadata):
        self.metadata.update(metadata)
        if self.handle:
            safe_method(self.handle, "update", metadata=self.metadata.copy())

    def payload(self, **fields):
        if self.handle:
            safe_method(self.handle, "update", **fields)


@dataclass
class Request:
    request_id: str
    session_id: str | None
    backend: object = None
    root: Observation = field(default_factory=Observation)
    trace_id: str | None = None
    costs: list = field(default_factory=list)
    llm_calls: list = field(default_factory=list)


@contextmanager
def observation(name, kind="span", **metadata):
    request = _current.get()
    handle = None
    if request and request.backend and request.root.handle:
        handle = safe_method(request.backend, "start", name, kind, request.root.handle)
    obs = Observation(handle)
    started = perf_counter()
    obs.update(**metadata)
    active_token = _active.set(obs)
    try:
        yield obs
    except BaseException as exc:
        obs.update(
            success=False,
            error_type=type(exc).__name__,
            http_status=getattr(exc, "status_code", None),
            api_error_type=type(exc).__name__,
        )
        obs.payload(level="ERROR", status_message=type(exc).__name__)
        raise
    else:
        obs.update(success=True)
    finally:
        obs.update(latency_ms=(perf_counter() - started) * 1000)
        if handle:
            safe_method(handle, "end")
        _active.reset(active_token)


@contextmanager
def request_trace(request_id=None, session_id=None):
    request = Request(request_id or str(uuid4()), session_id)
    request.backend = safe(get_backend)
    propagation = None
    if request.backend:
        propagation = safe(
            lambda: request.backend.context(request.request_id, session_id)
        )
        if propagation:
            safe_method(propagation, "__enter__")
        request.root.handle = safe_method(
            request.backend, "start", "rag_request", "span"
        )
    if request.root.handle:
        request.trace_id = safe(lambda: request.root.handle.trace_id)
    token = _current.set(request)
    started = perf_counter()
    request.root.update(request_id=request.request_id, session_id=session_id)
    try:
        yield request
    except BaseException as exc:
        request.root.update(success=False, error_type=type(exc).__name__)
        request.root.payload(level="ERROR", status_message=type(exc).__name__)
        raise
    else:
        request.root.update(success=True, error_type=None)
    finally:
        complete = len(request.costs) == 2 and all(c is not None for c in request.costs)
        known = sum((Decimal(c) for c in request.costs if c is not None), Decimal(0))
        request.root.update(
            total_latency_ms=(perf_counter() - started) * 1000,
            total_llm_cost=str(known) if complete else None,
            known_llm_cost=str(known),
            cost_complete=complete,
            currency="USD",
        )
        if request.root.handle:
            safe_method(request.root.handle, "end")
        _current.reset(token)
        if propagation:
            safe_method(propagation, "__exit__", None, None, None)


def record_llm(metadata):
    request = _current.get()
    if request:
        request.costs.append(metadata.get("cost_usd"))
        request.llm_calls.append(metadata.copy())


def chunk_metadata(documents):
    return [
        {
            "doc_id": d.metadata.get("doc_id"),
            "chunk_id": str(d.metadata.get("_id", "")),
            "similarity_score": d.metadata.get("similarity_score"),
            "branch_ids": d.metadata.get("branch_ids", []),
            "reranker_scores": d.metadata.get("branch_reranker_scores", {}),
        }
        for d in documents
    ]


def record_reranker_scores(scores):
    obs = _active.get()
    if obs:
        obs.update(scored_pairs=scores)
