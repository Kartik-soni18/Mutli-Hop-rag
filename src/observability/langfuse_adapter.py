"""The only module that knows the Langfuse SDK."""

import atexit
import os
from functools import cache
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv


class LangfuseBackend:
    def __init__(self):
        from langfuse import Langfuse

        self.client = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            base_url=os.environ["LANGFUSE_BASE_URL"],
            timeout=5,
        )

    def context(self, request_id, session_id):
        from langfuse import propagate_attributes

        return propagate_attributes(
            session_id=session_id,
            trace_name="rag_request",
            metadata={"request_id": request_id},
        )

    def start(self, name, kind, parent=None):
        if parent is None:
            return self.client.start_observation(
                name=name, as_type=kind, trace_context={"trace_id": uuid4().hex}
            )
        return parent.start_observation(name=name, as_type=kind)

    def flush(self):
        self.client.flush()


@cache
def get_backend():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    if os.getenv("LANGFUSE_ENABLED", "true").lower() != "true":
        return None
    if not all(
        os.getenv(k)
        for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
    ):
        return None
    try:
        backend = LangfuseBackend()
        atexit.register(flush)
        return backend
    except Exception:
        return None


def flush():
    try:
        backend = get_backend()
        if backend:
            backend.flush()
    except Exception:
        pass
