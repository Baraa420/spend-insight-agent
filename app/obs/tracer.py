"""Lightweight observability / AgentOps-style tracing.

Every request gets a ``Trace`` with a unique id. Spans record the planner and each
tool call with timing and attributes. Traces are written as JSON Lines to ``trace_dir``
so they can be inspected with zero infrastructure. If LangFuse credentials are present,
spans are also mirrored there (import is guarded so the dependency stays optional).
"""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Span:
    name: str
    start: float
    end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        if self.end is None:
            return 0.0
        return round((self.end - self.start) * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "elapsed_ms": self.elapsed_ms,
            "attributes": self.attributes,
        }


class Trace:
    """Collects spans for a single request."""

    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        self.spans: list[Span] = []
        self.start = time.perf_counter()

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        s = Span(name=name, start=time.perf_counter(), attributes=dict(attributes))
        try:
            yield s
        finally:
            s.end = time.perf_counter()
            self.spans.append(s)

    @property
    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.start) * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "elapsed_ms": self.elapsed_ms,
            "spans": [s.to_dict() for s in self.spans],
        }


class Tracer:
    """Creates traces and persists them; optionally mirrors to LangFuse."""

    def __init__(self, trace_dir: str = "traces", langfuse_client: Any | None = None) -> None:
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._langfuse = langfuse_client

    def new_trace(self) -> Trace:
        return Trace(trace_id=uuid.uuid4().hex[:12])

    def emit(self, trace: Trace) -> None:
        record = trace.to_dict()
        out = self.trace_dir / f"{trace.trace_id}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        if self._langfuse is not None:  # pragma: no cover - optional path
            try:
                self._langfuse.trace(id=trace.trace_id, name="chat", metadata=record)
            except Exception:
                pass


def build_tracer(settings: Any) -> Tracer:
    """Factory that wires optional LangFuse if configured."""
    langfuse_client = None
    if getattr(settings, "langfuse_enabled", False):  # pragma: no cover - optional path
        try:
            from langfuse import Langfuse

            langfuse_client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        except Exception:
            langfuse_client = None
    return Tracer(trace_dir=settings.trace_dir, langfuse_client=langfuse_client)
