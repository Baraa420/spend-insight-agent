"""LLM provider abstraction.

Defines a minimal, provider-agnostic chat interface plus the typed structures the
agent relies on: a routing ``Plan`` (which tools to call) and the final answer text.
Concrete providers live in ``offline.py`` (deterministic, no key) and
``openai_llm.py`` (any OpenAI-compatible endpoint).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class ToolCall:
    """A single planned tool invocation."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    """Planner output: an ordered list of tool calls to satisfy the question."""

    steps: list[ToolCall] = field(default_factory=list)
    rationale: str = ""


class BaseLLM:
    """Provider interface. Implementations must be safe to construct offline."""

    name: str = "base"

    def plan(self, question: str, tools: list[dict[str, Any]]) -> Plan:
        """Decide which tool(s) to call for the given question."""
        raise NotImplementedError

    def synthesize(self, question: str, tool_outputs: list[dict[str, Any]]) -> str:
        """Compose the final natural-language answer grounded in tool outputs."""
        raise NotImplementedError
