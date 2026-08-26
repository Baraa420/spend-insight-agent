"""OpenAI-compatible LLM provider.

Uses the ``openai`` SDK against any OpenAI-compatible endpoint (set ``OPENAI_BASE_URL``).
The planner uses JSON-mode function-style routing; the synthesiser produces a grounded
answer. Import of the SDK is done lazily so the package works without it installed.

This path is exercised only when ``LLM_PROVIDER=openai`` and a key is configured; the
offline provider covers all tests/CI.
"""
from __future__ import annotations

import json
from typing import Any

from app.llm.base import BaseLLM, Plan, ToolCall

_PLANNER_SYSTEM = (
    "You are a routing planner for a Spend-domain assistant. "
    "Given a user question and the list of available tools, respond with STRICT JSON "
    'of the form {"steps":[{"tool":"<name>","args":{...}}],"rationale":"..."}. '
    "Available tools: "
    "spend_analytics(operation in [total, top_n, anomalies], category?, quarter?, by?, n?), "
    "policy_retriever(query), calculator(expression). "
    "Choose the minimal set of tools needed."
)

_SYNTH_SYSTEM = (
    "You are a precise finance assistant. Using ONLY the provided tool outputs, "
    "answer the user's question concisely. When policy text is provided, cite the "
    "document and section. Do not invent numbers."
)


class OpenAILLM(BaseLLM):
    name = "openai"

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        from openai import OpenAI  # lazy import

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def plan(self, question: str, tools: list[dict[str, Any]]) -> Plan:
        resp = self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PLANNER_SYSTEM},
                {"role": "user", "content": question},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        steps = [
            ToolCall(tool=s["tool"], args=s.get("args", {})) for s in data.get("steps", [])
        ]
        return Plan(steps=steps, rationale=data.get("rationale", ""))

    def synthesize(self, question: str, tool_outputs: list[dict[str, Any]]) -> str:
        context = json.dumps(tool_outputs, ensure_ascii=False, indent=2)
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYNTH_SYSTEM},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nTool outputs:\n{context}",
                },
            ],
        )
        return (resp.choices[0].message.content or "").strip()
