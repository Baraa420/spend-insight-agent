"""Deterministic offline LLM.

Provides just enough "reasoning" to run the full agent with no API key and no network:
- ``plan`` uses transparent keyword heuristics to route to tools.
- ``synthesize`` composes a grounded answer from the tool outputs using templates.

This keeps the project fully local-first and makes tests reproducible. The same
interface is implemented by the OpenAI provider, so switching is configuration-only.
"""
from __future__ import annotations

import re
from typing import Any

from app.llm.base import BaseLLM, Plan, ToolCall

_CATEGORIES = [
    "Travel",
    "Software",
    "Hardware",
    "Consulting",
    "Marketing",
    "Facilities",
    "Training",
]
_QUARTER_RE = re.compile(r"\bq([1-4])\b", re.IGNORECASE)


class OfflineLLM(BaseLLM):
    name = "offline"

    # ---- planning -----------------------------------------------------------
    def plan(self, question: str, tools: list[dict[str, Any]]) -> Plan:
        q = question.lower()
        steps: list[ToolCall] = []

        wants_analytics = any(
            kw in q
            for kw in ("spend", "spent", "total", "top", "vendor", "anomal", "budget", "how much")
        )
        wants_policy = any(
            kw in q
            for kw in (
                "policy",
                "allowed",
                "can i",
                "limit",
                "approve",
                "approval",
                "reimburse",
                "per diem",
                "class flight",
                "business class",
                "rule",
            )
        )

        if wants_analytics:
            steps.append(ToolCall(tool="spend_analytics", args=self._analytics_args(q)))
        if wants_policy or not steps:
            # Policy retrieval is the safe default when nothing else matched.
            steps.append(ToolCall(tool="policy_retriever", args={"query": question}))

        rationale = "offline keyword routing: " + ", ".join(s.tool for s in steps)
        return Plan(steps=steps, rationale=rationale)

    def _analytics_args(self, q: str) -> dict[str, Any]:
        if "anomal" in q:
            return {"operation": "anomalies"}
        if "top" in q:
            n = 3
            m = re.search(r"top\s+(\d+)", q)
            if m:
                n = int(m.group(1))
            by = "vendor" if "vendor" in q else "category"
            return {"operation": "top_n", "n": n, "by": by}

        args: dict[str, Any] = {"operation": "total"}
        for cat in _CATEGORIES:
            if cat.lower() in q:
                args["category"] = cat
                break
        qm = _QUARTER_RE.search(q)
        if qm:
            args["quarter"] = int(qm.group(1))
        return args

    # ---- synthesis ----------------------------------------------------------
    def synthesize(self, question: str, tool_outputs: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for out in tool_outputs:
            tool = out.get("tool")
            result = out.get("result", {})
            if tool == "spend_analytics":
                parts.append(self._render_analytics(result))
            elif tool == "policy_retriever":
                parts.append(self._render_policy(result))
            elif tool == "calculator":
                parts.append(f"The result of the calculation is {result.get('value')}.")
        if not parts:
            return "I could not find enough information to answer that."
        return " ".join(p for p in parts if p)

    def _render_analytics(self, result: dict[str, Any]) -> str:
        op = result.get("operation")
        if op == "total":
            scope = []
            if result.get("category"):
                scope.append(f"category '{result['category']}'")
            if result.get("quarter"):
                scope.append(f"Q{result['quarter']}")
            scope_txt = (" for " + " in ".join(scope)) if scope else ""
            return f"Total spend{scope_txt} is {result.get('total_eur')} EUR across {result.get('count')} transactions."
        if op == "top_n":
            rows = result.get("rows", [])
            listed = "; ".join(f"{r['key']} ({r['total_eur']} EUR)" for r in rows)
            return f"Top {len(rows)} by {result.get('by')}: {listed}."
        if op == "anomalies":
            rows = result.get("anomalies", [])
            if not rows:
                return "No spend anomalies were detected (no month exceeded 2 standard deviations)."
            listed = "; ".join(
                f"{r['category']} in {r['month']} ({r['total_eur']} EUR, z={r['z_score']})"
                for r in rows
            )
            return f"Detected {len(rows)} anomaly(ies): {listed}."
        return ""

    def _render_policy(self, result: dict[str, Any]) -> str:
        chunks = result.get("chunks", [])
        if not chunks:
            return "No relevant policy section was found."
        top = chunks[0]
        cite = f"{top['doc']} \u2014 {top['section']}"
        return f"According to policy ({cite}): {top['text']}"
