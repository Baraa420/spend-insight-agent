"""Agentic orchestration built on LangGraph.

The agent is a small ``StateGraph`` with three nodes:

    planner  ->  executor  ->  synthesizer  ->  END

- **planner**     asks the LLM (offline or live) which tool(s) to call.
- **executor**    runs each planned tool and collects results + citations.
- **synthesizer** composes the final grounded answer from the tool outputs.

Each node is wrapped in a tracer span so every run produces an inspectable trace
(AgentOps-style). The agent degrades gracefully if LangGraph is unavailable by
falling back to a direct plan->execute->synthesize call (kept simple and robust).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from app.config import Settings
from app.llm import get_llm
from app.llm.base import Plan
from app.obs.tracer import Trace, build_tracer
from app.rag.retriever import TfidfRetriever
from app.tools import CalculatorTool, PolicyRetrieverTool, SpendAnalyticsTool


class AgentState(TypedDict, total=False):
    question: str
    plan: Plan
    tool_outputs: list[dict[str, Any]]
    tools_used: list[str]
    citations: list[str]
    answer: str
    trace: Trace


class ChatResult(TypedDict):
    answer: str
    tools_used: list[str]
    citations: list[str]
    trace_id: str
    elapsed_ms: float
    plan_rationale: str


class SpendInsightAgent:
    """Coordinates the LLM planner, tools, and synthesizer via a LangGraph pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        data_dir = Path(self.settings.data_dir)

        self.retriever = TfidfRetriever(data_dir / "policies")
        self.llm = get_llm(self.settings)
        self.tracer = build_tracer(self.settings)

        self.tools = {
            t.name: t
            for t in (
                PolicyRetrieverTool(self.retriever, top_k=self.settings.top_k),
                SpendAnalyticsTool(data_dir / "spend_transactions.csv"),
                CalculatorTool(),
            )
        }
        self._graph = self._build_graph()

    # ---- graph construction -------------------------------------------------
    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception:  # pragma: no cover - fallback if langgraph missing
            return None

        graph = StateGraph(AgentState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("executor", self._executor_node)
        graph.add_node("synthesizer", self._synthesizer_node)
        graph.set_entry_point("planner")
        graph.add_edge("planner", "executor")
        graph.add_edge("executor", "synthesizer")
        graph.add_edge("synthesizer", END)
        return graph.compile()

    # ---- nodes --------------------------------------------------------------
    def _tool_schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def _planner_node(self, state: AgentState) -> AgentState:
        trace = state["trace"]
        with trace.span("planner", provider=self.llm.name) as sp:
            plan = self.llm.plan(state["question"], self._tool_schemas())
            sp.attributes["steps"] = [s.tool for s in plan.steps]
        state["plan"] = plan
        return state

    def _executor_node(self, state: AgentState) -> AgentState:
        trace = state["trace"]
        outputs: list[dict[str, Any]] = []
        tools_used: list[str] = []
        citations: list[str] = []

        for step in state["plan"].steps:
            tool = self.tools.get(step.tool)
            if tool is None:
                continue
            with trace.span(f"tool:{step.tool}", args=step.args) as sp:
                result = tool.run(**step.args)
                sp.attributes["ok"] = "error" not in result
            outputs.append({"tool": step.tool, "result": result})
            tools_used.append(step.tool)
            citations.extend(result.get("citations", []))

        state["tool_outputs"] = outputs
        state["tools_used"] = tools_used
        state["citations"] = citations
        return state

    def _synthesizer_node(self, state: AgentState) -> AgentState:
        trace = state["trace"]
        with trace.span("synthesizer", provider=self.llm.name):
            state["answer"] = self.llm.synthesize(state["question"], state["tool_outputs"])
        return state

    # ---- public API ---------------------------------------------------------
    def chat(self, question: str) -> ChatResult:
        trace = self.tracer.new_trace()
        state: AgentState = {"question": question, "trace": trace}

        if self._graph is not None:
            state = self._graph.invoke(state)
        else:  # pragma: no cover - fallback path
            state = self._synthesizer_node(self._executor_node(self._planner_node(state)))

        self.tracer.emit(trace)
        return ChatResult(
            answer=state.get("answer", ""),
            tools_used=state.get("tools_used", []),
            citations=state.get("citations", []),
            trace_id=trace.trace_id,
            elapsed_ms=trace.elapsed_ms,
            plan_rationale=state.get("plan").rationale if state.get("plan") else "",
        )

    def reindex(self) -> int:
        """Rebuild the retrieval index from disk; returns chunk count."""
        return self.retriever.build()
