"""Unit tests for the individual tools and retriever."""
from __future__ import annotations

from pathlib import Path

from app.rag.retriever import TfidfRetriever
from app.tools import CalculatorTool, PolicyRetrieverTool, SpendAnalyticsTool

DATA = Path("data")


def test_retriever_builds_and_searches():
    r = TfidfRetriever(DATA / "policies")
    assert r.size > 0
    hits = r.search("purchase order approval limit", k=3)
    assert hits
    assert {"doc", "section", "text", "score"} <= set(hits[0])
    # procurement policy should rank for approval-limit query
    assert any("procurement" in h["doc"] for h in hits)


def test_policy_retriever_tool_returns_citations():
    r = TfidfRetriever(DATA / "policies")
    tool = PolicyRetrieverTool(r, top_k=2)
    out = tool.run(query="can I fly business class?")
    assert out["chunks"]
    assert out["citations"]
    assert " \u2014 " in out["citations"][0]


def test_spend_total_all():
    tool = SpendAnalyticsTool(DATA / "spend_transactions.csv")
    out = tool.run(operation="total")
    assert out["total_eur"] > 0
    assert out["count"] == 24


def test_spend_total_by_category():
    tool = SpendAnalyticsTool(DATA / "spend_transactions.csv")
    out = tool.run(operation="total", category="Travel")
    assert out["category"] == "Travel"
    assert out["total_eur"] > 0
    assert out["count"] >= 6


def test_spend_total_by_month():
    tool = SpendAnalyticsTool(DATA / "spend_transactions.csv")
    out = tool.run(operation="total", category="Travel", month="2025-02")
    # Only the February Travel transaction (980.00) should be counted.
    assert out["month"] == "2025-02"
    assert out["count"] == 1
    assert out["total_eur"] == 980.00


def test_spend_month_differs_from_year_total():
    tool = SpendAnalyticsTool(DATA / "spend_transactions.csv")
    year = tool.run(operation="total", category="Travel")
    month = tool.run(operation="total", category="Travel", month="2025-01")
    assert month["total_eur"] < year["total_eur"]


def test_spend_top_n_vendor():
    tool = SpendAnalyticsTool(DATA / "spend_transactions.csv")
    out = tool.run(operation="top_n", by="vendor", n=3)
    assert out["by"] == "vendor"
    assert len(out["rows"]) == 3
    # rows sorted descending
    totals = [r["total_eur"] for r in out["rows"]]
    assert totals == sorted(totals, reverse=True)


def test_spend_anomaly_detection_flags_travel():
    tool = SpendAnalyticsTool(DATA / "spend_transactions.csv")
    out = tool.run(operation="anomalies")
    # The April Travel spike (GlobeTrek 11800) should be flagged.
    assert any(a["category"] == "Travel" for a in out["anomalies"])


def test_calculator_safe_math():
    tool = CalculatorTool()
    out = tool.run(expression="(11800-1200)/1200*100")
    assert "value" in out
    assert round(out["value"], 2) == 883.33


def test_calculator_rejects_unsafe_input():
    tool = CalculatorTool()
    out = tool.run(expression="__import__('os').system('echo hacked')")
    assert "error" in out
