"""Tests for the offline LLM routing and settings mode."""
from __future__ import annotations

from app.config import Settings
from app.llm.offline import OfflineLLM


def test_offline_routes_to_analytics_for_spend_question():
    llm = OfflineLLM()
    plan = llm.plan("How much did we spend on Travel in Q1?", tools=[])
    tools = [s.tool for s in plan.steps]
    assert "spend_analytics" in tools
    step = next(s for s in plan.steps if s.tool == "spend_analytics")
    assert step.args.get("category") == "Travel"
    assert step.args.get("quarter") == 1


def test_offline_routes_to_policy_for_policy_question():
    llm = OfflineLLM()
    plan = llm.plan("What is the purchase order approval limit?", tools=[])
    assert "policy_retriever" in [s.tool for s in plan.steps]


def test_offline_routes_top_n_vendor():
    llm = OfflineLLM()
    plan = llm.plan("Show the top 3 vendors by spend", tools=[])
    step = next(s for s in plan.steps if s.tool == "spend_analytics")
    assert step.args["operation"] == "top_n"
    assert step.args["by"] == "vendor"
    assert step.args["n"] == 3


def test_offline_routes_anomalies():
    llm = OfflineLLM()
    plan = llm.plan("Are there any spend anomalies?", tools=[])
    step = next(s for s in plan.steps if s.tool == "spend_analytics")
    assert step.args["operation"] == "anomalies"


def test_settings_mode_offline_by_default():
    s = Settings(llm_provider="offline")
    assert s.mode == "offline"


def test_settings_mode_offline_without_key_even_if_openai():
    s = Settings(llm_provider="openai", openai_api_key=None)
    assert s.mode == "offline"


def test_settings_mode_live_with_key():
    s = Settings(llm_provider="openai", openai_api_key="sk-test")
    assert s.mode == "live"
