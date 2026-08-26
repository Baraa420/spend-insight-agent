"""End-to-end agent tests (offline) and FastAPI endpoint tests."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import app


def test_agent_answers_spend_question(agent):
    result = agent.chat("How much did we spend on Travel?")
    assert "spend_analytics" in result["tools_used"]
    # answer should contain a number and EUR
    assert "EUR" in result["answer"]
    assert result["trace_id"]


def test_agent_answers_policy_question_with_citation(agent):
    result = agent.chat("What is the purchase order approval limit?")
    assert "policy_retriever" in result["tools_used"]
    assert result["citations"], "expected at least one policy citation"


def test_agent_detects_anomaly(agent):
    result = agent.chat("Are there any spend anomalies?")
    assert "spend_analytics" in result["tools_used"]
    assert "anomal" in result["answer"].lower() or "Travel" in result["answer"]


# ---- API tests -------------------------------------------------------------
client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["mode"] == "offline"
    assert body["index_documents"] > 0


def test_ingest_endpoint():
    resp = client.post("/ingest")
    assert resp.status_code == 200
    assert resp.json()["index_documents"] > 0


def test_chat_endpoint_spend():
    resp = client.post("/chat", json={"question": "How much did we spend on Software?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "spend_analytics" in body["tools_used"]
    assert "EUR" in body["answer"]


def test_chat_endpoint_validation():
    resp = client.post("/chat", json={"question": ""})
    assert resp.status_code == 422  # empty question rejected by schema
