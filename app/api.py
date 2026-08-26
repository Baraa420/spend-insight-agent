"""FastAPI service exposing the Spend Insights Agent.

Endpoints:
    GET  /health  - service status, mode (offline/live), index size
    POST /ingest  - rebuild the RAG index from the docs folder
    POST /chat    - ask a natural-language question; returns grounded answer + citations
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app import __version__
from app.agent import SpendInsightAgent
from app.config import get_settings


# ---- schemas ---------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["How much did we spend on Travel?"])


class ChatResponse(BaseModel):
    answer: str
    tools_used: list[str]
    citations: list[str]
    trace_id: str
    elapsed_ms: float
    plan_rationale: str


class HealthResponse(BaseModel):
    status: str
    mode: str
    index_documents: int
    version: str


class IngestResponse(BaseModel):
    status: str
    index_documents: int


# ---- app -------------------------------------------------------------------
@lru_cache
def get_agent() -> SpendInsightAgent:
    return SpendInsightAgent(get_settings())


app = FastAPI(
    title="Spend Insights Agent",
    version=__version__,
    description="Local-first agentic AI for the Spend domain (RAG + analytics tools).",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    agent = get_agent()
    return HealthResponse(
        status="ok",
        mode=agent.settings.mode,
        index_documents=agent.retriever.size,
        version=__version__,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    agent = get_agent()
    count = agent.reindex()
    return IngestResponse(status="reindexed", index_documents=count)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    agent = get_agent()
    result = agent.chat(req.question)
    return ChatResponse(**result)
