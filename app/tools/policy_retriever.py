"""Policy retrieval tool (RAG): returns grounded policy chunks with citations."""
from __future__ import annotations

from typing import Any

from app.rag.retriever import BaseRetriever
from app.tools.base import BaseTool


class PolicyRetrieverTool(BaseTool):
    name = "policy_retriever"
    description = (
        "Retrieve relevant company spend/procurement/travel policy text for a natural-language "
        "query. Returns ranked chunks with document and section citations."
    )

    def __init__(self, retriever: BaseRetriever, top_k: int = 3) -> None:
        self._retriever = retriever
        self._top_k = top_k

    def run(self, query: str, k: int | None = None, **_: Any) -> dict[str, Any]:
        chunks = self._retriever.search(query, k=k or self._top_k)
        citations = [f"{c['doc']} \u2014 {c['section']}" for c in chunks]
        return {"chunks": chunks, "citations": citations}
