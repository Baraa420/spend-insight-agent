# Requirements — Spend Insights Agent

## 1. Purpose

Spend Insights Agent is a local-first, enterprise-style **agentic AI service** for the
Corporate Functions & Analytics (CFA) *Spend* domain. It lets a finance/procurement user ask
natural-language questions and get grounded answers by combining:

- **Retrieval-Augmented Generation (RAG)** over company spend/procurement policy documents, and
- **Structured analytics** over a spend-transactions dataset (aggregations, top-N, anomalies).

The project demonstrates production-minded AI engineering: a multi-step agent (LangGraph),
tool/function calling, a microservice API (FastAPI), observability/tracing (AgentOps-style),
tests, CI, and containerisation.

## 2. Users & Key Use Cases

| User | Example question | Expected behaviour |
|---|---|---|
| Finance analyst | "What did we spend on Travel in Q1?" | Calls the spend-analytics tool, returns a figure + short explanation. |
| Procurement officer | "What is the approval limit for a single purchase order?" | Retrieves the procurement policy (RAG), answers with a citation. |
| Cost controller | "Show the top 3 vendors by spend and flag anomalies." | Analytics tool for top-N + anomaly detection, summarised answer. |
| Employee | "Can I expense a business-class flight?" | Retrieves travel/expense policy (RAG), answers with citation. |

## 3. Functional Requirements

- **FR-1 Chat endpoint.** `POST /chat` accepts a user question and returns an answer, the tools
  used, and source citations (when RAG is involved).
- **FR-2 Agentic orchestration.** A LangGraph agent routes each question to the right tool(s):
  policy retrieval, spend analytics, or calculator; it may use more than one tool per question.
- **FR-3 RAG over policies.** Policy documents are ingested, chunked, indexed, and retrieved by
  semantic/lexical similarity; answers grounded in retrieved chunks include citations.
- **FR-4 Spend analytics tool.** Query a transactions dataset: total by category/vendor/period,
  top-N, and simple statistical anomaly detection.
- **FR-5 Calculator tool.** Safe arithmetic for follow-up math (e.g., budget variance).
- **FR-6 Ingestion endpoint.** `POST /ingest` (re)builds the retrieval index from the docs folder.
- **FR-7 Health endpoint.** `GET /health` reports service status, mode (offline/live), and index size.
- **FR-8 Observability.** Every request emits a structured trace (spans for planner + each tool
  call, with timings and token/`step` metadata) to a local JSONL sink; optional LangFuse export.

## 4. Non-Functional Requirements

- **NFR-1 Local-first / offline.** The full pipeline must run and pass tests **without any API key**
  or external network call, using a deterministic offline LLM and a local (TF-IDF) retriever.
- **NFR-2 Pluggable LLM.** When `LLM_PROVIDER=openai` and an API key are set, the agent uses a real
  OpenAI-compatible chat model. No code change required — configuration only.
- **NFR-3 Reproducible & tested.** `pytest` suite runs green offline; GitHub Actions CI runs it on
  every push.
- **NFR-4 Containerised.** `docker build` + `docker compose up` start the service locally.
- **NFR-5 Config via environment.** All settings via env / `.env` (12-factor), validated with
  Pydantic settings.
- **NFR-6 Clean architecture.** Clear separation: API ← agent ← tools ← providers; each unit
  independently testable.
- **NFR-7 Secure by default.** No secrets in code; calculator is sandboxed (no `eval` of arbitrary
  code); inputs validated.

## 5. Out of Scope

- Authentication/authorisation (documented as a production follow-up).
- Real vector database and hosted embeddings (local TF-IDF used instead; design allows swapping).
- Fine-tuning or model training.

## 6. Acceptance Criteria

1. `pytest` passes with zero external dependencies/keys.
2. `GET /health` returns `mode: "offline"` when no key is configured.
3. `POST /chat` with "How much did we spend on Travel?" returns a numeric total and names the
   `spend_analytics` tool in the response.
4. `POST /chat` with "What is the PO approval limit?" returns an answer with at least one policy
   citation and names the `policy_retriever` tool.
5. CI workflow is green on GitHub.
6. `docker compose up` serves the API on the configured port.
