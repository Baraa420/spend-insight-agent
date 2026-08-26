# Tasks — Spend Insights Agent

Implementation plan derived from `requirements.md` and `design.md`. Checked off as completed.

## Milestone 1 — Foundation
- [x] T1.1 Repo layout, `pyproject.toml` / `requirements.txt`, `.gitignore`, `LICENSE`.
- [x] T1.2 `app/config.py` — Pydantic `Settings`, `.env.example`.
- [x] T1.3 `app/obs/tracer.py` — JSONL tracer with spans + optional LangFuse hook.

## Milestone 2 — LLM providers
- [x] T2.1 `app/llm/base.py` — `BaseLLM`, message/response types.
- [x] T2.2 `app/llm/offline.py` — deterministic offline planner + synthesiser.
- [x] T2.3 `app/llm/openai_llm.py` — OpenAI-compatible provider (guarded import).
- [x] T2.4 `app/llm/factory.py` — `get_llm()` with fallback-to-offline.

## Milestone 3 — RAG + tools
- [x] T3.1 Sample data: `data/policies/*.md`, `data/spend_transactions.csv`.
- [x] T3.2 `app/rag/retriever.py` — chunking + TF-IDF search + citations.
- [x] T3.3 `app/tools/policy_retriever.py`.
- [x] T3.4 `app/tools/spend_analytics.py` (pandas: total / top_n / anomalies).
- [x] T3.5 `app/tools/calculator.py` (AST-safe).

## Milestone 4 — Agent + API
- [x] T4.1 `app/agent.py` — LangGraph planner→executor→synthesiser.
- [x] T4.2 `app/api.py` — FastAPI `/health`, `/ingest`, `/chat`.
- [x] T4.3 `app/main.py` / `run.py` entrypoint.

## Milestone 5 — Quality & delivery
- [x] T5.1 `tests/` — unit + integration + API (offline).
- [x] T5.2 `.github/workflows/ci.yml` — lint + tests matrix.
- [x] T5.3 `Dockerfile` + `docker-compose.yml`.
- [x] T5.4 `README.md` — quickstart, architecture, examples.

## Milestone 6 — Verify locally
- [x] T6.1 Create venv, install, `pytest` green.
- [x] T6.2 Smoke-test API (`/health`, `/chat`) locally.
