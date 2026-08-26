# Spend Insights Agent

A **local-first, production-minded agentic AI service** for the Corporate Functions &
Analytics *Spend* domain. Ask questions in natural language and get grounded answers by
combining **Retrieval-Augmented Generation (RAG)** over policy documents with **structured
analytics** over a spend-transactions dataset.

Built to mirror a real enterprise AI-engineering stack: **LangGraph** agent orchestration,
**FastAPI** microservice, tool/function calling, **observability/tracing** (AgentOps-style),
tests, CI, and Docker.

> **Local-first by design:** the whole pipeline runs and all tests pass with **no API key and
> no network** thanks to a deterministic offline LLM and a local TF-IDF retriever. Point it at a
> real OpenAI-compatible model with a single environment variable — no code changes.

---

## Why this project

It demonstrates the skills required to build enterprise Generative/Agentic AI:

- **Agentic, multi-step orchestration** — a LangGraph `planner → executor → synthesizer` graph
  that routes each question to the right tool(s).
- **RAG** — heading-aware chunking + TF-IDF retrieval with citations; interface designed to swap
  in embeddings + a vector DB.
- **Tool / function calling** — three tools (policy retrieval, spend analytics, safe calculator),
  each with a schema usable by a live LLM.
- **MLOps / AgentOps** — every request emits an inspectable JSONL trace (spans + timings);
  optional LangFuse export.
- **Software engineering** — typed config (Pydantic), clean layering, unit/integration/API tests,
  GitHub Actions CI, Dockerised, secure-by-default (sandboxed calculator, non-root container).

---

## Architecture

```
FastAPI (/health /ingest /chat)
        │
        ▼
LangGraph agent:  planner ──▶ executor ──▶ synthesizer
                     │            │
        LLM (offline│live)   tools: policy_retriever (RAG/TF-IDF)
                                    spend_analytics  (pandas)
                                    calculator       (AST-safe)
Cross-cutting: Settings · Tracer (JSONL | LangFuse)
```

See [`docs/design.md`](docs/design.md) for the full design, and
[`docs/requirements.md`](docs/requirements.md) / [`docs/tasks.md`](docs/tasks.md) for the
spec-driven breakdown.

---

## Quickstart

```bash
# 1. Create a virtualenv and install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Run the tests (fully offline, no key needed)
pytest -q

# 3. Try it from the CLI
python -m app.main "How much did we spend on Travel?"
python -m app.main "What is the purchase order approval limit?"
python -m app.main "Show the top 3 vendors by spend"
python -m app.main "Are there any spend anomalies?"

# 4. Run the API
uvicorn app.api:app --reload
# then:
curl localhost:8000/health
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
     -d '{"question":"How much did we spend on Travel?"}'
```

### With Docker

```bash
docker compose up --build
curl localhost:8000/chat -X POST -H 'content-type: application/json' \
     -d '{"question":"What is the PO approval limit?"}'
```

---

## Configuration

All settings come from environment variables / `.env` (see [`.env.example`](.env.example)).

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `offline` | `offline` (no key) or `openai` |
| `OPENAI_API_KEY` | – | required when `LLM_PROVIDER=openai` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | any OpenAI-compatible endpoint |
| `OPENAI_MODEL` | `gpt-4o-mini` | model name |
| `TOP_K` | `3` | retrieved chunks per query |
| `LANGFUSE_*` | – | optional AgentOps tracing |

### Switching to a live model

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
# optional: export OPENAI_BASE_URL / OPENAI_MODEL
uvicorn app.api:app
```

`GET /health` reports the active `mode` (`offline` / `live`).

---

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | status, mode, index size, version |
| POST | `/ingest` | rebuild the RAG index from `data/policies/` |
| POST | `/chat` | `{ "question": "..." }` → answer, tools_used, citations, trace_id |

---

## Observability

Each `/chat` request writes a trace to `traces/<trace_id>.jsonl` containing a span for the
planner and for every tool call, with timings and attributes — an AgentOps-style audit trail.
Set `LANGFUSE_*` to also export to LangFuse.

---

## Tests & CI

```bash
pytest -q --cov=app
ruff check app tests
```

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs lint + tests on
Python 3.10 / 3.11 / 3.12 for every push and PR.

---

## Production follow-ups (documented, not built)

- AuthN/AuthZ (OAuth2/JWT) on the API.
- Swap TF-IDF → embeddings + managed vector DB; offline LLM → SAP AI Core / hosted model.
- Rate limiting, response caching, PII redaction in traces.
- Kubernetes/Helm deployment manifests.

## License

MIT — see [`LICENSE`](LICENSE).
