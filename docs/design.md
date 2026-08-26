# Design — Spend Insights Agent

## 1. Architecture Overview

```
                       ┌──────────────────────────────────────────────┐
                       │                FastAPI service               │
                       │  /health   /ingest   /chat                    │
                       └───────────────┬──────────────────────────────┘
                                       │ (ChatRequest)
                                       ▼
                       ┌──────────────────────────────────────────────┐
                       │           LangGraph Agent (planner)           │
                       │  plan → route → call tool(s) → synthesise      │
                       └───┬───────────────┬───────────────┬───────────┘
                           │               │               │
                 ┌─────────▼──┐   ┌────────▼────────┐   ┌──▼──────────┐
                 │ policy_    │   │ spend_analytics │   │ calculator  │
                 │ retriever  │   │  (pandas)       │   │ (safe math) │
                 │ (RAG,TFIDF)│   └─────────────────┘   └─────────────┘
                 └─────┬──────┘
                       │ retrieves
                 ┌─────▼──────┐
                 │ doc index  │ (policies/*.md → chunks → TF-IDF matrix)
                 └────────────┘

   Cross-cutting:  LLMProvider (offline | openai)   ·   Tracer (JSONL | LangFuse)   ·   Settings
```

## 2. Components

### 2.1 API layer (`app/api.py`)
- FastAPI app with Pydantic request/response models.
- Endpoints:
  - `GET /health` → `{status, mode, index_documents, version}`
  - `POST /ingest` → rebuilds index, returns chunk count.
  - `POST /chat` → `{answer, tools_used, citations, trace_id, elapsed_ms}`
- Dependency-injects a singleton `Agent` built from `Settings`.

### 2.2 Agent (`app/agent.py`)
- Built on **LangGraph** `StateGraph`. State = `{question, plan, tool_results, answer, citations,
  tools_used}`.
- Nodes:
  1. **planner** — asks the LLM (or offline heuristic) which tool(s) to use and with what args;
     produces a typed `Plan`.
  2. **tool executor** — runs the selected tools, appends results + citations to state.
  3. **synthesiser** — asks the LLM (or offline template) to produce the final grounded answer.
- Conditional edge: planner → executor → (loop if plan has more steps) → synthesiser → END.
- Every node wrapped in a tracer span.

### 2.3 Tools (`app/tools/`)
- **`policy_retriever`** — RAG. Given a query, returns top-k policy chunks with `doc`/`section`
  citations. Backed by `Retriever`.
- **`spend_analytics`** — pandas over `data/spend_transactions.csv`. Supports:
  `total` (by optional category/vendor/quarter), `top_n` (by category or vendor),
  `anomalies` (z-score on monthly totals).
- **`calculator`** — safe arithmetic via AST evaluation (no `eval`). Whitelisted operators only.
- Each tool has a name, JSON-serialisable input schema (Pydantic), and a `run()` method, so it can
  be exposed to a real LLM as a function-calling tool later without change.

### 2.4 Retriever (`app/rag/retriever.py`)
- Loads `data/policies/*.md`, splits into overlapping chunks (heading-aware).
- Vectoriser: **scikit-learn `TfidfVectorizer`** (fully local, deterministic, no downloads).
- `search(query, k)` → cosine similarity → top-k chunks with scores + citation metadata.
- Interface `BaseRetriever` allows a future swap to embeddings + a vector DB.

### 2.5 LLM provider (`app/llm/`)
- `BaseLLM.chat(messages, tools=None) -> LLMResponse`.
- **`OfflineLLM`** — deterministic, rule-based. Implements just enough "reasoning" for the planner
  (keyword routing) and the synthesiser (templated grounded answer). Guarantees offline operation
  and stable tests.
- **`OpenAILLM`** — uses `openai` SDK against any OpenAI-compatible endpoint (`OPENAI_BASE_URL`),
  supports tool/function calling. Selected when `LLM_PROVIDER=openai` and key present.
- `get_llm(settings)` factory picks the implementation; on any import/auth error it falls back to
  offline and logs a warning (robustness).

### 2.6 Observability (`app/obs/tracer.py`)
- `Tracer` creates a `trace_id` per request and records spans `{name, start, end, elapsed_ms,
  attributes}` to `traces/*.jsonl` (AgentOps-style).
- If `LANGFUSE_*` env vars are present, also emits to LangFuse (optional import, guarded).
- Chosen JSONL so reviewers can inspect traces with zero infra.

### 2.7 Config (`app/config.py`)
- Pydantic `Settings` (`.env` supported). Keys:
  `LLM_PROVIDER` (offline|openai), `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`,
  `TOP_K`, `DATA_DIR`, `TRACE_DIR`, `LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST`, `APP_PORT`.
- `mode` property = `"live"` if a usable key is configured else `"offline"`.

## 3. Data Flow (chat request)

1. API receives question → creates tracer trace.
2. Planner node: LLM/offline decides tool plan (e.g. `[spend_analytics(total, category=Travel)]`).
3. Executor node: runs tool(s); analytics returns number, retriever returns chunks+citations.
4. Synthesiser node: LLM/offline composes final answer grounded in tool outputs + citations.
5. API returns answer, `tools_used`, `citations`, `trace_id`, `elapsed_ms`.

## 4. Key Design Decisions

| Decision | Rationale |
|---|---|
| Local-first with offline LLM + TF-IDF | Runs and tests **anywhere** with no keys/network; reviewers can `pip install && pytest` and it just works. |
| LangGraph for orchestration | Matches the JD; gives explicit, inspectable multi-step agent state (planner/executor/synth). |
| Provider & retriever behind interfaces | Swap to a hosted LLM or vector DB via config only — shows production extensibility. |
| Tracing to JSONL, optional LangFuse | Demonstrates AgentOps/observability without forcing external infra. |
| Pydantic settings + schemas | 12-factor config, typed I/O, testability. |
| AST-based calculator | Security-by-default (no arbitrary `eval`). |

## 5. Testing Strategy

- **Unit:** retriever ranking, each tool, offline LLM routing, calculator safety, settings mode.
- **Integration:** build agent, run `/chat` questions end-to-end in offline mode; assert tool
  selection, citations, and numeric answers.
- **API:** FastAPI `TestClient` for `/health`, `/ingest`, `/chat`.
- All tests run offline in CI (GitHub Actions, Python 3.10/3.11 matrix).

## 6. Production Follow-ups (documented, not built)
- AuthN/AuthZ (OAuth2/JWT) on the API.
- Swap TF-IDF → embeddings + managed vector DB; swap offline LLM → SAP AI Core / hosted model.
- Rate limiting, caching, and PII redaction in traces.
- Deployment manifests (Helm/K8s) — mirrors my existing OpenShift experience.
