"""Entrypoints: run the API server, or a quick CLI chat for local testing."""
from __future__ import annotations

import sys

from app.config import get_settings


def main() -> None:
    """Run the FastAPI server with uvicorn (used by the `spend-agent` script)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.api:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )


def cli() -> None:
    """Minimal offline CLI: `python -m app.main "your question"`."""
    from app.agent import SpendInsightAgent

    question = " ".join(sys.argv[1:]).strip() or "How much did we spend on Travel?"
    agent = SpendInsightAgent(get_settings())
    result = agent.chat(question)
    print(f"Q: {question}")
    print(f"A: {result['answer']}")
    print(f"   tools_used={result['tools_used']} citations={result['citations']}")
    print(f"   trace_id={result['trace_id']} elapsed_ms={result['elapsed_ms']}")


if __name__ == "__main__":
    # If args are passed, run the CLI; otherwise start the server.
    if len(sys.argv) > 1:
        cli()
    else:
        main()
