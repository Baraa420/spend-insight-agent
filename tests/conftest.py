"""Shared pytest fixtures."""
from __future__ import annotations

import pytest

from app.agent import SpendInsightAgent
from app.config import Settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    # Force offline mode regardless of environment.
    return Settings(llm_provider="offline", data_dir="data", trace_dir="traces")


@pytest.fixture(scope="session")
def agent(settings: Settings) -> SpendInsightAgent:
    return SpendInsightAgent(settings)
