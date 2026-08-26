"""Application configuration via environment / .env (12-factor)."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings. All values are overridable via environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    llm_provider: str = Field(default="offline", description="offline | openai")
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # Retrieval
    top_k: int = 3

    # Paths
    data_dir: str = "data"
    trace_dir: str = "traces"

    # API
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # Optional LangFuse tracing
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def mode(self) -> str:
        """Effective operating mode: 'live' only if openai is selected AND a key is present."""
        if self.llm_provider.lower() == "openai" and self.openai_api_key:
            return "live"
        return "offline"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
