"""LLM factory: choose provider from settings, with safe fallback to offline."""
from __future__ import annotations

import logging

from app.llm.base import BaseLLM
from app.llm.offline import OfflineLLM

logger = logging.getLogger(__name__)


def get_llm(settings) -> BaseLLM:
    """Return the configured LLM provider.

    Falls back to the deterministic offline provider if the live provider cannot be
    constructed (missing key, missing SDK, etc.), so the service never hard-fails.
    """
    if settings.mode == "live":
        try:
            from app.llm.openai_llm import OpenAILLM

            return OpenAILLM(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                model=settings.openai_model,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Falling back to offline LLM: %s", exc)
    return OfflineLLM()
