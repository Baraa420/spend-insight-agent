"""Base tool interface shared by all agent tools."""
from __future__ import annotations

from typing import Any


class BaseTool:
    name: str = "base"
    description: str = ""

    def run(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def schema(self) -> dict[str, Any]:
        """A JSON-serialisable description usable for LLM function-calling."""
        return {"name": self.name, "description": self.description}
