"""Query prompt templates.

The prompt text lives in ``<name>/*.md`` and is rendered by ``PromptSet`` via
``{{ key }}`` substitution. The ``PromptSet`` name selects the subdirectory
(default: ``default``).
"""
from __future__ import annotations

from pathlib import Path


class PromptSet:
    def __init__(self, name: str = "default", *, root: Path | None = None) -> None:
        base = root or Path(__file__).resolve().parent
        self.root = base / name
        self.name = name

    def render(self, template: str, **values: object) -> str:
        path = self.root / f"{template}.md"
        text = path.read_text(encoding="utf-8")
        for key, value in values.items():
            text = text.replace("{{ " + key + " }}", str(value))
        return text


__all__ = ["PromptSet"]
