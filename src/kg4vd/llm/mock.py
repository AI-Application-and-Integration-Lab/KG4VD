"""Deterministic mock LLM. No network, no GPU.

Replays canned responses keyed by a substring of the prompt, falling back to a
trivial echo. Used by tests and CI to exercise the pipeline end-to-end.
"""

from __future__ import annotations

import hashlib
from typing import Any

from kg4vd.config.schema import LLMCfg
from kg4vd.core.registry import register
from kg4vd.llm.base import LLMResponse


@register("llm", "mock")
class MockLLM:
    name = "mock"
    supports_vision = True

    def __init__(self, cfg: LLMCfg):
        self.cfg = cfg
        self.model = cfg.model
        self._responses: dict[str, str] = {}
        self.call_count = 0

    def set_response(self, key_substr: str, text: str) -> None:
        """Register a canned response: returned when ``key_substr`` appears in
        the user prompt."""
        self._responses[key_substr] = text

    async def acomplete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        images: list[str] | None = None,
        history: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.call_count += 1
        text = self._lookup(prompt)

        # cheap, deterministic token count: words * 1.3
        ptokens = max(1, int(len(prompt.split()) * 1.3))
        ctokens = max(1, int(len(text.split()) * 1.3))
        return LLMResponse(
            text=text,
            prompt_tokens=ptokens,
            completion_tokens=ctokens,
            total_tokens=ptokens + ctokens,
            image_tokens=200 * (len(images) if images else 0),
            finish_reason="stop",
            raw={"mock": True, "call": self.call_count},
        )

    async def aclose(self) -> None:
        return None

    def _lookup(self, prompt: str) -> str:
        for k, v in self._responses.items():
            if k in prompt:
                return v
        # Default: a small JSON object so structured-answer paths can parse it.
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        return (
            '{"answer": "[mock-llm:' + digest + '] this is a deterministic mock answer.", '
            '"claims": [], "cited_pages": [], "cited_chunks": [], '
            '"cited_entities": [], "cited_relations": [], "cited_subgraphs": []}'
        )
