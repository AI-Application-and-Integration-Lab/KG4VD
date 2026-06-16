"""OpenRouter - one key, many providers (GPT, Claude, Gemini, Qwen, Llama, ...)."""

from __future__ import annotations

import os

from kg4vd.config.schema import LLMCfg
from kg4vd.core.registry import register
from kg4vd.llm.openai_compatible import OpenAICompatibleClient


@register("llm", "openrouter")
class OpenRouterLLM(OpenAICompatibleClient):
    name = "openrouter"
    default_base_url = "https://openrouter.ai/api/v1"

    def __init__(self, cfg: LLMCfg):
        if cfg.base_url is None:
            cfg = cfg.model_copy(update={"base_url": self.default_base_url})
        if cfg.api_key_env in {"OPENAI_API_KEY", "VLLM_API_KEY", ""}:
            cfg = cfg.model_copy(update={"api_key_env": "OPENROUTER_API_KEY"})
        # Set per-call default headers so OpenRouter shows our app in their dashboard.
        self.default_headers = {
            "HTTP-Referer": os.environ.get(
                "OPENROUTER_HTTP_REFERER", "https://github.com/Amiannn/KG4VD"
            ),
            "X-Title": os.environ.get("OPENROUTER_X_TITLE", "KG4VD"),
        }
        super().__init__(cfg)
