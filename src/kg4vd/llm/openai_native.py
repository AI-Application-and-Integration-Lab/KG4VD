"""OpenAI direct client."""

from __future__ import annotations

from kg4vd.config.schema import LLMCfg
from kg4vd.core.registry import register
from kg4vd.llm.openai_compatible import OpenAICompatibleClient


@register("llm", "openai")
class OpenAILLM(OpenAICompatibleClient):
    name = "openai"
    default_base_url = None  # let OpenAI client use its default

    def __init__(self, cfg: LLMCfg):
        # Force the env-var to OPENAI_API_KEY unless explicitly overridden in cfg.
        if cfg.api_key_env in {"OPENROUTER_API_KEY", "VLLM_API_KEY", ""}:
            cfg = cfg.model_copy(update={"api_key_env": "OPENAI_API_KEY"})
        super().__init__(cfg)
