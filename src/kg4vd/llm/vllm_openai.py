"""vLLM / SGLang / Ollama (any OpenAI-compatible local server)."""

from __future__ import annotations

from kg4vd.config.schema import LLMCfg
from kg4vd.core.registry import register
from kg4vd.llm.openai_compatible import OpenAICompatibleClient


@register("llm", "vllm")
class VLLMOpenAILLM(OpenAICompatibleClient):
    name = "vllm"
    default_base_url = "http://localhost:8000/v1"

    def __init__(self, cfg: LLMCfg):
        if cfg.base_url is None:
            cfg = cfg.model_copy(update={"base_url": self.default_base_url})
        if cfg.api_key_env in {"OPENAI_API_KEY", "OPENROUTER_API_KEY", ""}:
            cfg = cfg.model_copy(update={"api_key_env": "VLLM_API_KEY"})
        super().__init__(cfg)
