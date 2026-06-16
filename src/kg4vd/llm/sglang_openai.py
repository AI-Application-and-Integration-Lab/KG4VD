"""SGLang local server (OpenAI-compatible).

Talks to a local ``sglang.launch_server`` instance - see
``scripts/launch_qwen36_sglang.sh`` (defaults to a Qwen3.6 MoE FP8 model on
``http://127.0.0.1:8004/v1``). Handy for cheap, offline development /
testing: point ``generator.llm.kind: sglang`` at the running server instead
of paying for OpenRouter.

The server ignores the API key, so no env var is required (blank
``api_key_env`` -> a dummy key downstream). Concurrency is governed by the
build stages' ``runtime.max_async`` caps (set ``extract``/``align`` up to the
server's batch size, e.g. 16).
"""

from __future__ import annotations

from kg4vd.config.schema import LLMCfg
from kg4vd.core.registry import register
from kg4vd.llm.openai_compatible import OpenAICompatibleClient


@register("llm", "sglang")
class SGLangLLM(OpenAICompatibleClient):
    name = "sglang"
    default_base_url = "http://127.0.0.1:8004/v1"

    def __init__(self, cfg: LLMCfg):
        if cfg.base_url is None:
            cfg = cfg.model_copy(update={"base_url": self.default_base_url})
        # sglang's OpenAI-compatible server needs no key. Don't force users
        # to export one - blank api_key_env yields a dummy key in the base
        # client. (Only override the cloud defaults so an explicit env var
        # still wins.)
        if cfg.api_key_env in {"OPENAI_API_KEY", "OPENROUTER_API_KEY", "VLLM_API_KEY"}:
            cfg = cfg.model_copy(update={"api_key_env": ""})
        super().__init__(cfg)
