"""Factory: ``build_llm(cfg)`` -> a concrete LLM client."""

from __future__ import annotations

from kg4vd.config.schema import LLMCfg
from kg4vd.core.registry import Registry


def build_llm(cfg: LLMCfg):
    """Resolve and instantiate an LLM backend from config.

    The backend is keyed by ``cfg.kind`` (e.g. ``openrouter``, ``openai``,
    ``vllm``, ``mock``).
    """

    cls = Registry.get("llm", cfg.kind)
    return cls(cfg)
