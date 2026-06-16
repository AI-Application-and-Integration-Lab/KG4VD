"""LLM backends.

Concrete clients are registered via setuptools entry points (see
pyproject.toml). Use ``build_llm(cfg)`` to instantiate one from a config.
"""

from kg4vd.llm.base import LLMResponse
from kg4vd.llm.factory import build_llm

__all__ = ["LLMResponse", "build_llm"]
