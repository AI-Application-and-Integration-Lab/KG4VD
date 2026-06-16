"""Common LLMResponse + base class shared by OpenAI-compatible clients."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    image_tokens: int = 0
    finish_reason: str = "stop"
    raw: Any = None

    @classmethod
    def empty(cls, *, finish_reason: str = "empty") -> "LLMResponse":
        return cls(text="", finish_reason=finish_reason)


def env_or_raise(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(
            f"Required environment variable {var!r} is not set; "
            f"see .env.example"
        )
    return val
