"""Shared OpenAI-compatible client used by openai / openrouter / vllm clients.

All three backends speak the same Chat Completions API; they only differ in
``base_url``, headers, and which env-var holds the key. This module factors
the common code out so each backend file is ~30 lines.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from kg4vd.config.schema import LLMCfg
from kg4vd.core.errors import LLMError, StreamingNotSupported
from kg4vd.llm.base import LLMResponse, env_or_raise
from kg4vd.obs.tracer import record_llm_usage
from kg4vd.utils.images import encode_image_to_data_url

logger = logging.getLogger(__name__)


class OpenAICompatibleClient:
    """Base class. Subclasses set ``name``, ``default_base_url`` (or rely on
    OpenAI's default), and ``default_headers``."""

    name: str = "openai_compatible"
    supports_vision: bool = True
    default_base_url: str | None = None
    default_headers: dict[str, str] = {}

    def __init__(self, cfg: LLMCfg):
        self.cfg = cfg
        self.model = cfg.model
        # A blank api_key_env means "this server needs no key" (local
        # vLLM / SGLang / Ollama) - the OpenAI SDK still wants a non-empty
        # string, so pass a dummy.
        api_key = env_or_raise(cfg.api_key_env) if cfg.api_key_env else "EMPTY"
        base_url = cfg.base_url or self.default_base_url
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=cfg.timeout_s,
            default_headers=self.default_headers or None,
        )

    # ----- public --------------------------------------------------------

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
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        if stream:
            raise StreamingNotSupported(
                "Streaming is not supported by this client."
            )

        messages = await self._build_messages(prompt, system, images, history)
        call_kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            call_kwargs["response_format"] = response_format
        for k, v in kwargs.items():
            if v is not None and k not in {"hashing_kv", "keyword_extraction"}:
                call_kwargs[k] = v
        # Forward provider-specific knobs (e.g. sglang/vLLM
        # `chat_template_kwargs={"enable_thinking": False}` to switch off a
        # reasoning model's thinking) via the OpenAI SDK's extra_body.
        if self.cfg.extra:
            call_kwargs["extra_body"] = {
                **self.cfg.extra, **call_kwargs.get("extra_body", {})
            }

        retrier = AsyncRetrying(
            stop=stop_after_attempt(self.cfg.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            retry=retry_if_exception_type(
                (RateLimitError, APIConnectionError, APITimeoutError, LLMError)
            ),
            reraise=True,
        )

        async for attempt in retrier:
            with attempt:
                try:
                    response = await self._client.chat.completions.create(
                        **call_kwargs
                    )
                except (RateLimitError, APIConnectionError, APITimeoutError):
                    raise
                except Exception as e:  # noqa: BLE001
                    logger.error("LLM call failed: %r", e)
                    raise LLMError(repr(e)) from e

                if (
                    not response
                    or not response.choices
                    or not response.choices[0].message
                    or not response.choices[0].message.content
                ):
                    raise LLMError("Empty response from LLM")

                content = response.choices[0].message.content or ""
                usage = response.usage
                result = LLMResponse(
                    text=content,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                    completion_tokens=(
                        getattr(usage, "completion_tokens", 0) if usage else 0
                    ),
                    total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
                    image_tokens=0,  # OpenAI doesn't separate; treat as in prompt_tokens
                    finish_reason=response.choices[0].finish_reason or "stop",
                    raw=response,
                )
                # Attribute token usage to the open tracer stage so the run
                # manifest's per-stage token / llm-call totals are real.
                record_llm_usage(result)
                return result
        # tenacity reraises; this is unreachable
        return LLMResponse.empty()

    # ----- helpers -------------------------------------------------------

    async def _build_messages(
        self,
        prompt: str,
        system: str | None,
        images: list[str] | None,
        history: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)

        if images:
            content: list[dict[str, Any]] = []
            for img in images:
                url = await encode_image_to_data_url(img)
                content.append({"type": "image_url", "image_url": {"url": url}})
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        return messages

    async def aclose(self) -> None:
        await self._client.close()
