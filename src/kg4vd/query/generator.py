"""Answer generation: text / image / fusion.

Generates answers via ``llm.acomplete`` using the text_answer / image_answer /
fusion_answer / answer_guidelines prompt templates.
"""
from __future__ import annotations

from typing import Any

from kg4vd.core.types import Answer, PageHit, Query, TextItem
from kg4vd.query.context import ContextBuilder
from kg4vd.query.prompts import PromptSet
from kg4vd.utils.json_repair import parse_json_object_loose

_MAX_TOKENS = 1024


def _usage(resp: Any) -> dict[str, Any]:
    return {
        "prompt_tokens": getattr(resp, "prompt_tokens", 0),
        "completion_tokens": getattr(resp, "completion_tokens", 0),
        "total_tokens": getattr(resp, "total_tokens", 0),
    }


class AnswerGenerator:
    def __init__(self, llm: Any, prompts: PromptSet) -> None:
        self.llm = llm
        self.prompts = prompts
        self.guidelines = prompts.render("answer_guidelines")

    async def from_images(self, query: Query, pages: list[PageHit]) -> Answer:
        prompt = self.prompts.render(
            "image_answer",
            query=query.text,
            pages=", ".join(str(p.page_id) for p in pages),
        )
        resp = await self.llm.acomplete(
            prompt,
            images=[p.image_path for p in pages if p.image_path],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=_MAX_TOKENS,
        )
        return _answer_from_parsed(
            _parse_answer(resp.text), raw=resp.text, usage=_usage(resp),
            default_pages=[p.page_id for p in pages],
        )

    async def from_texts(self, query: Query, items: list[TextItem]) -> Answer:
        prompt = self.prompts.render(
            "text_answer",
            answer_guidelines=str(
                query.answer_guidelines
                or "No additional answer guidelines were provided."
            ),
            query=query.text,
            text_context=ContextBuilder.format_texts(items) or "(No evidence.)",
        )
        resp = await self.llm.acomplete(
            prompt, response_format={"type": "json_object"},
            temperature=0.0, max_tokens=_MAX_TOKENS,
        )
        cited = sorted({p for item in items for p in item.page_ids})
        return _answer_from_parsed(
            _parse_answer(resp.text), raw=resp.text, usage=_usage(resp),
            default_pages=cited,
        )

    async def fuse(self, query: Query, image_answer: Answer, text_answer: Answer) -> Answer:
        prompt = self.prompts.render(
            "fusion_answer",
            query=query.text,
            image_draft=_answer_block(image_answer),
            text_draft=_answer_block(text_answer),
        )
        resp = await self.llm.acomplete(
            prompt, response_format={"type": "json_object"},
            temperature=0.0, max_tokens=_MAX_TOKENS,
        )
        cited = sorted(set(image_answer.cited_pages) | set(text_answer.cited_pages))
        return _answer_from_parsed(
            _parse_answer(resp.text), raw=resp.text, usage=_usage(resp),
            default_pages=cited,
        )


def _parse_answer(text: str) -> dict[str, Any]:
    try:
        parsed = parse_json_object_loose(text)
    except Exception:  # noqa: BLE001
        parsed = None
    if isinstance(parsed, dict) and parsed:
        nested = parsed.get("answer")
        if isinstance(nested, str) and nested.lstrip().startswith("{"):
            try:
                nested_parsed = parse_json_object_loose(nested)
            except Exception:  # noqa: BLE001
                nested_parsed = None
            if isinstance(nested_parsed, dict) and nested_parsed:
                return nested_parsed
        return parsed
    return {
        "answer": text.strip(), "reasoning": "",
        "cited_pages": [], "confidence": "low", "failure_reason": None,
    }


def _answer_from_parsed(
    parsed: dict[str, Any], *, raw: str, usage: dict[str, Any], default_pages: list[int]
) -> Answer:
    cited: list[int] = []
    for value in parsed.get("cited_pages") or []:
        try:
            cited.append(int(value))
        except (TypeError, ValueError):
            continue
    return Answer(
        text=str(parsed.get("answer") or ""),
        cited_pages=cited or default_pages,
        confidence=_confidence(parsed.get("confidence")),
        raw=raw,
        usage=usage,
        metadata={
            "reasoning": parsed.get("reasoning"),
            "failure_reason": parsed.get("failure_reason"),
        },
    )


def _confidence(value: object) -> str:
    return str(value) if value in {"high", "medium", "low"} else "low"


def _answer_block(answer: Answer) -> str:
    return (
        f"answer: {answer.text}\n"
        f"reasoning: {answer.metadata.get('reasoning') or ''}\n"
        f"cited_pages: {answer.cited_pages}\n"
        f"confidence: {answer.confidence}\n"
    )
