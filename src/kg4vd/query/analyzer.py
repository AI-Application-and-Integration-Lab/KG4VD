"""Query analyzer - route a query single / multi / document_level."""
from __future__ import annotations

from typing import Any

from kg4vd.core.types import Query, QueryAnalysis
from kg4vd.query.prompts import PromptSet
from kg4vd.utils.json_repair import parse_json_object_loose

_ROUTES = {"single", "multi", "document_level"}


class QueryAnalyzer:
    def __init__(self, llm: Any, prompts: PromptSet) -> None:
        self.llm = llm
        self.prompts = prompts

    async def analyze(self, query: Query) -> QueryAnalysis:
        prompt = self.prompts.render("query_analyzer", query=query.text)
        resp = await self.llm.acomplete(
            prompt,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=256,
        )
        try:
            data = parse_json_object_loose(resp.text)
        except Exception:  # noqa: BLE001
            data = {}
        if not isinstance(data, dict):
            data = {}
        route = data.get("route", "multi")
        if route not in _ROUTES:
            route = "multi"
        return QueryAnalysis(
            route=route,
            rationale=str(data.get("reason") or data.get("rationale") or ""),
            confidence=data.get("confidence"),
            answer_format=query.answer_format,
        )
