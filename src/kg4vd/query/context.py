"""Assemble the answer context: text items + image/text budgeting.

`TextItemBuilder`: per ranked page it emits a page-summary item, an OCR-text
item, and up to `max_per_page` entity/relation items, then scores each by
blending the graph (QGGE bridge) score with the query↔item embedding
similarity. `ContextBuilder` handles the token budgeting and formatting.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from kg4vd.core.types import PageHit, TextItem


def _clip(text: str, max_chars: int) -> str:
    value = " ".join(text.split())
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 1)].rstrip() + "…"


def _norm(vec: np.ndarray | None) -> np.ndarray | None:
    if vec is None:
        return None
    q = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(q))
    return q / n if n > 0 else q


def _card_text(card: Any) -> str:
    """Readable text for an entity/relation card - drop the structural header
    lines ([ENTITY]/[RELATION] / Document:) the embedding payload carries."""
    raw = (card.text_payload or "").strip()
    body = [
        ln for ln in raw.splitlines()
        if ln and not ln.startswith(("[ENTITY]", "[RELATION]", "Document:"))
    ]
    return " ".join(body).strip() or raw


class TextItemBuilder:
    """Build + score text items for the answer context from the index + pages."""

    def __init__(self, index: Any, pages_by_id: dict[int, Any]) -> None:
        self._pages = pages_by_id
        cards = index.cards()
        self._vectors = index.vectors()
        self._row_by_id = {c.evidence_id: i for i, c in enumerate(cards)}
        self._page_row_by_id: dict[int, int] = {}
        self._entities_by_page: dict[int, list[Any]] = defaultdict(list)
        self._relations_by_page: dict[int, list[Any]] = defaultdict(list)
        for i, c in enumerate(cards):
            if c.evidence_type == "page" and c.page_ids:
                self._page_row_by_id[int(c.page_ids[0])] = i
            elif c.evidence_type == "entity":
                for p in c.page_ids:
                    self._entities_by_page[int(p)].append(c)
            elif c.evidence_type == "relation":
                for p in c.page_ids:
                    self._relations_by_page[int(p)].append(c)

    def page_text_items(
        self, page_ids: Iterable[int], *, max_per_page: int = 4
    ) -> list[TextItem]:
        out: list[TextItem] = []
        for page_id in page_ids:
            page = self._pages.get(page_id)
            summary = (page.page_summary or "") if page else ""
            if summary:
                out.append(TextItem(
                    type="page", text=f"[p{page_id}] {summary}", score=0.0,
                    page_ids=[page_id], source_id=f"page:{page_id}",
                ))
            ocr = (page.text or "") if page else ""
            if ocr and ocr.strip() != summary.strip():
                out.append(TextItem(
                    type="page", text=f"[p{page_id}] text: {_clip(ocr, 1200)}",
                    score=0.0, page_ids=[page_id], source_id=f"page_text:{page_id}",
                ))
            for kind, by_page in (
                ("entity", self._entities_by_page),
                ("relation", self._relations_by_page),
            ):
                count = 0
                for card in by_page.get(page_id, []):
                    if count >= max_per_page:
                        break
                    text = _card_text(card)
                    if not text:
                        continue
                    out.append(TextItem(
                        type=kind, text=f"[p{page_id}] {text}", score=0.0,
                        page_ids=[int(p) for p in card.page_ids],
                        source_id=card.evidence_id,
                    ))
                    count += 1
        return out

    def _similarity(self, item: TextItem, q: np.ndarray | None) -> float | None:
        if q is None or item.source_id is None:
            return None
        if item.type in ("entity", "relation"):
            row = self._row_by_id.get(item.source_id)
        else:  # page
            page_id = item.page_ids[0] if item.page_ids else None
            row = self._page_row_by_id.get(page_id) if page_id is not None else None
        if row is None:
            return None
        return float(self._vectors[row] @ q)

    def build(
        self,
        node_scores: dict[str, float],
        *,
        query_vector: np.ndarray | None = None,
        page_ids: Iterable[int],
        graph_weight: float = 0.7,
        max_per_page: int = 4,
    ) -> list[TextItem]:
        items = self.page_text_items(page_ids, max_per_page=max_per_page)
        max_graph = max(node_scores.values(), default=0.0) or 1.0
        q = _norm(query_vector)
        for item in items:
            graph_score = node_scores.get(item.source_id or "", 0.0) / max_graph
            sim = self._similarity(item, q)
            if sim is None:
                item.score = float(graph_score)
            else:
                item.score = float(
                    graph_weight * graph_score + (1.0 - graph_weight) * max(sim, 0.0)
                )
        best: dict[str, TextItem] = {}
        for item in items:
            key = item.source_id or item.text
            old = best.get(key)
            if old is None or item.score > old.score:
                best[key] = item
        deduped = list(best.values())
        deduped.sort(key=lambda x: x.score, reverse=True)
        return deduped


class ContextBuilder:
    """Budget + format the retrieved evidence for the answer prompts."""

    def __init__(self, *, token_budget: int, image_count: int = 10) -> None:
        self.token_budget = token_budget
        self.image_count = image_count

    def images(self, pages: list[PageHit]) -> list[PageHit]:
        return [p for p in pages if p.image_path][: self.image_count]

    def texts(self, items: list[TextItem]) -> list[TextItem]:
        kept: list[TextItem] = []
        used = 0
        for item in items:
            approx = max(1, len(item.text) // 4)
            if kept and used + approx > self.token_budget:
                break
            kept.append(item)
            used += approx
        return kept

    @staticmethod
    def format_texts(items: list[TextItem]) -> str:
        return "\n".join(
            f"{i + 1}. [{item.type}; {_page_tag(item)}; score={item.score:.3f}] {item.text}"
            for i, item in enumerate(items)
        )


def _page_tag(item: TextItem) -> str:
    page_id = item.page_ids[0] if item.page_ids else None
    return f"p{page_id:03d}" if page_id is not None else "p?"
