"""QGGE page propagation over entity/relation "bridges".

The KG's entities and relations act as *bridges*: each links the pages it appears on
(its ``page_ids``). Starting from the anchor pages, each round expands the page
frontier through the bridges touching it, scoring a bridge by
``sim(bridge_embedding, query)``. Final page score blends the anchor (vector)
score and the best bridge score via ``lambda_page``.

This uses only the index - entity/relation ``page_ids`` + their embeddings - so
it needs no KG node-node edges (the RetrievalGraph is not consulted).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kg4vd.core.types import PageHit


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return {key: 1.0 for key in scores}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


@dataclass
class _BridgeIndex:
    page_to_bridges: dict[int, list[int]]
    bridge_to_pages: list[list[int]]
    bridge_emb_row: list[int]       # matrix row of the bridge's embedding
    bridge_source_id: list[str]     # evidence_id of the entity/relation


class GraphPropagator:
    """Query-guided graph expansion over page bridges (entities/relations)."""

    def __init__(self, index: Any) -> None:
        self._matrix = index.vectors()
        cards = index.cards()
        self._page_image: dict[int, str | None] = {}
        valid_pages: set[int] = set()
        for c in cards:
            if c.evidence_type == "page" and c.page_ids:
                pid = int(c.page_ids[0])
                valid_pages.add(pid)
                self._page_image[pid] = c.image_payload
        self._valid_pages = valid_pages
        self._bridges = self._build_bridges(cards, valid_pages)

    @staticmethod
    def _build_bridges(cards: list[Any], valid_pages: set[int]) -> _BridgeIndex:
        page_to_bridges: dict[int, list[int]] = {p: [] for p in valid_pages}
        bridge_to_pages: list[list[int]] = []
        bridge_emb_row: list[int] = []
        bridge_source_id: list[str] = []
        for row, c in enumerate(cards):
            if c.evidence_type not in ("entity", "relation"):
                continue
            pages = sorted(
                {int(p) for p in (c.page_ids or []) if int(p) in valid_pages}
            )
            if not pages:
                continue
            idx = len(bridge_to_pages)
            bridge_to_pages.append(pages)
            bridge_emb_row.append(row)
            bridge_source_id.append(c.evidence_id)
            for p in pages:
                page_to_bridges.setdefault(p, []).append(idx)
        for p in list(page_to_bridges):
            page_to_bridges[p] = sorted(set(page_to_bridges[p]))
        return _BridgeIndex(
            page_to_bridges, bridge_to_pages, bridge_emb_row, bridge_source_id
        )

    def _bridge_score(self, bridge_idx: int, q: np.ndarray) -> float | None:
        if not self._bridges.bridge_to_pages[bridge_idx]:
            return None
        emb = self._matrix[self._bridges.bridge_emb_row[bridge_idx]]
        return float(emb @ q)

    def propagate(
        self,
        anchors: list[PageHit],
        *,
        query_vector: np.ndarray,
        rounds: int,
        page_beam: int,
        lambda_page: float,
        page_scores: dict[int, float] | None = None,
        final_k: int | None = None,
    ) -> tuple[list[PageHit], dict[str, float]]:
        b = self._bridges
        if not b.page_to_bridges:
            return anchors[: final_k or page_beam], {}

        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm

        anchor_ids = [hit.page_id for hit in anchors]
        frontier = list(dict.fromkeys(anchor_ids))
        best_bridge_by_page: dict[int, float] = {}
        scored_bridges: dict[int, float] = {}

        for _ in range(max(rounds, 0)):
            touched: set[int] = set()
            for page_id in frontier:
                touched.update(b.page_to_bridges.get(page_id, ()))
            round_pages: dict[int, float] = {}
            for bridge_idx in touched:
                score = scored_bridges.get(bridge_idx)
                if score is None:
                    score = self._bridge_score(bridge_idx, q)
                    if score is None:
                        continue
                    scored_bridges[bridge_idx] = score
                for page_id in b.bridge_to_pages[bridge_idx]:
                    old = round_pages.get(page_id)
                    if old is None or score > old:
                        round_pages[page_id] = score
            for page_id, score in round_pages.items():
                old = best_bridge_by_page.get(page_id)
                if old is None or score > old:
                    best_bridge_by_page[page_id] = score
            if not round_pages:
                break
            frontier = [
                page_id
                for page_id, _ in sorted(
                    round_pages.items(), key=lambda x: (-x[1], x[0])
                )[:page_beam]
            ]

        anchor_scores = page_scores or {hit.page_id: hit.score for hit in anchors}
        candidates = set(anchor_ids) | set(best_bridge_by_page)
        page_norm = _minmax(
            {page_id: anchor_scores.get(page_id, 0.0) for page_id in candidates}
        )
        bridge_norm = _minmax(best_bridge_by_page)
        joint = {
            page_id: lambda_page * page_norm.get(page_id, 0.0)
            + (1.0 - lambda_page) * bridge_norm.get(page_id, 0.0)
            for page_id in candidates
        }
        ranked = sorted(joint.items(), key=lambda x: (-x[1], x[0]))
        limit = final_k or page_beam

        hits: list[PageHit] = []
        for rank, (page_id, score) in enumerate(ranked[:limit], start=1):
            if page_id not in self._valid_pages:
                continue
            hits.append(
                PageHit(
                    page_id=page_id,
                    score=float(score),
                    rank=rank,
                    image_path=self._page_image.get(page_id),
                )
            )

        # Bridge scores attributed to their source entity/relation (by
        # evidence_id), consumed by the context builder (P3) for text items.
        node_scores: dict[str, float] = {}
        for bridge_idx, score in scored_bridges.items():
            sid = b.bridge_source_id[bridge_idx]
            node_scores[sid] = max(node_scores.get(sid, 0.0), float(score))
        return hits, node_scores


class QGGEPropagator:
    """Adapter giving QGGE the uniform ``propagate(anchors, query_vector)``
    contract (reads its knobs from ``cfg.query.qgge``), so the query pipeline
    can swap it with ``PPRPropagator``."""

    def __init__(self, index: Any, cfg: Any) -> None:
        self._prop = GraphPropagator(index)
        self._c = cfg

    def propagate(
        self, anchors: list[PageHit], *, query_vector: Any = None
    ) -> tuple[list[PageHit], dict[str, float]]:
        return self._prop.propagate(
            anchors, query_vector=query_vector,
            rounds=self._c.rounds, page_beam=self._c.page_beam,
            lambda_page=self._c.lambda_page,
        )
