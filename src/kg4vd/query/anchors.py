"""Anchor retrieval: encode the query and cosine-search the page cards.

Returns the top-k anchor pages that seed graph propagation, using
`encode.encode_query` + the nano index filtered to `evidence_type=page`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kg4vd.core.types import PageHit, Query


@dataclass(slots=True)
class AnchorResult:
    pages: list[PageHit]
    query_vector: np.ndarray


class AnchorRetriever:
    """Encode a query and retrieve its top-k anchor pages from the index."""

    def __init__(self, index: Any, encoder: Any) -> None:
        self.index = index
        self.encoder = encoder

    async def retrieve(
        self,
        query: Query,
        *,
        top_k: int,
        doc_id: str | None = None,
        query_vector: np.ndarray | None = None,
    ) -> AnchorResult:
        # Precomputed vector (staged GME-encode phase) skips the encoder, so the
        # LLM phase can run with GME unloaded. self.encoder may be None then.
        if query_vector is not None:
            qv = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        else:
            qv = np.asarray(
                await self.encoder.encode_query(query), dtype=np.float32
            ).reshape(-1)
        filters: dict[str, Any] = {"evidence_type": "page"}
        if doc_id:
            filters["doc_id"] = doc_id
        hits = await self.index.search(qv, top_k=top_k, filters=filters)

        pages: list[PageHit] = []
        for rank, hit in enumerate(hits, start=1):
            card = hit.card
            if not card.page_ids:
                continue
            pages.append(
                PageHit(
                    page_id=int(card.page_ids[0]),
                    score=float(hit.score),
                    rank=rank,
                    image_path=card.image_payload,
                    metadata={"evidence_id": card.evidence_id, "doc_id": card.doc_id},
                )
            )
        return AnchorResult(pages=pages, query_vector=qv)
