"""Page reranker - Qwen3-VL-Reranker over HTTP.

Reranks the QGGE page candidates against the query with a dedicated multimodal
CrossEncoder served by ``services/reranker`` (see scripts/launch_reranker.sh).
It promotes gold pages from ranks 11..20 into the top-10 the generator sees.

Default-off (``query.reranker.enabled``). The server runs out-of-process (it
needs transformers>=4.57, incompatible with the GME encoder's env), reached via
``query.reranker.url`` / ``KG4VD_RERANKER_URL``. Because it contends with sglang
for GPU, reranking runs in the retrieval phase (run_query_batch), not alongside
generation.
"""
from __future__ import annotations

import os
from typing import Any, Protocol

import httpx

from kg4vd.core.types import PageHit, Query


class RerankerProtocol(Protocol):
    async def rerank(self, query: Query, pages: list[PageHit]) -> list[PageHit]: ...


class NoOpReranker:
    async def rerank(self, query: Query, pages: list[PageHit]) -> list[PageHit]:
        del query
        return pages


def _doc(page: PageHit) -> dict[str, Any]:
    """OpenAI-style multimodal doc (page image + id text) for /score."""
    content: list[dict[str, Any]] = [{"type": "text", "text": f"page {page.page_id}"}]
    if page.image_path:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"file://{os.path.abspath(page.image_path)}"},
        })
    return {"content": content}


class Qwen3VLReranker:
    """HTTP client for the Qwen3-VL-Reranker /score endpoint."""

    def __init__(
        self,
        *,
        url: str,
        model: str = "Qwen/Qwen3-VL-Reranker-2B",
        candidate_size: int = 20,
        top_k: int = 10,
        timeout: float = 120.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.candidate_size = max(1, int(candidate_size))
        self.top_k = max(1, int(top_k))
        self._timeout = timeout

    async def rerank(self, query: Query, pages: list[PageHit]) -> list[PageHit]:
        candidates = [p for p in pages if p.image_path][: self.candidate_size]
        if len(candidates) <= 1:
            return pages
        body = {
            "model": self.model,
            "text_1": query.text,
            "text_2": [_doc(p) for p in candidates],
        }
        async with httpx.AsyncClient(
            base_url=self.url, timeout=httpx.Timeout(self._timeout, connect=5.0)
        ) as client:
            resp = await client.post("/score", json=body)
            resp.raise_for_status()
            scores = {item["index"]: float(item["score"]) for item in resp.json()["data"]}

        order = sorted(range(len(candidates)), key=lambda i: scores.get(i, -1e9), reverse=True)
        reranked = [candidates[i] for i in order]
        selected = reranked[: self.top_k]
        selected_ids = {p.page_id for p in selected}
        tail = [p for p in pages if p.page_id not in selected_ids]
        out = [*selected, *tail]
        for rank, page in enumerate(out, start=1):
            page.rank = rank
        return out


def build_reranker(cfg: Any) -> RerankerProtocol:
    """Construct the reranker for ``cfg.query.reranker`` (NoOp when disabled)."""
    rc = cfg.query.reranker
    if not rc.enabled:
        return NoOpReranker()
    url = rc.url or os.environ.get("KG4VD_RERANKER_URL", "http://127.0.0.1:8003")
    return Qwen3VLReranker(
        url=url, model=rc.model, candidate_size=rc.candidate_size, top_k=rc.top_k
    )
