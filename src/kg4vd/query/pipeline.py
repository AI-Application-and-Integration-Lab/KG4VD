"""Unified query pipeline: analyze → anchor → propagate → (rerank) → context
→ generate.
"""
from __future__ import annotations

import asyncio
from typing import Any

from kg4vd.core.types import Answer, Query, QueryResult
from kg4vd.obs.tracer import stage
from kg4vd.query.anchors import AnchorRetriever
from kg4vd.query.analyzer import QueryAnalyzer
from kg4vd.query.context import ContextBuilder, TextItemBuilder
from kg4vd.query.generator import AnswerGenerator
from kg4vd.query.propagation import GraphPropagator
from kg4vd.query.reranker import RerankerProtocol


class QueryPipeline:
    def __init__(
        self,
        *,
        analyzer: QueryAnalyzer,
        anchors: AnchorRetriever,
        propagator: GraphPropagator,
        text_builder: TextItemBuilder,
        reranker: RerankerProtocol,
        context: ContextBuilder,
        generator: AnswerGenerator,
        cfg: Any,
    ) -> None:
        self.analyzer = analyzer
        self.anchors = anchors
        self.propagator = propagator
        self.text_builder = text_builder
        self.reranker = reranker
        self.context = context
        self.generator = generator
        self.cfg = cfg

    async def _retrieve(self, query: Query, *, query_vector: Any, timings: dict):
        """Anchor -> propagate -> (rerank).

        Returns ``(anchor_pages, ranked_pages, node_scores, query_vector)``.
        This is the GPU-free phase (anchor/propagate are index ops; the reranker
        is an HTTP call), so the staged batch runs it before sglang starts.
        """
        q = self.cfg.query
        anchor_result = await _timed(
            "anchor", timings,
            self.anchors.retrieve(
                query, top_k=q.retrieval.anchor_size, query_vector=query_vector
            ),
        )
        anchor_pages = anchor_result.pages
        qvec = anchor_result.query_vector
        if not anchor_pages:
            return anchor_pages, [], {}, qvec
        ranked_pages, node_scores = await _timed(
            "graph", timings,
            _async(self.propagator.propagate)(anchor_pages, query_vector=qvec),
        )
        if not ranked_pages:
            ranked_pages = anchor_pages[: q.retrieval.anchor_size]
        if q.reranker.enabled:
            ranked_pages = await _timed(
                "rerank", timings, self.reranker.rerank(query, ranked_pages)
            )
        return anchor_pages, ranked_pages, node_scores, qvec

    async def retrieve(self, query: Query, *, query_vector: Any = None):
        """Retrieval only - used by run_query_batch's GPU-free phase so the
        reranker server need not coexist with sglang."""
        return await self._retrieve(query, query_vector=query_vector, timings={})

    async def answer(
        self, query: Query, *, query_vector: Any = None, precomputed: Any = None
    ) -> QueryResult:
        """Answer a query. If ``precomputed`` = (ranked_pages, node_scores) is
        given, retrieval is skipped - the staged batch reranks in an earlier
        phase and injects the result here."""
        q = self.cfg.query
        timings: dict[str, float] = {}

        analysis = await _timed("analyze", timings, self.analyzer.analyze(query))
        if precomputed is not None:
            ranked_pages, node_scores = precomputed
            anchor_pages, qvec = ranked_pages, query_vector
        else:
            anchor_pages, ranked_pages, node_scores, qvec = await self._retrieve(
                query, query_vector=query_vector, timings=timings
            )
        if not ranked_pages:
            return QueryResult(
                query=query, analysis=analysis, answer=Answer(text="", confidence="low"),
                anchor_pages=[], ranked_pages=[],
                diagnostics={"failure": "no_anchor", "timings_ms": timings},
            )

        preset = q.ppr if q.propagation == "ppr" else q.qgge
        text_items = self.text_builder.build(
            node_scores, query_vector=qvec,
            page_ids=[p.page_id for p in ranked_pages],
            graph_weight=preset.lambda_page, max_per_page=preset.max_text_per_page,
        )

        mode = self._resolve_mode(analysis.route)
        selected_texts = self.context.texts(text_items)
        selected_images = self.context.images(ranked_pages)

        image_draft: Answer | None = None
        text_draft: Answer | None = None
        if mode == "images":
            answer = await _timed(
                "answer_images", timings,
                self.generator.from_images(query, selected_images),
            )
        elif mode == "texts" or not selected_images:
            answer = await _timed(
                "answer_texts", timings,
                self.generator.from_texts(query, selected_texts),
            )
            mode = "texts"
        else:
            # Each draft gets its own span (opened inside its own task) so its
            # LLM tokens attribute correctly under concurrency - a single span
            # wrapping the gather() would be pushed after the tasks are created.
            image_draft, text_draft = await asyncio.gather(
                _timed(
                    "answer_images", timings,
                    self.generator.from_images(query, selected_images),
                ),
                _timed(
                    "answer_texts", timings,
                    self.generator.from_texts(query, selected_texts),
                ),
            )
            answer = await _timed(
                "fuse", timings, self.generator.fuse(query, image_draft, text_draft)
            )

        return QueryResult(
            query=query, analysis=analysis, answer=answer,
            anchor_pages=anchor_pages, ranked_pages=ranked_pages,
            text_items=selected_texts, image_draft=image_draft, text_draft=text_draft,
            diagnostics={
                "route": analysis.route, "answer_mode": mode,
                "timings_ms": timings, "graph_nodes_scored": len(node_scores),
            },
        )

    def _resolve_mode(self, route: str) -> str:
        configured = self.cfg.query.answer.mode
        if configured != "auto":
            return configured
        return "fusion" if route in {"single", "multi"} else "texts"


def _async(fn):
    """Wrap a sync call so it can be awaited by _timed (propagate is sync)."""
    async def _run(*args, **kwargs):
        return fn(*args, **kwargs)
    return _run


async def _timed(name: str, timings: dict[str, float], awaitable):
    """Await ``awaitable`` inside a ``query.<name>`` tracer span.

    The span records elapsed time + (for LLM stages) token usage to the active
    tracer so ``kg4vd report`` covers queries; it no-ops when no tracer is set
    (e.g. unit tests). The per-stage ms is also kept in ``diagnostics``.
    """
    with stage(f"query.{name}") as rec:
        result = await awaitable
    timings[f"{name}_ms"] = round(rec.elapsed_ms or 0.0, 3)
    return result
