"""Query path - retrieve over a built MMKG and answer.

Reads the build artifacts (the nano index, evidence cards, and pages) and
answers questions over them.

Flow: analyze → anchor → propagate (QGGE) → rerank → context → generate.
"""

from __future__ import annotations

from kg4vd.query.analyzer import QueryAnalyzer
from kg4vd.query.anchors import AnchorResult, AnchorRetriever
from kg4vd.query.context import ContextBuilder, TextItemBuilder
from kg4vd.query.generator import AnswerGenerator
from kg4vd.query.loader import QueryArtifacts, load_query_artifacts
from kg4vd.query.pipeline import QueryPipeline
from kg4vd.query.prompts import PromptSet
from kg4vd.query.ppr import PPRPropagator
from kg4vd.query.propagation import GraphPropagator, QGGEPropagator
from kg4vd.query.reranker import NoOpReranker, Qwen3VLReranker, build_reranker

__all__ = [
    "AnchorResult",
    "AnchorRetriever",
    "AnswerGenerator",
    "ContextBuilder",
    "GraphPropagator",
    "NoOpReranker",
    "PPRPropagator",
    "PromptSet",
    "QGGEPropagator",
    "Qwen3VLReranker",
    "QueryAnalyzer",
    "QueryArtifacts",
    "QueryPipeline",
    "TextItemBuilder",
    "build_reranker",
    "load_query_artifacts",
]
