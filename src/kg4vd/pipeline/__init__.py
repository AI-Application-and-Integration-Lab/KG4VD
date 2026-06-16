"""Pipeline orchestration: ``run_build`` (construction) + ``run_query``.

The query path lands on the same build artifacts (index + cards + pages) and
uses QGGE propagation over entity/relation bridges; it doesn't import the
construction stages, so the build path stays light.
"""

from kg4vd.pipeline.align_embed import precompute_align_embeddings
from kg4vd.pipeline.build import BuildArtifacts, run_build
from kg4vd.pipeline.query import build_query_pipeline, run_query, run_query_batch

__all__ = [
    "BuildArtifacts",
    "build_query_pipeline",
    "precompute_align_embeddings",
    "run_build",
    "run_query",
    "run_query_batch",
]
