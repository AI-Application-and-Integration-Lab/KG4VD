"""Load a build's query artifacts from its work_dir.

Loads the persisted nano index (page/entity/relation cards + embeddings) and
the pages (``pages.jsonl``, for page summaries / OCR text used by the context
builder). QGGE propagation needs only the index; PPR propagation additionally
needs the RetrievalGraph (``retrieval_graph.json``), loaded when present.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kg4vd.config.schema import KG4VDConfig
from kg4vd.core.errors import IndexError as KGIndexError
from kg4vd.core.types import Page
from kg4vd.index import build_index
from kg4vd.kg.retrieval_graph import RetrievalGraph

logger = logging.getLogger(__name__)


@dataclass
class QueryArtifacts:
    """Loaded artifacts a query runs against."""

    index: Any
    work_dir: Path
    pages_by_id: dict[int, Page] = field(default_factory=dict)
    retrieval_graph: RetrievalGraph | None = None


def _load_pages(work_dir: Path) -> dict[int, Page]:
    path = work_dir / "pages.jsonl"
    if not path.is_file():
        return {}
    out: dict[int, Page] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            page = Page.model_validate_json(line)
            out[page.page_id] = page
    return out


async def load_query_artifacts(cfg: KG4VDConfig) -> QueryArtifacts:
    """Load the persisted index + pages for ``cfg.dataset.work_dir``.

    Raises ``IndexError`` if the build hasn't produced an index yet.
    """
    work_dir = Path(cfg.dataset.work_dir)
    index_dir = work_dir / "index"
    if not (index_dir / "meta.json").is_file():
        raise KGIndexError(
            f"No index at {index_dir}; run `kg4vd build` (embed+index) first."
        )
    index = build_index(cfg.index, dim=cfg.encoder.dim)
    await index.load(index_dir)

    rg_path = work_dir / "retrieval_graph.json"
    retrieval_graph = None
    if rg_path.is_file():
        retrieval_graph = RetrievalGraph.load(rg_path)
        logger.info("Loaded retrieval graph: %d nodes", len(retrieval_graph.nodes))

    return QueryArtifacts(
        index=index, work_dir=work_dir,
        pages_by_id=_load_pages(work_dir), retrieval_graph=retrieval_graph,
    )
