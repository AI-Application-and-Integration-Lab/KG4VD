"""Build-time retrieval graph for the PAGE-Rank query path.

A `RetrievalGraph` is a heterogeneous view of the finalised KG that adds
page nodes and the typed edges PAGE-Rank's PPR walker needs. It is built
once during ``kg4vd build`` (opt-in stage ``retrieval_graph``) and
persisted to ``<work_dir>/retrieval_graph.json``. The query runtime
mmaps it; the build pipeline never reads it back.

Node types:

    page      - one per ingested PDF page
    entity    - one per KGNode
    relation  - one per non-``same_as`` KGEdge

Edge types (every typed edge is persisted with explicit direction):

    page_mentions_entity     PageNode      → EntityNode
    entity_source_page       EntityNode    → PageNode
    page_supports_relation   PageNode      → RelationNode
    relation_source_page     RelationNode  → PageNode
    relation_head_entity     RelationNode  → EntityNode  (head/src)
    entity_head_relation     EntityNode    → RelationNode
    relation_tail_entity     RelationNode  → EntityNode  (tail/tgt)
    entity_tail_relation     EntityNode    → RelationNode
    same_as                  EntityNode    ↔ EntityNode  (symmetric pair)

`external_id` mirrors `EvidenceCard.evidence_id` so the query runtime can
cross-reference a PPR node back to its retrieval card.

Edge weights: all edges have weight = 1.0. The transition matrix is purely
structural; query-dependent behaviour lives in the seed (personalization)
vector - see ``query.ppr.PPRPropagator``.

``edge_type`` and edge ``metadata.confidence`` remain on each edge as
**tags** for packet rendering and diagnostics, but PPR doesn't read
them for weighting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kg4vd.core.types import KGEdge, KGNode, Page

logger = logging.getLogger(__name__)

FORMAT_VERSION = "page_rank/retrieval_graph/v1"


NodeType = Literal["page", "entity", "relation"]
EdgeType = Literal[
    "page_mentions_entity",
    "entity_source_page",
    "page_supports_relation",
    "relation_source_page",
    "relation_head_entity",
    "entity_head_relation",
    "relation_tail_entity",
    "entity_tail_relation",
    "same_as",
]


# ---------------------------------------------------------------------------
# Persisted schemas
# ---------------------------------------------------------------------------


class RGNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: int
    node_type: NodeType
    external_id: str
    doc_id: str
    page_ids: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RGEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src: int
    dst: int
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# RetrievalGraph
# ---------------------------------------------------------------------------


@dataclass
class RetrievalGraph:
    """In-memory retrieval graph with derived adjacency + lookup tables.

    Only `doc_id`, `nodes`, `edges`, and `stats` are persisted. The
    indexes (`_adj`, `_node_by_id`, etc.) are rebuilt on load.
    """

    doc_id: str
    nodes: list[RGNode] = field(default_factory=list)
    edges: list[RGEdge] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    _node_by_id: dict[int, RGNode] = field(default_factory=dict, repr=False, compare=False)
    _node_by_external_id: dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    _pages_by_page_id: dict[tuple[str, int], int] = field(default_factory=dict, repr=False, compare=False)
    _adj: dict[int, list[tuple[int, float, str]]] = field(default_factory=dict, repr=False, compare=False)
    _degree: dict[int, int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._rebuild_indexes()

    # ---- index construction ----------------------------------------------
    def _rebuild_indexes(self) -> None:
        self._node_by_id = {n.node_id: n for n in self.nodes}
        self._node_by_external_id = {n.external_id: n.node_id for n in self.nodes}
        self._pages_by_page_id = {}
        for n in self.nodes:
            if n.node_type == "page" and n.page_ids:
                # Page nodes always carry exactly one page_id.
                self._pages_by_page_id[(n.doc_id, n.page_ids[0])] = n.node_id
        self._adj = {n.node_id: [] for n in self.nodes}
        for e in self.edges:
            self._adj.setdefault(e.src, []).append((e.dst, e.weight, e.edge_type))
        self._degree = {nid: len(nbrs) for nid, nbrs in self._adj.items()}

    # ---- read API ---------------------------------------------------------
    @property
    def adj(self) -> dict[int, list[tuple[int, float, str]]]:
        return self._adj

    def degree(self, node_id: int) -> int:
        return self._degree.get(node_id, 0)

    def get_node(self, node_id: int) -> RGNode | None:
        return self._node_by_id.get(node_id)

    def get_node_by_external_id(self, external_id: str) -> RGNode | None:
        nid = self._node_by_external_id.get(external_id)
        return self._node_by_id.get(nid) if nid is not None else None

    def get_page_node(self, doc_id: str, page_id: int) -> RGNode | None:
        nid = self._pages_by_page_id.get((doc_id, page_id))
        return self._node_by_id.get(nid) if nid is not None else None

    def get_entities_on_page(self, doc_id: str, page_id: int) -> list[RGNode]:
        page = self.get_page_node(doc_id, page_id)
        if not page:
            return []
        return [
            self._node_by_id[dst]
            for dst, _w, et in self._adj.get(page.node_id, ())
            if et == "page_mentions_entity"
        ]

    def get_relations_on_page(self, doc_id: str, page_id: int) -> list[RGNode]:
        page = self.get_page_node(doc_id, page_id)
        if not page:
            return []
        return [
            self._node_by_id[dst]
            for dst, _w, et in self._adj.get(page.node_id, ())
            if et == "page_supports_relation"
        ]

    def get_source_pages(self, node_id: int) -> list[int]:
        n = self._node_by_id.get(node_id)
        if not n:
            return []
        if n.node_type == "page":
            return list(n.page_ids)
        seen: set[int] = set()
        out: list[int] = []
        for dst, _w, et in self._adj.get(node_id, ()):
            if et in ("entity_source_page", "relation_source_page"):
                page = self._node_by_id.get(dst)
                if page:
                    for pid in page.page_ids:
                        if pid not in seen:
                            seen.add(pid)
                            out.append(pid)
        return out

    # ---- IO ---------------------------------------------------------------
    def persist(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": FORMAT_VERSION,
            "doc_id": self.doc_id,
            "nodes": [n.model_dump(mode="json") for n in self.nodes],
            "edges": [e.model_dump(mode="json") for e in self.edges],
            "stats": self.stats,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "RetrievalGraph":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"retrieval_graph.json not found at {path}. "
                f"Run: kg4vd build <recipe> --stages retrieval_graph"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = payload.get("version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported retrieval_graph version {version!r} at {path}; "
                f"expected {FORMAT_VERSION!r}. Rebuild with "
                f"--stages retrieval_graph."
            )
        return cls(
            doc_id=payload["doc_id"],
            nodes=[RGNode.model_validate(n) for n in payload["nodes"]],
            edges=[RGEdge.model_validate(e) for e in payload["edges"]],
            stats=payload.get("stats", {}),
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_retrieval_graph(
    pages: list[Page],
    nodes: list[KGNode],
    edges: list[KGEdge],
) -> RetrievalGraph:
    """Construct a `RetrievalGraph` from finalised KG + Pages.

    Assumes `nodes`/`edges` have already passed through ``align`` and
    (optionally) ``canonicalize_same_as``. Surviving ``same_as`` edges
    are persisted as symmetric same_as pairs; non-same_as edges become
    relation nodes plus the four directed entity-relation edges.
    """
    rg_nodes: list[RGNode] = []
    rg_edges: list[RGEdge] = []
    next_id = 0

    # 1. Page nodes ---------------------------------------------------------
    page_key_to_node_id: dict[tuple[str, int], int] = {}
    for p in pages:
        nid = next_id
        next_id += 1
        rg_nodes.append(
            RGNode(
                node_id=nid,
                node_type="page",
                external_id=f"page:{p.doc_id}:{p.page_id}",
                doc_id=p.doc_id,
                page_ids=[p.page_id],
                metadata={
                    "image_path": p.page_image_path,
                    "has_image": bool(p.page_image_path),
                    "has_summary": bool(p.page_summary),
                    "n_text_chars": len(p.text or ""),
                    "n_figures": len(p.figure_image_paths),
                },
            )
        )
        page_key_to_node_id[(p.doc_id, p.page_id)] = nid

    # 2. Entity nodes -------------------------------------------------------
    entity_id_to_node_id: dict[str, int] = {}
    entity_name_by_id: dict[int, str] = {}    # for head/tail resolution below
    for n in nodes:
        nid = next_id
        next_id += 1
        doc_id = (n.metadata.get("doc_id") if n.metadata else None) or _resolve_doc_id(n, pages)
        rg_nodes.append(
            RGNode(
                node_id=nid,
                node_type="entity",
                external_id=f"entity:{n.entity_id}",
                doc_id=doc_id,
                page_ids=list(n.source_pages),
                metadata={
                    "name": n.name,
                    "entity_type": n.entity_type,
                    "modality": n.modality,
                    "visual_type": n.visual_type,
                    "description": n.description or "",
                    "visual_description": n.visual_description or "",
                },
            )
        )
        entity_id_to_node_id[n.entity_id] = nid
        entity_name_by_id[nid] = n.name

    # 3. Relation nodes (non-same_as only) ----------------------------------
    edge_id_to_node_id: dict[str, int] = {}
    for e in edges:
        if e.edge_type == "same_as":
            continue
        nid = next_id
        next_id += 1
        head_nid = entity_id_to_node_id.get(e.src_id)
        tail_nid = entity_id_to_node_id.get(e.tgt_id)
        # Doc id resolution mirrors cards.builders.build_relation_cards:
        # prefer edge metadata, then head node, then tail.
        doc_id = (e.metadata or {}).get("doc_id")
        if not doc_id and head_nid is not None:
            doc_id = rg_nodes[head_nid].doc_id
        if not doc_id and tail_nid is not None:
            doc_id = rg_nodes[tail_nid].doc_id
        doc_id = doc_id or "UNKNOWN"
        rel_meta: dict[str, Any] = {
            "relation": e.relation,
            "edge_type": e.edge_type,
            "confidence": float(e.confidence),
            "head_node_id": head_nid,
            "tail_node_id": tail_nid,
            # Resolved names so downstream renderers don't need to walk
            # back to the entity nodes themselves.
            "head_name": entity_name_by_id.get(head_nid, "?") if head_nid is not None else "?",
            "tail_name": entity_name_by_id.get(tail_nid, "?") if tail_nid is not None else "?",
            "source_text": e.description or "",
            "visual_evidence_hint": e.visual_evidence_hint,
        }
        # Carry forward cross-page-align audit tags (rescue_reason, origin,
        # fused/fused_from) so PPR diagnostics + future weighting can use them.
        for k in ("origin", "rescue_reason", "fused", "fused_from"):
            v = (e.metadata or {}).get(k)
            if v is not None:
                rel_meta[k] = v
        rg_nodes.append(
            RGNode(
                node_id=nid,
                node_type="relation",
                external_id=f"relation:{e.edge_id}",
                doc_id=doc_id,
                page_ids=list(e.source_pages),
                metadata=rel_meta,
            )
        )
        edge_id_to_node_id[e.edge_id] = nid

    # 4. Entity ↔ Page edges ------------------------------------------------
    for n in nodes:
        ent_nid = entity_id_to_node_id[n.entity_id]
        ent_doc = rg_nodes[ent_nid].doc_id
        for pid in n.source_pages:
            page_nid = page_key_to_node_id.get((ent_doc, pid))
            if page_nid is None:
                continue
            rg_edges.append(RGEdge(
                src=page_nid, dst=ent_nid,
                edge_type="page_mentions_entity", weight=1.0,
            ))
            rg_edges.append(RGEdge(
                src=ent_nid, dst=page_nid,
                edge_type="entity_source_page", weight=1.0,
            ))

    # 5. Relation/same_as edges --------------------------------------------
    # All edges carry weight 1.0 - PPR's query-aware behaviour now lives
    # in the personalization restart vector, so the transition
    # matrix is purely structural. ``confidence`` survives as edge
    # metadata for packet rendering / diagnostics.
    same_as_pairs = 0
    same_as_dropped = 0

    for e in edges:
        if e.edge_type == "same_as":
            head_nid = entity_id_to_node_id.get(e.src_id)
            tail_nid = entity_id_to_node_id.get(e.tgt_id)
            if head_nid is None or tail_nid is None or head_nid == tail_nid:
                same_as_dropped += 1
                continue
            meta = {
                "rescue_reason": (e.metadata or {}).get("rescue_reason"),
                "confidence": float(e.confidence),
                "origin": (e.metadata or {}).get("origin", "cross_page_align"),
            }
            rg_edges.append(RGEdge(
                src=head_nid, dst=tail_nid,
                edge_type="same_as", weight=1.0, metadata=meta,
            ))
            rg_edges.append(RGEdge(
                src=tail_nid, dst=head_nid,
                edge_type="same_as", weight=1.0, metadata=meta,
            ))
            same_as_pairs += 1
            continue

        rel_nid = edge_id_to_node_id.get(e.edge_id)
        if rel_nid is None:
            continue
        rel_doc = rg_nodes[rel_nid].doc_id

        # Relation ↔ Page (use the relation's source_pages within the
        # relation's resolved doc; multi-doc edges would split here but
        # that case is rare and tolerated by skipping unmatched pages).
        for pid in e.source_pages:
            page_nid = page_key_to_node_id.get((rel_doc, pid))
            if page_nid is None:
                continue
            rg_edges.append(RGEdge(
                src=page_nid, dst=rel_nid,
                edge_type="page_supports_relation", weight=1.0,
            ))
            rg_edges.append(RGEdge(
                src=rel_nid, dst=page_nid,
                edge_type="relation_source_page", weight=1.0,
            ))

        # Relation ↔ Entity (head + tail in both directions).
        head_nid = entity_id_to_node_id.get(e.src_id)
        tail_nid = entity_id_to_node_id.get(e.tgt_id)
        if head_nid is not None:
            rg_edges.append(RGEdge(
                src=rel_nid, dst=head_nid,
                edge_type="relation_head_entity", weight=1.0,
            ))
            rg_edges.append(RGEdge(
                src=head_nid, dst=rel_nid,
                edge_type="entity_head_relation", weight=1.0,
            ))
        if tail_nid is not None and tail_nid != head_nid:
            rg_edges.append(RGEdge(
                src=rel_nid, dst=tail_nid,
                edge_type="relation_tail_entity", weight=1.0,
            ))
            rg_edges.append(RGEdge(
                src=tail_nid, dst=rel_nid,
                edge_type="entity_tail_relation", weight=1.0,
            ))

    stats = {
        "n_pages": sum(1 for n in rg_nodes if n.node_type == "page"),
        "n_entities": sum(1 for n in rg_nodes if n.node_type == "entity"),
        "n_relations": sum(1 for n in rg_nodes if n.node_type == "relation"),
        "n_edges": len(rg_edges),
        "n_same_as_pairs": same_as_pairs,
        "n_same_as_dropped": same_as_dropped,
    }
    if same_as_dropped:
        logger.info(
            "build_retrieval_graph: dropped %d same_as edges with missing or "
            "self-referential endpoints",
            same_as_dropped,
        )
    return RetrievalGraph(
        doc_id=pages[0].doc_id if pages else "UNKNOWN",
        nodes=rg_nodes,
        edges=rg_edges,
        stats=stats,
    )


def _resolve_doc_id(n: KGNode, pages: list[Page]) -> str:
    if n.source_pages and pages:
        wanted = set(n.source_pages)
        for p in pages:
            if p.page_id in wanted:
                return p.doc_id
    return "UNKNOWN"
