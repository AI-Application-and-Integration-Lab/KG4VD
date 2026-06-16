"""RawPatch → KGPatch resolver.

The component-cued extractor's validator works on `RawPatch` (LLM-emitted,
name-keyed). Persistence and downstream merging speak `KGPatch`
(canonical, ID-keyed). The resolver bridges the two by minting stable
IDs from canonical names via `_entity_id_for(doc_id, name)`, so equal
names across pages collapse to one node downstream.

Principle: name-based canonicalisation up front (entity_id =
stable_hash(doc_id, name)), then ID-based merging downstream.
"""

from __future__ import annotations

from typing import Iterable

from kg4vd.core.types import BBoxRef, KGEdge, KGNode, KGPatch
from kg4vd.ingest.components import Component
from kg4vd.kg.extract._common import _edge_id_for, _entity_id_for
from kg4vd.kg.extract.raw_ops import (
    RawDeleteEdge,
    RawEdgeOp,
    RawNodeOp,
    RawPatch,
)


def resolve_raw_patch(
    raw: RawPatch,
    *,
    doc_id: str,
    page_id: int,
    components: Iterable[Component],
) -> KGPatch:
    """Convert a validated `RawPatch` into a canonical `KGPatch`.

    - Entity IDs are minted from canonical names via a stable hash, so
      equal names collapse to one node downstream.
    - Each emitted KGNode/KGEdge carries `source_components` AND
      `bboxes` (looked up from `components` via cited cid).
    - The original LLM-emitted name is stashed in
      `metadata["src_name"] / ["tgt_name"]` on edges so the scorecard
      and downstream eval scripts can render edges by name.
    - `delete_node_ids` are minted from the names in `raw.delete_nodes`.
    - `delete_edge_ids` are minted from `(src_id, tgt_id, relation)`
      tuples; deletes with a `None` relation get a wildcard
      `f"{src_name}->{tgt_name}"` synthetic key so the existing
      apply-patch logic can match every edge between (src, tgt).
    - `controller_summary` defaults empty here - it's the validator's
      output, not the resolver's. The caller is responsible for
      attaching it before persistence.
    """
    cid_to_bbox: dict[str, tuple[float, float, float, float]] = {
        c.component_id: tuple(c.bbox) for c in components
    }

    return KGPatch(
        add_nodes=[_to_kg_node(n, doc_id=doc_id, page_id=page_id,
                               cid_to_bbox=cid_to_bbox)
                   for n in raw.add_nodes],
        add_edges=[_to_kg_edge(e, doc_id=doc_id, page_id=page_id,
                               cid_to_bbox=cid_to_bbox)
                   for e in raw.add_edges],
        replace_nodes=[_to_kg_node(n, doc_id=doc_id, page_id=page_id,
                                   cid_to_bbox=cid_to_bbox)
                       for n in raw.replace_nodes],
        replace_edges=[_to_kg_edge(e, doc_id=doc_id, page_id=page_id,
                                   cid_to_bbox=cid_to_bbox)
                       for e in raw.replace_edges],
        delete_node_ids=[_entity_id_for(doc_id, name)
                         for name in raw.delete_nodes if name],
        delete_edge_ids=[_resolve_delete_edge_key(de, doc_id=doc_id)
                         for de in raw.delete_edges],
        reason=raw.reason,
    )


def _to_kg_node(
    n: RawNodeOp,
    *,
    doc_id: str,
    page_id: int,
    cid_to_bbox: dict[str, tuple[float, float, float, float]],
) -> KGNode:
    eid = _entity_id_for(doc_id, n.name)
    return KGNode(
        entity_id=eid,
        name=n.name,
        entity_type=n.entity_type,
        modality=n.modality,
        description=n.description or "",
        visual_description=n.visual_description,
        visual_type=n.visual_type,
        source_components=list(n.source_components),
        bboxes=_bboxes_from_cids(n.source_components, cid_to_bbox, page_id),
        source_pages=[page_id],
    )


def _to_kg_edge(
    e: RawEdgeOp,
    *,
    doc_id: str,
    page_id: int,
    cid_to_bbox: dict[str, tuple[float, float, float, float]],
) -> KGEdge:
    src_id = _entity_id_for(doc_id, e.src)
    tgt_id = _entity_id_for(doc_id, e.tgt)
    return KGEdge(
        edge_id=_edge_id_for(doc_id, src_id, tgt_id, e.relation),
        src_id=src_id,
        tgt_id=tgt_id,
        relation=e.relation,
        edge_type="semantic",
        description=e.description or "",
        visual_evidence_hint=e.visual_evidence_hint,
        confidence=float(e.confidence),
        source_components=list(e.source_components),
        bboxes=_bboxes_from_cids(e.source_components, cid_to_bbox, page_id),
        source_pages=[page_id],
        # Preserve the original names so the scorecard renderer and
        # eval scripts can show "src -[rel]-> tgt" without reverse-
        # lookup against the node table.
        metadata={"src_name": e.src, "tgt_name": e.tgt},
    )


def _bboxes_from_cids(
    cids: list[str],
    cid_to_bbox: dict[str, tuple[float, float, float, float]],
    page_id: int,
) -> list[BBoxRef]:
    out: list[BBoxRef] = []
    seen: set[str] = set()
    for cid in cids:
        if cid in seen:
            continue
        seen.add(cid)
        bb = cid_to_bbox.get(cid)
        if bb is not None:
            out.append(BBoxRef(component_id=cid, bbox=bb, page_id=page_id))
    return out


def _resolve_delete_edge_key(de: RawDeleteEdge, *, doc_id: str) -> str:
    """Compute the key the existing `_apply_patch` recognises.

    `_apply_patch.delete_edge_ids` accepts EITHER:
      - a real edge_id (when relation is known), OR
      - a synthetic `"src_name->tgt_name"` string (wildcard relation).
    """
    if de.relation:
        src_id = _entity_id_for(doc_id, de.src)
        tgt_id = _entity_id_for(doc_id, de.tgt)
        return _edge_id_for(doc_id, src_id, tgt_id, de.relation)
    # Wildcard form - _apply_patch splits on "->" and matches by name.
    return f"{de.src}->{de.tgt}"


__all__ = ["resolve_raw_patch"]
