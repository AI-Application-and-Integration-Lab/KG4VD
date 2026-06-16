"""Shared extraction primitives.

The KG extractor (``kg.extract.extractor.Extractor``) and the raw-patch
resolver build on these building blocks:

  * per-round audit / snapshot / result dataclasses,
  * deterministic entity / edge id hashing,
  * patch application (``_apply_patch``) + node/edge merging,
  * the async stage context helper.

They are kept module-level so the extractor, controller, and resolver
share one copy.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from kg4vd.config.schema import KG4VDConfig
from kg4vd.core.types import KGEdge, KGNode, KGPatch, Page
from kg4vd.obs.tracer import stage as stage_ctx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-round audit / snapshot / result
# ---------------------------------------------------------------------------


@dataclass
class _RoundAudit:
    round_idx: int
    add_nodes: int = 0
    add_edges: int = 0
    replace_nodes: int = 0
    replace_edges: int = 0
    delete_nodes: int = 0
    delete_edges: int = 0
    stop: bool = False
    reason: str = ""


@dataclass
class _RoundSnapshot:
    """Per-round snapshot: the audit + the post-round node/edge state.

    Lets the inspect script reconstruct the page-level graph as it evolves
    through rounds 0..N (round 0 = post-init, round k = post-kth-reflector).
    """
    round_idx: int
    audit: _RoundAudit
    nodes: list[KGNode]
    edges: list[KGEdge]


@dataclass
class ExtractionResult:
    page_id: int
    nodes: list[KGNode]
    edges: list[KGEdge]
    rounds: list[_RoundAudit]
    snapshots: list[_RoundSnapshot]


# ---------------------------------------------------------------------------
# Deterministic ids
# ---------------------------------------------------------------------------


def _entity_id_for(doc_id: str, name: str) -> str:
    h = uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}::{name.lower()}")
    return f"E:{h.hex[:12]}"


def _edge_id_for(doc_id: str, src_id: str, tgt_id: str, rel: str) -> str:
    h = uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}::{src_id}->{tgt_id}::{rel.lower()}")
    return f"R:{h.hex[:12]}"


# ---------------------------------------------------------------------------
# Patch application + merging
# ---------------------------------------------------------------------------


def _apply_patch(
    nodes: list[KGNode],
    edges: list[KGEdge],
    patch: KGPatch,
    *,
    page: Page,
    cfg: KG4VDConfig,
) -> tuple[list[KGNode], list[KGEdge]]:
    by_id = {n.entity_id: n for n in nodes}
    edge_by_id = {e.edge_id: e for e in edges}

    # add
    for n in patch.add_nodes:
        if n.entity_id in by_id:
            by_id[n.entity_id] = _merge_nodes(by_id[n.entity_id], n, cfg=cfg)
        else:
            by_id[n.entity_id] = n

    for e in patch.add_edges:
        if e.edge_id in edge_by_id:
            edge_by_id[e.edge_id] = _merge_edges(edge_by_id[e.edge_id], e, cfg=cfg)
        else:
            edge_by_id[e.edge_id] = e

    # replace (same entity_id wins; description gets merged)
    for n in patch.replace_nodes:
        # Re-resolve id by name, in case the LLM gave a name without our ID.
        eid = _resolve_node_id_by_name(by_id, n.name) or n.entity_id
        n2 = n.model_copy(update={"entity_id": eid})
        if eid in by_id:
            by_id[eid] = _merge_nodes(by_id[eid], n2, cfg=cfg, prefer_new=True)
        else:
            by_id[eid] = n2

    for e in patch.replace_edges:
        if e.edge_id in edge_by_id:
            edge_by_id[e.edge_id] = _merge_edges(
                edge_by_id[e.edge_id], e, cfg=cfg, prefer_new=True
            )
        else:
            edge_by_id[e.edge_id] = e

    # delete
    for name_or_id in patch.delete_node_ids:
        # Accept either a raw name or a real entity_id.
        target_id = name_or_id if name_or_id in by_id else _resolve_node_id_by_name(by_id, name_or_id)
        if target_id and target_id in by_id:
            by_id.pop(target_id, None)
            # Cascade: drop edges touching the deleted node.
            for e_id in list(edge_by_id):
                e = edge_by_id[e_id]
                if e.src_id == target_id or e.tgt_id == target_id:
                    edge_by_id.pop(e_id, None)

    for synthetic in patch.delete_edge_ids:
        # synthetic = "src_name->tgt_name" or a real edge_id.
        if synthetic in edge_by_id:
            edge_by_id.pop(synthetic, None)
            continue
        if "->" in synthetic:
            src_name, tgt_name = synthetic.split("->", 1)
            src_id = _resolve_node_id_by_name(by_id, src_name)
            tgt_id = _resolve_node_id_by_name(by_id, tgt_name)
            for e_id in list(edge_by_id):
                e = edge_by_id[e_id]
                if e.src_id == src_id and e.tgt_id == tgt_id:
                    edge_by_id.pop(e_id, None)

    return list(by_id.values()), list(edge_by_id.values())


def _resolve_node_id_by_name(
    by_id: dict[str, KGNode], name: str
) -> str | None:
    name_low = (name or "").strip().lower()
    if not name_low:
        return None
    for eid, n in by_id.items():
        if n.name.strip().lower() == name_low:
            return eid
    return None


def _union_bboxes(a, b):
    """Union two `BBoxRef` lists, deduping by (page_id, component_id, bbox)
    tuple. Preserves order: a first, then b's new entries.

    page_id matters because `canonicalize_same_as` calls this after
    merging per-page nodes into one canonical node - two pages both
    have a "P1"/"IM1" component, and similar bbox coords across pages
    are common (most documents have uniform page sizes, so the layout
    parser often reports near-identical figure bboxes on different
    pages). Without page_id in the key, cross-page bbox evidence would
    collapse to a single entry, hiding the entity's true multi-page
    presence from `_resolve_visual_crop` and from any caller that
    walks `n.bboxes` for grounding.
    """
    seen = set()
    out = []
    for ref in list(a) + list(b):
        key = (ref.page_id, ref.component_id, tuple(ref.bbox))
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def _merge_nodes(
    a: KGNode, b: KGNode, *, cfg: KG4VDConfig, prefer_new: bool = False
) -> KGNode:
    """Merge two nodes with the same entity_id.

    The grounding fields ``source_components`` / ``bboxes`` are
    accumulated (unioned) - a stub merging into a typed node, or two
    rounds adding evidence to the same entity, both end up with the
    superset. Nodes without these fields have empty defaults, so this is
    a no-op for them.
    """

    desc = _merge_description(
        a.description, b.description,
        max_chars=cfg.kg.description_max_chars,
        strategy=cfg.kg.description_merge,
    )
    return KGNode(
        entity_id=a.entity_id,
        name=b.name if prefer_new and b.name else a.name,
        entity_type=b.entity_type if prefer_new and b.entity_type else a.entity_type,
        modality=b.modality if b.modality != "text" or prefer_new else a.modality,
        description=desc,
        visual_description=b.visual_description or a.visual_description,
        visual_type=b.visual_type or a.visual_type,
        bbox=b.bbox or a.bbox,
        source_components=sorted(set(a.source_components) | set(b.source_components)),
        bboxes=_union_bboxes(a.bboxes, b.bboxes),
        source_pages=sorted(set(a.source_pages) | set(b.source_pages)),
        source_chunks=sorted(set(a.source_chunks) | set(b.source_chunks)),
        metadata={**a.metadata, **b.metadata},
    )


def _merge_edges(
    a: KGEdge, b: KGEdge, *, cfg: KG4VDConfig, prefer_new: bool = False
) -> KGEdge:
    desc = _merge_description(
        a.description, b.description,
        max_chars=cfg.kg.description_max_chars,
        strategy=cfg.kg.description_merge,
    )
    return KGEdge(
        edge_id=a.edge_id,
        src_id=a.src_id,
        tgt_id=a.tgt_id,
        relation=b.relation if prefer_new and b.relation else a.relation,
        edge_type=a.edge_type if a.edge_type != "semantic" else b.edge_type,
        description=desc,
        visual_evidence_hint=b.visual_evidence_hint or a.visual_evidence_hint,
        confidence=max(a.confidence, b.confidence),
        source_components=sorted(set(a.source_components) | set(b.source_components)),
        bboxes=_union_bboxes(a.bboxes, b.bboxes),
        source_pages=sorted(set(a.source_pages) | set(b.source_pages)),
        source_chunks=sorted(set(a.source_chunks) | set(b.source_chunks)),
        metadata={**a.metadata, **b.metadata},
    )


def _merge_description(a: str, b: str, *, max_chars: int, strategy: str) -> str:
    if not a:
        return b or ""
    if not b or b in a:
        return a
    if strategy == "concat_sep":
        candidate = a + " <SEP> " + b
        # Run the same sentence-level dedup the card builders do, so
        # the persisted node.description doesn't accumulate
        # near-duplicate phrasings across reflector rounds /
        # cross-page merges. Idempotent: clean input stays clean.
        try:
            from kg4vd.cards.builders import _compact_description
            candidate = _compact_description(candidate, max_chars=max_chars)
        except ImportError:                                              # pragma: no cover
            pass
        if len(candidate) <= max_chars:
            return candidate
        # Belt-and-suspenders: even after compact, descriptions can
        # exceed max_chars when both inputs are large and orthogonal.
        # Truncate from the older description, keep the newer one whole.
        keep_old = max_chars - len(b) - len(" <SEP> ")
        if keep_old <= 0:
            return b
        return a[:keep_old] + " <SEP> " + b
    # "llm_summarize" is not implemented; fall back to concat_sep.
    return _merge_description(a, b, max_chars=max_chars, strategy="concat_sep")


# ---------------------------------------------------------------------------
# async stage helper
# ---------------------------------------------------------------------------


class _astage:
    def __init__(self, name: str, **tags: Any):
        self.name = name
        self.tags = tags

    async def __aenter__(self):
        self._cm = stage_ctx(self.name, **self.tags)
        return self._cm.__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        return self._cm.__exit__(exc_type, exc, tb)
