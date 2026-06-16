"""Post-alignment canonicalisation.

Standard ER pipeline final step: turn pairwise `same_as` decisions into entity
clusters via connected-component closure, then contract each cluster into one
canonical node.

Why a separate stage rather than merging during alignment? Because the
alignment judge produces *pairwise* decisions; clustering needs the *global*
view to handle transitive cases (A ≡ B, B ≡ C ⇒ A ≡ B ≡ C even if the judge
never compared A vs C directly).

This is the standard "blocking → matching → clustering → canonicalisation"
ER pipeline (Fellegi & Sunter 1969, Magellan, JedAI, GraphRAG, etc.) - only
the *matching* step uses an LLM judge instead of a string-similarity
classifier.

Inputs:
    nodes, edges (with `same_as` edges from CrossPageAligner)
Outputs:
    canonical_nodes, canonical_edges, mapping (member_id → canonical_id)
"""

from __future__ import annotations

import logging

import networkx as nx

from kg4vd.config.schema import KG4VDConfig
from kg4vd.core.types import KGEdge, KGNode
from kg4vd.kg.extract import merge_edges, merge_nodes

logger = logging.getLogger(__name__)


def canonicalize_same_as(
    nodes: list[KGNode],
    edges: list[KGEdge],
    cfg: KG4VDConfig,
) -> tuple[list[KGNode], list[KGEdge], dict[str, str]]:
    """Contract `same_as` clusters into canonical nodes.

    Returns ``(nodes, edges, mapping)`` where mapping maps every member entity
    id to its canonical entity id (members include canonical itself).
    """

    align_cfg = cfg.cross_page_alignment
    if not align_cfg.canonicalize_same_as:
        return list(nodes), list(edges), {n.entity_id: n.entity_id for n in nodes}

    nodes_by_id: dict[str, KGNode] = {n.entity_id: n for n in nodes}

    # ---- Step 1. Build same_as graph (undirected) ----
    same_as = nx.Graph()
    for n in nodes:
        same_as.add_node(n.entity_id)

    # The aligner stores rs/10 in `confidence`; the threshold here is in 0..10.
    rs_threshold_same_as = align_cfg.rs_threshold_same_as / 10.0

    for e in edges:
        if e.edge_type != "same_as":
            continue
        if e.confidence < rs_threshold_same_as:
            continue
        if e.src_id not in nodes_by_id or e.tgt_id not in nodes_by_id:
            continue
        same_as.add_edge(e.src_id, e.tgt_id)

    # ---- Step 2. Connected components → clusters ----
    canonical: dict[str, str] = {}     # member_id → canonical_id
    new_nodes_by_id: dict[str, KGNode] = {}

    for cluster in nx.connected_components(same_as):
        cluster_list = list(cluster)
        if len(cluster_list) == 1:
            cid = cluster_list[0]
            canonical[cid] = cid
            new_nodes_by_id[cid] = nodes_by_id[cid]
            continue

        # Pick canonical: most source_pages, then longest description,
        # then lex-first name (deterministic).
        members_sorted = sorted(
            cluster_list,
            key=lambda x: (
                -len(nodes_by_id[x].source_pages),
                -len(nodes_by_id[x].description or ""),
                nodes_by_id[x].name.lower(),
            ),
        )
        head = members_sorted[0]
        merged = nodes_by_id[head]
        for m in members_sorted[1:]:
            merged = merge_nodes(merged, nodes_by_id[m], cfg=cfg)
        for m in members_sorted:
            canonical[m] = head
        new_nodes_by_id[head] = merged

    # ---- Step 3. Migrate edges ----
    # Drop `same_as` edges (consumed by the merge).
    # Rewrite all other edges' src_id/tgt_id through the canonical mapping.
    # Drop self-loops produced by the contraction.
    # Dedupe edges by (src, tgt, edge_type, relation), merging confidence/etc.
    migrated_by_key: dict[tuple, KGEdge] = {}
    for e in edges:
        if e.edge_type == "same_as":
            continue
        new_src = canonical.get(e.src_id, e.src_id)
        new_tgt = canonical.get(e.tgt_id, e.tgt_id)
        if new_src == new_tgt:
            continue
        new_e = e.model_copy(update={"src_id": new_src, "tgt_id": new_tgt})
        key = (new_src, new_tgt, e.edge_type, e.relation.lower().strip())
        if key in migrated_by_key:
            migrated_by_key[key] = merge_edges(
                migrated_by_key[key], new_e, cfg=cfg
            )
        else:
            migrated_by_key[key] = new_e

    canonical_nodes = list(new_nodes_by_id.values())
    canonical_edges = list(migrated_by_key.values())

    n_dropped = len(nodes) - len(canonical_nodes)
    if n_dropped > 0:
        logger.info(
            "canonicalize_same_as: %d nodes -> %d canonical (%d clusters merged); "
            "%d edges -> %d edges",
            len(nodes), len(canonical_nodes), n_dropped,
            len(edges), len(canonical_edges),
        )
    return canonical_nodes, canonical_edges, canonical
