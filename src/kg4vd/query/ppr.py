"""Personalized-PageRank page propagation over the retrieval graph.

An alternative to QGGE (``propagation.py``). Where QGGE expands the anchor pages
through the index's page<->bridge incidence, PPR runs a restart random walk over
``kg/retrieval_graph.json`` (page/entity/relation nodes + typed edges), seeded by
the anchor pages. Page-node mass is the graph signal; it is blended with the GME
anchor (vector) score via ``lambda_page`` - the same output contract as QGGE so
the rest of the query path is unchanged.

Power iteration runs inside a bounded neighbourhood
(``collect_neighborhood``):

    r_{t+1} = (1 − α)·s + α·Pᵀ·r_t + α·dangling_mass·s,    α = 1 − restart_probability

with ``s`` the (normalised) seed vector. Edge weights are structural and the
walk is purely seed-personalised. Uses dense numpy over the bounded subgraph -
no scipy dep.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from kg4vd.core.types import PageHit
from kg4vd.query.propagation import _minmax


def _collect_neighborhood(adj, seed_nodes, *, max_depth, max_nodes):
    """Bounded BFS from ``seed_nodes`` (≤ max_depth hops, ≤ max_nodes total)."""
    if max_nodes <= 0:
        return set()
    visited = set(seed_nodes)
    if len(visited) >= max_nodes:
        return visited
    frontier = set(seed_nodes)
    for _ in range(max_depth):
        nxt: set[int] = set()
        for node in frontier:
            for nbr, _w, _et in adj.get(node, ()):
                if nbr in visited:
                    continue
                visited.add(nbr)
                nxt.add(nbr)
                if len(visited) >= max_nodes:
                    return visited
        frontier = nxt
        if not frontier:
            break
    return visited


class PPRPropagator:
    """Personalized PageRank over the retrieval graph; QGGE-compatible output."""

    def __init__(self, graph: Any, index: Any, cfg: Any) -> None:
        self._graph = graph
        self._c = cfg
        # Page image + validity from the index (so PageHits carry image paths).
        self._page_image: dict[int, str | None] = {}
        for c in index.cards():
            if c.evidence_type == "page" and c.page_ids:
                self._page_image[int(c.page_ids[0])] = c.image_payload
        # page_id -> graph page-node id (single bundled doc: page_id is unique).
        self._page_node: dict[int, int] = {}
        self._node_meta: dict[int, tuple[str, str, int | None]] = {}
        for n in graph.nodes:
            page_id = int(n.page_ids[0]) if (n.node_type == "page" and n.page_ids) else None
            self._node_meta[n.node_id] = (n.node_type, n.external_id, page_id)
            if page_id is not None:
                self._page_node[page_id] = n.node_id

    def propagate(
        self, anchors: list[PageHit], *, query_vector: Any = None
    ) -> tuple[list[PageHit], dict[str, float]]:
        del query_vector  # PPR restart is seed-personalised; the vector is unused
        c = self._c
        adj = self._graph.adj

        # Seeds: anchor pages -> page nodes, weighted by anchor (vector) score.
        seed_scores: dict[int, float] = {}
        for hit in anchors:
            nid = self._page_node.get(hit.page_id)
            if nid is not None:
                seed_scores[nid] = max(seed_scores.get(nid, 0.0), max(hit.score, 0.0))
        if not seed_scores or sum(seed_scores.values()) <= 0:
            return anchors[: c.final_k], {}

        sub = _collect_neighborhood(
            adj, set(seed_scores), max_depth=c.max_depth, max_nodes=c.max_nodes
        )
        node_list = sorted(sub)
        n = len(node_list)
        idx = {nid: i for i, nid in enumerate(node_list)}

        s = np.zeros(n, dtype=np.float64)
        for nid, sc in seed_scores.items():
            if nid in idx:
                s[idx[nid]] = sc
        if s.sum() <= 0:
            return anchors[: c.final_k], {}
        s /= s.sum()

        # Row-normalised transition matrix over the bounded subgraph (dense).
        P = np.zeros((n, n), dtype=np.float64)
        for nid, i in idx.items():
            outs = [(idx[nbr], float(w)) for nbr, w, _ in adj.get(nid, ()) if nbr in idx]
            tot = sum(w for _, w in outs)
            if tot > 0:
                for j, w in outs:
                    P[i, j] = w / tot
        dangling = P.sum(axis=1) < 1e-12
        pt = P.T

        alpha = 1.0 - float(c.restart_probability)
        r = s.copy()
        for _ in range(max(c.max_iter, 1)):
            new_r = (1.0 - alpha) * s + alpha * (pt @ r)
            dm = float(r[dangling].sum()) if dangling.any() else 0.0
            if dm > 0:
                new_r += alpha * dm * s
            delta = float(np.abs(new_r - r).sum())
            r = new_r
            if delta < c.tol:
                break

        # Split mass into page scores (graph signal) + entity/relation scores.
        page_mass: dict[int, float] = {}
        node_scores: dict[str, float] = {}
        for nid, i in idx.items():
            ntype, ext_id, page_id = self._node_meta.get(nid, ("", "", None))
            mass = float(r[i])
            if ntype == "page" and page_id is not None:
                page_mass[page_id] = mass
            elif ntype in ("entity", "relation"):
                node_scores[ext_id] = mass

        # Blend GME anchor score + PPR page mass (min-max), like QGGE.
        anchor_scores = {h.page_id: h.score for h in anchors}
        candidates = set(anchor_scores) | set(page_mass)
        page_norm = _minmax({p: anchor_scores.get(p, 0.0) for p in candidates})
        ppr_norm = _minmax({p: page_mass.get(p, 0.0) for p in candidates})
        lam = float(c.lambda_page)
        joint = {
            p: lam * page_norm.get(p, 0.0) + (1.0 - lam) * ppr_norm.get(p, 0.0)
            for p in candidates
        }
        ranked = sorted(joint.items(), key=lambda x: (-x[1], x[0]))[: c.final_k]

        hits = [
            PageHit(
                page_id=p, score=float(score), rank=rank,
                image_path=self._page_image.get(p),
            )
            for rank, (p, score) in enumerate(ranked, start=1)
        ]
        return hits, node_scores
