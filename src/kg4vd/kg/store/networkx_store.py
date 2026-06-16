"""In-process NetworkX KG store.

Implements the API expected by ``retrieve.expand`` and the build pipeline:
  - upsert_node / upsert_edge (idempotent on entity_id / edge_id)
  - delete_node / delete_edge
  - iter_neighbours(entity_id, hops=1) → (edge_id, neighbour_id) pairs
  - aligned_entities(entity_id) → entity_ids connected by `same_as` edges
  - relation_meta(edge_id) → {src_id, tgt_id, ...}
  - communities(algorithm) → {community_id: list[entity_id]}
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from kg4vd.core.types import KGEdge, KGNode


class NetworkXKGStore:
    """Multi-graph store: multiple edges allowed between the same pair."""

    def __init__(self):
        self.g: nx.MultiDiGraph = nx.MultiDiGraph()
        self._edge_id_to_key: dict[str, tuple[str, str, int]] = {}

    # ----- mutators ---------------------------------------------------

    async def upsert_node(self, node: KGNode) -> None:
        self.g.add_node(node.entity_id, **_node_attrs(node))

    async def upsert_edge(self, edge: KGEdge) -> None:
        # If edge_id already exists, replace the edge attributes in place.
        if edge.edge_id in self._edge_id_to_key:
            u, v, k = self._edge_id_to_key[edge.edge_id]
            self.g.remove_edge(u, v, key=k)
        if not self.g.has_node(edge.src_id):
            self.g.add_node(edge.src_id)
        if not self.g.has_node(edge.tgt_id):
            self.g.add_node(edge.tgt_id)
        key = self.g.add_edge(edge.src_id, edge.tgt_id, **_edge_attrs(edge))
        self._edge_id_to_key[edge.edge_id] = (edge.src_id, edge.tgt_id, key)

    async def delete_node(self, entity_id: str) -> None:
        if self.g.has_node(entity_id):
            # Remove edges first to keep the lookup clean.
            for u, v, key in list(self.g.edges(entity_id, keys=True)):
                eid = self.g.edges[u, v, key].get("edge_id")
                self._edge_id_to_key.pop(eid, None)
                self.g.remove_edge(u, v, key=key)
            self.g.remove_node(entity_id)

    async def delete_edge(self, edge_id: str) -> None:
        key = self._edge_id_to_key.pop(edge_id, None)
        if key is None:
            return
        u, v, k = key
        if self.g.has_edge(u, v, key=k):
            self.g.remove_edge(u, v, key=k)

    # ----- queries ----------------------------------------------------

    def has_entity(self, entity_id: str) -> bool:
        return self.g.has_node(entity_id)

    def get_node(self, entity_id: str) -> dict[str, Any] | None:
        if not self.g.has_node(entity_id):
            return None
        return dict(self.g.nodes[entity_id])

    def all_nodes(self) -> list[dict[str, Any]]:
        return [{"entity_id": n, **dict(d)} for n, d in self.g.nodes(data=True)]

    def all_edges(self) -> list[dict[str, Any]]:
        out = []
        for u, v, key, data in self.g.edges(keys=True, data=True):
            out.append({"src_id": u, "tgt_id": v, **dict(data)})
        return out

    def iter_neighbours(
        self, entity_id: str, hops: int = 1
    ) -> Iterable[tuple[str, str]]:
        """Yield (edge_id, neighbour_id) for 1..hops outgoing/incoming edges."""

        if not self.g.has_node(entity_id):
            return
        # Multi-hop expansion can chain this 1-hop iterator externally.
        seen: set[str] = set()
        # outgoing
        for _, tgt, data in self.g.out_edges(entity_id, data=True):
            eid = data.get("edge_id")
            if eid and eid not in seen:
                seen.add(eid)
                yield eid, tgt
        # incoming
        for src, _, data in self.g.in_edges(entity_id, data=True):
            eid = data.get("edge_id")
            if eid and eid not in seen:
                seen.add(eid)
                yield eid, src

    def aligned_entities(self, entity_id: str) -> list[str]:
        out: set[str] = set()
        if not self.g.has_node(entity_id):
            return []
        for _, tgt, data in self.g.out_edges(entity_id, data=True):
            if data.get("edge_type") == "same_as":
                out.add(tgt)
        for src, _, data in self.g.in_edges(entity_id, data=True):
            if data.get("edge_type") == "same_as":
                out.add(src)
        return sorted(out)

    def relation_meta(self, edge_id: str) -> dict[str, Any] | None:
        key = self._edge_id_to_key.get(edge_id)
        if key is None:
            return None
        u, v, k = key
        if not self.g.has_edge(u, v, key=k):
            return None
        d = dict(self.g.edges[u, v, k])
        d["src_id"] = u
        d["tgt_id"] = v
        return d

    # ----- community detection ---------------------------------------

    async def communities(
        self, *, algorithm: str = "label_propagation", min_size: int = 3
    ) -> dict[str, list[str]]:
        if self.g.number_of_nodes() == 0:
            return {}
        # Run on the undirected projection so semantic clusters surface.
        ug = self.g.to_undirected(as_view=False)

        def _run() -> list[set[str]]:
            if algorithm == "label_propagation":
                return list(
                    nx.algorithms.community.label_propagation_communities(ug)
                )
            if algorithm == "greedy_modularity":
                return list(
                    nx.algorithms.community.greedy_modularity_communities(ug)
                )
            if algorithm == "leiden":
                # Optional dependency; fall back to label propagation if missing.
                try:
                    import igraph as ig  # noqa: F401
                    import leidenalg  # noqa: F401
                    return _leiden(ug)
                except ImportError:
                    return list(
                        nx.algorithms.community.label_propagation_communities(ug)
                    )
            raise ValueError(f"Unknown community algorithm {algorithm!r}")

        groups = await asyncio.to_thread(_run)
        out: dict[str, list[str]] = {}
        for i, members in enumerate(sorted(groups, key=lambda s: -len(s))):
            members = sorted(members)
            if len(members) < min_size:
                continue
            out[f"C{i}"] = members
        return out

    # ----- persistence ------------------------------------------------

    async def persist(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_gpickle(self.g, path)  # type: ignore[attr-defined]


def _leiden(ug: nx.Graph) -> list[set[str]]:
    """Lazy-imported Leiden community detection."""

    import igraph as ig
    import leidenalg

    nodes = list(ug.nodes())
    name_to_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(name_to_idx[u], name_to_idx[v]) for u, v in ug.edges()]
    g = ig.Graph(edges=edges, directed=False)
    parts = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
    return [{nodes[i] for i in p} for p in parts]


def _node_attrs(node: KGNode) -> dict[str, Any]:
    return {
        "entity_id": node.entity_id,
        "name": node.name,
        "entity_type": node.entity_type,
        "modality": node.modality,
        "description": node.description,
        "visual_description": node.visual_description,
        "visual_type": node.visual_type,
        "bbox": node.bbox,
        "source_pages": list(node.source_pages),
        "source_chunks": list(node.source_chunks),
        "metadata": dict(node.metadata),
    }


def _edge_attrs(edge: KGEdge) -> dict[str, Any]:
    return {
        "edge_id": edge.edge_id,
        "relation": edge.relation,
        "edge_type": edge.edge_type,
        "description": edge.description,
        "visual_evidence_hint": edge.visual_evidence_hint,
        "confidence": edge.confidence,
        "source_pages": list(edge.source_pages),
        "source_chunks": list(edge.source_chunks),
        "metadata": dict(edge.metadata),
    }
