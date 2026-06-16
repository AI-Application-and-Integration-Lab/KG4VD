"""Per-component scorecard renderer for extraction review."""

from __future__ import annotations

from typing import Iterable

from kg4vd.core.types import KGEdge, KGNode
from kg4vd.ingest.components import Component
from kg4vd.kg.extract.controller import is_stub
from kg4vd.kg.extract.raw_ops import RawEdgeOp, RawNodeOp


_NODE = KGNode | RawNodeOp
_EDGE = KGEdge | RawEdgeOp


def render_component_scorecards(
    components: Iterable[Component],
    nodes: Iterable[_NODE],
    edges: Iterable[_EDGE],
) -> str:
    """Return a multi-line scorecard string, one card per component +
    an `== unanchored ==` section for items with no valid grounding.
    """
    components = list(components)
    nodes = list(nodes)
    edges = list(edges)
    valid_cids = {c.component_id for c in components}

    nodes_per: dict[str, list[_NODE]] = {}
    edges_per: dict[str, list[KGEdge]] = {}
    for n in nodes:
        for cid in getattr(n, "source_components", []) or []:
            nodes_per.setdefault(cid, []).append(n)
    for e in edges:
        for cid in getattr(e, "source_components", []) or []:
            edges_per.setdefault(cid, []).append(e)

    parts: list[str] = []
    for c in components:
        cid = c.component_id
        ctype = c.type
        pos = c.position or "?"
        parts.append(f"== {cid} ({ctype}, {pos}) ==")
        # Manifest content: text for textual; html line for tables;
        # caption for captioned visuals; "see annotated page" otherwise.
        if c.text_full:
            text = c.text_full.replace("\n", " ").strip()
            if len(text) > 400:
                text = text[:400] + "..."
            parts.append(f"  manifest text: {text}")
        elif c.caption:
            parts.append(f"  caption: {c.caption}")
        else:
            parts.append(f"  manifest: <{ctype} - see annotated page>")
        # Tables: surface MinerU's HTML so the Reflector's enumeration
        # density check has something to count.
        if ctype == "table" and c.html:
            from kg4vd.kg.extract.manifest import clean_table_html
            html = clean_table_html(c.html)
            if len(html) > 1200:
                html = html[:1200] + "..."
            parts.append(f"  table_html: {html}")

        ns = nodes_per.get(cid, [])
        if ns:
            parts.append(
                f"  nodes grounded here ({len(ns)}): "
                + "; ".join(_fmt_node(n) for n in ns)
            )
        else:
            parts.append("  nodes grounded here (0): (none)")

        es = edges_per.get(cid, [])
        if es:
            parts.append(
                f"  edges grounded here ({len(es)}): "
                + "; ".join(_fmt_edge(e) for e in es)
            )
        else:
            parts.append("  edges grounded here (0): (none)")
        parts.append("")  # blank between cards

    # Catch entities whose source_components don't match any known
    # component. Validator R1 should prevent this; if it shows up,
    # something upstream produced ungrounded ops.
    orphan_nodes = [
        n for n in nodes
        if not any(cid in valid_cids
                   for cid in (getattr(n, "source_components", []) or []))
    ]
    orphan_edges = [
        e for e in edges
        if not any(cid in valid_cids
                   for cid in (getattr(e, "source_components", []) or []))
    ]
    if orphan_nodes or orphan_edges:
        parts.append("== unanchored ==")
        for n in orphan_nodes:
            parts.append(
                f"  - NODE {getattr(n, 'name', '?')} "
                f"({getattr(n, 'entity_type', '?')}) "
                f"cited={getattr(n, 'source_components', None)}"
            )
        for e in orphan_edges:
            parts.append(
                f"  - EDGE {_edge_endpoints(e)} "
                f"cited={getattr(e, 'source_components', None)}"
            )

    return "\n".join(parts).rstrip() or "(empty - no nodes or edges yet)"


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _fmt_node(n: _NODE) -> str:
    et = getattr(n, "entity_type", "?")
    mod = getattr(n, "modality", "?")
    name = getattr(n, "name", "?")
    marker = " [STUB - upgrade via replace_nodes]" if is_stub(n) else ""
    return f"{name} ({et}, {mod}){marker}"


def _fmt_edge(e: _EDGE) -> str:
    return _edge_endpoints(e)


def _edge_endpoints(e) -> str:
    """Render an edge as ``src -[rel]-> tgt``. Works for both
    `KGEdge` (post-resolution: src_id/tgt_id + optional metadata
    src_name/tgt_name) and `RawEdgeOp` (pre-resolution: src/tgt are
    the names directly)."""
    md = getattr(e, "metadata", {}) or {}
    # RawEdgeOp first (cheaper attr lookup), then metadata stash, then
    # fall back to the resolved IDs.
    src = (
        getattr(e, "src", None)
        or md.get("src_name")
        or getattr(e, "src_id", "?")
    )
    tgt = (
        getattr(e, "tgt", None)
        or md.get("tgt_name")
        or getattr(e, "tgt_id", "?")
    )
    rel = getattr(e, "relation", "?")
    return f"{src} -[{rel}]-> {tgt}"


__all__ = ["render_component_scorecards"]
