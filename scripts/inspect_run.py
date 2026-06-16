"""Inspect a build artifact directory.

Reads ``<work_dir>/{kg/nodes.jsonl, kg/edges.jsonl, index/cards.jsonl,
pages.jsonl, run_manifest.json}`` and produces:

  1. A statistics summary printed to stdout (per-page, per-stage, per-type).
  2. ``<out_dir>/local_page_<N>.png``       - local subgraph for one page.
  3. ``<out_dir>/doc_refined.png``           - doc-level graph, semantic edges only.
  4. ``<out_dir>/doc_with_alignment.png``    - doc-level graph, all edge types,
                                                nodes coloured by community.

Usage:
    python scripts/inspect_run.py <work_dir> [--page N] [--out-dir <dir>]

Example:
    python scripts/inspect_run.py recipes/test/exp --page 1 \\
                                  --out-dir recipes/test/exp/inspect
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def _load_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_run(work_dir: Path) -> tuple[list, list, list, list, dict]:
    nodes = _load_jsonl(work_dir / "kg" / "nodes.jsonl")
    edges = _load_jsonl(work_dir / "kg" / "edges.jsonl")
    cards = _load_jsonl(work_dir / "index" / "cards.jsonl")
    pages = _load_jsonl(work_dir / "pages.jsonl")
    manifest = {}
    if (work_dir / "run_manifest.json").is_file():
        manifest = json.loads((work_dir / "run_manifest.json").read_text())
    return nodes, edges, cards, pages, manifest


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------


def print_stats(nodes, edges, cards, pages, manifest) -> None:
    print("=" * 72)
    print(f"  Run: {manifest.get('run_id', '?')}   "
          f"llm={manifest.get('llm_id', '?')}   "
          f"encoder={manifest.get('encoder_id', '?')}")
    print("=" * 72)

    # ----- pipeline-level totals -----
    t = manifest.get("totals", {})
    print("\n[1] Pipeline totals")
    print(f"    pages              : {len(pages)}")
    print(f"    nodes              : {len(nodes)}")
    print(f"    edges              : {len(edges)}")
    print(f"    cards              : {len(cards)}")
    print(f"    llm_calls          : {t.get('llm_calls', 0)}")
    print(f"    total_tokens       : {t.get('total_tokens', 0):,}")
    print(f"    wall_clock         : {t.get('total_wall_ms', 0)/1000:.1f} s")

    # ----- per-page extraction (the per-page local graph) -----
    print("\n[2] Per-page local graph (LOCAL view - what each page contributes)")
    print(f"    {'page':>4} {'nodes':>6} {'sem_edges':>10} {'visual_ent':>10}")
    pid_to_nodes: dict[int, set[str]] = defaultdict(set)
    pid_to_visual: dict[int, set[str]] = defaultdict(set)
    pid_to_sem_edges: dict[int, set[str]] = defaultdict(set)
    for n in nodes:
        for pid in n["source_pages"]:
            pid_to_nodes[pid].add(n["entity_id"])
            if n.get("modality") == "visual":
                pid_to_visual[pid].add(n["entity_id"])
    for e in edges:
        if e["edge_type"] != "semantic":
            continue
        for pid in e["source_pages"]:
            pid_to_sem_edges[pid].add(e["edge_id"])
    for p in pages:
        pid = p["page_id"]
        print(f"    {pid:>4} {len(pid_to_nodes[pid]):>6} "
              f"{len(pid_to_sem_edges[pid]):>10} {len(pid_to_visual[pid]):>10}")

    # ----- name-merge stats (Tier 1) -----
    print("\n[3] Name-based cross-page merge (Tier 1)")
    sp_lengths = Counter(len(n["source_pages"]) for n in nodes)
    multi_page = sum(c for k, c in sp_lengths.items() if k > 1)
    print(f"    multi-page nodes   : {multi_page} / {len(nodes)} "
          f"({100*multi_page/max(1,len(nodes)):.1f}%)")
    print(f"    source_pages distribution: {dict(sorted(sp_lengths.items()))}")
    if multi_page == 0 and len(nodes) > 0 and len(pages) > 1:
        print("    ⚠️  every node touches exactly one page - Tier 1 cross-page merge")
        print("        is broken in the current build pipeline (see V2_PLAN A4 / build.py:_extract).")

    # ----- alignment edges (Tier 2) -----
    print("\n[4] Embedding-based cross-page alignment (Tier 2)")
    et_counter = Counter(e["edge_type"] for e in edges)
    print(f"    edge_type distribution: {dict(et_counter.most_common())}")
    align = [e for e in edges if e["edge_type"] in
             {"same_as", "aligns_with", "visually_refers_to"}]
    if align:
        confs = [e["confidence"] for e in align]
        print(f"    alignment confidence min/avg/max: "
              f"{min(confs):.2f} / {sum(confs)/len(confs):.2f} / {max(confs):.2f}")

    # ----- entity / card type distributions -----
    print("\n[5] Schema distributions")
    et_n = Counter(n["entity_type"] for n in nodes)
    print(f"    entity_type        : {dict(et_n.most_common())}")
    mod_n = Counter(n["modality"] for n in nodes)
    print(f"    modality           : {dict(mod_n.most_common())}")
    ct = Counter(c["evidence_type"] for c in cards)
    print(f"    card.evidence_type : {dict(ct.most_common())}")

    # ----- communities (Tier 3) -----
    print("\n[6] Community detection (Tier 3 - doc-level evidence)")
    comm_cards = [c for c in cards if c["evidence_type"] == "community_summary"]
    print(f"    communities found  : {len(comm_cards)}")
    if comm_cards:
        sizes = sorted(
            ((c.get("graph_refs") or {}).get("community_id", "?"),
             len((c.get("graph_refs") or {}).get("node_ids", []) or []))
            for c in comm_cards
        )
        print(f"    sizes              : "
              f"{', '.join(f'{cid}:{n}' for cid, n in sizes)}")

    # ----- top-degree hubs -----
    g = build_doc_graph(nodes, edges, include_alignment=False)
    if g.number_of_nodes() > 0:
        print("\n[7] Top hubs (semantic graph only)")
        deg = sorted(g.degree(), key=lambda x: -x[1])[:8]
        name_by_id = {n["entity_id"]: n["name"] for n in nodes}
        for nid, d in deg:
            print(f"    deg={d:>3}  {name_by_id.get(nid, nid)}")


# --------------------------------------------------------------------------
# Graph construction + rendering
# --------------------------------------------------------------------------


def build_doc_graph(
    nodes, edges, *, include_alignment: bool = True
) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for n in nodes:
        g.add_node(n["entity_id"], **{k: v for k, v in n.items() if k != "metadata"})
    for e in edges:
        if not include_alignment and e["edge_type"] != "semantic":
            continue
        if e["src_id"] in g and e["tgt_id"] in g:
            g.add_edge(
                e["src_id"], e["tgt_id"],
                relation=e["relation"], edge_type=e["edge_type"],
            )
    return g


def communities_from_cards(cards) -> dict[str, set[str]]:
    """Return {entity_id: community_id} from community_summary cards."""
    out: dict[str, set[str]] = {}
    for c in cards:
        if c["evidence_type"] != "community_summary":
            continue
        cid = (c.get("graph_refs") or {}).get("community_id")
        members = (c.get("graph_refs") or {}).get("node_ids") or []
        if not cid:
            continue
        for m in members:
            out.setdefault(m, set()).add(cid)
    return out


# ---------------------------------------------------------------------------
# Common style constants
# ---------------------------------------------------------------------------

MODALITY_COLORS = {
    "text":   "#88B4FF",   # soft blue
    "visual": "#FFB07B",   # soft orange
}

# Cross-page alignment edges - bright + thick so they stand out from the
# semantic background.
EDGE_PALETTE = {
    "semantic":           {"color": "#888888", "width": 0.8,  "alpha": 0.4,  "style": "solid",   "zorder": 1},
    "same_as":            {"color": "#1f78b4", "width": 2.6,  "alpha": 0.95, "style": "solid",   "zorder": 4},
    "aligns_with":        {"color": "#e31a1c", "width": 1.8,  "alpha": 0.85, "style": "dashed",  "zorder": 3},
    "visually_refers_to": {"color": "#33a02c", "width": 2.0,  "alpha": 0.95, "style": "dashdot", "zorder": 5},
    "part_of":            {"color": "#6a3d9a", "width": 1.4,  "alpha": 0.85, "style": "dotted",  "zorder": 2},
    "supports":           {"color": "#b15928", "width": 1.0,  "alpha": 0.7,  "style": "dotted",  "zorder": 2},
}


def _modality_colors(g: nx.Graph) -> list[str]:
    return [
        MODALITY_COLORS.get(g.nodes[n].get("modality", "text"), "#cccccc")
        for n in g.nodes
    ]


def _format_pages(pids) -> str:
    """Compact page-list label, e.g. [1,4-6] for [1,4,5,6]."""
    if not pids:
        return ""
    pids = sorted(set(int(p) for p in pids))
    runs: list[tuple[int, int]] = []
    s = e = pids[0]
    for p in pids[1:]:
        if p == e + 1:
            e = p
        else:
            runs.append((s, e))
            s = e = p
    runs.append((s, e))
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


# ---------------------------------------------------------------------------
# Per-page subgraph panel (used by the grid)
# ---------------------------------------------------------------------------


def _render_page_panel(ax, nodes, edges, page_id: int) -> tuple[int, int]:
    """Render one page's local subgraph onto an axes. Returns (n_nodes, n_edges)."""

    page_nodes = {n["entity_id"]: n for n in nodes if page_id in n["source_pages"]}
    g = nx.MultiDiGraph()
    for nid, n in page_nodes.items():
        g.add_node(nid, **n)
    for e in edges:
        if e["edge_type"] != "semantic":
            continue
        if page_id not in e["source_pages"]:
            continue
        if e["src_id"] in page_nodes and e["tgt_id"] in page_nodes:
            g.add_edge(e["src_id"], e["tgt_id"], relation=e["relation"])

    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()

    if n_nodes == 0:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center",
                transform=ax.transAxes, color="#999", fontsize=11)
    else:
        # Layout - generous spacing for small panels.
        pos = nx.spring_layout(
            g, seed=42, k=1.4/max(1, n_nodes**0.5), iterations=60
        )
        nx.draw_networkx_nodes(
            g, pos, ax=ax, node_size=320,
            node_color=_modality_colors(g),
            edgecolors="#333", linewidths=0.5,
        )
        # Labels: short name only; pages are obvious from the panel title.
        labels = {nid: g.nodes[nid].get("name", nid)[:18] for nid in g.nodes}
        nx.draw_networkx_labels(
            g, pos, labels=labels, ax=ax, font_size=6,
        )
        nx.draw_networkx_edges(
            g, pos, ax=ax, edge_color="#666", alpha=0.45,
            arrows=True, width=0.8,
            connectionstyle="arc3,rad=0.08", arrowsize=8,
        )
        edge_labels = {(u, v): d.get("relation", "")[:14]
                       for u, v, d in g.edges(data=True)}
        if edge_labels:
            nx.draw_networkx_edge_labels(
                g, pos, edge_labels=edge_labels, ax=ax,
                font_size=5, alpha=0.6,
            )
    ax.set_title(f"p{page_id}  ({n_nodes}n, {n_edges}e)", fontsize=10)
    ax.axis("off")
    return n_nodes, n_edges


def render_pages_grid(nodes, edges, pages, out_path: Path) -> None:
    """Multi-panel grid: one local subgraph per page.

    Modality colour: text = blue, visual = orange. Useful for spotting
    extraction-quality differences across pages at a glance.
    """

    n_pages = len(pages)
    if n_pages == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "(no pages)", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    # Choose a near-square grid.
    import math
    cols = min(5, math.ceil(math.sqrt(n_pages)))
    rows = math.ceil(n_pages / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.6))
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

    pages_sorted = sorted(pages, key=lambda p: p["page_id"])
    for idx, p in enumerate(pages_sorted):
        _render_page_panel(axes_flat[idx], nodes, edges, p["page_id"])
    for k in range(len(pages_sorted), len(axes_flat)):
        axes_flat[k].axis("off")

    fig.suptitle(
        "PER-PAGE LOCAL GRAPHS  (text = blue, visual = orange)",
        fontsize=12, y=1.0,
    )
    # Legend (top-right of figure).
    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor=MODALITY_COLORS["text"],   edgecolor="#333", label="text entity"),
            Patch(facecolor=MODALITY_COLORS["visual"], edgecolor="#333", label="visual entity"),
        ],
        loc="upper right", fontsize=9, framealpha=0.95,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Backwards-compat single-page render (used by --page CLI flag)
# ---------------------------------------------------------------------------


def render_local_page(nodes, edges, page_id: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    n_nodes, n_edges = _render_page_panel(ax, nodes, edges, page_id)
    ax.set_title(
        f"LOCAL GRAPH - page {page_id}  ({n_nodes} nodes, {n_edges} edges)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Doc-level renders
# ---------------------------------------------------------------------------


def render_doc_refined(nodes, edges, out_path: Path) -> None:
    """Doc-level semantic-only graph. Modality coloured. Page IDs on labels."""
    g = build_doc_graph(nodes, edges, include_alignment=False)
    _render_full_graph(
        g, out_path,
        title=(f"REFINED DOC GRAPH (semantic edges only) - "
               f"{g.number_of_nodes()} nodes, {g.number_of_edges()} edges"),
        color_mode="modality",
        show_edge_types=False,
        show_page_ids_on_labels=True,
    )


def render_doc_with_alignment(nodes, edges, cards, out_path: Path) -> None:
    """Doc-level graph with cross-page edges drawn prominently.

    Node colour = community.  Cross-page alignment edges drawn THICK and BRIGHT
    on top of the semantic background. Top-K nodes labelled with name + pages;
    top-K alignment edges labelled with relation + pages.
    """
    g = build_doc_graph(nodes, edges, include_alignment=True)
    membership = communities_from_cards(cards)
    _render_full_graph(
        g, out_path,
        title=(f"DOC GRAPH + ALIGNMENT  (coloured by community; "
               f"cross-page edges highlighted) - "
               f"{g.number_of_nodes()} nodes, {g.number_of_edges()} edges"),
        community_membership=membership,
        color_mode="community",
        show_edge_types=True,
        show_page_ids_on_labels=True,
        show_alignment_edge_labels=True,
    )


def render_doc_modality(nodes, edges, out_path: Path) -> None:
    """Doc-level graph, modality-coloured, all edges shown.

    Useful for spotting visual vs text entities across the entire document.
    """
    g = build_doc_graph(nodes, edges, include_alignment=True)
    _render_full_graph(
        g, out_path,
        title=(f"DOC GRAPH (modality-coloured: text = blue, visual = orange) - "
               f"{g.number_of_nodes()} nodes, {g.number_of_edges()} edges"),
        color_mode="modality",
        show_edge_types=True,
        show_page_ids_on_labels=True,
    )


def _render_full_graph(
    g: nx.MultiDiGraph,
    out_path: Path,
    title: str,
    *,
    community_membership: dict[str, set[str]] | None = None,
    color_mode: str = "modality",   # "modality" | "community"
    show_edge_types: bool = False,
    show_page_ids_on_labels: bool = False,
    show_alignment_edge_labels: bool = False,
    label_top_k: int = 25,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 11))
    if g.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "(empty graph)", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    # Layout
    pos = nx.spring_layout(
        g, seed=42, k=1.6/max(1, g.number_of_nodes()**0.5), iterations=100
    )

    # ---- node colour ----
    if color_mode == "community" and community_membership:
        cids = sorted({c for s in community_membership.values() for c in s})
        cmap = plt.cm.tab20
        cid_to_color = {cid: cmap(i % 20) for i, cid in enumerate(cids)}
        node_colors = []
        for n in g.nodes:
            if n in community_membership and community_membership[n]:
                cid = sorted(community_membership[n])[0]
                node_colors.append(cid_to_color[cid])
            else:
                node_colors.append("#cccccc")
    else:
        node_colors = _modality_colors(g)

    # ---- node size by degree ----
    deg = dict(g.degree())
    sizes = [140 + 70 * deg.get(n, 0) for n in g.nodes]

    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_size=sizes, node_color=node_colors,
        alpha=0.9, linewidths=0.6, edgecolors="#333",
    )

    # ---- edges ----
    # Draw semantic edges first (background), then alignment edges on top
    # so the cross-page links are unmistakably the dominant visual cue.
    if show_edge_types:
        edges_by_type: dict[str, list] = {}
        for u, v, d in g.edges(data=True):
            edges_by_type.setdefault(d.get("edge_type", "semantic"), []).append((u, v))

        for et in sorted(edges_by_type, key=lambda x: EDGE_PALETTE.get(x, {}).get("zorder", 0)):
            style = EDGE_PALETTE.get(et, EDGE_PALETTE["semantic"])
            nx.draw_networkx_edges(
                g, pos, edgelist=edges_by_type[et], ax=ax,
                edge_color=style["color"], width=style["width"],
                alpha=style["alpha"], style=style["style"],
                arrows=True, arrowsize=10,
                connectionstyle="arc3,rad=0.10",
            )

        # Legend (top-right): only edge types that appear
        from matplotlib.lines import Line2D
        handles = [
            Line2D(
                [0], [0], color=EDGE_PALETTE[et]["color"],
                lw=EDGE_PALETTE[et]["width"] + 0.5,
                linestyle=EDGE_PALETTE[et]["style"].replace("dashdot", "-."),
                label=et,
            )
            for et in EDGE_PALETTE if et in edges_by_type
        ]
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.95)
    else:
        nx.draw_networkx_edges(
            g, pos, ax=ax,
            edge_color="#666", width=0.9, alpha=0.55, arrows=True,
            connectionstyle="arc3,rad=0.10", arrowsize=10,
        )

    # ---- node labels (top-K by degree to avoid clutter) ----
    label_keep = sorted(g.degree(), key=lambda x: -x[1])[:label_top_k]
    label_ids = {nid for nid, _ in label_keep}
    labels = {}
    for nid in g.nodes:
        if nid not in label_ids:
            continue
        name = g.nodes[nid].get("name", nid)[:22]
        if show_page_ids_on_labels:
            pages = g.nodes[nid].get("source_pages", []) or []
            page_str = _format_pages(pages)
            labels[nid] = f"{name}\n[p{page_str}]" if page_str else name
        else:
            labels[nid] = name
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7)

    # ---- alignment-edge labels (only the most prominent ones) ----
    if show_alignment_edge_labels:
        align_edges = [
            (u, v, d) for u, v, d in g.edges(data=True)
            if d.get("edge_type") in {"same_as", "visually_refers_to"}
        ]
        # Limit clutter: top 10 alignment edges by endpoint-degree.
        align_edges.sort(
            key=lambda t: -(deg.get(t[0], 0) + deg.get(t[1], 0))
        )
        edge_labels = {}
        for u, v, d in align_edges[:10]:
            et_short = d.get("edge_type", "")[:6]
            edge_labels[(u, v)] = et_short
        if edge_labels:
            nx.draw_networkx_edge_labels(
                g, pos, edge_labels=edge_labels, ax=ax,
                font_size=6, alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.7),
            )

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Adaptive-extraction grid (rows = pages, cols = rounds)
# ---------------------------------------------------------------------------


def _render_adaptive_panel(
    ax,
    record: dict,
    prev_record: dict | None,
    layout_pos: dict,
) -> None:
    """One cell: post-round state for one page, with diff coloring."""

    nodes = record["nodes"]
    edges = record["edges"]
    prev_node_ids: set[str] = set()
    prev_edge_keys: set[tuple] = set()
    if prev_record is not None:
        prev_node_ids = {n["entity_id"] for n in prev_record["nodes"]}
        prev_edge_keys = {
            (e["src_id"], e["tgt_id"], e.get("relation", ""))
            for e in prev_record["edges"]
        }

    g = nx.MultiDiGraph()
    for n in nodes:
        g.add_node(n["entity_id"], **{k: v for k, v in n.items() if k != "metadata"})
    for e in edges:
        if e["src_id"] in g and e["tgt_id"] in g:
            g.add_edge(
                e["src_id"], e["tgt_id"],
                relation=e.get("relation", ""),
                edge_type=e.get("edge_type", "semantic"),
            )

    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()

    if n_nodes == 0:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center",
                transform=ax.transAxes, color="#999", fontsize=9)
    else:
        pos = {nid: layout_pos[nid] for nid in g.nodes if nid in layout_pos}
        for nid in g.nodes:
            if nid not in pos:
                pos[nid] = (0.0, 0.0)

        new_node_ids = {nid for nid in g.nodes if nid not in prev_node_ids}
        node_face = [
            MODALITY_COLORS.get(g.nodes[nid].get("modality", "text"), "#cccccc")
            for nid in g.nodes
        ]
        node_edgec = [
            "#2ca02c" if nid in new_node_ids else "#555"
            for nid in g.nodes
        ]
        node_lw = [1.6 if nid in new_node_ids else 0.4 for nid in g.nodes]

        nx.draw_networkx_nodes(
            g, pos, ax=ax, node_size=240,
            node_color=node_face, edgecolors=node_edgec, linewidths=node_lw,
        )

        new_edges, old_edges = [], []
        for u, v, d in g.edges(data=True):
            key = (u, v, d.get("relation", ""))
            (old_edges if key in prev_edge_keys else new_edges).append((u, v))

        if old_edges:
            nx.draw_networkx_edges(
                g, pos, edgelist=old_edges, ax=ax,
                edge_color="#aaa", alpha=0.5, width=0.7,
                arrows=True, arrowsize=7,
                connectionstyle="arc3,rad=0.08",
            )
        if new_edges:
            nx.draw_networkx_edges(
                g, pos, edgelist=new_edges, ax=ax,
                edge_color="#2ca02c", alpha=0.95, width=1.6,
                arrows=True, arrowsize=8,
                connectionstyle="arc3,rad=0.08",
            )

        labels = {nid: g.nodes[nid].get("name", nid)[:12] for nid in g.nodes}
        nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=5)

    audit = record.get("audit", {}) or {}
    page_id = record.get("page_id")
    round_idx = record.get("round_idx", 0)
    if round_idx == 0:
        op_str = f"+{audit.get('add_nodes', 0)}n +{audit.get('add_edges', 0)}e"
    else:
        op_str = (
            f"+{audit.get('add_nodes', 0)}/-{audit.get('delete_nodes', 0)}n "
            f"+{audit.get('add_edges', 0)}/-{audit.get('delete_edges', 0)}e"
        )
    ax.set_title(
        f"p{page_id}  r{round_idx}   {op_str}\n({n_nodes}n, {n_edges}e)",
        fontsize=8,
    )
    ax.axis("off")


def render_adaptive_grid(snap_dir: Path, out_path: Path) -> None:
    """Pages × rounds grid showing how each page's local graph evolves.

    Reads ``<snap_dir>/page_*.jsonl`` (written by ``build._persist_page_snapshots``).
    Each row = one page, each column = one adaptive round (round 0 = post-init,
    round k = post-kth reflector). Edges/nodes added in this round are green;
    preserved ones are grey. Title shows the audit counts for that round.
    """

    files = sorted(snap_dir.glob("page_*.jsonl"))
    if not files:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "(no snapshots)", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    page_records: dict[int, list[dict]] = {}
    for fp in files:
        recs = _load_jsonl(fp)
        if not recs:
            continue
        page_records[recs[0]["page_id"]] = sorted(recs, key=lambda r: r["round_idx"])

    if not page_records:
        return

    page_ids = sorted(page_records)
    n_rows = len(page_ids)
    n_cols = max(len(rs) for rs in page_records.values())

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.0, n_rows * 2.6),
        squeeze=False,
    )

    for r, pid in enumerate(page_ids):
        records = page_records[pid]

        # Per-page layout: union of all rounds, so node positions are stable
        # across columns and additions are easy to spot.
        union = nx.MultiDiGraph()
        for rec in records:
            for n in rec["nodes"]:
                if n["entity_id"] not in union:
                    union.add_node(
                        n["entity_id"],
                        **{k: v for k, v in n.items() if k != "metadata"},
                    )
            for e in rec["edges"]:
                if e["src_id"] in union and e["tgt_id"] in union:
                    union.add_edge(
                        e["src_id"], e["tgt_id"],
                        relation=e.get("relation", ""),
                    )
        if union.number_of_nodes() > 0:
            layout_pos = nx.spring_layout(
                union, seed=42,
                k=1.4 / max(1, union.number_of_nodes() ** 0.5),
                iterations=80,
            )
        else:
            layout_pos = {}

        prev_record = None
        for c in range(n_cols):
            ax = axes[r][c]
            if c < len(records):
                _render_adaptive_panel(ax, records[c], prev_record, layout_pos)
                prev_record = records[c]
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "-", ha="center", va="center",
                        transform=ax.transAxes, color="#ccc", fontsize=14)

    fig.suptitle(
        "ADAPTIVE EXTRACTION  (rows = pages, cols = rounds; "
        "green = new in round, grey = preserved)",
        fontsize=12, y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work_dir", type=Path,
                    help="Recipe work_dir (e.g. recipes/test/exp)")
    ap.add_argument("--page", type=int, default=1,
                    help="Page to render as a local-graph PNG (default: 1)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write PNGs (default: <work_dir>/inspect)")
    args = ap.parse_args()

    nodes, edges, cards, pages, manifest = load_run(args.work_dir)
    print_stats(nodes, edges, cards, pages, manifest)

    out_dir = args.out_dir or (args.work_dir / "inspect")
    out_dir.mkdir(parents=True, exist_ok=True)

    p_local = out_dir / f"local_page_{args.page:02d}.png"
    p_grid = out_dir / "pages_grid.png"
    p_adaptive = out_dir / "adaptive_grid.png"
    p_refined = out_dir / "doc_refined.png"
    p_aligned = out_dir / "doc_with_alignment.png"
    p_modality = out_dir / "doc_modality.png"

    render_local_page(nodes, edges, args.page, p_local)
    render_pages_grid(nodes, edges, pages, p_grid)
    render_doc_refined(nodes, edges, p_refined)
    render_doc_with_alignment(nodes, edges, cards, p_aligned)
    render_doc_modality(nodes, edges, p_modality)

    snap_dir = args.work_dir / "snapshots" / "per_page"
    if snap_dir.is_dir():
        render_adaptive_grid(snap_dir, p_adaptive)

    print("\n[8] Wrote PNGs:")
    print(f"    {p_local}            (one page, large view)")
    print(f"    {p_grid}             (all pages in a grid, modality-coloured)")
    if snap_dir.is_dir():
        print(f"    {p_adaptive}         (per-page graph evolution across adaptive rounds)")
    else:
        print(f"    (skipped {p_adaptive} - no snapshots/per_page in work_dir)")
    print(f"    {p_refined}          (doc-level, semantic edges only)")
    print(f"    {p_aligned}          (doc-level, all edges + cross-page highlighted)")
    print(f"    {p_modality}         (doc-level, modality-coloured)")


if __name__ == "__main__":
    main()
