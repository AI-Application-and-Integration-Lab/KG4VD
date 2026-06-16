"""Disk persistence for pipeline artifacts (extract snapshots, pages, KG).

Pure I/O helpers extracted from pipeline/build.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kg4vd.core.types import KGEdge, KGNode, Page


def _persist_page_snapshots(snap_dir: Path, page: Page, res: Any) -> None:
    """Dump one JSONL per page; one record per adaptive round.

    Record schema::

        {"page_id": int, "round_idx": int, "audit": {...},
         "nodes": [<KGNode JSON>, ...], "edges": [<KGEdge JSON>, ...]}
    """

    from dataclasses import asdict

    out = snap_dir / f"page_{page.page_id:04d}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for snap in res.snapshots:
            rec = {
                "page_id": page.page_id,
                "round_idx": snap.round_idx,
                "audit": asdict(snap.audit),
                "nodes": [n.model_dump(mode="json") for n in snap.nodes],
                "edges": [e.model_dump(mode="json") for e in snap.edges],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _persist_pages(work_dir: Path, pages: list[Page]) -> None:
    p = work_dir / "pages.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for page in pages:
            f.write(json.dumps(page.model_dump(mode="json"), ensure_ascii=False) + "\n")


def _load_pages(work_dir: Path) -> list[Page]:
    p = work_dir / "pages.jsonl"
    if not p.is_file():
        return []
    out: list[Page] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Page.model_validate_json(line))
    return out


def _persist_kg(work_dir: Path, nodes: list[KGNode], edges: list[KGEdge]) -> None:
    p = work_dir / "kg"
    p.mkdir(parents=True, exist_ok=True)
    with (p / "nodes.jsonl").open("w", encoding="utf-8") as f:
        for n in nodes:
            f.write(json.dumps(n.model_dump(mode="json"), ensure_ascii=False) + "\n")
    with (p / "edges.jsonl").open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e.model_dump(mode="json"), ensure_ascii=False) + "\n")


def _load_kg(work_dir: Path) -> tuple[list[KGNode], list[KGEdge]]:
    """Load persisted KG nodes + edges. Returns ([], []) if not present."""
    kg_dir = work_dir / "kg"
    nodes: list[KGNode] = []
    edges: list[KGEdge] = []
    np_path = kg_dir / "nodes.jsonl"
    ep_path = kg_dir / "edges.jsonl"
    if np_path.is_file():
        with np_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    nodes.append(KGNode.model_validate_json(line))
    if ep_path.is_file():
        with ep_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    edges.append(KGEdge.model_validate_json(line))
    return nodes, edges


