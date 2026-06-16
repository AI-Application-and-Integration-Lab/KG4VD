"""MinerU layout blocks → typed `Component` objects with stable IDs.

A `Component` is the unit the component-cued extractor cites via
`source_components`. Each non-chrome block from MinerU's
``preproc_blocks`` becomes one Component with a type-prefixed ID
(T1/P3/IM2/TB1/F1/L4) that the VLM can reference naturally.

Pipeline expectations:

  - Bboxes are in MinerU pixel space (the same coord system as
    ``page_size_mineru_px`` in middle.json). Downstream consumers must
    scale to ``page_size_render_px`` before drawing on the rendered page.
  - Chrome blocks (page_header / page_number / page_footer) are dropped
    here; the VLM never sees them.
  - The optional ``merge_close_components`` pass merges *same-type*
    textual blocks whose boxes are vertically close + horizontally
    aligned. It NEVER merges across types, and visual / structured
    blocks (image / chart / table / formula) are never merged at all.
  - Image prefix is ``IM``, NOT ``I``: ``I1`` is visually
    indistinguishable from ``11`` in bold sans-serif at small label
    sizes.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Static mapping tables - kept module-level so tests can import them.
# ---------------------------------------------------------------------------

TYPE_PREFIX: dict[str, str] = {
    "title": "T",
    "paragraph": "P",
    "text": "P",
    "list": "L",
    "index": "L",
    "image": "IM",
    "chart": "IM",
    "table": "TB",
    "formula": "F",
    "equation": "F",
    "interline_equation": "F",
}

CHROME_TYPES: set[str] = {"page_header", "page_number", "page_footer"}

# Block types that may be merged with adjacent same-type siblings.
# Visual / structured components (image / chart / table / formula) are
# NEVER merged - they represent atomic visual evidence.
MERGEABLE_TYPES: set[str] = {"paragraph", "text", "list", "index", "title"}


_ROLE_HINTS: dict[str, str] = {
    "title": "section/subsection title",
    "paragraph": "body text",
    "text": "body text",
    "list": "bulleted/numbered list",
    "index": "table-of-contents / index list",
    "image": "visual candidate; possible photo, figure, diagram, or chart",
    "chart": "chart / data visualization",
    "table": "structured evidence candidate",
    "formula": "math/formula region",
}


# ---------------------------------------------------------------------------
# Component data model
# ---------------------------------------------------------------------------


class Component(BaseModel):
    """One layout region the VLM can cite via its `component_id`.

    `bbox` is in MinerU pixel coords; downstream renderers must scale.
    """

    model_config = ConfigDict(extra="forbid")

    component_id: str
    block_ids: list[int] = Field(default_factory=list)
    type: str
    bbox: tuple[float, float, float, float]
    score: float | None = None
    text_full: str | None = None       # full OCR (textual blocks only)
    html: str | None = None            # MinerU table HTML (tables only)
    image_path: str | None = None      # MinerU figure crop path (image/chart only)
    caption: str | None = None         # downstream-set caption / hint
    role_hint: str = ""
    position: str = ""                 # downstream-set spatial label
    neighbours: dict[str, str] = Field(default_factory=dict)
    merged: bool = False               # True when produced by merge_close_components


# ---------------------------------------------------------------------------
# Builder - MinerU preproc_blocks → list[Component]
# ---------------------------------------------------------------------------


def build_components_from_middle(preproc_blocks: list[dict]) -> list[Component]:
    """Turn one page's ``preproc_blocks`` (from middle.json) into Components.

    Reading order follows MinerU's ``index`` field. Chrome blocks are
    dropped. Component IDs are numbered per-type-prefix in reading
    order (T1, T2, ..., P1, P2, ..., IM1, ...).
    """
    out: list[Component] = []
    counters: dict[str, int] = {}
    blocks = sorted(preproc_blocks or [], key=lambda b: b.get("index", 0))
    for block_id, blk in enumerate(blocks):
        btype = blk.get("type")
        if btype in CHROME_TYPES or btype is None:
            continue
        bbox = blk.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        prefix = TYPE_PREFIX.get(btype, "X")
        counters[prefix] = counters.get(prefix, 0) + 1
        cid = f"{prefix}{counters[prefix]}"
        out.append(Component(
            component_id=cid,
            block_ids=[block_id],
            type=btype,
            bbox=tuple(bbox),  # type: ignore[arg-type]
            score=blk.get("score"),
            text_full=_block_text_full(blk),
            html=_table_html_from_middle(blk) if btype == "table" else None,
            image_path=_image_path_from_middle(blk) if btype in ("image", "chart") else None,
            role_hint=_ROLE_HINTS.get(btype, btype or "unknown"),
        ))
    return out


def load_mineru_page(middle_json_path: str | Path, page_index: int) -> list[Component]:
    """Return components for one page (0-indexed) from a ``*_middle.json``."""
    data = json.loads(Path(middle_json_path).read_text())
    blocks = data["pdf_info"][page_index].get("preproc_blocks", [])
    return build_components_from_middle(blocks)


# ---------------------------------------------------------------------------
# Optional merge pass - combats label collision on dense pages
# ---------------------------------------------------------------------------


def merge_close_components(
    components: list[Component],
    *,
    y_gap_threshold: int = 30,
    x_overlap_min: float = 0.5,
) -> list[Component]:
    """Greedy merge of adjacent same-type textual components.

    Only merges blocks whose bboxes are vertically close
    (``gap <= y_gap_threshold``) and horizontally aligned (overlap
    ratio ≥ ``x_overlap_min`` of the narrower box). Same-type only.
    Re-numbers per-type IDs after merging.

    NOTE: merging a long list can fool the model
    into emitting one group entity for the whole list. If your page
    has dense enumerations, pass ``y_gap_threshold=0`` (only adjacent
    touching boxes merge) or filter ``MERGEABLE_TYPES`` upstream.
    """
    if not components:
        return components

    items = sorted(
        components,
        key=lambda c: (c.bbox[1], c.bbox[0]),
    )

    merged: list[Component] = []
    for c in items:
        if not merged:
            merged.append(c.model_copy(deep=True))
            continue
        last = merged[-1]
        if _should_merge(last, c, y_gap_threshold, x_overlap_min):
            merged[-1] = _merge_pair(last, c)
        else:
            merged.append(c.model_copy(deep=True))

    counters: dict[str, int] = {}
    for c in merged:
        prefix = TYPE_PREFIX.get(c.type, "X")
        counters[prefix] = counters.get(prefix, 0) + 1
        c.component_id = f"{prefix}{counters[prefix]}"
    return merged


def _should_merge(
    a: Component, b: Component, y_gap: int, x_overlap_min: float
) -> bool:
    if a.type != b.type:
        return False
    if a.type not in MERGEABLE_TYPES:
        return False
    ab, bb = a.bbox, b.bbox
    # b should come after a in reading order; check vertical gap.
    gap = bb[1] - ab[3]
    if gap > y_gap or gap < -2:
        return False
    overlap = max(0.0, min(ab[2], bb[2]) - max(ab[0], bb[0]))
    smaller_w = min(ab[2] - ab[0], bb[2] - bb[0])
    if smaller_w <= 0:
        return False
    return overlap / smaller_w >= x_overlap_min


def _merge_pair(a: Component, b: Component) -> Component:
    union = (
        min(a.bbox[0], b.bbox[0]),
        min(a.bbox[1], b.bbox[1]),
        max(a.bbox[2], b.bbox[2]),
        max(a.bbox[3], b.bbox[3]),
    )
    text_a = a.text_full or ""
    text_b = b.text_full or ""
    combined = (text_a + " " + text_b).strip() if (text_a or text_b) else None
    return a.model_copy(update={
        "bbox": union,
        "text_full": combined,
        "block_ids": [*a.block_ids, *b.block_ids],
        # Spatial annotations will be recomputed later.
        "position": "",
        "neighbours": {},
        "merged": True,
    })


# ---------------------------------------------------------------------------
# Internals - extract content from MinerU's nested span structure
# ---------------------------------------------------------------------------


def _block_text_full(block: dict) -> str | None:
    """Return the full OCR/text-layer string for a block (no truncation)."""
    parts: list[str] = []
    for line in block.get("lines", []) or []:
        for span in line.get("spans", []) or []:
            if span.get("type") == "text" and span.get("content"):
                parts.append(span["content"])
    for sub in block.get("blocks", []) or []:
        for line in sub.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                if span.get("type") == "text" and span.get("content"):
                    parts.append(span["content"])
    s = " ".join(parts).strip()
    return s or None


def _image_path_from_middle(block: dict) -> str | None:
    for sub in block.get("blocks", []) or []:
        for line in sub.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                if span.get("type") == "image" and span.get("image_path"):
                    return span["image_path"]
    return None


def _table_html_from_middle(block: dict) -> str | None:
    for sub in block.get("blocks", []) or []:
        for line in sub.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                if span.get("type") == "table" and span.get("html"):
                    return span["html"]
    return None


# ---------------------------------------------------------------------------
# Spatial annotations - position bucket + nearest-neighbour cids
# ---------------------------------------------------------------------------


def annotate_spatial(
    components: list[Component],
    *,
    page_w: float,
    page_h: float,
) -> list[Component]:
    """Attach `position` (upper-left / middle-center / ...) and
    `neighbours` (nearest cid per side) to each component.

    Returns a new list (does not mutate input). Used by ingest to give
    the VLM a coarse spatial hint without forcing it to reason about
    raw pixel coords.
    """
    if not components:
        return list(components)

    enriched: list[Component] = []
    bbox_centres: list[tuple[float, float]] = []
    for c in components:
        bb = c.bbox
        cx = (bb[0] + bb[2]) / 2
        cy = (bb[1] + bb[3]) / 2
        bbox_centres.append((cx, cy))
        col = "left" if cx < page_w / 3 else (
            "right" if cx > 2 * page_w / 3 else "center"
        )
        row = "upper" if cy < page_h / 3 else (
            "lower" if cy > 2 * page_h / 3 else "middle"
        )
        enriched.append(c.model_copy(update={"position": f"{row}-{col}"}))

    for i, c in enumerate(enriched):
        cx, cy = bbox_centres[i]
        cands: dict[str, tuple[float, str] | None] = {
            "above": None, "below": None, "left": None, "right": None,
        }
        for j, other in enumerate(enriched):
            if j == i:
                continue
            ox, oy = bbox_centres[j]
            dx, dy = ox - cx, oy - cy
            if abs(dy) >= abs(dx):
                key = "above" if dy < 0 else "below"
            else:
                key = "left" if dx < 0 else "right"
            dist = (dx * dx + dy * dy) ** 0.5
            cur = cands[key]
            if cur is None or dist < cur[0]:
                cands[key] = (dist, other.component_id)
        c.neighbours = {k: v[1] for k, v in cands.items() if v is not None}

    return enriched


__all__ = [
    "Component",
    "TYPE_PREFIX",
    "CHROME_TYPES",
    "MERGEABLE_TYPES",
    "build_components_from_middle",
    "merge_close_components",
    "load_mineru_page",
    "annotate_spatial",
]
