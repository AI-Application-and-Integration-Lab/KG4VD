"""Compact component manifest builder for the component-cued extractor.

The manifest is the textual half of what the Extractor / Reflector see
alongside the annotated page image. Design contract:

  - For TEXTUAL components (title / paragraph / list / index): include
    the FULL OCR/text-layer content. Truncating to a preview led to
    under-extraction on dense paragraphs (e.g. missing affiliation
    relations from contributor lists).
  - For VISUAL components (image / chart / formula / equation): do
    NOT inline a textual proxy. Leave a "see annotated image" pointer
    so the VLM is forced to interpret the actual visual content.
  - For TABLE components: surface MinerU's parsed HTML as
    ``table_html`` so the Reflector's enumeration-density check has
    something to count. The annotated image is still the primary cue.
  - Saturated components have their inner content suppressed and
    replaced by a ``[saturated - N nodes ...]`` marker; the
    component_id / type / position / neighbours are kept so the model
    still has spatial context for cross-component edges.
"""

from __future__ import annotations

import re
from typing import Iterable

from kg4vd.ingest.components import Component


TEXTUAL_TYPES: set[str] = {"title", "paragraph", "text", "list", "index"}
VISUAL_TYPES: set[str] = {"image", "chart", "table", "formula", "equation"}


def clean_table_html(html: str) -> str:
    """Strip MinerU's embedded ``<img src="...checksum.jpg"/>`` cell
    artefacts and collapse whitespace so the rendered table is
    enumerable. The ``<table>/<tr>/<td>`` structure is preserved.
    """
    if not html:
        return ""
    cleaned = re.sub(r"<img[^>]*/?>", "", html)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_manifest_text(
    components: Iterable[Component],
    *,
    saturated_cids: set[str] | None = None,
    nodes_per_cid: dict[str, int] | None = None,
) -> str:
    """Render a compact YAML-ish manifest the VLM can parse.

    `saturated_cids` replaces a component's inner content with a
    "see scorecard" marker; spatial context (type / position /
    neighbours) is kept so the model can still build cross-component
    edges using the surviving entities in the scorecard.
    """
    saturated = saturated_cids or set()
    counts = nodes_per_cid or {}
    lines: list[str] = []
    for c in components:
        cid = c.component_id
        pos = c.position or "?"
        lines.append(f"- {cid} ({c.type}, {pos})")
        if c.role_hint:
            lines.append(f"    role: {c.role_hint}")
        if c.neighbours:
            nbr_str = ", ".join(f"{v} {k}" for k, v in c.neighbours.items())
            lines.append(f"    neighbours: {nbr_str}")
        if cid in saturated:
            n = counts.get(cid, 0)
            lines.append(
                f"    [saturated - {n} nodes already grounded here; "
                f"see scorecard. Do NOT re-extract; use these entities "
                f"as edge endpoints only.]"
            )
            continue
        if c.type in TEXTUAL_TYPES and c.text_full:
            text = c.text_full.replace("\n", " ").strip()
            lines.append(f"    text: {text}")
        elif c.type in VISUAL_TYPES:
            lines.append(f"    content: <{c.type} - see annotated page>")
            if c.caption:
                lines.append(f"    caption: {c.caption}")
            if c.type == "table" and c.html:
                lines.append(f"    table_html: {clean_table_html(c.html)}")
        else:
            # Unknown / fallback - emit whatever text we have.
            if c.text_full:
                lines.append(f"    text: {c.text_full}")
    return "\n".join(lines)


__all__ = [
    "TEXTUAL_TYPES",
    "VISUAL_TYPES",
    "clean_table_html",
    "build_manifest_text",
]
