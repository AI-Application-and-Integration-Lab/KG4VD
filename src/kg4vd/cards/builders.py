"""Build EvidenceCards from Pages and KG state.

Each builder returns a list[EvidenceCard]. Builders are pure (no I/O).
Per-type emit toggles live in the config (`evidence_cards.emit.*``).

Payload-shape contract:
  Every text_payload starts with a one-line ``[<TYPE>]`` tag, then a
  short header that anchors the card to its (doc_id, page) provenance.
  The retriever's encoder embeds text_payload, so anything we want to
  affect retrieval must live there -- pure metadata fields are NOT
  embedded.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from kg4vd.config.schema import EvidenceCardsCfg, _EntityCardCfg
from kg4vd.core.types import EvidenceCard, KGEdge, KGNode, Page

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page / text-chunk / page-summary
# ---------------------------------------------------------------------------


def build_page_cards(pages: list[Page], cfg: EvidenceCardsCfg) -> list[EvidenceCard]:
    if not cfg.emit.page:
        return []
    # image_only: encode just the rendered page image (no text), as long as
    # the page actually has an image. Pages with no render fall back to text
    # so they don't end up with an empty embedding.
    image_only = cfg.page_card.image_only
    out = []
    for p in pages:
        # Page card embeds the raw page text - no LLM-generated `page_summary`
        # (kept distinct so the two card types don't collapse onto the same
        # embedding cluster in retrieval).
        text = (
            f"[PAGE]\nDocument: {p.doc_id}\nPage number: {p.page_id}\n"
            + (f"Snippet:\n{(p.text or '')[:2000]}" if p.text else "")
        )
        if image_only and p.page_image_path:
            text = ""
        out.append(
            EvidenceCard(
                evidence_id=f"page:{p.doc_id}:{p.page_id}",
                evidence_type="page",
                doc_id=p.doc_id,
                page_ids=[p.page_id],
                text_payload=text,
                image_payload=p.page_image_path,
                metadata={
                    "has_image": bool(p.page_image_path),
                    "n_figures": len(p.figure_image_paths),
                },
            )
        )
    return out


# ---------------------------------------------------------------------------
# Entity / Relation
# ---------------------------------------------------------------------------


def build_entity_card_for_node(
    n: KGNode,
    *,
    cfg: EvidenceCardsCfg | None = None,
    pages_by_id: "dict[int, Page] | None" = None,
    crop_out_dir: "Path | None" = None,
) -> EvidenceCard:
    """Build the entity card for a single node - single source of truth.

    Used both by `build_entity_cards` (cards stage, fed to the index)
    and by `CrossPageAligner` (align stage, in-memory similarity
    only). Keeping both call sites on the same payload format keeps
    the align-time embedding space aligned with the retrieval-time
    one, which matters for visual entities (so they bring their
    `visual_type` / `visual_description`) and for description
    compaction (so `<SEP>` duplicates don't pollute the vector).

    When `cfg.entity_card.use_visual_crop` is True, visual-modality
    entities get an `image_payload` set to a JPEG crop of their
    largest bbox region across all source pages - `pages_by_id` lets
    the helper resolve `bbox.page_id` back to the right page render.
    The encoder then produces a fused text+image embedding instead of
    text-only.
    """
    doc_id = _doc_id_from_pages(n.source_pages, n.metadata)
    # Compact the description: extract+merge stages emit
    # "<sentence>. <SEP> <near-duplicate sentence>. <SEP> ..." which
    # bloats the embedding with redundant tokens. Sentence-level
    # dedup via stem-set subset matching keeps semantic variants but
    # drops exact / near-duplicate phrasings.
    compact_desc = _compact_description(n.description or "")
    body_lines = [
        f"name: {n.name}",
        f"entity_type: {n.entity_type}",
        f"modality: {n.modality}",
    ]
    if n.modality == "visual":
        if n.visual_type:
            body_lines.append(f"visual_type: {n.visual_type}")
        if n.visual_description:
            body_lines.append(f"visual_description: {n.visual_description}")
    body_lines.append(f"description: {compact_desc}")
    body_lines.append(f"mentioned_on_pages: {n.source_pages}")
    # Anchor every entity card to its document so retrieval has
    # multi-doc disambiguation signal in the embedded text.
    text = (
        f"[ENTITY]\nDocument: {doc_id}\n"
        + "\n".join(body_lines)
    )
    metadata: dict[str, Any] = {
        "entity_type": n.entity_type,
        "modality": n.modality,
    }
    # Component-cued grounding: surface citation IDs + boxes so a UI can
    # draw "this entity comes from these regions" on the rendered page.
    # Nodes without grounding have empty arrays - skip the keys entirely
    # so those cards keep slim metadata.
    if n.source_components:
        metadata["source_components"] = list(n.source_components)
    if n.bboxes:
        metadata["bboxes"] = [b.model_dump() for b in n.bboxes]
    image_payload: str | None = None
    if cfg is not None and cfg.entity_card.use_visual_crop and n.modality == "visual":
        image_payload = _resolve_visual_crop(
            n, cfg.entity_card, pages_by_id or {}, crop_out_dir,
        )
    return EvidenceCard(
        evidence_id=f"entity:{n.entity_id}",
        evidence_type="entity",
        doc_id=doc_id,
        page_ids=list(n.source_pages),
        text_payload=text,
        image_payload=image_payload,
        metadata=metadata,
        graph_refs={"entity_id": n.entity_id},
    )


def build_entity_cards(
    nodes: list[KGNode],
    cfg: EvidenceCardsCfg,
    *,
    pages_by_id: "dict[int, Page] | None" = None,
    crop_out_dir: "Path | None" = None,
) -> list[EvidenceCard]:
    if not cfg.emit.entity:
        return []
    pages_by_id = pages_by_id or {}
    out: list[EvidenceCard] = []
    for n in nodes:
        out.append(build_entity_card_for_node(
            n, cfg=cfg, pages_by_id=pages_by_id, crop_out_dir=crop_out_dir,
        ))
    return out


def build_relation_cards(
    edges: list[KGEdge],
    nodes_by_id: dict[str, KGNode],
    cfg: EvidenceCardsCfg,
) -> list[EvidenceCard]:
    if not cfg.emit.relation:
        return []
    out = []
    for e in edges:
        # Skip `same_as` edges from the relation evidence - they are
        # purely a graph-merge signal, not a fact. Cross-page-align's
        # semantic edges (LLM-named, edge_type="semantic") DO produce
        # relation cards: they're exactly the kind of cross-page fact
        # this system is meant to surface.
        if e.edge_type == "same_as":
            continue
        head = nodes_by_id.get(e.src_id)
        tail = nodes_by_id.get(e.tgt_id)
        head_name = head.name if head else e.src_id
        tail_name = tail.name if tail else e.tgt_id
        # doc_id resolution: edges don't carry metadata.doc_id during
        # build, so prefer the head node's, then the tail's, then fall
        # back. Without this, every relation card lands at
        # doc_id="UNKNOWN", breaking any per-doc filtering.
        doc_id = "UNKNOWN"
        for cand in (e.metadata, head.metadata if head else {}, tail.metadata if tail else {}):
            if cand and cand.get("doc_id"):
                doc_id = cand["doc_id"]
                break
        # Compact the relation evidence text the same way as entity
        # descriptions - multi-page LLM emissions of the same fact
        # otherwise stack up via <SEP>.
        evidence_compact = _compact_description(e.description or "")
        body_lines = [
            f"head: {head_name}",
            f"relation: {e.relation}",
            f"tail: {tail_name}",
            f"evidence: {evidence_compact}",
        ]
        if e.visual_evidence_hint:
            body_lines.append(f"visual_evidence_hint: {e.visual_evidence_hint}")
        body_lines.append(f"source_pages: {e.source_pages}")
        # Document anchor in the embedded text, same reason as the
        # entity card: keeps multi-doc retrieval from collapsing.
        text = (
            f"[RELATION]\nDocument: {doc_id}\n"
            + "\n".join(body_lines)
        )

        metadata: dict[str, Any] = {
            "relation_type": e.edge_type,
            "confidence": e.confidence,
        }
        if e.source_components:
            metadata["source_components"] = list(e.source_components)
        if e.bboxes:
            metadata["bboxes"] = [b.model_dump() for b in e.bboxes]
        out.append(
            EvidenceCard(
                evidence_id=f"relation:{e.edge_id}",
                evidence_type="relation",
                doc_id=doc_id,
                page_ids=list(e.source_pages),
                text_payload=text,
                metadata=metadata,
                graph_refs={"edge_id": e.edge_id, "src_id": e.src_id, "tgt_id": e.tgt_id},
            )
        )
    return out




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _doc_id_from_pages(_pages, metadata: dict) -> str:
    return metadata.get("doc_id") or "UNKNOWN"


# ---------------------------------------------------------------------------
# Description compaction (entity / relation card payload cleanup)
# ---------------------------------------------------------------------------


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\s*<SEP>\s*")


def _compact_description(desc: str, *, max_chars: int = 1200) -> str:
    """Drop near-duplicate sentences from an LLM-merged description.

    Extract+merge concatenates per-round and per-page descriptions with
    ``<SEP>`` (or just appends across reflector rounds), so the same
    fact ends up phrased 3-5 ways in the same string -- e.g.::

        "Fei-Fei Li is a co-instructor of CS231n.
         <SEP> Fei-Fei Li is a co-instructor for the CS231n course.
         <SEP> Fei-Fei Li is a co-instructor of the CS231n course on Deep Learning."

    All three sentences embed almost identical token sets, so they
    bloat the cards without adding retrieval signal. Strategy: split
    on ``<SEP>`` and sentence boundaries, normalise each fragment to a
    stemmed token set, and keep a fragment only if its tokens are NOT
    a subset of an already-kept fragment. This preserves the most
    informative phrasings and drops the redundant variants.
    """
    if not desc:
        return ""
    fragments = [f.strip() for f in _SENT_SPLIT_RE.split(desc) if f.strip()]
    if len(fragments) <= 1:
        return desc.strip()
    # Stem helper -- mirrors pipeline/build._normalise_relation but
    # works at the sentence (not relation-phrase) granularity. Cheap
    # suffix list, falls back to snowballstemmer when available.
    try:
        import snowballstemmer  # type: ignore[import-not-found]
        _porter = snowballstemmer.stemmer("english")
        def _stem(t: str) -> str:
            return _porter.stemWord(t)
    except ImportError:
        _suffixes = (
            "ization", "ational", "fulness", "ousness", "iveness",
            "ication", "ation", "ement", "ities", "iness",
            "ence", "ance", "ment", "ness", "tion", "sion",
            "ing", "est", "ity", "or", "er", "ed", "es", "ly", "al", "ic", "s",
        )
        def _stem(t: str) -> str:
            for suf in _suffixes:
                if len(t) > len(suf) + 2 and t.endswith(suf):
                    return t[: -len(suf)]
            return t
    _stop = frozenset({
        "the", "a", "an", "of", "to", "in", "on", "for", "by", "at", "with",
        "as", "is", "are", "was", "were", "be", "been", "being", "and", "or",
        "that", "which", "this", "these", "those", "it", "its",
    })
    def _toks(s: str) -> frozenset[str]:
        out: set[str] = set()
        for tok in re.findall(r"[a-z0-9]+", s.lower()):
            if tok in _stop or len(tok) < 3:
                continue
            out.add(_stem(tok))
        return frozenset(out)
    # Rank fragments by token-set size (most informative first), keep
    # those not subsumed by an already-kept fragment.
    scored = sorted(
        ((_toks(f), f) for f in fragments),
        key=lambda x: (-len(x[0]), -len(x[1])),
    )
    kept: list[tuple[frozenset[str], str]] = []
    for toks, frag in scored:
        if not toks:
            continue
        if any(toks.issubset(k_toks) for k_toks, _ in kept):
            continue
        kept.append((toks, frag))
    # Restore original order from the input (stable sort by first-seen
    # position) -- gives a more readable output than the rank order.
    order = {frag: i for i, frag in enumerate(fragments)}
    kept.sort(key=lambda kf: order.get(kf[1], 1 << 30))
    out = " ".join(f for _, f in kept)
    if len(out) > max_chars:
        out = out[: max_chars - 3].rstrip() + "..."
    return out


# ---------------------------------------------------------------------------
# Visual-modality crop for fused entity-card encoding (Phase 1)
# ---------------------------------------------------------------------------


def _resolve_visual_crop(
    n: KGNode,
    ec_cfg: _EntityCardCfg,
    pages_by_id: dict[int, Page],
    out_dir: Path | None,
) -> str | None:
    """Return a path to a JPEG crop of `n`'s largest bbox region across
    all source pages, or None if any precondition isn't met.

    For multi-page entities (post-`canonicalize_same_as`), we score every
    `(page_id, bbox)` pair and pick the absolute largest by area -
    that's the most pixel-rich "iconic" appearance of the entity. The
    chosen bbox's `page_id` then routes us to the correct page render
    + scaling metadata, so a bbox from page 5 is cropped from page 5's
    image (not page 1's).

    Pre-flight checks (any failure → None, falls back to text-only):
      - `n.bboxes` non-empty (nodes without grounding skipped)
      - chosen bbox's `page_id` resolves to a page with render-px
        metadata + a page_image_path on disk
      - largest bbox area after scaling is at least `crop_min_area_px`

    The crop is cached by `entity_id` so repeated calls are free.
    """
    # Hard gate: only visual-modality entities get a crop. Text-modality
    # nodes' bbox is just a block of text and the encoder would re-OCR
    # what's already in `text_payload`. Without this guard, callers
    # that don't pre-filter (e.g. the cross-page aligner) would burn
    # disk + encoder cycles on useless crops.
    if n.modality != "visual":
        return None
    if not n.bboxes:
        return None
    try:
        from PIL import Image
    except ImportError:                                       # pragma: no cover
        return None

    # Pick the bbox with the largest area in MinerU px across *all*
    # pages this entity appears on.
    largest = max(
        n.bboxes,
        key=lambda b: max(0.0, b.bbox[2] - b.bbox[0]) * max(0.0, b.bbox[3] - b.bbox[1]),
    )
    page = pages_by_id.get(largest.page_id)
    if page is None or not page.page_image_path:
        return None
    md = page.metadata or {}
    mineru_size = md.get("page_size_mineru_px")
    render_size = md.get("page_size_render_px")
    if not (mineru_size and render_size):
        return None  # non-component-cued page
    mw, mh = float(mineru_size[0]), float(mineru_size[1])
    rw, rh = float(render_size[0]), float(render_size[1])
    if mw <= 0 or mh <= 0 or rw <= 0 or rh <= 0:
        return None
    sx, sy = rw / mw, rh / mh
    x0, y0, x1, y1 = largest.bbox
    rx0, ry0, rx1, ry1 = x0 * sx, y0 * sy, x1 * sx, y1 * sy

    # Pad each side by crop_padding_frac of the box size; clamp to image.
    pad_x = (rx1 - rx0) * ec_cfg.crop_padding_frac
    pad_y = (ry1 - ry0) * ec_cfg.crop_padding_frac
    cx0 = max(0.0, rx0 - pad_x)
    cy0 = max(0.0, ry0 - pad_y)
    cx1 = min(rw, rx1 + pad_x)
    cy1 = min(rh, ry1 + pad_y)
    crop_w, crop_h = cx1 - cx0, cy1 - cy0
    if crop_w * crop_h < ec_cfg.crop_min_area_px:
        return None

    # Cache by entity_id; same input → same path so re-runs reuse.
    cache_dir = out_dir or (Path(page.page_image_path).parent.parent.parent / "cards" / "entity_crops")
    safe_id = n.entity_id.replace("/", "_").replace(":", "_")
    out_path = cache_dir / f"{safe_id}.jpg"
    if out_path.is_file():
        return str(out_path)

    try:
        with Image.open(page.page_image_path) as im:
            im = im.convert("RGB")
            crop = im.crop((int(cx0), int(cy0), int(cx1), int(cy1)))
            # Downscale to bound encoder cost.
            m = max(crop.size)
            if m > ec_cfg.crop_max_long_side:
                s = ec_cfg.crop_max_long_side / m
                crop = crop.resize((max(1, int(crop.size[0] * s)),
                                     max(1, int(crop.size[1] * s))))
            cache_dir.mkdir(parents=True, exist_ok=True)
            crop.save(out_path, format="JPEG", quality=85)
    except (OSError, ValueError) as e:                        # pragma: no cover
        logger.warning("entity crop failed for %s: %s", n.entity_id, e)
        return None
    return str(out_path)
