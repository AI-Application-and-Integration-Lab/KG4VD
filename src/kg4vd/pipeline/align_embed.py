"""Precompute cross-page-alignment node embeddings with GME on the GPU.

The align stage needs two GPU consumers: GME (to embed entities and pick
candidate pairs) and the judge LLM. On a 32 GB card a large LLM server (e.g.
sglang TP=2) and GME-7B (~16 GB) cannot co-reside, but the two phases don't
overlap in time - align embeds *all* nodes upfront, then judges. So they can be
split:

  1. Stop the LLM server.                          (frees VRAM)
  2. ``kg4vd align-embed <recipe>`` - GME embeds every node, writes
     ``<work_dir>/kg/node_embs.npz``.
  3. Restart the LLM server.
  4. ``kg4vd build --stages align --resume`` - ``_align`` auto-detects the
     ``.npz``, skips loading GME, and judges via the LLM server.

Node cards are built with the exact helper the live aligner uses
(``build_entity_card_for_node``, same crop dir), so the embeddings are
bit-for-bit what the in-process path would have produced.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from kg4vd.cards.builders import build_entity_card_for_node
from kg4vd.config.schema import KG4VDConfig
from kg4vd.core.types import KGNode, Page
from kg4vd.encode import build_encoder

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def precompute_align_embeddings(cfg: KG4VDConfig) -> Path:
    """Embed every KG node with GME and write ``kg/node_embs.npz``.

    Returns the path to the written archive.
    """
    work_dir = Path(cfg.dataset.work_dir)
    kg_dir = work_dir / "kg"

    nodes = [KGNode.model_validate(d) for d in _read_jsonl(kg_dir / "nodes.jsonl")]
    pages = [Page.model_validate(d) for d in _read_jsonl(work_dir / "pages.jsonl")]
    pages_by_id = {p.page_id: p for p in pages}
    logger.info("align-embed: nodes=%d pages=%d", len(nodes), len(pages))

    crop_out_dir = None
    if cfg.evidence_cards.entity_card.use_visual_crop and pages:
        crop_out_dir = (
            Path(pages[0].page_image_path).parent.parent.parent
            / "cards" / "entity_crops"
        )

    cards = [
        build_entity_card_for_node(
            n, cfg=cfg.evidence_cards, pages_by_id=pages_by_id,
            crop_out_dir=crop_out_dir,
        )
        for n in nodes
    ]
    n_img = sum(1 for c in cards if c.image_payload)
    logger.info("align-embed: built %d node cards (%d with image crop)", len(cards), n_img)

    encoder = build_encoder(cfg.encoder)
    try:
        t = time.time()
        embs = await encoder.encode_cards_batch(cards)
        logger.info("align-embed: embedded %s in %.0fs", tuple(embs.shape), time.time() - t)
    finally:
        close = getattr(encoder, "close", None)
        if close:
            close()

    out = kg_dir / "node_embs.npz"
    np.savez(
        out,
        embs=embs.astype(np.float32, copy=False),
        node_ids=np.array([n.entity_id for n in nodes], dtype=object),
    )
    logger.info("align-embed: wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out
