"""Deterministic mock encoder.

Hashes the (text, image_path) of each EvidenceCard / Query to produce a
stable, low-dimensional embedding. No GPU, no network, no flakiness - used by
unit tests and CI smoke runs.

Despite being a "mock", it produces *reasonable* relative similarity: cards
with overlapping word sets will have higher cosine similarity than disjoint
ones, because we mix a hashed-bag-of-words signal with a global hash. This
makes integration tests realistic enough to catch wiring bugs.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from kg4vd.config.schema import EncoderCfg
from kg4vd.core.registry import register
from kg4vd.core.types import EvidenceCard, Query

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@register("encoder", "mock")
class MockEncoder:
    name = "mock"

    def __init__(self, cfg: EncoderCfg):
        self.cfg = cfg
        self.dim = cfg.dim
        self.normalize = cfg.normalize

    async def encode_query(self, query: Query) -> np.ndarray:
        return self._embed(query.text, query.images)

    async def encode_card(self, card: EvidenceCard) -> np.ndarray:
        text = card.text_payload or ""
        text = f"[{card.evidence_type}] {text}"
        imgs = [card.image_payload] if card.image_payload else []
        return self._embed(text, imgs)

    async def encode_cards_batch(self, cards: list[EvidenceCard]) -> np.ndarray:
        if not cards:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.stack([self._embed_for_card(c) for c in cards], axis=0)
        return out

    # ----- internals --------------------------------------------------

    def _embed_for_card(self, card: EvidenceCard) -> np.ndarray:
        text = f"[{card.evidence_type}] {card.text_payload or ''}"
        imgs = [card.image_payload] if card.image_payload else []
        return self._embed(text, imgs)

    def _embed(self, text: str, images: list[str] | None) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)

        for tok in _TOKEN_RE.findall((text or "").lower()):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:16], 16)
            idx = h % self.dim
            sign = 1.0 if (h & 1) else -1.0
            v[idx] += sign

        # Image signal: deterministic but distinct from text.
        if images:
            for p in images:
                if not p:
                    continue
                h = int(hashlib.sha1(p.encode("utf-8")).hexdigest()[:16], 16)
                idx = (h + 7) % self.dim
                v[idx] += 0.5
            # one extra dim to keep image-bearing cards separable
            v[(self.dim - 1) % self.dim] += 0.25

        if self.normalize:
            n = float(np.linalg.norm(v)) or 1.0
            v = v / n
        return v
