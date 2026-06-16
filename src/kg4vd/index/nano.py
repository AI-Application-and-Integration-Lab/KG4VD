"""Unified evidence index backed by nano-vectordb.

We store one numpy matrix and a parallel list of EvidenceCard JSON blobs.
Cosine similarity is computed by L2-normalising at upsert time and using
dot-product at query time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kg4vd.config.schema import IndexCfg
from kg4vd.core.errors import IndexError
from kg4vd.core.registry import register
from kg4vd.core.types import EvidenceCard, Hit


@register("index", "nano")
class NanoUnifiedIndex:
    """In-process numpy index. Pure dependency-free for the cosine path."""

    name = "nano"

    def __init__(self, cfg: IndexCfg, *, dim: int):
        self.cfg = cfg
        self.dim = dim
        self.metric = cfg.metric
        if self.metric not in {"cosine", "dot"}:
            raise IndexError(
                f"NanoUnifiedIndex supports cosine|dot, got {self.metric!r}. "
                "(maxsim is a multi-vector store; use a separate backend.)"
            )
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._cards: list[EvidenceCard] = []
        self._id_to_idx: dict[str, int] = {}

    async def upsert(
        self,
        cards: list[EvidenceCard],
        embeddings: np.ndarray,
    ) -> None:
        if len(cards) == 0:
            return
        if embeddings.shape[0] != len(cards):
            raise IndexError(
                f"upsert: cards ({len(cards)}) and embeddings ({embeddings.shape[0]}) "
                "must align"
            )
        if embeddings.shape[1] != self.dim:
            raise IndexError(
                f"upsert: embedding dim {embeddings.shape[1]} != index dim {self.dim}"
            )

        emb = embeddings.astype(np.float32, copy=False)
        if self.metric == "cosine":
            emb = _l2_normalize(emb)

        # Stage updates (replace) vs appends (new).
        replace_idxs: list[tuple[int, int]] = []   # (existing_idx, new_idx_in_emb)
        append_cards: list[EvidenceCard] = []
        append_embeddings: list[np.ndarray] = []
        for j, c in enumerate(cards):
            if c.evidence_id in self._id_to_idx:
                replace_idxs.append((self._id_to_idx[c.evidence_id], j))
            else:
                append_cards.append(c)
                append_embeddings.append(emb[j])
        for ex_idx, new_j in replace_idxs:
            self._matrix[ex_idx] = emb[new_j]
            self._cards[ex_idx] = cards[new_j]
        if append_cards:
            block = np.stack(append_embeddings, axis=0)
            self._matrix = (
                np.concatenate([self._matrix, block], axis=0)
                if self._matrix.size
                else block
            )
            for c in append_cards:
                self._id_to_idx[c.evidence_id] = len(self._cards)
                self._cards.append(c)

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[Hit]:
        if self._matrix.shape[0] == 0:
            return []
        q = query_embedding.astype(np.float32, copy=False).reshape(-1)
        if q.shape[0] != self.dim:
            raise IndexError(f"query dim {q.shape[0]} != index dim {self.dim}")
        if self.metric == "cosine":
            q = q / (np.linalg.norm(q) or 1.0)

        scores = self._matrix @ q  # (N,)

        # Filter
        keep = _mask_for_filters(self._cards, filters) if filters else None
        if keep is not None:
            scores = np.where(keep, scores, -np.inf)

        k = min(top_k, scores.shape[0])
        if k <= 0:
            return []
        idxs = np.argpartition(-scores, k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]

        hits: list[Hit] = []
        for rank, i in enumerate(idxs):
            s = float(scores[i])
            if s == -np.inf:
                continue
            hits.append(
                Hit(card=self._cards[int(i)], score=s, rank=rank, source="retrieve")
            )
        return hits

    async def persist(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "matrix.npy", self._matrix)
        with (path / "cards.jsonl").open("w", encoding="utf-8") as f:
            for c in self._cards:
                f.write(json.dumps(c.model_dump(mode="json"), ensure_ascii=False) + "\n")
        with (path / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"dim": self.dim, "metric": self.metric, "size": len(self._cards)},
                f,
            )

    async def load(self, path: Path) -> None:
        path = Path(path)
        if not (path / "meta.json").is_file():
            raise IndexError(f"No saved index at {path}")
        with (path / "meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta["dim"] != self.dim:
            raise IndexError(
                f"Saved index dim {meta['dim']} != index dim {self.dim}"
            )
        self._matrix = np.load(path / "matrix.npy")
        self._cards = []
        self._id_to_idx = {}
        with (path / "cards.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                card = EvidenceCard.model_validate_json(line)
                self._id_to_idx[card.evidence_id] = len(self._cards)
                self._cards.append(card)

    async def size(self) -> int:
        return len(self._cards)

    def cards(self) -> list[EvidenceCard]:
        """Read-only view of the indexed cards (row-aligned with ``vectors()``)."""
        return self._cards

    def vectors(self) -> np.ndarray:
        """Read-only view of the embedding matrix (rows aligned with ``cards()``)."""
        return self._matrix

    def get_embeddings(self, evidence_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Look up embeddings for a batch of evidence_ids.

        Returns ``(matrix, found_mask)``:
          - ``matrix``: shape ``(len(evidence_ids), dim)`` - rows for missing
            ids are zero vectors (which produce cosine_sim = 0, i.e. no
            topic bias contribution).
          - ``found_mask``: shape ``(len(evidence_ids),)`` boolean - True for
            ids that were resolved against the index.

        Used by PPR topic-sensitive personalization.
        Synchronous on purpose: this is pure RAM lookup, the async
        ``search`` API would only add ceremony.
        """
        n = len(evidence_ids)
        out = np.zeros((n, self.dim), dtype=np.float32)
        mask = np.zeros(n, dtype=bool)
        for i, eid in enumerate(evidence_ids):
            idx = self._id_to_idx.get(eid)
            if idx is not None:
                out[i] = self._matrix[idx]
                mask[i] = True
        return out, mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return m / n


def _mask_for_filters(
    cards: list[EvidenceCard], filters: dict[str, Any]
) -> np.ndarray:
    """Build a boolean mask for the cards. Supported filter keys:

    - ``evidence_type``: str | list[str]
    - ``doc_id``: str | list[str]
    - ``page_id_in``: list[int]  (any page_id intersects this list)
    """

    n = len(cards)
    mask = np.ones(n, dtype=bool)

    et = filters.get("evidence_type")
    if et is not None:
        et_set = {et} if isinstance(et, str) else set(et)
        for i, c in enumerate(cards):
            if c.evidence_type not in et_set:
                mask[i] = False

    did = filters.get("doc_id")
    if did is not None:
        did_set = {did} if isinstance(did, str) else set(did)
        for i, c in enumerate(cards):
            if c.doc_id not in did_set:
                mask[i] = False

    pids = filters.get("page_id_in")
    if pids is not None:
        wanted = set(pids)
        for i, c in enumerate(cards):
            if not (set(c.page_ids) & wanted):
                mask[i] = False

    return mask
