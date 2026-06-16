"""Cross-page Entity Alignment (build-time).

Per-page extraction produces nodes and relations in isolation; facts that
span pages get lost. This stage recovers them by:

  1. Finding cross-page candidate pairs by entity embedding similarity.
  2. Asking an LLM judge to either declare them the SAME entity
     (``same_as``) or to name the semantic relation between them in the
     SAME free-form vocabulary the per-page extractor uses
     (``edge_type="semantic"`` with an LLM-chosen ``relation`` string).

Only ``same_as`` is a reserved token - it triggers the canonical merge
in the build pipeline. Everything else lands in the regular KG as a
normal semantic edge and goes through the same dedup / card / embed
pipeline as extractor-produced relations.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import numpy as np

from kg4vd.cards.builders import build_entity_card_for_node, _resolve_visual_crop
from kg4vd.config.schema import CrossPageAlignCfg
from kg4vd.core.types import EvidenceCard, KGEdge, KGNode
from kg4vd.kg.align.prompts import (
    CROSS_PAGE_ALIGN_JUDGE,
    CROSS_PAGE_ALIGN_JUDGE_VISUAL,
)
from kg4vd.kg.prompts import PROMPTS_VERSION
from kg4vd.kg.relation import normalize_relation
from kg4vd.utils.json_repair import parse_json_loose

logger = logging.getLogger(__name__)


class CrossPageAligner:
    """Build-time cross-page entity aligner."""

    def __init__(
        self,
        *,
        encoder: Any,
        llm: Any,
        cfg: CrossPageAlignCfg,
        max_async: int = 12,
        evidence_cards_cfg: Any = None,
        pages_by_id: dict[int, Any] | None = None,
        crop_out_dir: Any = None,
    ):
        self.encoder = encoder
        self.llm = llm
        self.cfg = cfg
        self.max_async = max_async
        # When set, align-time entity cards are built with the same
        # (image_payload) treatment as the index-time cards, so the
        # cosine similarity sees the same vector space at align time
        # as retrieval time. Defaults are None → text-only embedding
        # (used when no crop config is supplied).
        self._ec_cfg = evidence_cards_cfg
        self._pages_by_id = pages_by_id or {}
        self._crop_out_dir = crop_out_dir

    async def align(
        self,
        nodes: list[KGNode],
        page_summaries: dict[tuple[str, int], str] | None = None,
        extract_edges: list[KGEdge] | None = None,
        precomputed_embeddings: np.ndarray | None = None,
    ) -> list[KGEdge]:
        if not self.cfg.enabled or len(nodes) < 2:
            return []
        self._page_summaries = page_summaries or {}

        # 1) Embed each node by turning it into a tiny EvidenceCard. This
        #    reuses the unified encoder, so the alignment query is in the
        #    same vector space as everything else.
        # When `precomputed_embeddings` is supplied (node-aligned), use it and
        # skip the encoder entirely - this lets a GPU-constrained box embed the
        # nodes in a separate GME-on-GPU pass, then judge here with the encoder
        # absent so its VRAM is free for a co-resident LLM server. The vectors
        # are the same GME space either way.
        if precomputed_embeddings is not None:
            if precomputed_embeddings.shape[0] != len(nodes):
                raise ValueError(
                    f"precomputed_embeddings rows ({precomputed_embeddings.shape[0]}) "
                    f"!= n_nodes ({len(nodes)})"
                )
            embs = precomputed_embeddings.astype(np.float32, copy=False)
        elif hasattr(self.encoder, "encode_cards_batch"):
            # Most real encoders (GME, ColQwen, Qwen3-VL-Embed) only implement
            # `encode_card`; only the mock encoder ships an explicit
            # `encode_cards_batch`. Fall back to a bounded-concurrency per-card
            # path when the batch method isn't available - same pattern as
            # `pipeline/build.py:_embed_and_index`.
            node_cards = [self._node_to_card(n) for n in nodes]
            embs = await self.encoder.encode_cards_batch(node_cards)
        else:
            node_cards = [self._node_to_card(n) for n in nodes]
            sem = asyncio.Semaphore(self.max_async)

            async def _enc(c: EvidenceCard) -> np.ndarray:
                async with sem:
                    return await self.encoder.encode_card(c)

            vecs = await asyncio.gather(*[_enc(c) for c in node_cards])
            embs = np.stack(vecs, axis=0) if vecs else np.zeros((0, 0))
        if embs.size == 0:
            return []
        embs = _l2(embs)

        # 2) For each node, find candidate matches by cosine similarity in the
        #    same doc OR cross-doc (depending on config).
        sim = embs @ embs.T  # (N, N)
        np.fill_diagonal(sim, -np.inf)

        # 3) Build candidate pool per node (top-K, excluding same-page nodes).
        K = self.cfg.candidate_top_k
        cand_per_node: list[list[int]] = []
        for i, node_i in enumerate(nodes):
            scores = sim[i]
            # Mask out same-page entities (alignment is *cross-page* by definition)
            for j, node_j in enumerate(nodes):
                if set(node_i.source_pages) & set(node_j.source_pages):
                    scores[j] = -np.inf
            order = np.argsort(-scores)
            top = [int(j) for j in order[:K] if scores[j] > -np.inf]
            cand_per_node.append(top)

        # 4) Judge with LLM in parallel (bounded).
        sem = asyncio.Semaphore(self.max_async)
        tasks = [
            self._judge_one(i, nodes, cand_per_node[i], sem)
            for i in range(len(nodes))
            if cand_per_node[i]
        ]
        results: list[list[KGEdge]] = await asyncio.gather(*tasks)
        edges: list[KGEdge] = []
        for group in results:
            edges.extend(group)

        # Post-judge filter: even with HARD RULE 2 in the prompt, the LLM
        # still tends to bind every entity to one of a few "document
        # root" hubs with generic relations like is_part_of /
        # supports_claim. Those edges add no retrieval signal and crowd
        # out the genuinely informative ones. Drop generic-relation
        # edges whose target is a hub (high align-in-degree). The
        # prompt's HARD RULE 2 catches the obvious cases; this catches
        # the rest with mechanical thresholding.
        edges = _drop_generic_edges_to_hubs(edges)

        # Tiered keep: strong (rs >= rs_threshold) always kept; weak
        # (rs >= rs_threshold_rescue but below strong) kept only if the
        # edge bridges components OR crosses modality. See
        # `_apply_rescue_filter`.
        edges = _apply_rescue_filter(
            edges, nodes, extract_edges or [], self.cfg,
        )

        # The LLM names the relation directly; downstream dedup handles
        # near-duplicate phrasing.
        return _dedupe_edges(edges)

    def _resolve_judge_crops(
        self, src: KGNode, cands: list[KGNode],
    ) -> tuple[str | None, list[str | None]]:
        """Resolve crop paths for the judge call. Returns (src_crop,
        per_candidate_crops). Each entry is None when:
          - flag off, or
          - cfg/pages not threaded into the aligner, or
          - the node fails the precondition checks in `_resolve_visual_crop`
            (text modality, no bboxes, missing page metadata, too small).
        """
        if self._ec_cfg is None or not getattr(
            self._ec_cfg.entity_card, "use_visual_crop", False
        ):
            return None, [None] * len(cands)

        def _resolve(n: KGNode) -> str | None:
            return _resolve_visual_crop(
                n, self._ec_cfg.entity_card, self._pages_by_id, self._crop_out_dir,
            )

        return _resolve(src), [_resolve(c) for c in cands]

    def _node_to_card(self, n: KGNode) -> EvidenceCard:
        """Build the entity card for align-time similarity. When
        `evidence_cards_cfg` + `pages_by_id` were supplied, the card
        gets the same image_payload treatment as the index-time card
        - keeping align-time and retrieval-time embeddings in the same
        vector space.
        """
        return build_entity_card_for_node(
            n,
            cfg=self._ec_cfg,
            pages_by_id=self._pages_by_id,
            crop_out_dir=self._crop_out_dir,
        )

    async def _judge_one(
        self,
        i: int,
        nodes: list[KGNode],
        cand_idxs: list[int],
        sem: asyncio.Semaphore,
    ) -> list[KGEdge]:
        if not cand_idxs:
            return []
        src = nodes[i]
        cands = [nodes[j] for j in cand_idxs]

        # Resolve crops for src + each candidate. Only attempted when the
        # `entity_card.use_visual_crop` flag is on and we have the page
        # map; otherwise stays empty (None entries) and the text-only
        # judge path runs.
        src_crop, cand_crops = self._resolve_judge_crops(src, cands)
        use_vision = (
            src_crop is not None or any(c is not None for c in cand_crops)
        )

        block = "\n\n".join(
            f"- name: {c.name!r}\n"
            f"  entity_type: {c.entity_type}\n"
            f"  modality: {c.modality}{_visual_lines(c, indent='  ')}\n"
            f"  description: {c.description}\n"
            f"  pages: {c.source_pages}\n"
            + ("  has_image: yes\n" if cc else "  has_image: no\n")
            + f"  page_context:\n{self._page_context_for(c)}"
            for c, cc in zip(cands, cand_crops)
        )

        if use_vision:
            # Attach images in order: src (if any), then candidates with
            # crops in their listed order. Skip None entries - the prompt
            # already says which candidates have images via `has_image`.
            images: list[str] = []
            image_lines: list[str] = []
            if src_crop is not None:
                images.append(src_crop)
                image_lines.append(f"  {len(images)}. SOURCE: {src.name!r}")
            for c, cc in zip(cands, cand_crops):
                if cc is not None:
                    images.append(cc)
                    image_lines.append(f"  {len(images)}. CANDIDATE: {c.name!r}")
            prompt = CROSS_PAGE_ALIGN_JUDGE_VISUAL.format(
                src_name=src.name,
                src_entity_type=src.entity_type,
                src_modality=src.modality,
                src_visual_block=_visual_lines(src, indent="  "),
                src_description=src.description,
                src_pages=src.source_pages,
                src_page_context=self._page_context_for(src),
                candidates_block=block,
                image_index_block="\n".join(image_lines) or "  (none)",
                rs_threshold_floor=int(self.cfg.rs_threshold_rescue),
                rs_threshold_same_as=int(self.cfg.rs_threshold_same_as),
            )
        else:
            prompt = CROSS_PAGE_ALIGN_JUDGE.format(
                src_name=src.name,
                src_entity_type=src.entity_type,
                src_modality=src.modality,
                src_visual_block=_visual_lines(src, indent="  "),
                src_description=src.description,
                src_pages=src.source_pages,
                src_page_context=self._page_context_for(src),
                candidates_block=block,
                rs_threshold_floor=int(self.cfg.rs_threshold_rescue),
                rs_threshold_same_as=int(self.cfg.rs_threshold_same_as),
            )
            images = None  # type: ignore[assignment]

        try:
            async with sem:
                resp = await self.llm.acomplete(
                    prompt,
                    system=f"prompt-set: {PROMPTS_VERSION}",
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    images=images,
                )
        except Exception as e:  # noqa: BLE001
            # One judge call failing (e.g. an over-length prompt that exceeds
            # the model's context, or a transient error that outlived the
            # client's retries) must not abort the whole align - it has no
            # mid-stage checkpoint and runs for hours. Skip this node's
            # candidates instead.
            logger.warning(
                "align judge failed for %r (%d candidates): %r - skipping",
                src.name, len(cands), e,
            )
            return []
        try:
            obj = parse_json_loose(resp.text)
        except ValueError:
            return []

        # Object may be a list directly, or wrapped (`{"decisions": [...]}`).
        if isinstance(obj, dict):
            obj = obj.get("decisions") or obj.get("results") or list(obj.values())[0] if obj else []

        if not isinstance(obj, list):
            return []

        rs_threshold_same_as = int(self.cfg.rs_threshold_same_as)
        rs_threshold_floor = int(self.cfg.rs_threshold_rescue)

        out: list[KGEdge] = []
        cand_by_name = {c.name: c for c in cands}
        for item in obj:
            if not isinstance(item, dict):
                continue
            cand_name = (item.get("candidate") or "").strip()
            cand = cand_by_name.get(cand_name)
            if cand is None:
                continue
            decision = (item.get("decision") or "").lower().strip()
            if decision == "no":
                continue
            try:
                rs = int(item.get("rs", 0))
            except (TypeError, ValueError):
                rs = 0

            if decision == "same_as":
                # Hard rule: same_as requires same entity_type. The prompt
                # already enforces this; drop violators rather than auto-
                # demoting to a guessed relation we don't have evidence for.
                if src.entity_type != cand.entity_type:
                    continue
                if rs < rs_threshold_same_as:
                    continue
                edge_type = "same_as"
                relation = "same_as"
            elif decision == "related":
                if rs < rs_threshold_floor:
                    continue
                relation = normalize_relation(item.get("relation"))
                if not relation:
                    continue
                edge_type = "semantic"
            else:
                continue

            out.append(
                KGEdge(
                    edge_id=_align_edge_id(src.entity_id, cand.entity_id, relation),
                    src_id=src.entity_id,
                    tgt_id=cand.entity_id,
                    relation=relation,
                    edge_type=edge_type,  # type: ignore[arg-type]
                    description=item.get("rationale") or "",
                    confidence=rs / 10.0,
                    source_pages=sorted(set(src.source_pages) | set(cand.source_pages)),
                    metadata={"origin": "cross_page_align"},
                )
            )
        return out

    def _page_context_for(self, node: KGNode) -> str:
        """Return indented page-summary lines for the pages this node spans."""
        summaries = getattr(self, "_page_summaries", {}) or {}
        if not summaries:
            return "    (no page summaries available)"
        doc_id = (node.metadata or {}).get("doc_id")
        lines: list[str] = []
        for p in node.source_pages:
            txt = None
            if doc_id is not None:
                txt = summaries.get((doc_id, p))
            if txt is None:
                # Fallback: match by page_id alone (single-doc builds).
                for (d, pp), s in summaries.items():
                    if pp == p:
                        txt = s
                        break
            if txt:
                snippet = txt.strip().replace("\n", " ")
                if len(snippet) > 400:
                    snippet = snippet[:400].rstrip() + "..."
                lines.append(f"    p{p}: {snippet}")
        return "\n".join(lines) if lines else "    (no page summaries available)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _visual_lines(n: KGNode, indent: str = "") -> str:
    """Format `visual_type` / `visual_description` lines for visual nodes.

    Returns "" for text-modality nodes so the prompt collapses cleanly.
    For visual nodes returns a leading newline + indented lines so the
    block slots in right after the `modality:` line.
    """
    if n.modality != "visual":
        return ""
    parts: list[str] = []
    if n.visual_type:
        parts.append(f"{indent}visual_type: {n.visual_type}")
    if n.visual_description:
        parts.append(f"{indent}visual_description: {n.visual_description}")
    if not parts:
        return ""
    return "\n" + "\n".join(parts)



def _l2(m: np.ndarray) -> np.ndarray:
    if m.size == 0:
        return m
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return m / n


def _align_edge_id(src_id: str, tgt_id: str, decision: str) -> str:
    h = uuid.uuid5(uuid.NAMESPACE_OID, f"ALIGN::{src_id}::{tgt_id}::{decision}")
    return f"A:{h.hex[:12]}"


def _dedupe_edges(edges: list[KGEdge]) -> list[KGEdge]:
    seen: set[tuple[str, str, str]] = set()
    out: list[KGEdge] = []
    for e in edges:
        key = (e.src_id, e.tgt_id, e.edge_type)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


# Generic-vocab relations that carry no real semantic content on their
# own. Empirically these are what the judge falls back to when it
# wants to bind an entity to a document-root / report / chapter hub.
# When the target node has many such inbound edges (it's a hub), the
# edge is almost certainly noise and we drop it.
_GENERIC_HUB_RELATIONS = frozenset({
    "is_part_of", "part_of",
    "supports_claim", "supports",
    "is_related_to", "relates_to", "related_to",
    "is_associated_with", "associated_with",
    "includes", "is_included_in",
    "is_linked_to", "linked_to",
})
# A target with this many or more generic inbound edges is a hub.
# Tuned from smoke_esg observation (top hub had 587 is_part_of
# inbound); a threshold of 5 keeps the genuinely specific part_of
# claims and drops the report/document-root fan-in.
_HUB_GENERIC_THRESHOLD = 5


def _drop_generic_edges_to_hubs(edges: list[KGEdge]) -> list[KGEdge]:
    """Drop generic-vocab edges that converge on hub targets."""
    from collections import Counter
    generic_in_degree: Counter[str] = Counter()
    for e in edges:
        if e.relation in _GENERIC_HUB_RELATIONS:
            generic_in_degree[e.tgt_id] += 1
    if not generic_in_degree:
        return edges
    hubs = {tid for tid, n in generic_in_degree.items()
            if n >= _HUB_GENERIC_THRESHOLD}
    if not hubs:
        return edges
    kept: list[KGEdge] = []
    dropped = 0
    for e in edges:
        if e.relation in _GENERIC_HUB_RELATIONS and e.tgt_id in hubs:
            dropped += 1
            continue
        kept.append(e)
    if dropped:
        logger.info(
            "_drop_generic_edges_to_hubs: dropped %d generic edges into %d "
            "hub target(s) (kept %d)",
            dropped, len(hubs), len(kept),
        )
    return kept


# ---------------------------------------------------------------------------
# Tiered keep: strong vs. weak-with-rescue
# ---------------------------------------------------------------------------


def _tag_rescue(edge: KGEdge, reason: str) -> KGEdge:
    md = dict(edge.metadata or {})
    md["rescue_reason"] = reason
    return edge.model_copy(update={"metadata": md})


def _apply_rescue_filter(
    edges: list[KGEdge],
    nodes: list[KGNode],
    extract_edges: list[KGEdge],
    cfg: CrossPageAlignCfg,
) -> list[KGEdge]:
    """Tiered keep with special-case rescue for sub-strong edges.

    Strong (rs >= cfg.rs_threshold) → kept (unless the pair already
        has an edge - see "pair-dedup" below).
    Weak  (rs in [cfg.rs_threshold_rescue, cfg.rs_threshold))
      → considered for rescue in TWO passes:

      Pass 1 - BRIDGE: if the edge connects two otherwise-
        disconnected components, keep it as `bridge`. This is
        structural and modality-agnostic - a sub-cross_modality-
        floor cross-modal edge still survives here when it's the
        only thing tying two components together.

      Pass 2 - PROPERTY: for edges not already kept as bridges,
        keep cross_modality (visual ↔ text) when rs >=
        cfg.rs_threshold_cross_modality (stricter floor - visual
        facts are easy to hallucinate).

    Below floor → already dropped at the judge decoder.

    Pair-dedup: cross-page edges are only added between node pairs
    that don't already have an edge (in either direction). This
    matches the intuition that one connecting edge per pair is
    enough - extract evidence wins over align evidence; if neither
    exists yet, the first surviving align rescue wins.
    """
    strong_t = cfg.rs_threshold / 10.0
    floor_t = cfg.rs_threshold_rescue / 10.0
    cross_mod_t = cfg.rs_threshold_cross_modality / 10.0
    node_by_id = {n.entity_id: n for n in nodes}

    # Undirected pair-set, seeded from extract edges. Cross-page
    # rescues will register their pair after keeping, so subsequent
    # rescues between the same pair are dropped.
    def pair_key(a: str, b: str) -> frozenset[str]:
        return frozenset({a, b})
    existing_pairs: set[frozenset[str]] = set()
    for e in extract_edges:
        existing_pairs.add(pair_key(e.src_id, e.tgt_id))

    # Partition. Among strong, sort by conf-desc so the highest-
    # confidence variant wins when multiple strong edges exist for the
    # same pair (rare but possible).
    raw_strong: list[KGEdge] = []
    same_as: list[KGEdge] = []
    weak: list[KGEdge] = []
    for e in edges:
        if e.edge_type == "same_as":
            same_as.append(e)
        elif e.confidence >= strong_t:
            raw_strong.append(e)
        elif e.confidence >= floor_t:
            weak.append(e)
        # else: shouldn't happen - judge floor already enforces this

    strong: list[KGEdge] = []
    for e in sorted(raw_strong, key=lambda x: -x.confidence):
        pk = pair_key(e.src_id, e.tgt_id)
        if pk in existing_pairs:
            continue
        strong.append(e)
        existing_pairs.add(pk)
    # same_as is structural (triggers merge); always register its pair.
    for e in same_as:
        existing_pairs.add(pair_key(e.src_id, e.tgt_id))

    # Union-Find over entity_ids. Seed from nodes, then union by every
    # edge that's already "in" (extract + strong + same_as).
    parent: dict[str, str] = {n.entity_id: n.entity_id for n in nodes}
    def find(x: str) -> str:
        # path-compressed find
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x
    def union(a: str, b: str) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    for e in extract_edges + strong + same_as:
        if e.src_id in parent and e.tgt_id in parent:
            union(e.src_id, e.tgt_id)

    rescued: list[KGEdge] = []
    rescued_ids: set[str] = set()

    # Pass 1: BRIDGE rescue first. Bridges are structural - if a weak
    # edge connects two otherwise-disconnected components it has high
    # information value regardless of confidence or modality. Tagging
    # bridge first lets sub-cross_modality-floor cross-modal edges
    # still survive when they happen to be the only structural
    # connection (per the user's design intent).
    #
    # Sort by degree-desc so bridges through high-degree (structurally
    # central) endpoints get processed first. Degree is computed ONCE
    # from the "in" graph: extract + strong + same_as.
    from collections import Counter
    deg: Counter[str] = Counter()
    for e in extract_edges + strong + same_as:
        deg[e.src_id] += 1
        if e.tgt_id != e.src_id:
            deg[e.tgt_id] += 1

    def _bridge_key(e: KGEdge) -> tuple[int, float]:
        # negate for descending sort
        return (-max(deg[e.src_id], deg[e.tgt_id]), -e.confidence)

    for e in sorted(weak, key=_bridge_key):
        pk = pair_key(e.src_id, e.tgt_id)
        if pk in existing_pairs:
            continue
        if union(e.src_id, e.tgt_id):
            rescued.append(_tag_rescue(e, "bridge"))
            rescued_ids.add(e.edge_id)
            existing_pairs.add(pk)

    # Pass 2: property rescues for non-bridge weak edges.
    # cross_modality requires conf >= cross_mod_t (stricter floor).
    # surface_grounded uses the global rescue floor.
    # Conf-desc order so the strongest variant of a pair wins.
    for e in sorted(weak, key=lambda x: -x.confidence):
        if e.edge_id in rescued_ids:
            continue
        pk = pair_key(e.src_id, e.tgt_id)
        if pk in existing_pairs:
            continue
        s, t = node_by_id.get(e.src_id), node_by_id.get(e.tgt_id)
        if s is None or t is None:
            continue
        reason = None
        if s.modality != t.modality and e.confidence >= cross_mod_t:
            reason = "cross_modality"
        if reason is not None:
            rescued.append(_tag_rescue(e, reason))
            rescued_ids.add(e.edge_id)
            existing_pairs.add(pk)
            union(e.src_id, e.tgt_id)

    dropped = len(weak) - len(rescued)
    if rescued or dropped:
        by_reason: dict[str, int] = {}
        for r in rescued:
            k = (r.metadata or {}).get("rescue_reason", "?")
            by_reason[k] = by_reason.get(k, 0) + 1
        logger.info(
            "_apply_rescue_filter: kept %d strong + %d same_as + %d rescued "
            "(%s); dropped %d sub-strong",
            len(strong), len(same_as), len(rescued),
            ", ".join(f"{k}={v}" for k, v in sorted(by_reason.items())),
            dropped,
        )
    return strong + same_as + rescued
