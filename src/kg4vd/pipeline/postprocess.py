"""Graph post-processing: entity-type sanitisation + relation/edge dedup.

Pure functions over (nodes, edges), extracted from pipeline/build.py.
"""

from __future__ import annotations

import logging

from kg4vd.config.schema import KG4VDConfig
from kg4vd.core.types import KGEdge, KGNode

logger = logging.getLogger(__name__)


def _sanitize_types(
    nodes: list[KGNode], cfg: KG4VDConfig
) -> list[KGNode]:
    """Fold any non-canonical entity_type / visual_type into safe defaults.

    The extraction prompt asks the LLM to pick from the recipe vocab,
    but in practice it sometimes invents types (e.g. ``data_metric``,
    ``subsection``, ``slide``). Downstream code doesn't gate on these,
    but they break per-recipe expected_metrics and weaken the claim
    that profiles constrain extraction. Fold offenders to ``other`` /
    ``None`` and log the rewrite count.
    """
    canon_text = set(cfg.kg.entity_types)
    canon_visual = set(cfg.kg.visual_entity_types)
    canon_all = canon_text | canon_visual
    et_rewrites: dict[str, int] = {}
    vt_rewrites: dict[str, int] = {}
    modality_conflicts = 0
    out: list[KGNode] = []
    for n in nodes:
        update: dict[str, object] = {}
        new_etype = n.entity_type
        # Modality / type conflict: visual node typed as a text entity
        # (e.g. unnamed "man" the LLM extracted from an illustration but
        # tagged person+visual). HARD RULE 1 in the extraction prompt
        # tells it to split into separate text+visual entities; this is
        # the safety net when the rule slips. Promote the visual to its
        # already-set visual_type, falling back to visual_object.
        if n.modality == "visual" and new_etype in canon_text:
            new_etype = n.visual_type if n.visual_type in canon_visual else "visual_object"
            modality_conflicts += 1
        if new_etype not in canon_all:
            et_rewrites[new_etype] = et_rewrites.get(new_etype, 0) + 1
            new_etype = "other"
        if new_etype != n.entity_type:
            update["entity_type"] = new_etype
        if (
            n.modality == "visual"
            and n.visual_type is not None
            and n.visual_type not in canon_visual
        ):
            vt_rewrites[n.visual_type] = vt_rewrites.get(n.visual_type, 0) + 1
            update["visual_type"] = None
        out.append(n.model_copy(update=update) if update else n)
    if et_rewrites:
        logger.info(
            "_sanitize_types: rewrote entity_type for %d nodes -> 'other' (%s)",
            sum(et_rewrites.values()),
            ", ".join(f"{k}={v}" for k, v in sorted(et_rewrites.items())),
        )
    if vt_rewrites:
        logger.info(
            "_sanitize_types: cleared visual_type for %d nodes (%s)",
            sum(vt_rewrites.values()),
            ", ".join(f"{k}={v}" for k, v in sorted(vt_rewrites.items())),
        )
    if modality_conflicts:
        logger.info(
            "_sanitize_types: rewrote entity_type for %d visual nodes that "
            "were tagged with a text-mode type",
            modality_conflicts,
        )
    return out


_REL_STOPWORDS = frozenset({
    "to", "of", "by", "for", "in", "on", "at", "with", "as", "the", "a", "an",
    "is", "are", "be", "being", "been", "was", "were",
    "and", "or", "that", "which", "this", "these", "those",
    "can", "could", "should", "would", "may", "might", "will", "shall",
})

# Suffix list, sorted longest-first so the greedy strip catches "ation"
# before "ion", "ing" before "es", etc. This is a degenerate Porter
# stemmer - strong enough to fold the LLM's common derivational variants
# ("publish" / "publisher" / "publication" -> "publish") without pulling
# in nltk or snowballstemmer as a hard dep. If snowballstemmer is
# available we use it instead (better edge-case behaviour).
_REL_STEM_SUFFIXES = (
    "ization", "ational", "ization", "fulness", "ousness", "iveness",
    "ication", "iciency", "ication",
    "ation", "ement", "ities", "iness",
    "ence", "ance", "ment", "ness", "tion", "sion",
    "ing", "est", "ity",
    "or", "er", "ed", "es", "ly", "al", "ic",
    "s",
)
try:
    import snowballstemmer as _ss   # type: ignore[import-not-found]
    _PORTER = _ss.stemmer("english")
except ImportError:                  # pragma: no cover
    _PORTER = None


def _stem_word(tok: str) -> str:
    """Stem a single token. Real Porter if available, else greedy suffix."""
    if _PORTER is not None:
        return _PORTER.stemWord(tok)
    for suf in _REL_STEM_SUFFIXES:
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def _normalise_relation(rel: str) -> frozenset[str]:
    """Map a relation string to a stemmed token set for dedup matching.

    Aggressive on cosmetic variation (case, whitespace, stop-words,
    derivational morphology), conservative on real semantic difference.
    Returns a frozenset so subset-matching can detect cases where the
    LLM emits e.g. ``"committed to"`` and ``"committed to achieving"``
    between the same pair, or ``"publishes"`` and ``"is the publisher
    of"`` (both stem to ``{"publish"}``).
    """
    if not rel:
        return frozenset()
    out: set[str] = set()
    for tok in rel.lower().strip().replace("-", " ").replace("_", " ").split():
        if not tok or tok in _REL_STOPWORDS:
            continue
        tok = _stem_word(tok)
        if len(tok) >= 3:
            out.add(tok)
    return frozenset(out)


def _collapse_near_dupe_edges(
    edges: list[KGEdge], cfg: KG4VDConfig
) -> list[KGEdge]:
    """Merge same-direction edges whose relations are token-set subsets.

    Two edges (same src, tgt, edge_type) are duplicates if one relation's
    normalised token set is equal to or a subset of the other. This
    catches the common LLM patterns:

      "committed to"          {commit}                ⊆ {commit, achiev}
      "committed to achieving" {commit, achiev}       (kept)

      "is part of"            {part}                  ⊆ {part}
      "part of"               {part}                  (collapsed)

    Different ``edge_type`` values are kept apart (a ``semantic`` edge
    and a ``same_as`` edge between the same pair are genuinely distinct
    in the schema). Cross-page-align relations now share the
    ``semantic`` bucket with extractor-produced ones, so this stage
    naturally collapses align/extractor duplicates against each other.
    Within a duplicate group: keep the edge with the most-informative
    relation (largest token set, then highest confidence), union
    ``source_pages``, concatenate descriptions.
    """
    # First pass: bucket by (src, tgt, edge_type) so subset matching
    # only runs within already-matched pairs.
    from collections import defaultdict
    triples: dict[tuple[str, str, str], list[KGEdge]] = defaultdict(list)
    for e in edges:
        triples[(e.src_id, e.tgt_id, e.edge_type)].append(e)

    out: list[KGEdge] = []
    merged = 0
    for triple, group in triples.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        # Build (edge, normalised-token-set) pairs and sort by
        # informativeness: larger token set first, then higher confidence,
        # then longer raw relation as a tiebreak.
        ranked = [
            (e, _normalise_relation(e.relation or ""))
            for e in group
        ]
        ranked.sort(
            key=lambda pair: (len(pair[1]), pair[0].confidence, len(pair[0].relation or "")),
            reverse=True,
        )
        # Greedy clustering: each candidate joins the most-informative
        # cluster whose token set is a superset of (or equal to) its own.
        # Empty token sets only match other empty sets.
        clusters: list[tuple[frozenset[str], list[KGEdge]]] = []
        for e, toks in ranked:
            placed = False
            for i, (rep_toks, members) in enumerate(clusters):
                # toks ⊆ rep_toks means e is a less-informative variant of rep.
                # rep_toks ⊆ toks would mean e is more informative - but ranking
                # ensures we encounter the more-informative one first, so we
                # only need the ⊆ direction here.
                if (toks and rep_toks and toks.issubset(rep_toks)) or (not toks and not rep_toks and rep_toks == toks):
                    members.append(e)
                    placed = True
                    break
            if not placed:
                clusters.append((toks, [e]))

        for rep_toks, members in clusters:
            if len(members) == 1:
                out.append(members[0])
                continue
            keeper = members[0]
            pages: set[int] = set(keeper.source_pages or [])
            chunks: set[str] = set(keeper.source_chunks or [])
            descs: list[str] = []
            if (keeper.description or "").strip():
                descs.append(keeper.description.strip())
            for e in members[1:]:
                pages.update(e.source_pages or [])
                chunks.update(e.source_chunks or [])
                d = (e.description or "").strip()
                if d and d not in descs:
                    descs.append(d)
            merged += len(members) - 1
            out.append(keeper.model_copy(update={
                "source_pages": sorted(pages),
                "source_chunks": sorted(chunks),
                "description": " ".join(descs)[: int(cfg.kg.description_max_chars)],
            }))
    if merged:
        logger.info(
            "_collapse_near_dupe_edges: merged %d duplicate edges (kept %d)",
            merged, len(out),
        )
    return out


def _drop_orphan_edges(
    nodes: list[KGNode], edges: list[KGEdge]
) -> list[KGEdge]:
    """Filter out edges whose src_id or tgt_id isn't in ``nodes``.

    Cause: when the LLM names an edge endpoint that wasn't also added
    as a node in the same patch, ``_edge_from_raw`` falls back to
    ``uuid5(doc_id, name)`` for the id, which can produce a dangling
    reference. We drop those here so downstream card builders never
    see orphans.
    """
    node_ids = {n.entity_id for n in nodes}
    kept: list[KGEdge] = []
    dropped = 0
    for e in edges:
        if e.src_id in node_ids and e.tgt_id in node_ids:
            kept.append(e)
        else:
            dropped += 1
    if dropped:
        logger.info(
            "Dropped %d orphan edges (target node not in KG); kept %d",
            dropped, len(kept),
        )
    return kept


