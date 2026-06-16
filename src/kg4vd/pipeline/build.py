"""Build pipeline.

Stages, in order:

  1. ingest    - PDF → Pages (mineru / pypdfium)
  2. augment   - per-page summary (VLM)
  3. extract   - adaptive per-page component-cued extractor + reflector
  4. align     - cross-page entity alignment + same_as canonicalize
  5. cards     - build EvidenceCards (page / entity / relation)
  6. embed     - multimodal encoder (GME) embeds all cards
  7. index     - upsert into the unified evidence index

  (opt-in) retrieval_graph - page+entity+relation graph view, built on demand
  via ``--stages retrieval_graph`` after a full build.

Each stage is independently runnable / resumable via ``run_build(..., stages=...)``.
Stage helpers live in sibling modules: persistence.py (disk I/O) and
postprocess.py (entity-type sanitisation + edge dedup).

Dead-letter list: chunk-level failures during extract get written to the run
manifest's `failed_chunks[]`.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from kg4vd.cards import (
    build_entity_cards,
    build_page_cards,
    build_relation_cards,
)
from kg4vd.augment import summarize_pages
from kg4vd.config.schema import KG4VDConfig
from kg4vd.core.types import EvidenceCard, KGEdge, KGNode, Page
from kg4vd.encode import build_encoder
from kg4vd.index import build_index
from kg4vd.ingest import build_ingest
from kg4vd.kg import canonicalize_same_as
from kg4vd.kg.align import CrossPageAligner
from kg4vd.kg.extract import Extractor, merge_edges, merge_nodes
from kg4vd.kg.retrieval_graph import build_retrieval_graph
from kg4vd.kg.store import NetworkXKGStore
from kg4vd.llm.factory import build_llm
from kg4vd.llm.openai_compatible import OpenAICompatibleClient  # noqa: F401  (touch for plugin)
from kg4vd.obs import RunManifest, Tracer
from kg4vd.obs.tracer import stage as stage_ctx
from kg4vd.pipeline.persistence import (
    _load_kg,
    _load_pages,
    _persist_kg,
    _persist_page_snapshots,
    _persist_pages,
)
from kg4vd.pipeline.postprocess import (
    _collapse_near_dupe_edges,
    _drop_orphan_edges,
    _sanitize_types,
)

logger = logging.getLogger(__name__)


ALL_STAGES = (
    "ingest",
    "augment",
    "extract",
    "align",
    "cards",
    "embed",
    "index",
)
# Opt-in stages: not run by default `kg4vd build`. Pass explicitly via
# `--stages retrieval_graph` (typically after a full build has populated
# pages.jsonl + kg/*.jsonl).
OPTIONAL_STAGES = ("retrieval_graph",)
_VALID_STAGES = ALL_STAGES + OPTIONAL_STAGES


@dataclass
class BuildArtifacts:
    pages: list[Page] = field(default_factory=list)
    nodes: list[KGNode] = field(default_factory=list)
    edges: list[KGEdge] = field(default_factory=list)
    cards: list[EvidenceCard] = field(default_factory=list)
    failed_chunks: list[dict[str, Any]] = field(default_factory=list)
    work_dir: Path | None = None
    manifest: RunManifest | None = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_build(
    cfg: KG4VDConfig,
    *,
    pdf_paths: list[str | Path],
    stages: Iterable[str] | None = None,
    resume: bool = False,
) -> BuildArtifacts:
    """Run the build pipeline end-to-end (or a subset of stages)."""

    selected = tuple(stages) if stages else ALL_STAGES
    for s in selected:
        if s not in _VALID_STAGES:
            raise ValueError(f"Unknown stage {s!r}; valid: {_VALID_STAGES}")

    work_dir = Path(cfg.dataset.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    trace_path = work_dir / cfg.obs.trace_path
    manifest_path = work_dir / cfg.obs.manifest_path
    started = datetime.now(timezone.utc).isoformat()
    run_id = uuid.uuid4().hex[:12]

    tracer = Tracer(
        run_id=run_id,
        trace_id=run_id,
        jsonl_path=trace_path,
        rich_progress=cfg.obs.rich_progress,
        write_jsonl=cfg.obs.tracer in {"jsonl", "both"},
        title=f"kg4vd build · {cfg.dataset.name}",
    )
    manifest = RunManifest.from_config(
        cfg, run_id=run_id, trace_id=tracer.trace_id, started_at=started
    )

    art = BuildArtifacts(work_dir=work_dir, manifest=manifest)
    t0 = time.time()
    try:
        with tracer:
            async with _astage("build", n_pdfs=len(pdf_paths)):
                # ----- 1. ingest --------------------------------------------------
                if "ingest" in selected:
                    art.pages = await _ingest(cfg, pdf_paths, work_dir, resume=resume)
                else:
                    art.pages = _load_pages(work_dir)

                # Apply skip_empty_pages (preserving page IDs).
                if cfg.kg.extract.skip_empty_pages:
                    pre = len(art.pages)
                    art.pages = [
                        p for p in art.pages if (p.text or "").strip()
                        or p.page_image_path
                        or p.figure_image_paths
                    ]
                    if len(art.pages) < pre:
                        logger.info("Skipped %d empty pages", pre - len(art.pages))

                # ----- 2. augment -------------------------------------------------
                if "augment" in selected:
                    art.pages = await _augment(cfg, art.pages)
                    _persist_pages(work_dir, art.pages)

                # ----- 3. extract -------------------------------------------------
                store = NetworkXKGStore()
                if "extract" not in selected:
                    # Stages run in subset (e.g. --stages cards,embed,index)
                    # need the persisted KG to be available without paying
                    # the LLM cost of re-extraction. The KG is persisted at
                    # the end of a full build to <work_dir>/kg/*.jsonl, so
                    # load it here when we're skipping extract.
                    art.nodes, art.edges = _load_kg(work_dir)
                    for n in art.nodes:
                        n.metadata.setdefault(
                            "doc_id", _doc_id_from_node(n, art.pages)
                        )
                        await store.upsert_node(n)
                    for e in art.edges:
                        await store.upsert_edge(e)
                if "extract" in selected:
                    async with _astage(
                        "extract", n_pages=len(art.pages), total=len(art.pages)
                    ):
                        nodes, edges, failed = await _extract(
                            cfg, art.pages, work_dir=work_dir
                        )
                        # Drop dangling edges before they propagate further.
                        # The reflector's parser uses uuid5(doc_id, name) as a
                        # fallback when an edge references a name that wasn't
                        # also added as a node in the same patch - those edges
                        # end up pointing at non-existent ids ("orphans").
                        edges = _drop_orphan_edges(nodes, edges)
                        # Fold any non-canonical entity_type / visual_type
                        # values from the LLM into safe defaults.
                        nodes = _sanitize_types(nodes, cfg)
                        # Collapse near-duplicate same-direction edges
                        # (LLM often emits "committed to" + "committed to
                        # achieving" between the same pair). Conservative:
                        # only merges when relations share a normalised
                        # form. Different edge_type values are kept apart.
                        edges = _collapse_near_dupe_edges(edges, cfg)
                        art.nodes = nodes
                        art.edges = edges
                        art.failed_chunks = failed
                        for n in nodes:
                            # Stash doc_id in node metadata for downstream card builders.
                            n.metadata.setdefault(
                                "doc_id", _doc_id_from_node(n, art.pages)
                            )
                            await store.upsert_node(n)
                        for e in edges:
                            await store.upsert_edge(e)

                # ----- 4. align ---------------------------------------------------
                if "align" in selected and art.nodes:
                    align_edges = await _align(
                        cfg, art.nodes, art.pages, art.edges
                    )
                    art.edges = list(art.edges) + list(align_edges)
                    for e in align_edges:
                        await store.upsert_edge(e)

                    # ----- 4b. canonicalize same_as clusters ---------------------
                    # Standard ER pipeline final step: connected-component
                    # closure on `same_as` edges → contract clusters → migrate
                    # other edges → drop self-loops + dedupe. Toggleable for
                    # ablation via cfg.cross_page_alignment.canonicalize_same_as.
                    if cfg.cross_page_alignment.canonicalize_same_as:
                        async with _astage("align.canonicalize"):
                            (
                                art.nodes, art.edges, _mapping
                            ) = canonicalize_same_as(art.nodes, art.edges, cfg)
                        # Cluster contraction often pulls together edges
                        # that have near-duplicate phrasings now pointing
                        # at the same canonical pair (e.g. "belong to" and
                        # "is belong to" originally between different surface
                        # variants). canonicalize_same_as's internal dedup
                        # uses a literal lower-case relation key, so it
                        # only catches exact-string duplicates -- run the
                        # token-set/stem dedup again to fold the rest.
                        art.edges = _collapse_near_dupe_edges(art.edges, cfg)
                        # Rebuild the store so downstream stages see the
                        # canonical graph (community detection in particular).
                        store = NetworkXKGStore()
                        for n in art.nodes:
                            n.metadata.setdefault(
                                "doc_id", _doc_id_from_node(n, art.pages)
                            )
                            await store.upsert_node(n)
                        for e in art.edges:
                            await store.upsert_edge(e)

                # ----- 5/6/7. cards + embed + index -------------------------------
                # The encoder (a multi-GB GME checkpoint) is only constructed
                # when a card/embed/index stage actually runs - so `--stages
                # ingest`/`extract`/`align` never load it onto the GPU. This
                # matters when those stages share a box with a separate LLM
                # server that already owns the VRAM.
                index = None
                if "cards" in selected or "embed" in selected or "index" in selected:
                    encoder = build_encoder(cfg.encoder)
                    index = build_index(cfg.index, dim=encoder.dim)
                    async with _astage("cards", n_pages=len(art.pages),
                                       n_nodes=len(art.nodes),
                                       n_edges=len(art.edges)):
                        cards = _make_basic_cards(
                            cfg, art.pages, art.nodes, art.edges
                        )
                        art.cards = cards
                        if "embed" in selected or "index" in selected:
                            await _embed_and_index(cfg, encoder, index, cards)

                # Persist the index - only when a stage that touches it
                # actually ran. Otherwise (e.g. `--stages retrieval_graph`)
                # we'd overwrite the on-disk index with an empty one.
                index_touching = {"cards", "embed", "index"}
                if index_touching.intersection(selected):
                    index_dir = work_dir / "index"
                    await index.persist(index_dir)
                # Persist the KG store as JSON for portability. Same
                # guard: only rewrite when the KG was touched this run.
                kg_touching = {"extract", "align"}
                if kg_touching.intersection(selected):
                    _persist_kg(work_dir, art.nodes, art.edges)

                # ----- 9. retrieval_graph (opt-in) ----------------------------
                # Builds a page+entity+relation graph view used by the
                # PAGE-Rank query path. Reads the finalised KG + Pages from
                # memory (loaded earlier from disk if those stages were
                # skipped). Writes to <work_dir>/retrieval_graph.json.
                if "retrieval_graph" in selected:
                    async with _astage(
                        "retrieval_graph",
                        n_pages=len(art.pages),
                        n_nodes=len(art.nodes),
                        n_edges=len(art.edges),
                    ):
                        rg = build_retrieval_graph(
                            pages=art.pages,
                            nodes=art.nodes,
                            edges=art.edges,
                        )
                        rg_path = work_dir / "retrieval_graph.json"
                        rg.persist(rg_path)
                        logger.info(
                            "retrieval_graph persisted to %s "
                            "(%d nodes / %d edges; %d pages, %d entities, "
                            "%d relations, %d same_as pairs)",
                            rg_path,
                            len(rg.nodes),
                            len(rg.edges),
                            rg.stats["n_pages"],
                            rg.stats["n_entities"],
                            rg.stats["n_relations"],
                            rg.stats["n_same_as_pairs"],
                        )

        # Manifest finalisation
        manifest.totals = tracer.totals()
        manifest.failed_chunks = art.failed_chunks
        manifest.ended_at = datetime.now(timezone.utc).isoformat()
        manifest.elapsed_s = time.time() - t0
        manifest.write(manifest_path)

    except Exception:
        logger.exception("Build failed")
        manifest.totals = tracer.totals()
        manifest.failed_chunks = art.failed_chunks
        manifest.ended_at = datetime.now(timezone.utc).isoformat()
        manifest.elapsed_s = time.time() - t0
        manifest.extra["status"] = "failed"
        manifest.write(manifest_path)
        raise

    return art


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


async def _ingest(
    cfg: KG4VDConfig, pdf_paths: list[str | Path], work_dir: Path, *, resume: bool
) -> list[Page]:
    ingester = build_ingest(cfg.ingest)
    pages_path = work_dir / "pages.jsonl"
    if resume and pages_path.is_file():
        return _load_pages(work_dir)

    out: list[Page] = []
    async with _astage("ingest", n_pdfs=len(pdf_paths)):
        for pdf in pdf_paths:
            async with _astage("ingest.one", pdf=str(pdf)):
                pages = await ingester.ingest(pdf, work_dir=work_dir)
                out.extend(pages)
    _persist_pages(work_dir, out)
    return out


async def _augment(cfg: KG4VDConfig, pages: list[Page]) -> list[Page]:
    if not cfg.augment.page_summary.enabled:
        return pages

    llm = build_llm(_llm_cfg_from_string(cfg.augment.page_summary.llm, cfg=cfg))
    try:
        async with _astage("augment.page_summary", n=len(pages)):
            pages = await summarize_pages(
                pages,
                llm=llm,
                max_async=cfg.runtime.max_async.get("extract", 8),
                max_tokens=cfg.augment.page_summary.max_tokens,
                temperature=cfg.augment.page_summary.temperature,
            )
    finally:
        # Best-effort close (no-op for some clients).
        close = getattr(llm, "aclose", None)
        if close:
            await close()
    return pages


async def _extract(
    cfg: KG4VDConfig, pages: list[Page], *, work_dir: Path | None = None,
) -> tuple[list[KGNode], list[KGEdge], list[dict[str, Any]]]:
    """Per-page extraction with cross-page **merge** (not just dedupe).

    When two pages emit nodes / edges with the same canonical id, they are
    merged via :func:`merge_nodes` / :func:`merge_edges`, which union
    `source_pages` / `source_chunks` and concatenate descriptions. This is the
    standard ER "blocking" stage - exact-name canonicalisation by UUID5 hash
    on lower-cased name. Surface-form variation across pages (which our LLM
    extraction usually produces) is handled separately by the alignment stage
    and post-alignment canonicalisation.

    If ``work_dir`` is given, also dumps per-page adaptive snapshots
    (one JSONL per page under ``snapshots/per_page/``) so the inspect script
    can render the round-by-round evolution of each page's local graph.
    """

    llm = build_llm(_llm_cfg_from_string(cfg.kg.extract.llm, cfg=cfg))
    # Single extraction pipeline: the component-cued, layout-aware adaptive
    # extractor. Requires MinerU-style layout components on each Page (see
    # kg4vd.kg.extract.extractor.load_components_for_page).
    extractor = Extractor(llm=llm, cfg=cfg)

    sem = asyncio.Semaphore(cfg.runtime.max_async.get("extract", 12))
    failed: list[dict[str, Any]] = []
    nodes_by_id: dict[str, KGNode] = {}
    edges_by_id: dict[str, KGEdge] = {}

    snap_dir: Path | None = None
    if work_dir is not None:
        snap_dir = Path(work_dir) / "snapshots" / "per_page"
        snap_dir.mkdir(parents=True, exist_ok=True)

    async def _one(p: Page) -> None:
        async with sem:
            try:
                async with _astage("extract.page", page=p.page_id):
                    res = await extractor.extract(p)
                # No `await` between the dict reads & writes below, so the
                # asyncio scheduler cannot interleave another _one() task
                # mid-merge. Safe without an explicit lock.
                for n in res.nodes:
                    if n.entity_id in nodes_by_id:
                        nodes_by_id[n.entity_id] = merge_nodes(
                            nodes_by_id[n.entity_id], n, cfg=cfg
                        )
                    else:
                        nodes_by_id[n.entity_id] = n
                for e in res.edges:
                    if e.edge_id in edges_by_id:
                        edges_by_id[e.edge_id] = merge_edges(
                            edges_by_id[e.edge_id], e, cfg=cfg
                        )
                    else:
                        edges_by_id[e.edge_id] = e
                if snap_dir is not None:
                    _persist_page_snapshots(snap_dir, p, res)
            except Exception as exc:  # noqa: BLE001
                logger.error("Extract failed for page %s: %r", p.page_id, exc)
                failed.append(
                    {
                        "stage": "extract",
                        "doc_id": p.doc_id,
                        "page_id": p.page_id,
                        "error": repr(exc),
                    }
                )

    try:
        await asyncio.gather(*[_one(p) for p in pages])
    finally:
        close = getattr(llm, "aclose", None)
        if close:
            await close()

    return list(nodes_by_id.values()), list(edges_by_id.values()), failed


async def _align(
    cfg: KG4VDConfig,
    nodes: list[KGNode],
    pages: list[Page] | None = None,
    extract_edges: list[KGEdge] | None = None,
) -> list[KGEdge]:
    if not cfg.cross_page_alignment.enabled:
        return []
    # Optional precomputed node embeddings (node-aligned .npz with `embs` +
    # `node_ids`). When present and matching, skip building the encoder - the
    # nodes were embedded in a separate GME-on-GPU pass, so we can judge here
    # with the encoder's VRAM freed for a co-resident LLM server. Produce it
    # with `kg4vd align-embed`.
    import numpy as np

    precomputed_embs: np.ndarray | None = None
    emb_path = Path(cfg.dataset.work_dir) / "kg" / "node_embs.npz"
    if emb_path.exists():
        data = np.load(emb_path, allow_pickle=True)
        ids = [str(x) for x in data["node_ids"].tolist()]
        if ids == [n.entity_id for n in nodes]:
            precomputed_embs = data["embs"]
            logger.info(
                "align: using precomputed embeddings %s (encoder not loaded)",
                tuple(precomputed_embs.shape),
            )
        else:
            logger.warning(
                "align: %s node_ids do not match current nodes - ignoring, "
                "encoding live", emb_path,
            )
    encoder = None if precomputed_embs is not None else build_encoder(cfg.encoder)
    llm = build_llm(_llm_cfg_from_string(cfg.cross_page_alignment.judge_llm, cfg=cfg))
    pages_by_id = {p.page_id: p for p in (pages or [])}
    # When `evidence_cards.entity_card.use_visual_crop` is on, the
    # aligner gets the same pages + crop dir so its in-memory entity
    # cards carry the same image_payload as the index-time ones.
    crop_out_dir = None
    if cfg.evidence_cards.entity_card.use_visual_crop and pages:
        crop_out_dir = Path(pages[0].page_image_path).parent.parent.parent / "cards" / "entity_crops"
        # Clean stale crops from any prior run - pre-canonicalize
        # entity_ids change after `canonicalize_same_as`, leaving
        # orphan JPEGs that no node points at any more.
        if crop_out_dir.exists():
            import shutil
            shutil.rmtree(crop_out_dir)
    aligner = CrossPageAligner(
        encoder=encoder,
        llm=llm,
        cfg=cfg.cross_page_alignment,
        max_async=cfg.runtime.max_async.get("align", 12),
        evidence_cards_cfg=cfg.evidence_cards,
        pages_by_id=pages_by_id,
        crop_out_dir=crop_out_dir,
    )
    page_summaries: dict[tuple[str, int], str] = {}
    for p in pages or []:
        if p.page_summary:
            page_summaries[(p.doc_id, p.page_id)] = p.page_summary
    try:
        async with _astage("align", n_nodes=len(nodes)):
            return await aligner.align(
                nodes,
                page_summaries=page_summaries,
                extract_edges=extract_edges or [],
                precomputed_embeddings=precomputed_embs,
            )
    finally:
        close = getattr(llm, "aclose", None)
        if close:
            await close()


def _make_basic_cards(
    cfg: KG4VDConfig,
    pages: list[Page],
    nodes: list[KGNode],
    edges: list[KGEdge],
) -> list[EvidenceCard]:
    nodes_by_id = {n.entity_id: n for n in nodes}
    out: list[EvidenceCard] = []
    out.extend(build_page_cards(pages, cfg.evidence_cards))
    pages_by_id = {p.page_id: p for p in pages}
    crop_out_dir = None
    if cfg.evidence_cards.entity_card.use_visual_crop and pages:
        crop_out_dir = Path(pages[0].page_image_path).parent.parent.parent / "cards" / "entity_crops"
    out.extend(build_entity_cards(
        nodes, cfg.evidence_cards,
        pages_by_id=pages_by_id, crop_out_dir=crop_out_dir,
    ))
    out.extend(build_relation_cards(edges, nodes_by_id, cfg.evidence_cards))
    return out


async def _embed_and_index(
    cfg: KG4VDConfig,
    encoder: Any,
    index: Any,
    cards: list[EvidenceCard],
) -> None:
    if not cards:
        return
    sem_size = cfg.runtime.max_async.get("embed", 16)
    async with _astage("embed", n_cards=len(cards)):
        # Encoder may implement encode_cards_batch; fall back to per-card.
        if hasattr(encoder, "encode_cards_batch"):
            embs = await encoder.encode_cards_batch(cards)
        else:
            sem = asyncio.Semaphore(sem_size)

            async def _enc(c: EvidenceCard):
                async with sem:
                    return await encoder.encode_card(c)

            embs_list = await asyncio.gather(*[_enc(c) for c in cards])
            import numpy as np
            embs = np.stack(embs_list, axis=0) if embs_list else None

    async with _astage("index.upsert", n_cards=len(cards)):
        if embs is not None:
            await index.upsert(cards, embs)


def _doc_id_from_node(n: KGNode, pages: list[Page]) -> str:
    if n.source_pages and pages:
        for p in pages:
            if p.page_id in n.source_pages:
                return p.doc_id
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_cfg_from_string(spec: str, *, cfg: KG4VDConfig | None = None):
    """Parse e.g. ``"openrouter:openai/gpt-4o-mini"`` → an LLMCfg.

    Special case: ``"default"`` (or empty) → fall back to ``cfg.generator.llm``
    so per-stage LLM specs can inherit the recipe's main LLM choice without
    repeating ``kind`` / ``model`` / ``api_key_env`` everywhere.
    """

    from kg4vd.config.schema import LLMCfg

    if spec in ("", "default") and cfg is not None:
        return cfg.generator.llm

    if ":" in spec:
        kind, model = spec.split(":", 1)
    else:
        kind, model = "openrouter", spec
    api_env = {
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "vllm": "VLLM_API_KEY",
        "sglang": "",          # local server, no key
        "hf_local": "",
        "mock": "MOCK_API_KEY",
    }.get(kind, "OPENROUTER_API_KEY")
    return LLMCfg(kind=kind, model=model, api_key_env=api_env)


# ---------------------------------------------------------------------------
# async stage helper
# ---------------------------------------------------------------------------


class _astage:
    def __init__(self, name: str, **tags: Any):
        self.name = name
        self.tags = tags

    async def __aenter__(self):
        self._cm = stage_ctx(self.name, **self.tags)
        return self._cm.__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        return self._cm.__exit__(exc_type, exc, tb)
