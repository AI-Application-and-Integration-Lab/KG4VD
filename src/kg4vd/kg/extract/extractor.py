"""Component-cued, layout-aware KG extractor.

This is the single extraction pipeline: per page it runs a loop of logical
rounds (each = one extractor+reflector pair), cued by the page's MinerU
layout components.

  round 0   INIT (annotated image + manifest)          → RawPatch
            REFLECT (image + manifest + scorecard)     → RevisionBrief
            → controller updates saturation + decides stop
  round 1   REVISE (saturated image + masked manifest +
                    scorecard + brief)                 → RawPatch
            REFLECT                                    → RevisionBrief
  ...
  round N-1 (max_rounds-1)

Stop conditions (whichever first):
  - ``cfg.kg.extract.max_rounds`` reached
  - ``RevisionBrief.stop_recommendation == True``
  - N consecutive REVISE rounds with empty ops (default 2)

Worst-case LLM calls per page = 2 * max_rounds (every round runs
both ext and reflect). Reflector typically stops earlier.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from kg4vd.config.schema import KG4VDConfig
from kg4vd.core.types import (
    KGEdge,
    KGNode,
    Page,
    RevisionBrief,
)
from kg4vd.ingest.components import Component
from kg4vd.kg.extract._common import (
    ExtractionResult,
    _apply_patch,
    _astage,
    _RoundAudit,
    _RoundSnapshot,
)
from kg4vd.kg.extract.annotate import annotate_page
from kg4vd.kg.extract.controller import Controller
from kg4vd.kg.extract.manifest import build_manifest_text
from kg4vd.kg.extract.prompts import (
    EXTRACTOR_INIT_PROMPT,
    EXTRACTOR_REVISE_PROMPT,
    PROMPTS_VERSION_COMPONENT_CUED,
    REFLECTOR_PROMPT,
)
from kg4vd.kg.extract.raw_ops import RawDeleteEdge, RawEdgeOp, RawNodeOp, RawPatch
from kg4vd.kg.extract.resolver import resolve_raw_patch
from kg4vd.kg.extract.scorecard import render_component_scorecards
from kg4vd.utils.json_repair import parse_json_object_loose


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page → Components: read the layout components the ingest stage
# attached to the Page, or fail loud if absent.
# ---------------------------------------------------------------------------


def load_components_for_page(page: Page) -> list[Component]:
    """Pull the layout components attached to a Page.

    The component-cued path requires a layout parser (MinerU) to have
    produced `components.json` next to the page artefacts and recorded
    its path in ``page.metadata['components_path']``. The MinerU ingest
    parser populates this automatically.
    """
    md = page.metadata or {}
    cp = md.get("components_path")
    if not cp or not Path(cp).is_file():
        raise RuntimeError(
            f"Page {page.page_id} has no components_path; the "
            f"component-cued extractor requires MinerU-style layout "
            f"components on disk. Either run a recipe with "
            f"`ingest.parser=mineru` or populate "
            f"`page.metadata['components_path']` upstream."
        )
    raw = json.loads(Path(cp).read_text())
    return [Component.model_validate(c) for c in raw]


# ---------------------------------------------------------------------------
# Output schemas - what we expect each prompt to JSON-emit
# ---------------------------------------------------------------------------


def _parse_extractor_output(text: str) -> RawPatch:
    """Parse {reason, ops, uncertainties} → RawPatch."""
    obj = parse_json_object_loose(text)
    raw_ops = obj.get("ops")
    ops = raw_ops if isinstance(raw_ops, dict) else {}
    return RawPatch(
        add_nodes=[RawNodeOp.model_validate(_strip_extra_node(n))
                   for n in (ops.get("add_nodes") or [])
                   if isinstance(n, dict) and n.get("name")],
        add_edges=[RawEdgeOp.model_validate(_strip_extra_edge(e))
                   for e in (ops.get("add_edges") or [])
                   if isinstance(e, dict) and e.get("src") and e.get("tgt")],
        replace_nodes=[RawNodeOp.model_validate(_strip_extra_node(n))
                       for n in (ops.get("replace_nodes") or [])
                       if isinstance(n, dict) and n.get("name")],
        replace_edges=[RawEdgeOp.model_validate(_strip_extra_edge(e))
                       for e in (ops.get("replace_edges") or [])
                       if isinstance(e, dict) and e.get("src") and e.get("tgt")],
        delete_nodes=[s for s in (ops.get("delete_nodes") or [])
                      if isinstance(s, str) and s],
        delete_edges=[RawDeleteEdge.model_validate(de)
                      for de in (ops.get("delete_edges") or [])
                      if isinstance(de, dict) and de.get("src") and de.get("tgt")],
        reason=(obj.get("reason") or "").strip(),
    )


def _parse_reflector_output(text: str, *, page_id: int) -> RevisionBrief:
    obj = parse_json_object_loose(text)
    # Ensure page_id is set (model sometimes omits it)
    if "page_id" not in obj:
        obj["page_id"] = page_id
    # Suggested ops uses the same name-keyed shape - convert names
    # → IDs only when the controller actually applies it (here we
    # keep it as a KGPatch with empty entity_ids, since suggestions
    # never get applied directly).
    obj.pop("suggested_ops", None)  # advisory-only; the validator
    # routes any actual upgrades through the next REVISE round
    try:
        return RevisionBrief.model_validate(obj)
    except Exception:
        # Best effort: return an empty brief but keep page_id set.
        return RevisionBrief(page_id=page_id)


# Whitelists of fields RawNodeOp/RawEdgeOp accept - used to drop any
# extra LLM-emitted fields (e.g. `confidence` on nodes, model-specific
# keys) before pydantic validation.
_NODE_FIELDS = set(RawNodeOp.model_fields)
_EDGE_FIELDS = set(RawEdgeOp.model_fields)


def _strip_extra_node(d: dict) -> dict:
    return {k: v for k, v in d.items() if k in _NODE_FIELDS}


def _strip_extra_edge(d: dict) -> dict:
    return {k: v for k, v in d.items() if k in _EDGE_FIELDS}


# ---------------------------------------------------------------------------
# Annotated image rendering - saturation-aware
# ---------------------------------------------------------------------------


def _scale_components_to_render_px(
    components: list[Component], page: Page, render_size: tuple[int, int]
) -> list[Component]:
    """MinerU bboxes live in `page_size_mineru_px`; the rendered page
    image is at `page_size_render_px`. Scale before annotating so the
    boxes line up.
    """
    md = page.metadata or {}
    mw, mh = (md.get("page_size_mineru_px") or render_size)
    rw, rh = render_size
    if (mw, mh) == (rw, rh):
        return components
    sx, sy = rw / mw, rh / mh
    scaled = []
    for c in components:
        b = c.bbox
        scaled.append(c.model_copy(update={
            "bbox": (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)
        }))
    return scaled


def _render_annotated_for_round(
    page: Page,
    components: list[Component],
    saturated_cids: set[str],
    out_dir: Path,
    revise_round_idx: int,
) -> str:
    """Render an annotated image with the current saturation set.

    Returns the on-disk path (caches per-round).
    """
    if not page.page_image_path:
        # No source image - return whatever the upstream provided
        return (page.metadata or {}).get("annotated_image_path", "")
    if not saturated_cids:
        return (page.metadata or {}).get("annotated_image_path") \
            or page.page_image_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"annotated_muted_r{revise_round_idx}.png"
    with Image.open(page.page_image_path) as im:
        render_size = im.size
    scaled = _scale_components_to_render_px(components, page, render_size)
    annotate_page(
        page.page_image_path,
        scaled,
        out_path,
        mute_cids=saturated_cids,
        expected_image_size=render_size,
    )
    return str(out_path)


# ---------------------------------------------------------------------------
# Stub-list builder for the RevisionBrief block
# ---------------------------------------------------------------------------


def _stub_brief_lines(nodes: list[KGNode]) -> list[str]:
    """Return ``"  - <name>  (cited: P1,P2)"`` lines for every stub."""
    lines: list[str] = []
    for n in nodes:
        if (n.entity_type or "").strip() != "entity":
            continue
        cited = ",".join(n.source_components or [])
        lines.append(f"  - {n.name}  (cited: {cited})")
    return lines


def _render_revision_brief(
    brief: RevisionBrief, current_nodes: list[KGNode]
) -> str:
    parts: list[str] = []
    if brief.summary:
        parts.append(f"Summary: {brief.summary}")
    if brief.focus_cues:
        parts.append("Focus cues:")
        for fc in brief.focus_cues:
            parts.append(
                f"  - {fc.component_id} (priority={fc.priority}): "
                f"{fc.reason} (requested: {fc.requested_input})"
            )
    if brief.critiques:
        parts.append("Critiques:")
        for c in brief.critiques:
            parts.append(
                f"  - [{c.severity}] {c.issue_type}: {c.comment} "
                f"(target: {c.target_ref})"
            )
    if brief.stop_recommendation:
        parts.append("(Reflector recommends STOP.)")
    stub_lines = _stub_brief_lines(current_nodes)
    if stub_lines:
        parts.append(
            f"Stubs to upgrade ({len(stub_lines)} - emit replace_nodes "
            "with the exact `name` below; pick a vocab type, NOT 'entity'):"
        )
        parts.extend(stub_lines)
    return "\n".join(parts) if parts else "(empty brief)"


# ---------------------------------------------------------------------------
# Audit shim - bump _RoundAudit with the controller summary so the
# build manifest can see drop / repair counts per round.
# ---------------------------------------------------------------------------


@dataclass
class _CCRoundAudit(_RoundAudit):
    kind: str = "init"             # "init" / "reflector" / "revise"
    controller_summary: dict[str, int] | None = None
    saturated_cids: list[str] | None = None
    # Full reflector RevisionBrief, dumped to a dict so it round-trips
    # through asdict() into the snapshots JSONL. Only set on reflector
    # rounds; None for init/revise rounds.
    brief: dict | None = None


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------


class Extractor:
    """The KG extractor: role-split (Extractor / Reflector / Controller)
    adaptive loop with component-cued grounding.

    Exposes ``extract(page) -> ExtractionResult``; driven by
    ``pipeline/build.py`` for every page.
    """

    prompts_version: str = PROMPTS_VERSION_COMPONENT_CUED

    def __init__(
        self,
        *,
        llm: Any,
        cfg: KG4VDConfig,
        annotated_cache_dir: Path | None = None,
    ):
        self.llm = llm
        self.cfg = cfg
        self.entity_types = cfg.kg.entity_types
        self.visual_entity_types = cfg.kg.visual_entity_types
        self.cc_cfg = cfg.kg.extract
        # Per-page muted-image cache dir; if None we write next to the
        # page artefact (or to a temp dir for offline page objects).
        self._cache_dir = annotated_cache_dir

    async def extract(self, page: Page) -> ExtractionResult:
        components = load_components_for_page(page)
        controller = Controller(components, cfg=self.cc_cfg)
        nodes: list[KGNode] = []
        edges: list[KGEdge] = []
        rounds: list[_CCRoundAudit] = []
        snapshots: list[_RoundSnapshot] = []

        # Each "round" is one extractor+reflector pair, so the budget
        # has units the UI cares about: round 0 is always init+reflect,
        # rounds 1..max_rounds-1 are revise+reflect. The reflector
        # decides whether the next revise actually fires.
        snap_idx = 0  # raw snapshot counter for trace consistency

        # ---- ROUND 0: init + reflect ---------------------------------
        async with _astage("kg.extract.init", page=page.page_id):
            init_raw = await self._call_extractor_init(page, components)
        nodes, edges, audit_e = self._validate_resolve_apply(
            init_raw, controller, nodes, edges, page=page,
            kind="init", round_idx=snap_idx,
        )
        rounds.append(audit_e)
        snapshots.append(_RoundSnapshot(
            round_idx=snap_idx, audit=audit_e,
            nodes=list(nodes), edges=list(edges),
        ))
        snap_idx += 1

        brief, sat_cids = await self._reflect_and_snapshot(
            page, components, controller, nodes, edges,
            rounds, snapshots, snap_idx,
        )
        snap_idx += 1
        if brief.stop_recommendation:
            return ExtractionResult(
                page_id=page.page_id, nodes=nodes, edges=edges,
                rounds=rounds, snapshots=snapshots,  # type: ignore[arg-type]
            )

        # ---- ROUNDS 1..max_rounds-1: revise + reflect ----------------
        empty_streak = 0
        revise_round_idx = 0
        for _logical_round in range(1, self.cc_cfg.max_rounds):
            # --- Extractor REVISE ---
            async with _astage("kg.extract.revise",
                                  page=page.page_id, round=snap_idx):
                rev_raw = await self._call_extractor_revise(
                    page, components, nodes, edges, brief,
                    saturated_cids=sat_cids,
                    revise_round_idx=revise_round_idx,
                )
            revise_round_idx += 1
            had_ops = not rev_raw.is_empty()
            nodes, edges, audit_e = self._validate_resolve_apply(
                rev_raw, controller, nodes, edges, page=page,
                kind="revise", round_idx=snap_idx,
            )
            rounds.append(audit_e)
            snapshots.append(_RoundSnapshot(
                round_idx=snap_idx, audit=audit_e,
                nodes=list(nodes), edges=list(edges),
            ))
            snap_idx += 1
            net_ops = (audit_e.add_nodes + audit_e.add_edges
                       + audit_e.replace_nodes + audit_e.replace_edges
                       + audit_e.delete_nodes + audit_e.delete_edges)
            if not had_ops or net_ops == 0:
                empty_streak += 1
            else:
                empty_streak = 0

            # --- Reflector ---
            brief, sat_cids = await self._reflect_and_snapshot(
                page, components, controller, nodes, edges,
                rounds, snapshots, snap_idx,
            )
            snap_idx += 1
            if brief.stop_recommendation:
                break
            if empty_streak >= self.cc_cfg.max_empty_revisions:
                break

        return ExtractionResult(
            page_id=page.page_id,
            nodes=nodes,
            edges=edges,
            rounds=rounds,  # type: ignore[arg-type]  -- _CCRoundAudit subclasses _RoundAudit
            snapshots=snapshots,
        )

    async def _reflect_and_snapshot(
        self,
        page: Page,
        components: list[Component],
        controller: Controller,
        nodes: list[KGNode],
        edges: list[KGEdge],
        rounds: list[_CCRoundAudit],
        snapshots: list[_RoundSnapshot],
        snap_idx: int,
    ) -> tuple[RevisionBrief, set[str]]:
        """One reflector pass + its snapshot. Returns (brief, sat_cids)
        so the caller can decide whether to enter the next revise."""
        async with _astage("kg.extract.reflect",
                              page=page.page_id, round=snap_idx):
            brief = await self._call_reflector(page, components, nodes, edges)
        sat_cids, _decisions = controller.update_saturation(brief, nodes)
        audit_r = _CCRoundAudit(
            round_idx=snap_idx,
            kind="reflector",
            stop=brief.stop_recommendation,
            reason=brief.summary,
            saturated_cids=sorted(sat_cids),
            brief=brief.model_dump(mode="json"),
        )
        rounds.append(audit_r)
        snapshots.append(_RoundSnapshot(
            round_idx=snap_idx, audit=audit_r,
            nodes=list(nodes), edges=list(edges),
        ))
        return brief, sat_cids

    # --- LLM call wrappers (overridable for tests) -----------------------

    async def _call_extractor_init(
        self, page: Page, components: list[Component]
    ) -> RawPatch:
        prompt = EXTRACTOR_INIT_PROMPT.format(
            entity_types=", ".join(self.entity_types),
            visual_entity_types=", ".join(self.visual_entity_types),
            page_id=page.page_id,
            total_pages=(page.metadata or {}).get("total_pages", "?"),
            page_text_block="Page text:\n" + (page.text or "(no extracted text)"),
            component_manifest=build_manifest_text(components) or "(no components)",
        )
        text = await self._llm_call(
            prompt,
            image_path=(page.metadata or {}).get("annotated_image_path"),
        )
        return _parse_extractor_output(text)

    async def _call_reflector(
        self,
        page: Page,
        components: list[Component],
        nodes: list[KGNode],
        edges: list[KGEdge],
    ) -> RevisionBrief:
        prompt = REFLECTOR_PROMPT.format(
            page_id=page.page_id,
            total_pages=(page.metadata or {}).get("total_pages", "?"),
            page_text_block="Page text:\n" + (page.text or "(no extracted text)"),
            component_manifest=build_manifest_text(components) or "(no components)",
            current_kg_block=render_component_scorecards(components, nodes, edges),
        )
        text = await self._llm_call(
            prompt,
            image_path=(page.metadata or {}).get("annotated_image_path"),
        )
        return _parse_reflector_output(text, page_id=page.page_id)

    async def _call_extractor_revise(
        self,
        page: Page,
        components: list[Component],
        nodes: list[KGNode],
        edges: list[KGEdge],
        brief: RevisionBrief,
        *,
        saturated_cids: set[str],
        revise_round_idx: int,
    ) -> RawPatch:
        nodes_per_cid: dict[str, int] = {}
        for n in nodes:
            for cid in n.source_components or []:
                nodes_per_cid[cid] = nodes_per_cid.get(cid, 0) + 1
        manifest = build_manifest_text(
            components,
            saturated_cids=saturated_cids,
            nodes_per_cid=nodes_per_cid,
        ) or "(no components)"
        prompt = EXTRACTOR_REVISE_PROMPT.format(
            page_id=page.page_id,
            total_pages=(page.metadata or {}).get("total_pages", "?"),
            page_text_block="Page text:\n" + (page.text or "(no extracted text)"),
            component_manifest=manifest,
            current_kg_block=render_component_scorecards(components, nodes, edges),
            revision_brief_block=_render_revision_brief(brief, nodes),
        )
        # Saturation-aware annotated image
        cache_dir = self._cache_dir or self._default_cache_dir(page)
        image_path = _render_annotated_for_round(
            page, components, saturated_cids,
            out_dir=cache_dir, revise_round_idx=revise_round_idx,
        )
        text = await self._llm_call(prompt, image_path=image_path)
        return _parse_extractor_output(text)

    # --- LLM I/O ---------------------------------------------------------

    async def _llm_call(self, prompt: str, *, image_path: str | None) -> str:
        images = [image_path] if image_path else None
        resp = await self.llm.acomplete(
            prompt,
            system=f"prompt-set: {self.prompts_version}",
            images=images,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=self.cc_cfg.max_tokens,
        )
        return getattr(resp, "text", "") or ""

    # --- internal helpers ------------------------------------------------

    def _validate_resolve_apply(
        self,
        raw: RawPatch,
        controller: Controller,
        nodes: list[KGNode],
        edges: list[KGEdge],
        *,
        page: Page,
        kind: str,
        round_idx: int,
    ) -> tuple[list[KGNode], list[KGEdge], _CCRoundAudit]:
        validated, summary = controller.validate(raw, nodes)
        kg = resolve_raw_patch(
            validated,
            doc_id=page.doc_id,
            page_id=page.page_id,
            components=controller.components,
        )
        kg.controller_summary = summary
        # Pre-apply counts; post-merge, dedup happens inside _apply_patch
        # but we record the patch shape so the trace shows what was emitted.
        before = (len(nodes), len(edges))
        nodes2, edges2 = _apply_patch(nodes, edges, kg, page=page, cfg=self.cfg)
        # Net counts (positive = added, negative = removed)
        net_nodes_added = len(nodes2) - before[0]
        net_edges_added = len(edges2) - before[1]
        audit = _CCRoundAudit(
            round_idx=round_idx,
            kind=kind,
            add_nodes=max(0, net_nodes_added) + summary.get("materialised_stub_nodes", 0),
            add_edges=max(0, net_edges_added),
            replace_nodes=len(kg.replace_nodes),
            replace_edges=len(kg.replace_edges),
            delete_nodes=len(kg.delete_node_ids),
            delete_edges=len(kg.delete_edge_ids),
            reason=raw.reason,
            controller_summary=summary,
        )
        return nodes2, edges2, audit

    def _default_cache_dir(self, page: Page) -> Path:
        """Cache muted images next to the page artefact when the
        caller didn't supply a dir explicitly."""
        if page.page_image_path:
            return Path(page.page_image_path).parent
        return Path("/tmp") / f"kg4vd_cc_{page.doc_id}_{page.page_id}"


__all__ = [
    "Extractor",
    "load_components_for_page",
]
