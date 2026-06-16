"""Controller - validator + saturation tracker for the component-cued
extractor.

The controller is **pure code, no LLM calls**. It enforces the six
guarantees that turn raw LLM output into a clean grounded patch
(rules R1-R6) and tracks per-page saturation state used
by the manifest masker / annotated-image masker on later rounds.

Validator rules R1-R6, applied in order:

  R1 Drop nodes whose `source_components` is missing or names an
     unknown component_id.
  R2 Repair edges whose `source_components` is empty by taking the
     union of the cited grounding of their src / tgt nodes (looked
     up in the post-R1 lookup). Drop edges that can't be repaired.
  R3 For surviving edges whose src/tgt name is missing from the
     post-R1+R4 lookup, materialise a stub node grounded to the
     edge's `source_components`.
  R4 Replace_nodes filter:
        * If target name is unknown → demote to `add_nodes`.
        * If target exists and is already typed (not a stub) and
          new `entity_type` is the same OR is itself "entity" →
          drop (no-op upgrade).
        * If target IS a stub but the new type is still "entity" →
          drop (pointless replace).
  R6 Rebuild the name-to-node lookup AFTER R1 (and again after R4)
     so subsequent edge logic only sees SURVIVING nodes. Without
     this, stub materialisation would skip orphan edges because
     `name_to_node` would still show the dropped node as "present".

R5 (saturation density guard) is on the saturation-tracker side -
it lives in ``Controller.update_saturation``.

The validator is a stateless function so it can be tested in isolation.
The ``Controller`` class wraps the per-page saturation set and offers
a thin wrapper that calls the validator with `self.valid_cids`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from kg4vd.config.schema import ExtractCfg
from kg4vd.core.types import KGNode, RevisionBrief
from kg4vd.ingest.components import Component
from kg4vd.kg.extract.raw_ops import RawEdgeOp, RawNodeOp, RawPatch, norm_name


# Exact summary keys are stable - downstream code (run manifest, eval
# scripts) reads these by name. Don't rename without sweeping callers.
_VALIDATOR_SUMMARY_KEYS = (
    "drop_nodes_no_grounding",
    "drop_nodes_invalid_ids",
    "drop_edges_no_grounding",
    "drop_edges_invalid_ids",
    "repair_edges_from_endpoints",
    "materialised_stub_nodes",
    "drop_invalid_replace",
    "demoted_replace_to_add",
)


_SAT_STATUS = {"covered", "irrelevant"}
_UNSAT_STATUS = {"partially_covered", "uncovered", "needs_visual_inspection"}


# ---------------------------------------------------------------------------
# Stub creation helper (kept module-level so tests can introspect it)
# ---------------------------------------------------------------------------


def make_stub_node(name: str, source_components: list[str]) -> RawNodeOp:
    """Construct the placeholder node materialised when an edge endpoint
    is missing. ``entity_type="entity"`` is the sentinel a later
    `replace_nodes` round upgrades."""
    return RawNodeOp(
        name=name,
        entity_type="entity",
        modality="text",
        description=(
            "Auto-materialised stub for orphan edge endpoint; "
            "semantic type to be refined."
        ),
        source_components=list(source_components),
    )


def is_stub(n: RawNodeOp | KGNode) -> bool:
    """True for nodes the controller materialised as a stub."""
    et = (getattr(n, "entity_type", "") or "").strip()
    if et == "entity":
        return True
    desc = getattr(n, "description", "") or ""
    return desc.startswith("Auto-materialised stub")


# ---------------------------------------------------------------------------
# R1-R6 validator
# ---------------------------------------------------------------------------


def _empty_summary() -> dict[str, int]:
    return {k: 0 for k in _VALIDATOR_SUMMARY_KEYS}


def validate_or_repair_ops(
    patch: RawPatch,
    current_nodes: Iterable[KGNode | RawNodeOp],
    valid_component_ids: set[str],
) -> tuple[RawPatch, dict[str, int]]:
    """Run R1-R6 over `patch`. Returns ``(filtered_patch, summary)``.

    `current_nodes` is the running per-page KG state (the patch hasn't
    been applied yet); the controller uses it for stub-target lookup
    and replace-validity checks.

    The returned RawPatch is a deep-copy-modified version of the
    input - the caller's `patch` is not mutated in place.
    """
    summary = _empty_summary()
    # Deep-copy so we don't mutate the caller's RawPatch.
    p = patch.model_copy(deep=True)

    def _valid_sc(sc: list[str]) -> list[str]:
        return [x for x in (sc or []) if x in valid_component_ids]

    # ---- R1: validate nodes' source_components -----------------------------
    for slot in ("add_nodes", "replace_nodes"):
        kept: list[RawNodeOp] = []
        for n in getattr(p, slot):
            sc = list(n.source_components or [])
            if not sc:
                summary["drop_nodes_no_grounding"] += 1
                continue
            sc_valid = _valid_sc(sc)
            if not sc_valid:
                summary["drop_nodes_invalid_ids"] += 1
                continue
            if len(sc_valid) != len(sc):
                n.source_components = sc_valid
            kept.append(n)
        setattr(p, slot, kept)

    # ---- R6 (first half): rebuild name_to_node from CURRENT nodes +
    # surviving add_nodes only. Replace_nodes hasn't been validated
    # yet so don't trust it in the lookup. -----------------------------------
    name_to_node: dict[str, RawNodeOp | KGNode] = {}
    for n in current_nodes:
        k = norm_name(getattr(n, "name", None))
        if k:
            name_to_node[k] = n
    for n in p.add_nodes:
        k = norm_name(n.name)
        if k and k not in name_to_node:
            name_to_node[k] = n

    # ---- R4: replace_nodes filter ------------------------------------------
    kept_replace: list[RawNodeOp] = []
    for rn in p.replace_nodes:
        name_k = norm_name(rn.name)
        new_type = (rn.entity_type or "").strip()
        if not name_k:
            summary["drop_invalid_replace"] += 1
            continue
        existing = name_to_node.get(name_k)
        if existing is None:
            # Promote to add - model named a new entity via replace.
            p.add_nodes.append(rn)
            summary["demoted_replace_to_add"] += 1
            name_to_node[name_k] = rn
            continue
        old_type = (getattr(existing, "entity_type", "") or "").strip()
        old_is_stub = old_type == "entity"
        new_is_stub = (not new_type) or new_type == "entity"
        if new_is_stub:
            # Replacing with a stub - pointless.
            summary["drop_invalid_replace"] += 1
            continue
        if not old_is_stub and old_type == new_type:
            # Re-affirming an already-correct type - wasted op.
            summary["drop_invalid_replace"] += 1
            continue
        kept_replace.append(rn)
    p.replace_nodes = kept_replace

    # Apply surviving replaces to the lookup (R6 second half) so edge
    # logic sees the post-replace state.
    for rn in kept_replace:
        k = norm_name(rn.name)
        if k:
            name_to_node[k] = rn

    # ---- R2: edge grounding repair -----------------------------------------
    for slot in ("add_edges", "replace_edges"):
        kept_e: list[RawEdgeOp] = []
        for e in getattr(p, slot):
            sc = list(e.source_components or [])
            sc_valid = _valid_sc(sc)
            if sc_valid:
                if len(sc_valid) != len(sc):
                    e.source_components = sc_valid
                kept_e.append(e)
                continue
            # Repair: union of src/tgt node citations.
            src = name_to_node.get(norm_name(e.src))
            tgt = name_to_node.get(norm_name(e.tgt))
            cand: set[str] = set()
            if src is not None:
                cand.update(getattr(src, "source_components", []) or [])
            if tgt is not None:
                cand.update(getattr(tgt, "source_components", []) or [])
            cand &= valid_component_ids
            if cand:
                e.source_components = sorted(cand)
                summary["repair_edges_from_endpoints"] += 1
                kept_e.append(e)
            else:
                if sc:
                    summary["drop_edges_invalid_ids"] += 1
                else:
                    summary["drop_edges_no_grounding"] += 1
        setattr(p, slot, kept_e)

    # ---- R3: materialise stub nodes for orphan edge endpoints -------------
    for slot in ("add_edges", "replace_edges"):
        for e in getattr(p, slot):
            for endpoint in (e.src, e.tgt):
                k = norm_name(endpoint)
                if not k or k in name_to_node:
                    continue
                stub = make_stub_node(endpoint, e.source_components)
                p.add_nodes.append(stub)
                name_to_node[k] = stub
                summary["materialised_stub_nodes"] += 1

    return p, summary


# ---------------------------------------------------------------------------
# Saturation tracker - instance state, density-guarded
# ---------------------------------------------------------------------------


@dataclass
class _SaturationState:
    saturated: set[str] = field(default_factory=set)
    last_decisions: list[dict] = field(default_factory=list)


class Controller:
    """Per-page controller - owns saturation state, exposes the
    validator, and helps the extractor render saturation-aware inputs.

    Stateless across pages; one Controller per page. The validator
    function `validate_or_repair_ops` is also exposed module-level
    for callers that just want the rule logic without saturation.
    """

    def __init__(
        self,
        components: list[Component],
        cfg: ExtractCfg | None = None,
    ):
        self.components = components
        self.cid_to_component = {c.component_id: c for c in components}
        self.valid_cids: set[str] = set(self.cid_to_component)
        self.cfg = cfg or ExtractCfg()
        self._sat = _SaturationState()

    # -- saturation -----------------------------------------------------

    @property
    def saturated_cids(self) -> set[str]:
        return set(self._sat.saturated)

    def update_saturation(
        self,
        brief: RevisionBrief,
        current_nodes: Iterable[KGNode | RawNodeOp],
    ) -> tuple[set[str], list[dict]]:
        """Update the cumulative saturation set from a Reflector brief.

        Rules:
          - status `covered` / `irrelevant` → propose to saturate.
          - status `partially_covered` / `uncovered` /
            `needs_visual_inspection` → unsaturate (Reflector override).
          - components NOT mentioned in this brief keep their previous
            state (set accumulates across rounds).
          - **R5 density guard** (controller-side): refuse to saturate
            paragraph / list / table components whose manifest length
            > `density_guard_min_chars` but grounded node count
            < `density_guard_min_nodes`. Catches the "one group node
            despite a 22-name list" failure mode.

        Returns the new saturation set + a per-component decision log
        suitable for trace inspection.
        """
        if not self.cfg.saturation.enabled:
            return set(self._sat.saturated), []

        sat = set(self._sat.saturated)
        decisions: list[dict] = []
        counts = _nodes_per_cid(current_nodes)
        min_chars = self.cfg.saturation.density_guard_min_chars
        min_nodes = self.cfg.saturation.density_guard_min_nodes

        for cr in brief.component_reviews:
            cid = cr.component_id
            status = cr.status
            c = self.cid_to_component.get(cid)
            if c is None:
                continue

            if status in _SAT_STATUS:
                content_len = (
                    len(c.text_full or "") + len(c.html or "")
                )
                n = counts.get(cid, 0)
                if (
                    c.type in {"paragraph", "list", "table"}
                    and content_len > min_chars
                    and n < min_nodes
                ):
                    # Density sanity blocks saturation regardless of
                    # what the Reflector said.
                    if cid in sat:
                        sat.discard(cid)
                    decisions.append({
                        "cid": cid,
                        "status": status,
                        "decision": "deny_sanity",
                        "content_len": content_len,
                        "n_grounded": n,
                    })
                    continue
                if cid not in sat:
                    decisions.append({
                        "cid": cid, "status": status, "decision": "saturate",
                    })
                sat.add(cid)
            elif status in _UNSAT_STATUS:
                if cid in sat:
                    decisions.append({
                        "cid": cid, "status": status, "decision": "unsaturate",
                    })
                    sat.discard(cid)

        self._sat.saturated = sat
        self._sat.last_decisions = decisions
        return set(sat), decisions

    # -- validator wrapper ---------------------------------------------

    def validate(
        self,
        patch: RawPatch,
        current_nodes: Iterable[KGNode | RawNodeOp],
    ) -> tuple[RawPatch, dict[str, int]]:
        return validate_or_repair_ops(patch, current_nodes, self.valid_cids)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _nodes_per_cid(
    nodes: Iterable[KGNode | RawNodeOp],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for n in nodes:
        for cid in (getattr(n, "source_components", None) or []):
            counts[cid] = counts.get(cid, 0) + 1
    return counts


__all__ = [
    "Controller",
    "validate_or_repair_ops",
    "make_stub_node",
    "is_stub",
]
