"""Raw operation types - what the LLM emits before ID resolution.

The component-cued Extractor / Reflector emit operations keyed by
**name**: nodes have `name`, edges have `src` / `tgt` names. The
controller's validator (``controller.validate_or_repair_ops``) works
on this layer so it can:

  - Match edges to nodes by canonical name (the dedup key).
  - Materialise stub nodes from edge endpoint names without needing
    an ID mint at this stage.
  - Stay simple to test (no ID-resolution coupling).

ID resolution (`name → entity_id`, `src/tgt → src_id/tgt_id`) is the
job of a later parser stage that converts a validated `RawPatch` into
the canonical `core.types.KGPatch` (see ``resolver.py``). RawPatch is the
shape the controller speaks; ``KGPatch`` is the persistence shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RawNodeOp(BaseModel):
    """LLM-emitted node before entity_id is resolved."""

    model_config = ConfigDict(extra="forbid")

    name: str
    entity_type: str
    modality: Literal["text", "visual"] = "text"
    description: str = ""
    visual_description: str | None = None
    visual_type: str | None = None
    source_components: list[str] = Field(default_factory=list)


class RawEdgeOp(BaseModel):
    """LLM-emitted edge before src_id / tgt_id are resolved."""

    model_config = ConfigDict(extra="forbid")

    src: str
    tgt: str
    relation: str
    description: str = ""
    confidence: float = 1.0
    visual_evidence_hint: str | None = None
    source_components: list[str] = Field(default_factory=list)


class RawDeleteEdge(BaseModel):
    """Name-keyed edge delete reference. ``relation`` may be omitted to
    delete every edge between (src, tgt) regardless of relation."""

    model_config = ConfigDict(extra="forbid")

    src: str
    tgt: str
    relation: str | None = None


class RawPatch(BaseModel):
    """Full LLM-emitted patch shape consumed by the validator.

    `delete_nodes` is a list of canonical names;
    `delete_edges` carries (src, tgt, optional relation). The
    validator does not interpret deletes itself - they pass through
    untouched and are applied later by ``apply_ops``.
    """

    model_config = ConfigDict(extra="forbid")

    add_nodes: list[RawNodeOp] = Field(default_factory=list)
    add_edges: list[RawEdgeOp] = Field(default_factory=list)
    replace_nodes: list[RawNodeOp] = Field(default_factory=list)
    replace_edges: list[RawEdgeOp] = Field(default_factory=list)
    delete_nodes: list[str] = Field(default_factory=list)
    delete_edges: list[RawDeleteEdge] = Field(default_factory=list)
    reason: str = ""

    def is_empty(self) -> bool:
        return not (
            self.add_nodes
            or self.add_edges
            or self.replace_nodes
            or self.replace_edges
            or self.delete_nodes
            or self.delete_edges
        )

    def operation_counts(self) -> dict[str, int]:
        return {
            "add_nodes": len(self.add_nodes),
            "add_edges": len(self.add_edges),
            "replace_nodes": len(self.replace_nodes),
            "replace_edges": len(self.replace_edges),
            "delete_nodes": len(self.delete_nodes),
            "delete_edges": len(self.delete_edges),
        }


def norm_name(s: str | None) -> str:
    """Canonical name key - case-folded + stripped."""
    if not isinstance(s, str):
        return ""
    return s.strip().lower()


__all__ = [
    "RawNodeOp",
    "RawEdgeOp",
    "RawDeleteEdge",
    "RawPatch",
    "norm_name",
]
