"""Core data types.

Every public structure used across module boundaries lives here, defined as
Pydantic BaseModel for cheap validation and JSON serialisation. Heavy ML
arrays (embeddings) stay outside Pydantic and are passed as numpy arrays.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _gen_qid() -> str:
    """Short, URL-safe query id used when callers don't supply one.

    Batch drivers (e.g. iterating ``questions.jsonl``) should set ``qid``
    explicitly to the dataset row id so diagnostics files line up.
    Interactive ``kg4vd ask`` paths get an auto-generated qid here.
    """
    return uuid.uuid4().hex[:12]

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Modality(str, Enum):
    text = "text"
    image = "image"
    graph = "graph"
    summary = "summary"


EvidenceType = Literal[
    "page",
    "entity",
    "relation",
]


# ---------------------------------------------------------------------------
# Document-level types
# ---------------------------------------------------------------------------


class Page(BaseModel):
    """A single PDF page after ingest + augmentation."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    page_id: int                    # original PDF page number, 1-indexed
    text: str = ""
    page_image_path: str | None = None
    figure_image_paths: list[str] = Field(default_factory=list)
    page_summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# KG types
# ---------------------------------------------------------------------------


class BBoxRef(BaseModel):
    """One (component_id, bbox) pair attached to a node/edge.

    A node/edge that cites N source_components carries N BBoxRefs, one
    per cited region. Bbox is in the layout parser's pixel space
    (MinerU's `page_size_mineru_px`); downstream visualisation re-scales
    via `Page.metadata["page_size_render_px"]`.

    `page_id` lets downstream code (cropping, retrieval) map a bbox back
    to its source page after canonicalize_same_as merges multi-page
    entities into one node - `component_id` like "P1"/"IM1" is only
    unique within a page, so it can't disambiguate on its own.
    """

    model_config = ConfigDict(extra="forbid")

    component_id: str
    bbox: tuple[float, float, float, float]
    page_id: int


class KGNode(BaseModel):
    """A node in the multimodal KG.

    `modality="visual"` triggers visual-entity-specific schema fields.
    """

    model_config = ConfigDict(extra="forbid")

    entity_id: str
    name: str
    entity_type: str                # text type or one of the visual_entity_types
    modality: Literal["text", "visual"] = "text"
    description: str = ""
    visual_description: str | None = None     # only for modality="visual"
    visual_type: str | None = None            # chart_element / diagram_component / ...
    bbox: tuple[float, float, float, float] | None = None  # single-bbox form (superseded by `bboxes`)
    # Every claim cites the layout component(s) that support it.
    # Required at write time by the controller's validator, defaulted
    # to [] so records without grounding still load.
    source_components: list[str] = Field(default_factory=list)
    bboxes: list[BBoxRef] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    source_chunks: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KGEdge(BaseModel):
    """An edge in the multimodal KG."""

    model_config = ConfigDict(extra="forbid")

    edge_id: str
    src_id: str
    tgt_id: str
    relation: str
    edge_type: Literal[
        "semantic",          # any LLM-named relation (per-page or cross-page)
        "same_as",           # cross-page alignment: same real entity (merge)
        "supports",          # text-chunk / page supports a relation
    ] = "semantic"
    description: str = ""
    visual_evidence_hint: str | None = None
    confidence: float = 1.0
    # Same grounding contract as KGNode.
    source_components: list[str] = Field(default_factory=list)
    bboxes: list[BBoxRef] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    source_chunks: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KGPatch(BaseModel):
    """A reflector patch - explicit add/replace/delete operations.

    Each round of adaptive extraction produces one patch. Patches are
    auditable; we record per-round counts of add/replace/delete for ablation.
    """

    model_config = ConfigDict(extra="forbid")

    add_nodes: list[KGNode] = Field(default_factory=list)
    add_edges: list[KGEdge] = Field(default_factory=list)
    replace_nodes: list[KGNode] = Field(default_factory=list)   # same entity_id, new fields
    replace_edges: list[KGEdge] = Field(default_factory=list)
    delete_node_ids: list[str] = Field(default_factory=list)
    delete_edge_ids: list[str] = Field(default_factory=list)
    stop: bool = False
    reason: str = ""
    # Controller-side audit counts emitted by validate_or_repair_ops.
    # The run manifest uses this to summarize dropped / repaired /
    # materialised ops per round.
    controller_summary: dict[str, int] = Field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (
            self.add_nodes
            or self.add_edges
            or self.replace_nodes
            or self.replace_edges
            or self.delete_node_ids
            or self.delete_edge_ids
        )

    def operation_counts(self) -> dict[str, int]:
        return {
            "add_nodes": len(self.add_nodes),
            "add_edges": len(self.add_edges),
            "replace_nodes": len(self.replace_nodes),
            "replace_edges": len(self.replace_edges),
            "delete_nodes": len(self.delete_node_ids),
            "delete_edges": len(self.delete_edge_ids),
        }


# ---------------------------------------------------------------------------
# RevisionBrief - reflector output for the component-cued extractor.
# ---------------------------------------------------------------------------


_PriorityLit = Literal["high", "medium", "low"]
_RequestedInputLit = Literal[
    "annotated_page_only",
    "component_crop",
    "expanded_manifest",
    "table_html",
    "neighbor_context",
]
_ComponentStatusLit = Literal[
    "covered",
    "partially_covered",
    "uncovered",
    "needs_visual_inspection",
    "irrelevant",
]
_CritiqueTargetKindLit = Literal["node", "edge", "component", "coverage"]


class FocusCue(BaseModel):
    """One Reflector cue pointing the Extractor at a component to re-examine."""

    model_config = ConfigDict(extra="forbid")

    component_id: str
    priority: _PriorityLit
    reason: str
    requested_input: _RequestedInputLit = "annotated_page_only"


class ComponentReview(BaseModel):
    """Reflector's per-component coverage verdict - drives saturation."""

    model_config = ConfigDict(extra="forbid")

    component_id: str
    status: _ComponentStatusLit
    notes: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    """A specific issue with an existing node / edge / component."""

    model_config = ConfigDict(extra="forbid")

    target_kind: _CritiqueTargetKindLit
    target_ref: dict[str, Any]
    issue_type: str
    severity: _PriorityLit
    comment: str


class RevisionBrief(BaseModel):
    """Reflector emits this; never writes ops itself.

    Suggested_ops is advisory only - the controller validates and the
    next Extractor pass may accept, modify, or reject each suggestion.
    `stop_recommendation: true` with empty critiques/focus_cues is a
    valid (and common) SUCCESS outcome.
    """

    model_config = ConfigDict(extra="forbid")

    page_id: int
    summary: str = ""
    focus_cues: list[FocusCue] = Field(default_factory=list)
    component_reviews: list[ComponentReview] = Field(default_factory=list)
    critiques: list[Critique] = Field(default_factory=list)
    suggested_ops: KGPatch = Field(default_factory=KGPatch)
    stop_recommendation: bool = False


# ---------------------------------------------------------------------------
# Evidence Card - the universal retrievable unit
# ---------------------------------------------------------------------------


class EvidenceCard(BaseModel):
    """A typed, retrievable piece of evidence.

    Every retrievable unit in the system - page, text chunk, entity, relation,
    subgraph, page summary, document summary, community summary - is normalised
    into an EvidenceCard. The `evidence_type` field lets downstream stages do
    type-aware selection while keeping a single shared encode/index/retrieve
    surface.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    evidence_type: EvidenceType
    doc_id: str
    page_ids: list[int] = Field(default_factory=list)
    text_payload: str | None = None
    image_payload: str | None = None        # path or data URL
    metadata: dict[str, Any] = Field(default_factory=dict)
    graph_refs: dict[str, Any] | None = None    # {node_ids: [...], edge_ids: [...], community_id: str}

    def card_hash(self) -> str:
        """Stable hash of card content (used for embedding cache keys)."""
        payload = self.model_dump(mode="json", exclude={"evidence_id"})
        s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Query / Hit
# ---------------------------------------------------------------------------


class Query(BaseModel):
    """A user question plus optional multimodal context.

    ``qid`` is the canonical query identifier used by diagnostics, run
    logs, and the per-question JSON files written under
    ``exp/page_rank_runs/<qid>.json``. Batch drivers set it from the
    dataset row id; interactive paths get an auto-generated short hex.

    ``answer_guidelines`` / ``min_words`` / ``max_words`` are populated
    by DLVQA-style benchmarks at inference time; MMLongBench-style
    questions leave them ``None``.

    ``answer_format`` is the dataset's gold answer-shape annotation -
    authoritative when present. Used as a routing hint by the analyzer
    (overrides its own LLM guess) and as a system-prompt selector by
    the generator. We accept the raw casing from each benchmark::

      MMLongBench-Doc  : "Str" | "Float" | "Int" | "List" | "None"
      DLVQA            : "prose" | "wiki"

    The analyzer normalises this internally; downstream code can read
    either via ``Query.answer_format`` or ``QueryAnalysis.answer_format``.
    """

    model_config = ConfigDict(extra="forbid")

    qid: str = Field(default_factory=_gen_qid)
    text: str
    images: list[str] = Field(default_factory=list)   # paths or data URLs
    audio: list[str] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    answer_guidelines: str | None = None
    min_words: int | None = None
    max_words: int | None = None
    answer_format: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Hit(BaseModel):
    """A scored evidence card returned by retrieval/rerank."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    card: EvidenceCard
    score: float
    rank: int = -1
    source: Literal["retrieve", "expand", "rerank"] = "retrieve"
    explanation: str | None = None


# ---------------------------------------------------------------------------
# Query-time types (retrieval + answer). Consumed by the `query` package.
# ---------------------------------------------------------------------------

QueryRoute = Literal["single", "multi", "document_level"]
AnswerMode = Literal["auto", "images", "texts", "fusion"]


class QueryAnalysis(BaseModel):
    """Router output: how a query should be handled."""

    model_config = ConfigDict(extra="forbid")

    route: QueryRoute
    rationale: str = ""
    confidence: str | None = None
    answer_format: str | None = None


class PageHit(BaseModel):
    """A scored page from anchor retrieval / graph propagation."""

    model_config = ConfigDict(extra="forbid")

    page_id: int
    score: float
    rank: int
    image_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextItem(BaseModel):
    """A scored text snippet (page / entity / relation) for the answer context."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["page", "entity", "relation"]
    text: str
    score: float
    page_ids: list[int] = Field(default_factory=list)
    source_id: str | None = None


class Answer(BaseModel):
    """A generated answer with citations and token usage."""

    model_config = ConfigDict(extra="forbid")

    text: str
    cited_pages: list[int] = Field(default_factory=list)
    confidence: str | None = None
    raw: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """Full result of one query: analysis + ranked evidence + answer."""

    model_config = ConfigDict(extra="forbid")

    query: Query
    analysis: QueryAnalysis
    answer: Answer
    anchor_pages: list[PageHit] = Field(default_factory=list)
    ranked_pages: list[PageHit] = Field(default_factory=list)
    text_items: list[TextItem] = Field(default_factory=list)
    image_draft: Answer | None = None
    text_draft: Answer | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
