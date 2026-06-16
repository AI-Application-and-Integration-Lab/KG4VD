"""Pydantic config schema."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DatasetCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    data_dir: str
    work_dir: str


# ---------------------------------------------------------------------------
# Ingest / Augment
# ---------------------------------------------------------------------------


class IngestCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    parser: Literal["mineru", "pypdfium"] = "pypdfium"
    dpi: int = 300
    jpeg_quality: int = 90
    max_image_long_side: int = 2048
    output_subdir: str = "dumps"


class _SubsummaryCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    # "default" uses cfg.generator.llm; otherwise use "<kind>:<model>".
    llm: str = "default"
    max_tokens: int = 1024
    temperature: float = 0.0


class AugmentCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Per-page summaries used by alignment and answer context.
    page_summary: _SubsummaryCfg = _SubsummaryCfg()


# ---------------------------------------------------------------------------
# KG construction
# ---------------------------------------------------------------------------


class _SaturationCfg(BaseModel):
    """Saturation guards for the extract loop."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    # Avoid marking long components complete when too few nodes are grounded.
    density_guard_min_chars: int = 200
    density_guard_min_nodes: int = 3


class ExtractCfg(BaseModel):
    """KG extraction settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # LLM spec for extraction ("default" uses generator.llm).
    llm: str = "default"
    # Bound each LLM call to avoid runaway generations.
    max_tokens: int = 8192
    skip_empty_pages: bool = True
    # Logical extraction rounds per page.
    max_rounds: int = 5
    # Consecutive empty revisions before stopping.
    max_empty_revisions: int = 2
    saturation: _SaturationCfg = _SaturationCfg()


class KGCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_types: list[str] = Field(
        default_factory=lambda: [
            "person", "organization", "group", "location",
            "event", "date", "work_of_art", "product",
            "scientific_concept",
            "chapter", "section", "unit", "assessment_type",
        ]
    )
    visual_entity_types: list[str] = Field(
        default_factory=lambda: [
            "visual_object", "chart_element", "diagram_component",
            "table_region", "layout_region", "figure_panel",
        ]
    )
    extract: ExtractCfg = ExtractCfg()
    description_merge: Literal["concat_sep", "llm_summarize"] = "concat_sep"
    description_max_chars: int = 4000


class CrossPageAlignCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    candidate_top_k: int = 20

    # Strong-keep threshold for cross-page related edges (rs 0..10).
    rs_threshold: float = 9.0
    rs_threshold_same_as: float = 9.0
    # Minimum score eligible for rescue heuristics below rs_threshold.
    rs_threshold_rescue: float = 7.0
    # Stricter rescue floor for cross-modality edges.
    rs_threshold_cross_modality: float = 8.0

    judge_llm: str = "default"

    # Contract same_as clusters after alignment.
    canonicalize_same_as: bool = True


# ---------------------------------------------------------------------------
# Evidence Cards
# ---------------------------------------------------------------------------


class _EmitCfg(BaseModel):
    """Evidence card types to build."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page: bool = True
    entity: bool = True
    relation: bool = True


class _EntityCardCfg(BaseModel):
    """Entity card settings."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    use_visual_crop: bool = False
    crop_padding_frac: float = 0.06
    crop_min_area_px: int = 60 * 60
    crop_max_long_side: int = 768


class _PageCardCfg(BaseModel):
    """Page card settings."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    image_only: bool = True


class EvidenceCardsCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    emit: _EmitCfg = _EmitCfg()
    page_card: _PageCardCfg = _PageCardCfg()
    entity_card: _EntityCardCfg = _EntityCardCfg()


# ---------------------------------------------------------------------------
# Encoder / Index
# ---------------------------------------------------------------------------


class EncoderCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "mock"
    model: str | None = None
    dim: int = 512
    normalize: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)


class IndexCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["nano", "faiss", "qdrant"] = "nano"
    metric: Literal["cosine", "dot", "maxsim"] = "cosine"


# ---------------------------------------------------------------------------
# LLM / Generator
# ---------------------------------------------------------------------------


class LLMCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["openai", "openrouter", "vllm", "sglang", "hf_local", "mock"] = "openrouter"
    model: str = "openai/gpt-4o-mini"
    base_url: str | None = None
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout_s: int = 300
    max_retries: int = 5
    extra: dict[str, Any] = Field(default_factory=dict)


class GeneratorCfg(BaseModel):
    """Default LLM used by build and query stages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    llm: LLMCfg = LLMCfg()


# ---------------------------------------------------------------------------
# Observability / Runtime
# ---------------------------------------------------------------------------


class ObservabilityCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tracer: Literal["jsonl", "rich", "both", "none"] = "both"
    trace_path: str = "trace.jsonl"
    manifest_path: str = "run_manifest.json"
    rich_progress: bool = True


class RuntimeCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_async: dict[str, int] = Field(
        default_factory=lambda: {
            "extract": 12,
            "align": 12,
            "embed": 16,
        }
    )


# ---------------------------------------------------------------------------
# Query settings are excluded from config_hash().
# ---------------------------------------------------------------------------


class _RetrievalCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    anchor_size: int = 30
    token_budget: int = 8000


class _QGGEPresetCfg(BaseModel):
    """Query-guided graph expansion settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rounds: int = 4
    page_beam: int = 100
    lambda_page: float = 0.7
    max_text_per_page: int = 4


class _PPRPresetCfg(BaseModel):
    """Personalized PageRank propagation settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    restart_probability: float = 0.15
    max_depth: int = 4
    max_nodes: int = 2000
    max_iter: int = 50
    tol: float = 1e-6
    lambda_page: float = 0.7
    final_k: int = 100
    max_text_per_page: int = 4


class _AnswerCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["auto", "images", "texts", "fusion"] = "auto"
    prompt_set: str = "default"


class _RerankerCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    candidate_size: int = 20
    top_k: int = 10
    # Empty url falls back to KG4VD_RERANKER_URL, then localhost:8003.
    url: str = ""
    model: str = "Qwen/Qwen3-VL-Reranker-2B"


class QueryCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # "qgge" uses the index; "ppr" uses retrieval_graph.json.
    propagation: Literal["qgge", "ppr"] = "qgge"
    retrieval: _RetrievalCfg = _RetrievalCfg()
    qgge: _QGGEPresetCfg = _QGGEPresetCfg()
    ppr: _PPRPresetCfg = _PPRPresetCfg()
    answer: _AnswerCfg = _AnswerCfg()
    reranker: _RerankerCfg = _RerankerCfg()


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


class RecipeCfg(BaseModel):
    """Recipe metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = ""
    description: str = ""
    extends: list[str] = Field(default_factory=list)


class KG4VDConfig(BaseModel):
    """Full KG4VD config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recipe: RecipeCfg = RecipeCfg()
    dataset: DatasetCfg
    ingest: IngestCfg = IngestCfg()
    augment: AugmentCfg = AugmentCfg()
    kg: KGCfg = KGCfg()
    cross_page_alignment: CrossPageAlignCfg = CrossPageAlignCfg()
    evidence_cards: EvidenceCardsCfg = EvidenceCardsCfg()
    encoder: EncoderCfg = EncoderCfg()
    index: IndexCfg = IndexCfg()
    generator: GeneratorCfg = GeneratorCfg()
    obs: ObservabilityCfg = ObservabilityCfg()
    runtime: RuntimeCfg = RuntimeCfg()
    query: QueryCfg = QueryCfg()
    prompt_set_version: str = "kg4vd-v1.0.0"

    def config_hash(self) -> str:
        """Return a stable build-artifact config hash."""
        payload = self.model_dump(mode="json")
        payload.pop("query", None)
        ds = payload.get("dataset")
        if isinstance(ds, dict):
            ds.pop("data_dir", None)
            ds.pop("work_dir", None)
        s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
