"""Core types, protocols, and registry."""

from kg4vd.core.errors import (
    ConfigError,
    EncoderError,
    IndexError,
    KG4VDError,
    PipelineError,
    RegistryError,
    RetrievalError,
)
from kg4vd.core.registry import Registry, register
from kg4vd.core.types import (
    Answer,
    EvidenceCard,
    EvidenceType,
    Hit,
    KGEdge,
    KGNode,
    KGPatch,
    Modality,
    Page,
    PageHit,
    Query,
    QueryAnalysis,
    QueryResult,
    TextItem,
)

__all__ = [
    "Answer",
    "ConfigError",
    "EncoderError",
    "EvidenceCard",
    "EvidenceType",
    "Hit",
    "IndexError",
    "KG4VDError",
    "KGEdge",
    "KGNode",
    "KGPatch",
    "Modality",
    "Page",
    "PageHit",
    "PipelineError",
    "Query",
    "QueryAnalysis",
    "QueryResult",
    "Registry",
    "RegistryError",
    "RetrievalError",
    "TextItem",
    "register",
]
