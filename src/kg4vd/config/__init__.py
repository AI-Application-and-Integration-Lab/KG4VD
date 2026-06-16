"""Config schema, YAML loader, and presets."""

from kg4vd.config.loader import load_config, resolve_overrides
from kg4vd.config.schema import (
    AugmentCfg,
    CrossPageAlignCfg,
    DatasetCfg,
    EncoderCfg,
    EvidenceCardsCfg,
    GeneratorCfg,
    IndexCfg,
    IngestCfg,
    KG4VDConfig,
    KGCfg,
    LLMCfg,
    ObservabilityCfg,
    RecipeCfg,
    RuntimeCfg,
)

__all__ = [
    "AugmentCfg",
    "CrossPageAlignCfg",
    "DatasetCfg",
    "EncoderCfg",
    "EvidenceCardsCfg",
    "GeneratorCfg",
    "IndexCfg",
    "IngestCfg",
    "KG4VDConfig",
    "KGCfg",
    "LLMCfg",
    "ObservabilityCfg",
    "RecipeCfg",
    "RuntimeCfg",
    "load_config",
    "resolve_overrides",
]
