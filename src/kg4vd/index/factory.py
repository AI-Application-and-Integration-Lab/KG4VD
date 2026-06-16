from __future__ import annotations

from kg4vd.config.schema import IndexCfg
from kg4vd.core.registry import Registry


def build_index(cfg: IndexCfg, *, dim: int):
    cls = Registry.get("index", cfg.backend)
    return cls(cfg, dim=dim)
