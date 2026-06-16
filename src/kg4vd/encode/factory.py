from __future__ import annotations

from kg4vd.config.schema import EncoderCfg
from kg4vd.core.registry import Registry


def build_encoder(cfg: EncoderCfg):
    cls = Registry.get("encoder", cfg.name)
    return cls(cfg)
