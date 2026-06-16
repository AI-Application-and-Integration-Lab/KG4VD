from __future__ import annotations

from kg4vd.config.schema import IngestCfg


def build_ingest(cfg: IngestCfg):
    if cfg.parser == "pypdfium":
        from kg4vd.ingest.pypdfium_parser import PyPdfiumIngest
        return PyPdfiumIngest(cfg)
    if cfg.parser == "mineru":
        from kg4vd.ingest.mineru_parser import MinerUIngest
        return MinerUIngest(cfg)
    raise ValueError(f"Unknown ingest parser: {cfg.parser!r}")
