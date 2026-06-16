"""Observability: tracer (with token + time accounting), run manifest."""

from kg4vd.obs.manifest import RunManifest
from kg4vd.obs.tracer import (
    StageRecord,
    Tracer,
    aggregate_trace,
    get_console,
    get_current_tracer,
    install_rich_logging,
    set_current_tracer,
)

__all__ = [
    "RunManifest",
    "StageRecord",
    "Tracer",
    "aggregate_trace",
    "get_console",
    "get_current_tracer",
    "install_rich_logging",
    "set_current_tracer",
]
