"""Run manifest.

A `run_manifest.json` is written next to each run output. It captures
everything needed to reproduce / cite a run:
  - config_hash, prompt_set_version
  - git_sha (best-effort; "unknown" if not in a git repo)
  - encoder / llm IDs
  - python / package version
  - dataset name + work_dir
  - totals from the tracer (tokens, time, llm_calls)
  - failed_chunks (from the build pipeline)
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kg4vd import __version__ as KG4VD_VERSION
from kg4vd.config.schema import KG4VDConfig
from kg4vd.kg.prompts import PROMPTS_VERSION


@dataclass
class RunManifest:
    run_id: str
    trace_id: str
    config_hash: str
    prompt_set_version: str
    dataset_name: str
    work_dir: str
    encoder_id: str
    index_backend: str
    llm_id: str
    git_sha: str
    git_dirty: bool
    kg4vd_version: str
    python_version: str
    platform: str
    started_at: str
    ended_at: str | None = None
    elapsed_s: float | None = None
    failed_chunks: list[dict[str, Any]] = field(default_factory=list)
    totals: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        cfg: KG4VDConfig,
        *,
        run_id: str,
        trace_id: str,
        started_at: str,
    ) -> "RunManifest":
        sha, dirty = _git_state()
        return cls(
            run_id=run_id,
            trace_id=trace_id,
            config_hash=cfg.config_hash(),
            # Read from the runtime constant in kg.prompts, NOT from
            # cfg.prompt_set_version (which is a stale schema default and
            # doesn't get auto-updated on prompt edits). The whole point
            # of the manifest is to identify which prompt set produced
            # this artefact -- it must reflect what actually shipped.
            prompt_set_version=PROMPTS_VERSION,
            dataset_name=cfg.dataset.name,
            work_dir=cfg.dataset.work_dir,
            encoder_id=f"{cfg.encoder.name}:{cfg.encoder.model or 'default'}",
            index_backend=cfg.index.backend,
            llm_id=f"{cfg.generator.llm.kind}:{cfg.generator.llm.model}",
            git_sha=sha,
            git_dirty=dirty,
            kg4vd_version=KG4VD_VERSION,
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            started_at=started_at,
        )

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_state() -> tuple[str, bool]:
    """Return (short_sha, is_dirty). Best effort - never raises."""

    if shutil.which("git") is None:
        return ("unknown", False)

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.getcwd(),
            timeout=2,
        ).decode().strip()
    except Exception:
        return ("unknown", False)

    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd(),
                timeout=2,
            ).decode().strip()
        )
    except Exception:
        dirty = False

    return (sha, dirty)
