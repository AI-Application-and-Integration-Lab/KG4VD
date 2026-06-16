"""Tracer with paper-grade accounting.

Every pipeline stage runs inside a ``Tracer.stage(name, **tags)`` context. The
tracer records:
  - prompt / completion / image / total tokens
  - LLM call count, cache hits / misses
  - wall-clock time
  - parent/child span hierarchy
  - arbitrary tags

Output:
  - JSONL sink (one record per stage exit, stream-friendly)
  - Optional Rich progress sink (live spinner) - shares one stderr Console
    with our log handler so the two don't clobber each other
  - In-memory stage tree (queryable for tests / reports)

The tracer is explicit: pipeline functions accept ``tracer`` as a parameter.
A contextvar-based "current tracer" is provided as a convenience for code paths
that can't easily thread it through (e.g. retry decorators).
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Shared Rich Console (used by logger + progress so they cooperate)
# ---------------------------------------------------------------------------

_SHARED_CONSOLE = None
_LOGGING_INSTALLED = False
_NOISY_LOGGERS = ("httpx", "httpcore", "openai", "urllib3", "asyncio")


def get_console():
    """Return the process-wide Rich Console (stderr-bound, lazy)."""
    global _SHARED_CONSOLE
    if _SHARED_CONSOLE is None:
        try:
            from rich.console import Console
            _SHARED_CONSOLE = Console(stderr=True, soft_wrap=False)
        except Exception:
            _SHARED_CONSOLE = None
    return _SHARED_CONSOLE


def install_rich_logging(level: int = logging.INFO) -> None:
    """Replace the root logger's handlers with a single RichHandler that
    routes through the shared Console, and quiet noisy 3rd-party libs.

    Idempotent - calling twice is harmless. Safe to call before a Tracer
    exists; the Tracer's progress later attaches to the same Console so
    log lines and the spinner don't fight.
    """
    global _LOGGING_INSTALLED
    if _LOGGING_INSTALLED:
        return
    console = get_console()
    if console is None:
        return
    try:
        from rich.logging import RichHandler
    except Exception:
        return
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=False,
        rich_tracebacks=False,
        log_time_format="[%H:%M:%S]",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    root.addHandler(handler)
    root.setLevel(level)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    _LOGGING_INSTALLED = True


# ---------------------------------------------------------------------------
# Stage record
# ---------------------------------------------------------------------------


@dataclass
class StageRecord:
    span_id: str
    parent_span_id: str | None
    stage: str
    start_ns: int
    end_ns: int = 0
    elapsed_ms: float = 0.0
    ok: bool = True
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    image_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    llm_cache_hits: int = 0
    llm_cache_misses: int = 0
    tags: dict[str, Any] = field(default_factory=dict)

    def to_record(self, trace_id: str, run_id: str) -> dict[str, Any]:
        d = asdict(self)
        d.update({
            "type": "stage_timing",
            "trace_id": trace_id,
            "run_id": run_id,
        })
        return d

    def add_llm_usage(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        image_tokens: int = 0,
        cache_hit: bool = False,
    ) -> None:
        self.llm_calls += 1
        if cache_hit:
            self.llm_cache_hits += 1
        else:
            self.llm_cache_misses += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.image_tokens += image_tokens
        self.total_tokens += prompt_tokens + completion_tokens + image_tokens


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


_CURRENT_TRACER: contextvars.ContextVar["Tracer | None"] = contextvars.ContextVar(
    "kg4vd_current_tracer", default=None
)
_SPAN_STACK: contextvars.ContextVar[tuple[StageRecord, ...]] = contextvars.ContextVar(
    "kg4vd_span_stack", default=()
)


class Tracer:
    """Per-run tracer. Construct one and pass it through the pipeline."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        trace_id: str | None = None,
        jsonl_path: str | Path | None = None,
        rich_progress: bool = False,
        write_jsonl: bool = True,
        title: str | None = None,
    ):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.trace_id = trace_id or self.run_id
        self.records: list[StageRecord] = []
        self._lock = threading.Lock()
        self._jsonl_fp = None
        self._rich_progress = None
        self._live = None
        self._progress_title = title or "kg4vd build"
        self._span_to_task: dict[str, int] = {}

        if write_jsonl and jsonl_path is not None:
            jsonl_path = Path(jsonl_path)
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            # line-buffered so each stage appears immediately
            self._jsonl_fp = jsonl_path.open("a", encoding="utf-8", buffering=1)

        # Only stages with depth <= this limit get a spinner row in the
        # live region. Deeper stages (e.g. `extract.page`, `align.judge_one`,
        # the dozens of per-page LLM calls) are still recorded for the
        # manifest but stay off-screen so the display doesn't flood.
        # 0 = only the build root; 1 = root + one level; etc.
        try:
            self._progress_depth = int(
                os.getenv("KG4VD_PROGRESS_DEPTH", "1")
            )
        except ValueError:
            self._progress_depth = 1

        if rich_progress:
            self._rich_progress = self._maybe_make_rich_progress()

    # ----- progress sink (optional) ----------------------------------------

    def _maybe_make_rich_progress(self):
        if os.getenv("KG4VD_PROGRESS", "1") in {"0", "false", "False"}:
            return None
        try:
            from rich.progress import (
                BarColumn,
                Progress,
                ProgressColumn,
                TextColumn,
                TimeElapsedColumn,
            )
            from rich.spinner import Spinner
            from rich.text import Text
        except Exception:
            return None
        console = get_console()
        if console is None:
            return None
        # Make sure logs route through the same console; otherwise the
        # built-in stderr logger would draw over the live region.
        install_rich_logging()

        class _StatusColumn(ProgressColumn):
            """Leading glyph: animated spinner while running, ✓/✗ when done."""

            def __init__(self) -> None:
                super().__init__()
                self._spinner = Spinner("dots", style="cyan")

            def render(self, task):  # noqa: ANN001
                state = task.fields.get("state", "run")
                if state == "ok":
                    return Text(" ✓", style="bold green")
                if state == "err":
                    return Text(" ✗", style="bold red")
                return self._spinner.render(task.get_time())

        class _CountColumn(ProgressColumn):
            """Right-of-bar text: ``done/total pct%`` when the stage has a
            known total, otherwise the dim tag summary."""

            def render(self, task):  # noqa: ANN001
                # Only stages opened with a real `total` show a count/%; the
                # rest (finished via a 1/1 sentinel) keep their tag summary.
                if task.fields.get("determinate") and task.total:
                    return Text(
                        f"{int(task.completed)}/{int(task.total)}"
                        f" {task.percentage:>3.0f}%",
                        style="green" if task.finished else "cyan",
                    )
                return Text(task.fields.get("tagstr", ""), style="dim")

        p = Progress(
            _StatusColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(
                bar_width=20,
                style="grey37",
                complete_style="cyan",
                finished_style="green",
                pulse_style="cyan",
            ),
            _CountColumn(),
            TextColumn("[dim]·"),
            TimeElapsedColumn(),
            console=console,
            expand=False,
        )
        # Render the bars inside a rounded, titled panel via a Live region.
        # (We drive the refresh through Live rather than Progress.start() so
        # the panel border redraws too.) Logs still print above the panel
        # because they route through the same console.
        try:
            from rich.box import ROUNDED
            from rich.live import Live
            from rich.panel import Panel

            panel = Panel(
                p,
                title=f"[bold]{self._progress_title}[/]",
                title_align="left",
                border_style="cyan",
                box=ROUNDED,
                padding=(0, 1),
                expand=False,
            )
            self._live = Live(
                panel, console=console, refresh_per_second=12, transient=False
            )
            self._live.start()
        except Exception:
            # Fall back to a bare (panel-less) progress if Live/Panel is
            # unavailable for any reason.
            self._live = None
            p.start()
        return p

    # ----- public API ------------------------------------------------------

    def __enter__(self) -> "Tracer":
        token = _CURRENT_TRACER.set(self)
        self._enter_token = token
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _CURRENT_TRACER.reset(self._enter_token)
        self.close()

    def close(self) -> None:
        if self._jsonl_fp is not None:
            try:
                self._jsonl_fp.close()
            finally:
                self._jsonl_fp = None
        if self._live is not None:
            # Panel path: the Progress is rendered by our Live, not started
            # on its own - just stop the Live.
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
            self._rich_progress = None
        elif self._rich_progress is not None:
            # Fallback path: the Progress was started directly.
            try:
                self._rich_progress.stop()
            except Exception:
                pass
            self._rich_progress = None

    @contextmanager
    def stage(self, name: str, **tags: Any) -> Iterator[StageRecord]:
        """Context manager that opens / closes a stage span.

        Pass ``total=<int>`` to render a determinate progress bar that fills
        as direct child spans complete (e.g. ``extract`` over its per-page
        children); stages without a total show a pulsing bar.
        """

        total = tags.pop("total", None)
        stack = _SPAN_STACK.get()
        parent = stack[-1] if stack else None
        depth = len(stack)
        rec = StageRecord(
            span_id=uuid.uuid4().hex[:12],
            parent_span_id=parent.span_id if parent else None,
            stage=name,
            start_ns=time.perf_counter_ns(),
            tags=dict(tags),
        )
        token = _SPAN_STACK.set(stack + (rec,))

        rich_task_id = None
        if (
            self._rich_progress is not None
            and depth <= self._progress_depth
        ):
            try:
                rich_task_id = self._rich_progress.add_task(
                    self._fmt_active_desc(name, depth),
                    total=total,
                    tagstr=self._fmt_tags(tags),
                    state="run",
                    determinate=total is not None,
                )
                self._span_to_task[rec.span_id] = rich_task_id
            except Exception:
                rich_task_id = None

        try:
            yield rec
            rec.ok = True
        except Exception as e:  # noqa: BLE001
            rec.ok = False
            rec.error = repr(e)
            raise
        finally:
            rec.end_ns = time.perf_counter_ns()
            rec.elapsed_ms = (rec.end_ns - rec.start_ns) / 1e6
            with self._lock:
                self.records.append(rec)
                if self._jsonl_fp is not None:
                    self._jsonl_fp.write(
                        json.dumps(
                            rec.to_record(self.trace_id, self.run_id),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            _SPAN_STACK.reset(token)

            if self._rich_progress is not None:
                # A direct child finishing advances its parent's bar. This is
                # a no-op visually for indeterminate parents (they keep
                # pulsing) and fills determinate ones - e.g. each `extract.page`
                # advances the `extract` bar toward its `total`.
                if rec.parent_span_id:
                    ptask = self._span_to_task.get(rec.parent_span_id)
                    if ptask is not None:
                        try:
                            self._rich_progress.advance(ptask, 1)
                        except Exception:
                            pass
                # Finish this stage's own row: solid bar + ✓/✗, kept on screen.
                if rich_task_id is not None:
                    try:
                        state = "ok" if rec.ok else "err"
                        if total is not None:
                            self._rich_progress.update(
                                rich_task_id, completed=total, state=state
                            )
                        else:
                            self._rich_progress.update(
                                rich_task_id, total=1, completed=1, state=state
                            )
                    except Exception:
                        pass
                    self._span_to_task.pop(rec.span_id, None)

    # ----- formatting helpers ---------------------------------------------

    # Tag keys that read naturally as "<value> <noun>" in the progress UI.
    _TAG_UNITS = {
        "n_pages": "pages",
        "n_pdfs": "pdfs",
        "n_summaries": "summaries",
        "n_nodes": "nodes",
        "n_edges": "edges",
        "n_cards": "cards",
        "n_chunks": "chunks",
        "merged": "merged",
    }

    @classmethod
    def _fmt_tags(cls, tags: dict[str, Any]) -> str:
        if not tags:
            return ""
        parts = []
        for k, v in tags.items():
            if v is None or v == "":
                continue
            sv = str(v)
            if len(sv) > 40:
                sv = sv[:37] + "..."
            unit = cls._TAG_UNITS.get(k)
            parts.append(f"{sv} {unit}" if unit else f"{k}={sv}")
        return " · ".join(parts)

    def _fmt_active_desc(self, name: str, depth: int) -> str:
        # Indent by depth and pad to a fixed width so the bars line up
        # in a column regardless of stage-name length.
        indent = "  " * depth
        return f"{indent}{name}".ljust(16)

    # ----- introspection ---------------------------------------------------

    def totals(self) -> dict[str, Any]:
        """Roll up tokens / calls / wall time across all recorded stages."""

        with self._lock:
            recs = list(self.records)
        agg = {
            "total_stages": len(recs),
            "total_wall_ms": sum(r.elapsed_ms for r in recs),
            "prompt_tokens": sum(r.prompt_tokens for r in recs),
            "completion_tokens": sum(r.completion_tokens for r in recs),
            "image_tokens": sum(r.image_tokens for r in recs),
            "total_tokens": sum(r.total_tokens for r in recs),
            "llm_calls": sum(r.llm_calls for r in recs),
            "llm_cache_hits": sum(r.llm_cache_hits for r in recs),
            "llm_cache_misses": sum(r.llm_cache_misses for r in recs),
            "errors": sum(0 if r.ok else 1 for r in recs),
        }
        per_stage: dict[str, dict[str, Any]] = {}
        for r in recs:
            slot = per_stage.setdefault(
                r.stage,
                {
                    "calls": 0, "wall_ms": 0.0,
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "image_tokens": 0, "total_tokens": 0,
                    "llm_calls": 0,
                },
            )
            slot["calls"] += 1
            slot["wall_ms"] += r.elapsed_ms
            slot["prompt_tokens"] += r.prompt_tokens
            slot["completion_tokens"] += r.completion_tokens
            slot["image_tokens"] += r.image_tokens
            slot["total_tokens"] += r.total_tokens
            slot["llm_calls"] += r.llm_calls
        agg["per_stage"] = per_stage
        return agg


# ---------------------------------------------------------------------------
# trace.jsonl aggregation (cumulative across invocations)
# ---------------------------------------------------------------------------


def aggregate_trace(trace_path: str | Path) -> dict[str, Any]:
    """Roll up per-stage totals from a ``trace.jsonl`` file.

    The run manifest reflects only the most recent ``kg4vd build`` invocation
    (it is rewritten each run), so a build done in separate ``--stages`` passes
    leaves a manifest covering just the last pass. ``trace.jsonl`` instead
    *appends* every span across every invocation, so aggregating it gives the
    true cumulative totals. Output matches ``Tracer.totals()`` exactly (summed
    per distinct stage name; child spans like ``kg.extract.init`` keep their
    own slot, which is where the LLM tokens are recorded).
    """
    path = Path(trace_path)
    fields = (
        "prompt_tokens", "completion_tokens", "image_tokens",
        "total_tokens", "llm_calls", "llm_cache_hits", "llm_cache_misses",
    )
    recs: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "stage_timing":
                recs.append(d)

    agg: dict[str, Any] = {
        "total_stages": len(recs),
        "total_wall_ms": sum(r.get("elapsed_ms", 0.0) for r in recs),
        "errors": sum(0 if r.get("ok", True) else 1 for r in recs),
    }
    for fld in fields:
        agg[fld] = sum(r.get(fld, 0) for r in recs)
    per_stage: dict[str, dict[str, Any]] = {}
    for r in recs:
        slot = per_stage.setdefault(
            r.get("stage", "?"),
            {"calls": 0, "wall_ms": 0.0, "prompt_tokens": 0,
             "completion_tokens": 0, "image_tokens": 0,
             "total_tokens": 0, "llm_calls": 0},
        )
        slot["calls"] += 1
        slot["wall_ms"] += r.get("elapsed_ms", 0.0)
        for fld in ("prompt_tokens", "completion_tokens", "image_tokens",
                    "total_tokens", "llm_calls"):
            slot[fld] += r.get(fld, 0)
    agg["per_stage"] = per_stage
    return agg


# ---------------------------------------------------------------------------
# contextvar accessors
# ---------------------------------------------------------------------------


def get_current_tracer() -> Tracer | None:
    return _CURRENT_TRACER.get()


def set_current_tracer(tracer: Tracer | None):
    return _CURRENT_TRACER.set(tracer)


def record_llm_usage(resp: Any) -> None:
    """Attribute an ``LLMResponse``'s token usage to the open stage.

    Called from the LLM client after every completion, so per-stage token /
    llm-call totals in the run manifest reflect real usage. No-op when called
    outside any stage (e.g. setup code) or with tracing disabled.
    """
    t = get_current_tracer()
    if t is None:
        return
    stack = _SPAN_STACK.get()
    rec = stack[-1] if stack else (t.records[-1] if t.records else None)
    if rec is None:
        return
    rec.add_llm_usage(
        prompt_tokens=getattr(resp, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(resp, "completion_tokens", 0) or 0,
        image_tokens=getattr(resp, "image_tokens", 0) or 0,
        cache_hit=bool(getattr(resp, "cache_hit", False)),
    )


@contextmanager
def stage(name: str, **tags: Any) -> Iterator[StageRecord]:
    """Convenience: open a stage on the current tracer if one is active.

    Falls back to a no-op record if no tracer is in scope. This lets code
    instrument itself defensively without forcing every caller to set up a
    tracer.
    """

    t = get_current_tracer()
    if t is None:
        rec = StageRecord(
            span_id=uuid.uuid4().hex[:12],
            parent_span_id=None,
            stage=name,
            start_ns=time.perf_counter_ns(),
            tags=dict(tags),
        )
        try:
            yield rec
        finally:
            rec.end_ns = time.perf_counter_ns()
            rec.elapsed_ms = (rec.end_ns - rec.start_ns) / 1e6
        return
    with t.stage(name, **tags) as rec:
        yield rec
