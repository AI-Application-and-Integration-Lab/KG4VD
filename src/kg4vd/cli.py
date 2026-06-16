"""Command line interface for KG4VD."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from kg4vd.config import load_config
from kg4vd.obs import aggregate_trace, get_console, install_rich_logging
from kg4vd.pipeline import run_build
from kg4vd.pipeline.build import ALL_STAGES, OPTIONAL_STAGES

# Avoid noisy tokenizer fork warnings after the GME encoder has loaded.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Keep Rich logs and progress output on one console.
_log_level = getattr(
    logging, os.environ.get("KG4VD_LOG", "INFO").upper(), logging.INFO
)
install_rich_logging(level=_log_level)

app = typer.Typer(help="Build and query multimodal document knowledge graphs.")
console = get_console() or Console()


def _version_cb(value: bool) -> None:
    if value:
        from kg4vd import __version__

        console.print(f"kg4vd {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=_version_cb, is_eager=True,
        help="Show the kg4vd version and exit.",
    ),
) -> None:
    """Build and query multimodal document knowledge graphs."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_overrides(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(
                f"--set expects key=value (got: {item!r})"
            )
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _load(recipe: Path, set_: list[str]) -> "KG4VDConfig":  # noqa: F821
    overrides = _parse_overrides(set_) if set_ else None
    return load_config(recipe, overrides=overrides)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def build(
    recipe: Path = typer.Argument(..., exists=True, dir_okay=False),
    pdf: List[Path] = typer.Option(
        None, "--pdf", help="PDF path. Repeat for multiple files.",
    ),
    stages: Optional[str] = typer.Option(
        None, "--stages",
        help=(
            f"Comma-separated stages: {','.join(ALL_STAGES + OPTIONAL_STAGES)}"
        ),
    ),
    resume: bool = typer.Option(False, "--resume", help="Reuse existing artifacts."),
    set_: List[str] = typer.Option([], "--set", help="Override config: key=value."),
):
    """Build artifacts for a recipe."""

    cfg = _load(recipe, set_)
    if not pdf:
        data_dir = Path(cfg.dataset.data_dir)
        pdf = sorted(data_dir.glob("*.pdf"))
        if not pdf:
            raise typer.BadParameter(
                f"No PDFs found in {data_dir}; pass --pdf explicitly."
            )

    stages_tup = stages.split(",") if stages else None

    _print_run_header(cfg, recipe, pdf, stages_tup, resume)

    art = asyncio.run(
        run_build(cfg, pdf_paths=pdf, stages=stages_tup, resume=resume)
    )
    _print_build_summary(art, cfg)


@app.command()
def align_embed(
    recipe: Path = typer.Argument(..., exists=True, dir_okay=False),
    set_: List[str] = typer.Option([], "--set", help="Override config: key=value."),
):
    """Precompute node embeddings for the align stage."""
    from kg4vd.pipeline import precompute_align_embeddings

    cfg = _load(recipe, set_)
    out = asyncio.run(precompute_align_embeddings(cfg))
    console.print(f"[green]✓[/] wrote node embeddings → {out}")


def _print_run_header(cfg, recipe, pdfs, stages, resume) -> None:
    from rich.panel import Panel
    from rich.table import Table

    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim")
    t.add_column()
    t.add_row("recipe   :", str(recipe))
    t.add_row("dataset  :", f"{cfg.dataset.name}  →  {cfg.dataset.work_dir}")
    t.add_row(
        "llm      :",
        f"{cfg.generator.llm.kind}:{cfg.generator.llm.model}",
    )
    t.add_row(
        "encoder  :",
        f"{cfg.encoder.name}  (dim={cfg.encoder.dim})",
    )
    t.add_row(
        "stages   :",
        ",".join(stages) if stages else "all",
    )
    t.add_row("pdfs     :", ", ".join(p.name for p in pdfs))
    from kg4vd.kg.prompts import PROMPTS_VERSION
    t.add_row(
        "options  :",
        f"resume={resume}  prompt_set={PROMPTS_VERSION}",
    )
    console.print(Panel(t, title="KG4VD build", border_style="cyan"))


def _print_build_summary(art, cfg) -> None:
    from rich.table import Table

    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(justify="right")
    t.add_row("pages",  str(len(art.pages)))
    t.add_row("nodes",  str(len(art.nodes)))
    t.add_row("edges",  str(len(art.edges)))
    t.add_row("cards",  str(len(art.cards)))
    if art.failed_chunks:
        t.add_row(
            "[yellow]failed chunks[/yellow]",
            f"[yellow]{len(art.failed_chunks)}[/yellow]",
        )
    if art.manifest and getattr(art.manifest, "totals", None):
        totals = art.manifest.totals or {}
        if totals.get("total_tokens"):
            t.add_row("tokens", f"{totals.get('total_tokens', 0):,}")
        if totals.get("llm_calls"):
            t.add_row("llm calls", str(totals.get("llm_calls", 0)))
        if totals.get("total_wall_ms"):
            t.add_row(
                "wall clock",
                f"{totals.get('total_wall_ms', 0)/1000:.1f}s",
            )
    console.print()
    console.print("[bold green]✓ Build done[/bold green]")
    console.print(t)
    if art.manifest:
        console.print(
            f"[dim]manifest:[/dim] "
            f"{Path(cfg.dataset.work_dir) / cfg.obs.manifest_path}"
        )


@app.command()
def show_config(
    recipe: Path = typer.Argument(..., exists=True, dir_okay=False),
    set_: List[str] = typer.Option([], "--set"),
    format_: str = typer.Option(
        "json", "--format", "-f",
        help="Output format: json or yaml.",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o",
        help="Write to a file instead of stdout.",
    ),
):
    """Show the resolved recipe config."""

    cfg = _load(recipe, set_)
    data = cfg.model_dump(mode="json")

    # Infer format from --out extension when caller didn't override.
    fmt = format_.lower()
    if out is not None and fmt == "json" and out.suffix.lower() in {".yaml", ".yml"}:
        fmt = "yaml"

    if fmt == "yaml":
        import yaml
        text = yaml.safe_dump(
            data, sort_keys=False, allow_unicode=True, default_flow_style=False,
        )
    elif fmt == "json":
        text = json.dumps(data, indent=2, ensure_ascii=False)
    else:
        raise typer.BadParameter(f"Unknown --format {format_!r}; expected 'json' or 'yaml'.")

    header = (
        f"# Resolved config for {recipe}\n"
        f"# config_hash: {cfg.config_hash()}\n"
        f"#\n"
        f"# Fully-expanded recipe snapshot. Generated by:\n"
        f"#   kg4vd show-config {recipe} -f yaml -o {out or '<path>'}\n"
    )
    if out is not None:
        if fmt == "yaml":
            out.write_text(header + text, encoding="utf-8")
        else:
            out.write_text(text + "\n", encoding="utf-8")
        console.print(f"Wrote resolved config to {out}  (format={fmt}, hash={cfg.config_hash()})")
    else:
        if fmt == "json":
            console.print_json(data=data)
        else:
            console.print(text)
        console.print(f"[bold]config_hash[/bold]: {cfg.config_hash()}")


@app.command()
def list_plugins():
    """List registered plugins."""

    from kg4vd.core.registry import Registry
    for kind in ("encoder", "index", "llm"):
        names = Registry.list(kind)
        console.print(f"[bold]{kind}[/bold]: {names}")


@app.command()
def report(
    recipe: Path = typer.Argument(..., exists=True, dir_okay=False),
    kind: str = typer.Option("totals", "--kind", help="totals | per_stage | csv"),
    set_: List[str] = typer.Option([], "--set"),
):
    """Summarize run cost and timing."""

    cfg = _load(recipe, set_)
    work_dir = Path(cfg.dataset.work_dir)
    trace_path = work_dir / cfg.obs.trace_path
    manifest_path = work_dir / cfg.obs.manifest_path
    if trace_path.is_file():
        totals = aggregate_trace(trace_path)
        source = f"{trace_path.name} - cumulative across all invocations"
    elif manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as f:
            totals = json.load(f).get("totals", {})
        source = f"{manifest_path.name} - most recent invocation only"
    else:
        raise typer.BadParameter(
            f"No trace.jsonl or manifest under {work_dir}; run `kg4vd build` first."
        )

    if kind == "totals":
        console.print(f"[dim]source: {source}[/]")
        console.print_json(data=totals)
    elif kind == "per_stage":
        per_stage = totals.get("per_stage", {})
        table = Table(title=f"Per-stage totals · {source}")
        for col in ("stage", "calls", "wall_ms", "total_tokens", "llm_calls"):
            table.add_column(col)
        for s, d in sorted(per_stage.items()):
            table.add_row(s, str(d.get("calls", 0)), f"{d.get('wall_ms', 0):.1f}",
                          str(d.get("total_tokens", 0)), str(d.get("llm_calls", 0)))
        console.print(table)
    elif kind == "csv":
        per_stage = totals.get("per_stage", {})
        rows = ["stage,calls,wall_ms,prompt_tokens,completion_tokens,image_tokens,total_tokens,llm_calls"]
        for s, d in sorted(per_stage.items()):
            rows.append(
                f"{s},{d.get('calls', 0)},{d.get('wall_ms', 0):.1f},"
                f"{d.get('prompt_tokens', 0)},{d.get('completion_tokens', 0)},"
                f"{d.get('image_tokens', 0)},{d.get('total_tokens', 0)},"
                f"{d.get('llm_calls', 0)}"
            )
        console.print("\n".join(rows))
    else:
        raise typer.BadParameter("--kind must be one of: totals | per_stage | csv")


def _print_result(question: str, result, *, show_evidence: bool) -> None:
    from kg4vd.query import ContextBuilder

    ans = result.answer
    console.print(f"\n[bold cyan]Q[/] {question}")
    console.print(f"[bold]A[/] {ans.text or '(no answer)'}")
    console.print(
        f"[dim]cited pages: {ans.cited_pages}  ·  confidence: {ans.confidence}"
        f"  ·  mode: {result.diagnostics.get('answer_mode', '?')}"
        f"  ·  route: {result.diagnostics.get('route', '?')}[/]"
    )
    if show_evidence:
        console.print("[bold]Top evidence[/]")
        console.print(ContextBuilder.format_texts(result.text_items[:8]))


@app.command()
def query(
    recipe: Path = typer.Argument(..., exists=True, dir_okay=False),
    question: str = typer.Option(..., "--q", "-q", help="Question to answer."),
    show_evidence: bool = typer.Option(
        False, "--evidence", help="Print retrieved evidence."
    ),
    manage_sglang: bool = typer.Option(
        True, "--manage-sglang/--no-manage-sglang",
        help="Start/stop the sglang server when needed.",
    ),
    manage_reranker: bool = typer.Option(
        True, "--manage-reranker/--no-manage-reranker",
        help="Start/stop the reranker server when enabled.",
    ),
    set_: List[str] = typer.Option([], "--set"),
):
    """Answer one question."""
    from kg4vd.pipeline import run_query

    cfg = _load(recipe, set_)
    result = asyncio.run(run_query(
        cfg, question, manage_sglang=manage_sglang, manage_reranker=manage_reranker
    ))
    _print_result(question, result, show_evidence=show_evidence)


@app.command()
def query_batch(
    recipe: Path = typer.Argument(..., exists=True, dir_okay=False),
    input_path: Path = typer.Option(
        ..., "--input", "-i", exists=True, dir_okay=False,
        help="Question JSONL.",
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--out", "-o",
        help="Output JSONL.",
    ),
    question_field: str = typer.Option("question", "--question-field"),
    qid_field: str = typer.Option("qid", "--qid-field"),
    manage_sglang: bool = typer.Option(True, "--manage-sglang/--no-manage-sglang"),
    manage_reranker: bool = typer.Option(True, "--manage-reranker/--no-manage-reranker"),
    set_: List[str] = typer.Option([], "--set"),
):
    """Answer questions from JSONL."""
    from kg4vd.pipeline import run_query_batch

    cfg = _load(recipe, set_)
    items: list[tuple[str, str]] = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = row.get(question_field)
            if not q:
                continue
            items.append((str(row.get(qid_field) or ""), str(q)))
    if not items:
        raise typer.BadParameter(f"No questions found in {input_path}.")

    if output_path is None:
        output_path = Path(cfg.dataset.work_dir) / "answers" / f"{input_path.stem}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Answering {len(items)} questions…[/]")
    results = asyncio.run(run_query_batch(
        cfg, items, manage_sglang=manage_sglang, manage_reranker=manage_reranker
    ))
    with output_path.open("w", encoding="utf-8") as f:
        for (qid, question), result in zip(items, results):
            f.write(json.dumps({
                "qid": qid,
                "question": question,
                "answer": result.answer.text,
                "cited_pages": result.answer.cited_pages,
                "confidence": result.answer.confidence,
                "answer_mode": result.diagnostics.get("answer_mode"),
                "route": result.diagnostics.get("route"),
            }, ensure_ascii=False) + "\n")
    console.print(f"[green]✓[/] wrote {len(results)} answers → {output_path}")


if __name__ == "__main__":
    app()
