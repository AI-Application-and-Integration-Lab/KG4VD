"""MinerU-based PDF parser.

Two-stage adapter:

  1. ``_run_mineru(pdf_path, out_root)`` - parse the PDF with MinerU 3.x.
     MinerU needs transformers>=4.57 (incompatible with the GME encoder's
     env), so it runs out-of-process: we invoke ``services/mineru/run.py`` with
     the MinerU env's python (``MINERU_PYTHON``), which calls MinerU's Python
     API (``do_parse``) directly - not the CLI binary. Produces
     ``{out_root}/{pdf_stem}/auto/{pdf_stem}_middle.json`` plus image crops.
  2. ``build_pages_from_mineru_output(...)`` - pure post-processing.
     For each page in middle.json:
        - Render the page image at fixed DPI via pypdfium2.
        - Build typed `Component` objects; merge close textual
          siblings to keep label circles legible.
        - Render the annotated page PNG (boxes + circle labels).
        - Write `components.json` and `page.json` next to the
          artefacts.
        - Emit a `Page` with all the metadata the component-cued
          extractor needs (`components_path`, `annotated_image_path`,
          `page_size_*` triple, etc.).

The split lets tests exercise the post-processing logic against a
checked-in fixture middle.json without requiring MinerU to be installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from kg4vd.config.schema import IngestCfg
from kg4vd.core.errors import PipelineError
from kg4vd.core.types import Page
from kg4vd.ingest.components import (
    annotate_spatial,
    build_components_from_middle,
    merge_close_components,
)
from kg4vd.kg.extract.annotate import annotate_page

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public adapter
# ---------------------------------------------------------------------------


class MinerUIngest:
    name = "mineru"

    def __init__(self, cfg: IngestCfg):
        self.cfg = cfg

    async def ingest(
        self, pdf_path: str | Path, *, work_dir: str | Path
    ) -> list[Page]:
        pdf_path = Path(pdf_path)
        work_dir = Path(work_dir)
        out_root = work_dir / self.cfg.output_subdir / pdf_path.stem
        out_root.mkdir(parents=True, exist_ok=True)

        # 1. Run MinerU (subprocess; offload from the event loop)
        await asyncio.to_thread(_run_mineru, pdf_path, out_root)

        # 2. Post-process: build per-page Pages + on-disk artefacts
        pages = await asyncio.to_thread(
            build_pages_from_mineru_output,
            pdf_path=pdf_path,
            mineru_out_root=out_root,
            pages_out_dir=work_dir / "pages",
            cfg=self.cfg,
        )
        logger.info("Ingested %d pages from %s via MinerU", len(pages), pdf_path)
        return pages


# ---------------------------------------------------------------------------
# Stage 1 - invoke MinerU 3.x (out-of-process, via its Python API runner)
# ---------------------------------------------------------------------------

_DEFAULT_MINERU_RUNNER = "services/mineru/run.py"


def _run_mineru(pdf_path: Path, out_root: Path) -> None:
    """Parse a PDF with MinerU 3.x via the out-of-process API runner.

    MinerU needs transformers>=4.57 (incompatible with the GME encoder's pin),
    so it lives in its own conda env and we shell out to ``services/mineru/run.py``
    with that env's python. The runner calls MinerU's ``do_parse`` API (not the
    CLI binary) and writes ``{out_root}/{stem}/auto/{stem}_middle.json``.

    Requires ``MINERU_PYTHON`` (path to the MinerU env's python; see
    ``services/mineru/README.md``). Override the runner with ``KG4VD_MINERU_RUNNER``.
    """
    python = os.environ.get("MINERU_PYTHON", "").strip()
    runner = os.environ.get("KG4VD_MINERU_RUNNER", _DEFAULT_MINERU_RUNNER)
    if not python:
        raise PipelineError(
            "MINERU_PYTHON is not set. Point it at a MinerU 3.x env's python "
            "(see services/mineru/README.md), e.g. in your .env."
        )
    if not Path(python).is_file():
        raise PipelineError(
            f"MinerU env python not found at {python!r}. Set MINERU_PYTHON to a "
            f"python in a MinerU 3.x env (see services/mineru/README.md)."
        )
    if not Path(runner).is_file():
        raise PipelineError(
            f"MinerU runner not found at {runner!r}. Set KG4VD_MINERU_RUNNER to "
            f"the path of services/mineru/run.py."
        )
    cmd = [python, runner, "--pdf", str(pdf_path), "--out", str(out_root), "--lang", "en"]
    logger.info("Running MinerU: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(
            f"MinerU runner exited with code {result.returncode}.\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )
    expected = out_root / pdf_path.stem / "auto" / f"{pdf_path.stem}_middle.json"
    if not expected.is_file():
        raise PipelineError(
            f"MinerU finished but the expected middle.json was not produced at "
            f"{expected}. Inspect {out_root} for what MinerU actually wrote."
        )


# ---------------------------------------------------------------------------
# Stage 2 - post-processing (pure; testable without subprocess)
# ---------------------------------------------------------------------------


def build_pages_from_mineru_output(
    *,
    pdf_path: Path,
    mineru_out_root: Path,
    pages_out_dir: Path,
    cfg: IngestCfg,
) -> list[Page]:
    """Turn a MinerU output dir + the original PDF into a list of
    `Page` objects with every artefact the component-cued extractor
    needs on disk.

    `mineru_out_root` is the directory MinerU wrote into (i.e. the
    ``-o`` argument of the previous CLI step). Inside it MinerU has
    placed ``{pdf_stem}/auto/{pdf_stem}_middle.json`` and friends.
    """
    middle_path = mineru_out_root / pdf_path.stem / "auto" / f"{pdf_path.stem}_middle.json"
    if not middle_path.is_file():
        raise PipelineError(
            f"middle.json not found at {middle_path}. "
            f"Did the MinerU CLI complete?"
        )

    middle = json.loads(middle_path.read_text())
    pdf_info = middle.get("pdf_info") or []
    if not pdf_info:
        raise PipelineError(
            f"middle.json at {middle_path} has no pdf_info - empty PDF?"
        )

    pages_out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    total_pages = len(pdf)
    pages: list[Page] = []
    try:
        for i, pinfo in enumerate(pdf_info):
            page_id = i + 1
            page_dir = pages_out_dir / f"page_{page_id:04d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            pages.append(_build_one_page(
                pdf, pinfo, page_id=page_id, page_idx=i,
                page_dir=page_dir, pdf_path=pdf_path, cfg=cfg,
                total_pages=total_pages,
            ))
    finally:
        pdf.close()
    return pages


def _build_one_page(
    pdf: pdfium.PdfDocument,
    pinfo: dict[str, Any],
    *,
    page_id: int,
    page_idx: int,
    page_dir: Path,
    pdf_path: Path,
    cfg: IngestCfg,
    total_pages: int,
) -> Page:
    page_obj = pdf[page_idx]
    page_size_pdf = page_obj.get_size()  # (w_pt, h_pt) in PDF points
    # Render page image at the configured DPI, capping the long side.
    bitmap = page_obj.render(scale=cfg.dpi / 72.0).to_pil()
    w, h = bitmap.size
    m = max(w, h)
    if m > cfg.max_image_long_side:
        s = cfg.max_image_long_side / m
        bitmap = bitmap.resize((max(1, int(w * s)), max(1, int(h * s))))
    page_image_path = page_dir / "page_image.png"
    bitmap.convert("RGB").save(page_image_path, format="PNG")
    render_size = bitmap.size

    # MinerU's blocks are in `page_size_mineru_px` coords.
    m_page_size = pinfo.get("page_size") or [page_size_pdf[0], page_size_pdf[1]]
    preproc_blocks = pinfo.get("preproc_blocks") or []

    # Build components from the MinerU middle.json blocks.
    components = build_components_from_middle(preproc_blocks)
    y_gap = max(20, int(m_page_size[1] * 0.03))
    components = merge_close_components(components, y_gap_threshold=y_gap)
    components = annotate_spatial(
        components, page_w=m_page_size[0], page_h=m_page_size[1],
    )

    # Persist components.json (the contract ComponentCuedExtractor reads).
    components_path = page_dir / "components.json"
    components_path.write_text(
        json.dumps([c.model_dump() for c in components],
                   indent=2, ensure_ascii=False)
    )

    # Render annotated page image with bboxes scaled to render space.
    sx, sy = render_size[0] / m_page_size[0], render_size[1] / m_page_size[1]
    scaled = [c.model_copy(update={
        "bbox": (c.bbox[0] * sx, c.bbox[1] * sy,
                 c.bbox[2] * sx, c.bbox[3] * sy),
    }) for c in components]
    annotated_path = page_dir / "annotated.png"
    annotate_page(
        page_image_path, scaled, annotated_path,
        expected_image_size=render_size,
    )

    # Page text - concatenate the OCR/text-layer spans.
    text_chunks: list[str] = []
    for blk in preproc_blocks:
        for line in blk.get("lines") or []:
            for span in line.get("spans") or []:
                if span.get("type") == "text" and span.get("content"):
                    text_chunks.append(span["content"])
    page_text = " ".join(text_chunks).strip()

    page = Page(
        doc_id=pdf_path.stem,
        page_id=page_id,
        text=page_text,
        page_image_path=str(page_image_path),
        figure_image_paths=[],
        metadata={
            "total_pages": total_pages,
            "mineru_page_idx": page_idx,
            "page_size_pdf": [float(page_size_pdf[0]), float(page_size_pdf[1])],
            "page_size_mineru_px": list(m_page_size),
            "page_size_render_px": list(render_size),
            "annotated_image_path": str(annotated_path),
            "components_path": str(components_path),
            "n_components": len(components),
        },
    )
    # Mirror the Page on disk for inspection / re-loading.
    (page_dir / "page.json").write_text(page.model_dump_json(indent=2))
    page_obj.close()
    return page


__all__ = [
    "MinerUIngest",
    "build_pages_from_mineru_output",
]
