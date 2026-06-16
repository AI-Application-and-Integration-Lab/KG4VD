"""Pure-Python PDF parser using pypdfium2."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pypdfium2 as pdfium

from kg4vd.config.schema import IngestCfg
from kg4vd.core.types import Page

logger = logging.getLogger(__name__)


class PyPdfiumIngest:
    name = "pypdfium"

    def __init__(self, cfg: IngestCfg):
        self.cfg = cfg

    async def ingest(self, pdf_path: str | Path, *, work_dir: str | Path) -> list[Page]:
        pdf_path = Path(pdf_path)
        work_dir = Path(work_dir)
        out_dir = work_dir / self.cfg.output_subdir / pdf_path.stem / "pages"
        out_dir.mkdir(parents=True, exist_ok=True)

        return await asyncio.to_thread(
            self._ingest_sync, pdf_path, out_dir
        )

    def _ingest_sync(self, pdf_path: Path, out_dir: Path) -> list[Page]:
        pdf = pdfium.PdfDocument(str(pdf_path))
        pages: list[Page] = []
        try:
            for i, page_obj in enumerate(pdf):
                page_id = i + 1
                # Render page image
                scale = self.cfg.dpi / 72.0
                bitmap = page_obj.render(scale=scale).to_pil()
                # Resize the long side to max_image_long_side.
                w, h = bitmap.size
                m = max(w, h)
                if m > self.cfg.max_image_long_side:
                    s = self.cfg.max_image_long_side / m
                    bitmap = bitmap.resize((max(1, int(w * s)), max(1, int(h * s))))
                img_path = out_dir / f"page_{page_id:04d}.jpg"
                bitmap.convert("RGB").save(
                    img_path, format="JPEG", quality=self.cfg.jpeg_quality, optimize=True
                )

                # Extract text
                try:
                    textpage = page_obj.get_textpage()
                    text = textpage.get_text_range()
                    textpage.close()
                except Exception:
                    text = ""

                pages.append(
                    Page(
                        doc_id=pdf_path.stem,
                        page_id=page_id,
                        text=text,
                        page_image_path=str(img_path),
                        figure_image_paths=[],
                        metadata={"total_pages": pdf.__len__()},
                    )
                )
                page_obj.close()
        finally:
            pdf.close()

        logger.info("Ingested %d pages from %s", len(pages), pdf_path)
        return pages
