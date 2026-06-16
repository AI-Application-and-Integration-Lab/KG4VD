"""Page-summary stage.

Each page summary is generated from `(page_text, page_image)` so the VLM can
notice charts/diagrams that the text alone misses. The summary text lives on
the Page and feeds the cross-page aligner + the downstream query answer.
"""

from __future__ import annotations

import asyncio
from typing import Any

from kg4vd.augment.prompts import PAGE_SUMMARY
from kg4vd.core.types import Page
from kg4vd.kg.prompts import PROMPTS_VERSION


async def summarize_pages(
    pages: list[Page],
    *,
    llm: Any,
    max_async: int = 8,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> list[Page]:
    sem = asyncio.Semaphore(max_async)

    async def _one(p: Page) -> Page:
        async with sem:
            prompt = PAGE_SUMMARY.format(
                page_text=(p.text or "")[:6000] or "(no text)",
            )
            images = [p.page_image_path] if p.page_image_path else None
            resp = await llm.acomplete(
                prompt,
                system=f"prompt-set: {PROMPTS_VERSION}",
                images=images,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return p.model_copy(update={"page_summary": resp.text.strip()})

    return await asyncio.gather(*[_one(p) for p in pages])
