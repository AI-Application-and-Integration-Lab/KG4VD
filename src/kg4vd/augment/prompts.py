"""Page-summary prompt (augment stage)."""

from __future__ import annotations

from textwrap import dedent


PAGE_SUMMARY = dedent(
    """
    Summarise this single page of a visually rich document. Mention any
    figures, charts, diagrams, or tables explicitly. Limit to ~5 sentences.

    Page text:
    {page_text}
    """
).strip()
