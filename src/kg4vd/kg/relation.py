"""Shared relation-name normalisation.

Both the per-page extractor (`kg/extract/adaptive.py`) and the
cross-page aligner (`kg/align/cross_page.py`) take an LLM-provided
relation string and have to coerce it into a canonical form before
emitting an edge. Without a shared helper the two paths drift:
extract used to emit `"applies to"` while align emitted `"applies_to"`
between the same pair, and dedup in `_collapse_near_dupe_edges`
treats them as distinct buckets only because the raw strings are
different. Routing both through this helper keeps the vocabulary
single-source-of-truth.
"""

from __future__ import annotations

from typing import Any

# Characters that are never valid inside a snake_case relation token -
# whitespace, punctuation, brackets. Replaced with `_`.
_REL_BAD_CHARS = set(" \t/\\(){}[]'\"`,;:!?")


def normalize_relation(raw: Any) -> str:
    """Canonicalise an LLM-provided relation string to snake_case.

    Rules:
      - lowercase
      - punctuation/whitespace/hyphens → underscore
      - collapse runs of underscores
      - strip leading/trailing underscores
      - reject empty input or >80 char relations (returns "" → caller
        drops the edge)

    Examples:
      "applies to"            → "applies_to"
      "Is Part Of"            → "is_part_of"
      "depicts"               → "depicts"
      "is-related-to"         → "is_related_to"
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip().lower()
    if not s:
        return ""
    buf = []
    for ch in s:
        if ch in _REL_BAD_CHARS or ch == "-":
            buf.append("_")
        else:
            buf.append(ch)
    out = "".join(buf)
    while "__" in out:
        out = out.replace("__", "_")
    out = out.strip("_")
    if not out or len(out) > 80:
        return ""
    return out
