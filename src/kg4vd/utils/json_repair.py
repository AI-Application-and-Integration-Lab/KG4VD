"""Loose JSON parser used for LLM outputs.

Order of attempts:
  1. ``json.loads`` directly.
  2. Find the first ``[`` ... last ``]`` (or first ``{`` ... last ``}``) and
     try ``json.loads`` on that slice.
  3. ``json_repair.repair_json``.

Raises ``ValueError`` if all three fail.
"""

from __future__ import annotations

import json
from typing import Any

import json_repair


def parse_json_loose(text: str) -> Any:
    if not isinstance(text, str):
        return text

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Heuristic: pull a JSON array or object out of surrounding chatter.
    for opener, closer in (("[", "]"), ("{", "}")):
        s = text.find(opener)
        e = text.rfind(closer)
        if 0 <= s < e:
            try:
                return json.loads(text[s : e + 1])
            except json.JSONDecodeError:
                continue

    try:
        return json_repair.repair_json(text, return_objects=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not parse LLM output as JSON: {exc!r}") from exc


def parse_json_object_loose(text: str) -> dict:
    """Like ``parse_json_loose`` but always returns a ``dict``.

    LLMs occasionally drift from the expected ``{"key": ...}`` shape
    and emit a top-level array (just the ``ops`` list) or a scalar.
    Callers that immediately do ``obj.get("...")`` then crash with
    ``AttributeError: 'list' object has no attribute 'get'``,
    silently dropping the entire page/section. Normalises to:

      - dict          → returned as-is
      - single-elem list whose only entry is a dict → unwrapped
        (covers the common ``[{...}]`` mistake)
      - anything else (list with >1 elem, scalar, parse failure)
        → ``{}`` (graceful no-op; caller treats as empty patch)

    Use this anywhere a downstream caller assumes a dict shape. For
    callers that legitimately accept either shape (e.g. an aligner
    judge that handles both ``[...]`` and ``{"decisions": [...]}``),
    use ``parse_json_loose`` directly and switch on ``isinstance``
    at the call site.
    """
    try:
        obj = parse_json_loose(text)
    except ValueError:
        return {}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        return obj[0]
    return {}
