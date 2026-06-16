"""YAML loader with `extends:` composition + dotted-path overrides.

Recipe files compose smaller preset YAMLs::

    extends:
      - presets/base.yaml
      - presets/encoder.gme_qwen2vl.yaml
      - presets/llm.openrouter.gpt4omini.yaml

    dataset:
      name: my_dataset
      ...

`extends` is processed in order, with later files overriding earlier keys
(deep merge for dicts, replace for lists). The recipe's own keys override
everything from `extends`.

CLI / programmatic overrides are applied last via ``resolve_overrides``::

    cfg = load_config("recipe.yaml", overrides={"encoder.name": "gme_qwen2vl"})
"""

from __future__ import annotations

from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from kg4vd.config.schema import KG4VDConfig
from kg4vd.core.errors import ConfigError


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> KG4VDConfig:
    """Load a recipe YAML and resolve all `extends:` refs and overrides."""

    path = Path(path).resolve()
    if not path.is_file():
        raise ConfigError(f"Recipe not found: {path}")

    raw = _load_with_extends(path, _seen=set())

    if overrides:
        raw = resolve_overrides(raw, overrides)

    try:
        return KG4VDConfig.model_validate(raw)
    except Exception as e:  # pydantic ValidationError
        raise ConfigError(f"Invalid config in {path}: {e}") from e


def resolve_overrides(
    raw: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Apply dotted-path overrides to a raw config dict.

    Supports either::

        {"encoder.name": "gme_qwen2vl", "retrieval.candidate_pool": 120}

    or already-nested dicts (deep-merged).
    """

    result = deepcopy(raw)
    for key, value in overrides.items():
        if "." in key:
            _set_dotted(result, key, value)
        elif isinstance(value, dict):
            result[key] = _deep_merge(result.get(key, {}), value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_with_extends(path: Path, *, _seen: set[Path]) -> dict[str, Any]:
    if path in _seen:
        raise ConfigError(f"Circular extends including {path}")
    _seen = _seen | {path}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a mapping at top level of {path}")

    extends = data.pop("extends", [])
    if extends and not isinstance(extends, list):
        raise ConfigError(f"`extends` must be a list in {path}")

    base: dict[str, Any] = {}
    for ext in extends:
        ext_path = _resolve_extends_path(ext, anchor=path)
        ext_data = _load_with_extends(ext_path, _seen=_seen)
        base = _deep_merge(base, ext_data)

    return _deep_merge(base, data)


def _resolve_extends_path(ext: str, *, anchor: Path) -> Path:
    """Resolve an `extends:` reference.

    Order of search:
      1. Path relative to the recipe file's directory.
      2. Path under built-in presets (kg4vd.config.presets).
    """

    relative = (anchor.parent / ext).resolve()
    if relative.is_file():
        return relative

    # Try built-in presets - path of the form "presets/foo.yaml" or "foo.yaml".
    leaf = ext.split("/")[-1]
    try:
        files = resources.files("kg4vd.config.presets")
        candidate = files.joinpath(leaf)
        # `as_file` lets us read packaged resources whether they live on disk
        # or inside a wheel.
        with resources.as_file(candidate) as p:
            if Path(p).is_file():
                return Path(p)
    except (FileNotFoundError, ModuleNotFoundError):
        pass

    raise ConfigError(f"Cannot resolve extends ref {ext!r} (anchor={anchor})")


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Recursive merge: dicts merge, everything else (incl. lists) replaces."""

    out = deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _set_dotted(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: Any = d
    for p in parts[:-1]:
        if not isinstance(cur, dict):
            raise ConfigError(
                f"Cannot apply override {dotted_key!r}: hit non-dict at {p!r}"
            )
        cur = cur.setdefault(p, {})
    if not isinstance(cur, dict):
        raise ConfigError(
            f"Cannot apply override {dotted_key!r}: parent is not a dict"
        )
    # Coerce common scalar types from string CLI values.
    cur[parts[-1]] = _coerce(value)


def _coerce(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    try:
        if s.startswith("0") and len(s) > 1 and not s.startswith("0."):
            return s  # zero-padded → keep string
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return v
