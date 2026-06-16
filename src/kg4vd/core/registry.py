"""Plugin registry.

Two ways to register an implementation:

1. Decorator at class definition site::

       from kg4vd.core.registry import register

       @register("encoder", "mock")
       class MockEncoder: ...

2. Setuptools entry-point in pyproject.toml under
   ``[project.entry-points."kg4vd.<kind>s"]``. These are loaded lazily on
   first ``Registry.get(kind, name)`` call.

Resolution order: in-memory decorator registrations win over entry points,
so tests can override published implementations cleanly.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from threading import RLock
from typing import Any, Callable

from kg4vd.core.errors import RegistryError

_REGISTRY: dict[str, dict[str, Any]] = {}
_LOCK = RLock()
_LOADED_ENTRY_POINTS: set[str] = set()

# Each kind maps to the entry-point group name in pyproject.toml.
_ENTRY_POINT_GROUPS = {
    "encoder":   "kg4vd.encoders",
    "index":     "kg4vd.indices",
    "reranker":  "kg4vd.rerankers",
    "llm":       "kg4vd.llms",
    "generator": "kg4vd.generators",
}


def register(kind: str, name: str) -> Callable[[type], type]:
    """Class decorator that registers an implementation under (kind, name)."""

    def _decorator(cls: type) -> type:
        with _LOCK:
            _REGISTRY.setdefault(kind, {})[name] = cls
        return cls

    return _decorator


class Registry:
    """Static accessor; not meant to be instantiated."""

    @staticmethod
    def get(kind: str, name: str) -> Any:
        with _LOCK:
            kind_map = _REGISTRY.get(kind, {})
            if name in kind_map:
                return kind_map[name]

        # Lazy-load entry points for this kind (once).
        Registry._load_entry_points(kind)

        with _LOCK:
            kind_map = _REGISTRY.get(kind, {})
            if name not in kind_map:
                available = sorted(kind_map.keys())
                raise RegistryError(
                    f"No {kind} named {name!r}; available: {available}"
                )
            return kind_map[name]

    @staticmethod
    def list(kind: str) -> list[str]:
        Registry._load_entry_points(kind)
        with _LOCK:
            return sorted(_REGISTRY.get(kind, {}).keys())

    @staticmethod
    def _load_entry_points(kind: str) -> None:
        group = _ENTRY_POINT_GROUPS.get(kind)
        if group is None:
            return
        if group in _LOADED_ENTRY_POINTS:
            return
        try:
            eps = entry_points(group=group)
        except TypeError:
            # Python <3.10 API; not supported but be defensive.
            eps = []
        with _LOCK:
            for ep in eps:
                # In-memory registrations take precedence; don't clobber.
                if ep.name in _REGISTRY.get(kind, {}):
                    continue
                try:
                    cls = ep.load()
                except Exception as e:  # noqa: BLE001
                    # Silently skip plugins whose deps aren't installed; surface
                    # only when actually requested via Registry.get(kind, name).
                    _REGISTRY.setdefault(kind, {})[ep.name] = _LoadFailed(
                        ep.name, e
                    )
                    continue
                _REGISTRY.setdefault(kind, {})[ep.name] = cls
            _LOADED_ENTRY_POINTS.add(group)


class _LoadFailed:
    """Marker placed in the registry when an entry point's class fails to load.

    Calling it re-raises the original error with context, so users get a
    helpful message at use-time rather than silent missing-feature behaviour.
    """

    def __init__(self, name: str, error: Exception):
        self._name = name
        self._error = error

    def __call__(self, *args: Any, **kwargs: Any):
        raise RegistryError(
            f"Plugin {self._name!r} could not be loaded "
            f"(probably a missing optional dependency): {self._error!r}"
        )
