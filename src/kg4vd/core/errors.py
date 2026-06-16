"""Error hierarchy.

All exceptions raised by kg4vd derive from KG4VDError so callers can catch
the package boundary cleanly.
"""

from __future__ import annotations


class KG4VDError(Exception):
    """Base for all kg4vd errors."""


class ConfigError(KG4VDError):
    """Raised when a config is invalid, missing, or inconsistent."""


class RegistryError(KG4VDError):
    """Raised when a plugin / implementation is not found in the registry."""


class EncoderError(KG4VDError):
    """Raised by encoders for missing weights / dimension mismatch / etc."""


class IndexError(KG4VDError):  # noqa: A001 - intentional shadow
    """Raised by the unified evidence index."""


class RetrievalError(KG4VDError):
    """Raised when retrieval cannot complete."""


class PipelineError(KG4VDError):
    """Raised by build / query pipelines for unrecoverable failures."""


class LLMError(KG4VDError):
    """Raised by LLM clients for parse/refuse/timeout/policy errors."""


class StreamingNotSupported(KG4VDError):
    """Raised when a client is called with stream=True."""
