"""PDF ingest. Two parsers are supported via plugin selection:

  - ``pypdfium``  - pure-Python; used as the no-dep default.
  - ``mineru``    - high-quality layout-aware parser (optional extra).
"""

from kg4vd.ingest.factory import build_ingest

__all__ = ["build_ingest"]
