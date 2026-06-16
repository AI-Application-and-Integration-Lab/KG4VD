"""Image utilities for VLM input."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from io import BytesIO
from pathlib import Path


def _is_data_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith("data:")


def _is_remote_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))


async def encode_image_to_data_url(
    path_or_url: str,
    *,
    max_long_side: int = 1024,
    jpeg_quality: int = 85,
) -> str:
    """Return a data: URL for vision LLM input.

    - Accepts a local path, a remote http(s) URL (returned as-is), or an
      already-formatted data: URL (returned as-is).
    - Resizes to at most ``max_long_side`` on the long edge.
    - Falls back to raw base64 if Pillow is unavailable.
    """

    if _is_data_url(path_or_url) or _is_remote_url(path_or_url):
        return path_or_url

    return await asyncio.to_thread(
        _encode_local, path_or_url, max_long_side, jpeg_quality
    )


def _encode_local(path: str, max_long_side: int, quality: int) -> str:
    try:
        from PIL import Image

        mime, _ = mimetypes.guess_type(path)
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            m = max(w, h)
            if m > max_long_side:
                s = max_long_side / m
                im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:{mime or 'image/jpeg'};base64,{b64}"
    except Exception:
        # Fallback without Pillow: raw base64 of the file.
        mime, _ = mimetypes.guess_type(path)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime or 'image/png'};base64,{b64}"


async def resize_for_vlm(path: str, *, max_long_side: int = 1024) -> str:
    """Resize an image in place; returns the path. Used by the encoder pipeline."""

    return await asyncio.to_thread(_resize_in_place, path, max_long_side)


def _resize_in_place(path: str, max_long_side: int) -> str:
    try:
        from PIL import Image
    except ImportError:
        return path
    p = Path(path)
    with Image.open(p) as im:
        im = im.convert("RGB")
        w, h = im.size
        m = max(w, h)
        if m <= max_long_side:
            return path
        s = max_long_side / m
        im.resize((max(1, int(w * s)), max(1, int(h * s)))).save(p, optimize=True)
    return path
