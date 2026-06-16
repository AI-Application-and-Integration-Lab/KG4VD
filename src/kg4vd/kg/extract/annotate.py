"""Render an annotated page image with component IDs overlaid as boxes.

Visual design:

  - Type-coded outline: BLUE for text-class components, RED for
    image / chart, GREEN for table / formula. Easy modal signal for
    the VLM in the vision channel.
  - Circle labels with the component_id, placed in the left margin
    when there's room, above the box as a fallback, inside the
    top-left corner as a last resort. Avoids occluding content.
  - Boxes are outline-only; the source image is otherwise unchanged.

Saturation masking:

  - Components passed via ``mute_cids`` are rendered with a thin grey
    outline, a hollow grey label circle, AND a ~78%-opacity dark
    overlay across the bbox so the content is barely legible. The
    outline + hollow label remain so the model can still cite the
    component_id for cross-component edges.

Implementation notes:

  - Coordinate spaces drift. Components carry bboxes in
    MinerU pixel space; the rendered page may be at a different DPI
    (typically 1700×2200 for a letter PDF). The caller must scale
    bboxes before calling this function, OR pass
    ``expected_image_size=(W, H)`` so we can refuse to render with a
    mismatched canvas instead of silently producing a wrong image.

  - PIL alpha. ``ImageDraw.rectangle(fill=(r,g,b,a))`` on an
    RGB canvas silently drops the alpha channel. We therefore work
    in RGBA throughout and composite strong overlays via
    ``Image.alpha_composite``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from kg4vd.ingest.components import Component


TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    # Text-class - BLUE.
    "title": (0, 80, 200),
    "paragraph": (0, 80, 200),
    "text": (0, 80, 200),
    "list": (0, 80, 200),
    "index": (0, 80, 200),
    # Image / chart - RED.
    "image": (200, 30, 30),
    "chart": (200, 30, 30),
    # Structured (table / formula) - GREEN.
    "table": (0, 130, 50),
    "formula": (0, 130, 50),
    "equation": (0, 130, 50),
}

_DEFAULT_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]

_MUTED_COLOR = (170, 170, 170)
_OVERLAY_RGBA = (40, 40, 40, 200)  # ~78% opacity near-black


def _load_font(size: int) -> ImageFont.ImageFont:
    for cand in _DEFAULT_FONT_CANDIDATES:
        if Path(cand).is_file():
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                pass
    return ImageFont.load_default()


def annotate_page(
    page_image_path: str | Path,
    components: Iterable[Component],
    out_path: str | Path,
    *,
    line_width: int = 4,
    mute_cids: set[str] | None = None,
    expected_image_size: tuple[int, int] | None = None,
) -> str:
    """Draw labelled boxes on the page image; write PNG to ``out_path``.

    Args:
        page_image_path: rendered (not source PDF) page image. Bboxes
            on ``components`` MUST be in this image's pixel space -
            caller is responsible for scaling from MinerU coords.
        components: iterable of Component (already type-prefixed).
        out_path: target PNG path (RGBA-compatible).
        line_width: outline thickness in pixels for active components.
        mute_cids: component_ids to render in muted style (thin grey
            outline + hollow label + dark overlay on the bbox).
        expected_image_size: optional ``(W, H)`` assertion against the
            opened image. Raises ``ValueError`` on mismatch - catches
            the "bboxes drawn at MinerU scale onto render-px
            canvas" failure mode early.

    Returns the absolute path of the written PNG.
    """
    base = Image.open(page_image_path).convert("RGBA")
    if expected_image_size is not None and base.size != tuple(expected_image_size):
        raise ValueError(
            f"annotate_page: image dimensions {base.size} do not match "
            f"expected {tuple(expected_image_size)}. Likely a coord-space "
            f"mismatch - bboxes may be in MinerU px but image is at render px."
        )

    W, H = base.size
    font_size = max(18, min(32, int(H * 0.014)))
    font = _load_font(font_size)

    components = list(components)
    muted = mute_cids or set()

    # ---- Pass 1: paint translucent overlays for every muted bbox.
    # PIL won't blend alpha onto an RGB canvas, so we build a separate
    # RGBA overlay layer and composite once at the end.
    if muted:
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        for c in components:
            if c.component_id not in muted:
                continue
            x0, y0, x1, y1 = c.bbox
            odraw.rectangle((x0, y0, x1, y1), fill=_OVERLAY_RGBA)
        base = Image.alpha_composite(base, overlay)

    img = base
    draw = ImageDraw.Draw(img, "RGBA")

    # ---- Pass 2: outlines + labels for every component (muted + active).
    for c in components:
        x0, y0, x1, y1 = c.bbox
        is_muted = c.component_id in muted
        color = _MUTED_COLOR if is_muted else TYPE_COLORS.get(c.type, (60, 60, 60))

        stroke = max(1, line_width // 2) if is_muted else line_width
        draw.rectangle((x0, y0, x1, y1), outline=color + (255,), width=stroke)

        label = c.component_id
        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            tb_offx, tb_offy = tb[0], tb[1]
        except AttributeError:
            tw, th = font_size * len(label) // 2, font_size
            tb_offx = tb_offy = 0

        radius = max(tw, th) // 2 + 6

        # Try LEFT (centre just left of box).
        cx = x0 - radius - 3
        cy = y0 + radius
        if cx - radius < 0:
            # Try ABOVE (centre just above box).
            cx = x0 + radius
            cy = y0 - radius - 3
            if cy - radius < 0:
                # Fall back INSIDE top-left.
                cx = x0 + radius + 2
                cy = y0 + radius + 2

        if is_muted:
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                outline=color + (255,),
                width=2,
            )
            text_color = color
        else:
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=color + (255,),
            )
            text_color = (255, 255, 255)

        text_x = cx - tw / 2 - tb_offx
        text_y = cy - th / 2 - tb_offy
        draw.text((text_x, text_y), label, fill=text_color, font=font)

    img.save(out_path, format="PNG")
    return str(out_path)


__all__ = ["TYPE_COLORS", "annotate_page"]
