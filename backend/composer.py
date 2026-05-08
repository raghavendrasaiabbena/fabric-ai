"""
Image Composition Engine — matches Kesarkava reference style:
  - Top-left:  Company logo in white box
  - Top-right: Product code / GSM / Width — stacked white rounded boxes
  - Left side: One white pill label per swatch, vertically centered on each band

Font sizing rule: ~4% of image width for color labels, ~4.5% for info boxes.
This mirrors the client reference (720px image → 28px labels → readable).
At 4x upscale (2880px) the same ratio gives 115px labels — still clear.
"""

import base64
import io
import logging
import os

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

FONT_BOLD = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/Arial Bold.ttf",
]
FONT_REG = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    paths = FONT_BOLD if bold else FONT_REG
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, max(10, size))
            except Exception:
                pass
    return ImageFont.load_default()


def _measure(draw: ImageDraw.ImageDraw, text: str, font, pad_x: int, pad_y: int):
    """Return (box_width, box_height) WITHOUT drawing anything."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return tw + pad_x * 2, th + pad_y * 2


def _draw_label(draw: ImageDraw.ImageDraw, text: str, x: int, y: int,
                font, pad_x: int, pad_y: int, radius: int, shadow: bool = True):
    """Draw a white rounded-rect label at (x, y). Returns (box_w, box_h)."""
    bw, bh = _measure(draw, text, font, pad_x, pad_y)

    if shadow:
        so = max(3, radius // 3)
        draw.rounded_rectangle(
            [x + so, y + so, x + bw + so, y + bh + so],
            radius=radius, fill=(0, 0, 0, 80),
        )
    draw.rounded_rectangle(
        [x, y, x + bw, y + bh],
        radius=radius, fill=(255, 255, 255, 245),
    )
    draw.text((x + pad_x, y + pad_y), text, font=font, fill=(15, 15, 15, 255))
    return bw, bh


def compose(
    enhanced_path: str,
    output_path: str,
    product_code: str,
    gsm: str,
    width: str,
    colors: list,
    swatches: list,
    logo_base64: str = "",
) -> str:
    img = Image.open(enhanced_path).convert("RGBA")
    W, H = img.size
    logger.info(f"Composing {W}x{H}, {len(swatches)} swatches")

    # ── Font sizes proportional to image width ────────────────────
    # Reference: client's 720px image uses ~28px labels (3.9% of width)
    label_fs = max(22, int(W * 0.039))   # color name labels
    info_fs  = max(26, int(W * 0.045))   # product/gsm/width boxes
    logo_fs  = max(20, int(W * 0.030))

    # Layout spacing proportional to image
    margin = max(14, int(W * 0.022))
    pad_x  = max(12, int(W * 0.018))
    pad_y  = max(6,  int(H * 0.007))
    radius = max(8,  int(W * 0.012))
    gap    = max(6,  int(H * 0.006))

    font_label = _load_font(label_fs, bold=True)
    font_info  = _load_font(info_fs,  bold=True)

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    # ── Logo — top left ───────────────────────────────────────────
    if logo_base64:
        try:
            logo = Image.open(io.BytesIO(base64.b64decode(logo_base64))).convert("RGBA")
            logo_w = int(W * 0.22)
            logo_h = int(logo.height * logo_w / logo.width)
            logo   = logo.resize((logo_w, logo_h), Image.LANCZOS)
            bg_w   = logo_w + pad_x * 2
            bg_h   = logo_h + pad_y * 2
            logo_bg = Image.new("RGBA", (bg_w, bg_h), (255, 255, 255, 230))
            overlay.paste(logo_bg, (margin, margin), logo_bg)
            overlay.paste(logo,    (margin + pad_x, margin + pad_y), logo)
        except Exception as e:
            logger.warning(f"Logo error: {e}")

    # ── Product info boxes — top right ────────────────────────────
    info_lines = []
    if product_code.strip(): info_lines.append(product_code.strip())
    if gsm.strip():           info_lines.append(f"Gsm: {gsm.strip()}")
    if width.strip():         info_lines.append(f"Width: {width.strip()}")

    y_info = margin
    for line in info_lines:
        bw, bh = _measure(draw, line, font_info, pad_x, pad_y)  # measure only, no draw
        x0 = W - margin - bw
        _draw_label(draw, line, x0, y_info, font_info, pad_x, pad_y, radius, shadow=True)
        y_info += bh + gap

    # ── Color labels — left side, one per swatch ─────────────────
    for i, swatch in enumerate(swatches):
        if i >= len(colors):
            break
        name = str(colors[i]).strip()
        if not name:
            continue

        y_center = int(swatch["y_percent"] / 100 * H)
        _, bh    = _measure(draw, name, font_label, pad_x, pad_y)  # height only
        y0 = y_center - bh // 2
        y0 = max(margin, min(H - margin - bh, y0))

        _draw_label(draw, name, margin, y0, font_label, pad_x, pad_y, radius, shadow=True)

    # ── Merge overlay onto image ──────────────────────────────────
    result = Image.alpha_composite(img, overlay).convert("RGB")
    result.save(output_path, "JPEG", quality=97, subsampling=0)
    logger.info(f"Saved: {output_path}")
    return output_path
