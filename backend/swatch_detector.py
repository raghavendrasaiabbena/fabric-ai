"""
Swatch Detector — uses Claude Vision to locate each fabric swatch in the image.
Returns ordered list of swatches with their vertical center positions.
"""

import base64
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

MEDIA_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",  ".webp": "image/webp",
}

PROMPT = """This image shows stacked fabric swatches/rolls photographed together.

Identify every distinct fabric color band visible from TOP to BOTTOM.

Return ONLY a valid JSON array — no explanation, no markdown:
[
  {"index": 0, "y_percent": 12, "color_description": "light beige"},
  {"index": 1, "y_percent": 30, "color_description": "mint green"},
  ...
]

Rules:
- y_percent = vertical CENTER of each swatch as % of image height (0=top, 100=bottom)
- color_description = 2–3 words, simple color name
- Order swatches top to bottom
- Only count clearly distinct fabric layers (ignore thin border lines)
- Return ONLY the JSON array"""


def detect_swatches(image_path: str, api_key: str) -> list[dict]:
    import anthropic

    ext = os.path.splitext(image_path)[1].lower()
    media_type = MEDIA_TYPES.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": img_b64},
                },
                {"type": "text", "text": PROMPT},
            ],
        }],
    )

    raw = response.content[0].text.strip()
    logger.info(f"Claude raw response: {raw[:200]}")

    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        logger.warning("No JSON array found in Claude response")
        return []

    swatches = json.loads(match.group())
    logger.info(f"Detected {len(swatches)} swatches")
    return swatches
