"""Image preprocessing.

Reimplements the ``smart_resize`` logic of Qwen2.5-VL (qwen_vl_utils) so the
package does not depend on it. The resized image fed to the model determines
the number of vision tokens; coordinates themselves are normalized bins and
are therefore independent of the resize.
"""

from __future__ import annotations

import math

from PIL import Image

IMAGE_FACTOR = 28  # vision patches are 28x28 pixels
DEFAULT_MIN_PIXELS = 16 * IMAGE_FACTOR * IMAGE_FACTOR
DEFAULT_MAX_PIXELS = 2560 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_ASPECT_RATIO = 200.0


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> tuple[int, int]:
    """Compute target (height, width) for the model input.

    Both sides are multiples of ``factor`` and the pixel count is kept within
    ``[min_pixels, max_pixels]`` while approximately preserving aspect ratio.

    Raises:
        ValueError: If the aspect ratio exceeds ``MAX_ASPECT_RATIO``.
    """
    if max(height, width) / min(height, width) > MAX_ASPECT_RATIO:
        raise ValueError(
            f"aspect ratio must not exceed {MAX_ASPECT_RATIO}, "
            f"got {max(height, width) / min(height, width):.1f}"
        )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def resize_for_model(
    image: Image.Image,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> Image.Image:
    """Return ``image`` resized to the dimensions chosen by smart_resize."""
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height, width, min_pixels=min_pixels, max_pixels=max_pixels
    )
    if (resized_width, resized_height) == (width, height):
        return image
    return image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
