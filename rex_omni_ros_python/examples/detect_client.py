#!/usr/bin/env python3
"""Example client for the rex_omni detect service (works on ROS1 and ROS2).

Usage (with the workspace built and sourced):
    python detect_client.py IMAGE CATEGORY [CATEGORY ...] [--gui] [--service NAME]
                            [-o RESULT_IMAGE]
"""

from __future__ import annotations

import argparse
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from rex_omni_msgs.srv import Detect

from rex_omni_ros.compat import RosNode
from rex_omni_ros.conversions import pil_to_image_msg

# Distinct colors cycled over categories (Set1-like palette).
PALETTE = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#17becf",
]


def visualize(image: Image.Image, detections: list[Any], path: str) -> None:
    """Draw labeled bounding boxes on a copy of ``image`` and save it."""
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, round(min(canvas.size) / 300))
    try:
        font = ImageFont.load_default(size=max(12, 6 * line_width))
    except TypeError:  # Pillow < 10.1 has no size argument
        font = ImageFont.load_default()

    colors: dict[str, str] = {}
    for detection in detections:
        color = colors.setdefault(
            detection.category, PALETTE[len(colors) % len(PALETTE)]
        )
        box = detection.bbox
        draw.rectangle((box.x0, box.y0, box.x1, box.y1), outline=color, width=line_width)

        label = f"{detection.category} {detection.confidence:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_height = bottom - top
        text_y = box.y0 - text_height - 2 * line_width
        if text_y < 0:  # keep the label inside the image at the top edge
            text_y = box.y0
        draw.rectangle(
            (
                box.x0,
                text_y,
                box.x0 + (right - left) + 2 * line_width,
                text_y + text_height + 2 * line_width,
            ),
            fill=color,
        )
        draw.text(
            (box.x0 + line_width, text_y + line_width),
            label,
            fill="white",
            font=font,
        )
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="path to the input image")
    parser.add_argument("categories", nargs="+", help="categories to detect")
    parser.add_argument(
        "--gui", action="store_true", help="use the GUI grounding variant"
    )
    parser.add_argument("--service", default="/rex_omni/detect")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="save a visualization of the detections to this image file",
    )
    args, _ = parser.parse_known_args()

    node = RosNode("rex_omni_detect_client")

    image = Image.open(args.image)
    request = RosNode.request_class(Detect)()
    request.image = pil_to_image_msg(image)
    request.categories = args.categories
    request.variant = (
        request.VARIANT_GUI_GROUNDING if args.gui else request.VARIANT_DETECTION
    )

    node.log_info(f"calling {args.service} ...")
    response = node.call_service(Detect, args.service, request, timeout=args.timeout)

    if not response.success:
        node.log_error(f"detection failed: {response.message}")
        return
    node.log_info(
        f"{len(response.detections)} detections "
        f"in {response.inference_time:.2f}s"
    )
    for detection in response.detections:
        box = detection.bbox
        node.log_info(
            f"  {detection.category}: "
            f"({box.x0:.0f}, {box.y0:.0f})-({box.x1:.0f}, {box.y1:.0f}) "
            f"conf={detection.confidence:.2f}"
        )

    if args.output:
        visualize(image, list(response.detections), args.output)
        node.log_info(f"saved visualization to {args.output}")


if __name__ == "__main__":
    main()
