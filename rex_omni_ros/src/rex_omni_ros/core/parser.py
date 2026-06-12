"""Parsing of Rex-Omni token output into typed predictions.

Standard tasks emit blocks like::

    <|object_ref_start|>person<|object_ref_end|><|box_start|><0><35><980><987>, <646><0><999><940><|box_end|>

where ``<N>`` are coordinate tokens quantized to bins ``[0, 999]``. A group of
2 numbers is a point, 4 a box, and an even count above 4 a polygon. The
keypoint task instead emits JSON mapping instance ids to a bbox and named
keypoints.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Sequence, Union

from rex_omni_ros.core.types import (
    Annotation,
    Box,
    Keypoint,
    KeypointInstance,
    Point,
    Polygon,
    coord_token_count,
)

logger = logging.getLogger(__name__)

COORD_BINS_MAX = 999

REF_BLOCK_PATTERN = re.compile(
    r"<\|object_ref_start\|>\s*([^<]+?)\s*<\|object_ref_end\|>"
    r"\s*<\|box_start\|>(.*?)<\|box_end\|>",
    re.DOTALL,
)
COORD_TOKEN_PATTERN = re.compile(r"<(\d+)>")
JSON_FENCE_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
INSTANCE_CATEGORY_PATTERN = re.compile(r"^([a-zA-Z_]+)")

END_OF_TEXT = "<|im_end|>"
BOX_END = "<|box_end|>"


def _bin_to_pixel(bin_value: int, size: int) -> float:
    return bin_value / COORD_BINS_MAX * size


def _coords_to_shape(
    bins: list[int], width: int, height: int
) -> Union[Box, Point, Polygon, None]:
    if len(bins) == 2:
        return Point(_bin_to_pixel(bins[0], width), _bin_to_pixel(bins[1], height))
    if len(bins) == 4:
        return Box(
            _bin_to_pixel(bins[0], width),
            _bin_to_pixel(bins[1], height),
            _bin_to_pixel(bins[2], width),
            _bin_to_pixel(bins[3], height),
        )
    if len(bins) > 4 and len(bins) % 2 == 0:
        points = [
            Point(_bin_to_pixel(bins[i], width), _bin_to_pixel(bins[i + 1], height))
            for i in range(0, len(bins), 2)
        ]
        return Polygon(points)
    return None


def parse_annotations(text: str, width: int, height: int) -> list[Annotation]:
    """Parse standard-task output into annotations, in emission order."""
    text = text.split(END_OF_TEXT)[0]
    if not text.endswith(BOX_END):
        text += BOX_END

    annotations: list[Annotation] = []
    for category, coords_text in REF_BLOCK_PATTERN.findall(text):
        for group in coords_text.split(","):
            bins = [int(n) for n in COORD_TOKEN_PATTERN.findall(group)]
            if not bins:
                continue
            shape = _coords_to_shape(bins, width, height)
            if shape is None:
                logger.warning(
                    "skipping coordinate group with unexpected length %d "
                    "for category %r",
                    len(bins),
                    category,
                )
                continue
            annotations.append(Annotation(category=category.strip(), shape=shape))
    return annotations


def _extract_json(text: str) -> str | None:
    fenced = JSON_FENCE_PATTERN.findall(text)
    if fenced:
        return fenced[0]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return None


def _parse_coord_pair(value: object, width: int, height: int) -> Point | None:
    if not isinstance(value, str):
        return None
    bins = [int(n) for n in COORD_TOKEN_PATTERN.findall(value)]
    if len(bins) != 2:
        return None
    return Point(_bin_to_pixel(bins[0], width), _bin_to_pixel(bins[1], height))


def parse_keypoint_instances(
    text: str, width: int, height: int
) -> list[KeypointInstance]:
    """Parse keypoint-task JSON output, in emission order."""
    json_str = _extract_json(text.split(END_OF_TEXT)[0])
    if json_str is None:
        logger.warning("keypoint output contains no JSON object")
        return []
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as error:
        logger.warning("failed to decode keypoint JSON: %s", error)
        return []
    if not isinstance(data, dict):
        logger.warning("keypoint JSON root is not an object")
        return []

    instances: list[KeypointInstance] = []
    for instance_id, instance in data.items():
        if not isinstance(instance, dict):
            continue
        bbox_value = instance.get("bbox")
        keypoints_value = instance.get("keypoints")
        if not isinstance(bbox_value, str) or not isinstance(keypoints_value, dict):
            logger.warning("instance %r lacks bbox/keypoints; skipping", instance_id)
            continue

        bins = [int(n) for n in COORD_TOKEN_PATTERN.findall(bbox_value)]
        if len(bins) != 4:
            logger.warning(
                "instance %r has invalid bbox %r; skipping", instance_id, bbox_value
            )
            continue
        box = Box(
            _bin_to_pixel(bins[0], width),
            _bin_to_pixel(bins[1], height),
            _bin_to_pixel(bins[2], width),
            _bin_to_pixel(bins[3], height),
        )

        keypoints = [
            Keypoint(name=name, position=_parse_coord_pair(value, width, height))
            for name, value in keypoints_value.items()
        ]

        category_match = INSTANCE_CATEGORY_PATTERN.match(str(instance_id))
        category = category_match.group(1) if category_match else "keypoint_instance"

        instances.append(
            KeypointInstance(category=category, box=box, keypoints=keypoints)
        )
    return instances


def assign_confidences(
    items: Sequence[Union[Annotation, KeypointInstance]],
    coord_token_probs: Sequence[float],
) -> None:
    """Assign per-item confidence as the mean probability of its coordinate tokens.

    ``coord_token_probs`` must list the probability of every coordinate token
    in the generated text, in emission order; items consume them positionally.
    Items whose tokens cannot be aligned keep confidence 0.
    """
    cursor = 0
    for item in items:
        count = coord_token_count(item)
        window = coord_token_probs[cursor : cursor + count]
        if count > 0 and len(window) == count:
            item.confidence = float(sum(window) / count)
        cursor += count
    if cursor != len(coord_token_probs):
        logger.debug(
            "coordinate token count mismatch: consumed %d of %d; "
            "confidences may be approximate",
            cursor,
            len(coord_token_probs),
        )
