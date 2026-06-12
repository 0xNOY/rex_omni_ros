"""Plain data types shared across the core layer.

All coordinates are absolute pixel values in the original (unresized) image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass
class Box:
    """Axis-aligned bounding box."""

    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class Point:
    """2D point."""

    x: float
    y: float


@dataclass
class Polygon:
    """Closed polygon."""

    points: list[Point] = field(default_factory=list)


Shape = Union[Box, Point, Polygon]


@dataclass
class Annotation:
    """One parsed prediction of a standard (non-keypoint) task.

    ``category`` holds the category name, referring expression, or, for OCR
    tasks, the recognized text.
    """

    category: str
    shape: Shape
    confidence: float = 0.0


@dataclass
class Keypoint:
    """One named keypoint; ``position`` is ``None`` when not visible."""

    name: str
    position: Point | None


@dataclass
class KeypointInstance:
    """One detected instance with its skeleton keypoints."""

    category: str
    box: Box
    keypoints: list[Keypoint] = field(default_factory=list)
    confidence: float = 0.0


def coord_token_count(item: Union[Annotation, KeypointInstance]) -> int:
    """Number of coordinate tokens the model emitted for ``item``.

    Used to align per-token logprobs with parsed predictions.
    """
    if isinstance(item, KeypointInstance):
        visible = sum(1 for kp in item.keypoints if kp.position is not None)
        return 4 + 2 * visible
    if isinstance(item.shape, Box):
        return 4
    if isinstance(item.shape, Point):
        return 2
    return 2 * len(item.shape.points)
