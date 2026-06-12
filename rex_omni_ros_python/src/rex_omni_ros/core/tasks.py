"""Task definitions and prompt construction.

Prompt templates must stay byte-identical to the ones Rex-Omni was trained
with (https://github.com/IDEA-Research/Rex-Omni), including the doubled quote
in the GUI grounding template.
"""

from __future__ import annotations

import json
from enum import Enum

from rex_omni_ros.core.types import Box

COORD_BINS = 1000  # coordinates are quantized to <0> .. <999>


class TaskType(Enum):
    """Tasks supported by Rex-Omni."""

    DETECTION = "detection"
    POINTING = "pointing"
    VISUAL_PROMPTING = "visual_prompting"
    KEYPOINT = "keypoint"
    OCR_BOX = "ocr_box"
    OCR_POLYGON = "ocr_polygon"
    GUI_GROUNDING = "gui_grounding"
    GUI_POINTING = "gui_pointing"


PROMPT_TEMPLATES: dict[TaskType, str] = {
    TaskType.DETECTION: (
        "Detect {categories}. Output the bounding box coordinates in "
        "[x0, y0, x1, y1] format."
    ),
    TaskType.POINTING: "Point to {categories}.",
    TaskType.VISUAL_PROMPTING: (
        "Given reference boxes {visual_prompt} indicating one or more objects, "
        "find all similar objects in the image and output their bounding boxes."
    ),
    TaskType.KEYPOINT: (
        "Can you detect each {categories} in the image using a [x0, y0, x1, y1] "
        "box format, and then provide the coordinates of its {keypoints} as "
        "[x0, y0]? Output the answer in JSON format."
    ),
    TaskType.OCR_BOX: "Detect all {categories} and recognize them.",
    TaskType.OCR_POLYGON: (
        "Can you detect all {categories} in this image in polygon format like "
        "[x0, y0, x1, y1, x2, y2 ...] and then recognize them?"
    ),
    TaskType.GUI_GROUNDING: 'Detect element "{categories}"" in the image.',
    TaskType.GUI_POINTING: 'Point to element "{categories}".',
}

# Keypoint names in the order the model was trained with.
KEYPOINT_SETS: dict[str, list[str]] = {
    "person": [
        "nose",
        "left eye",
        "right eye",
        "left ear",
        "right ear",
        "left shoulder",
        "right shoulder",
        "left elbow",
        "right elbow",
        "left wrist",
        "right wrist",
        "left hip",
        "right hip",
        "left knee",
        "right knee",
        "left ankle",
        "right ankle",
    ],
    "animal": [
        "left eye",
        "right eye",
        "nose",
        "neck",
        "root of tail",
        "left shoulder",
        "left elbow",
        "left front paw",
        "right shoulder",
        "right elbow",
        "right front paw",
        "left hip",
        "left knee",
        "left back paw",
        "right hip",
        "right knee",
        "right back paw",
    ],
}

# Conventional category strings for the OCR variants (from the upstream
# tutorials): word-level boxes, text-line boxes and text-line polygons.
OCR_WORD_CATEGORY = "word"
OCR_TEXTLINE_CATEGORY = "text line"


def boxes_to_bin_tokens(boxes: list[Box], width: int, height: int) -> list[str]:
    """Convert pixel boxes to quantized coordinate-token strings.

    Each box becomes ``"<x0><y0><x1><y1>"`` with bins in ``[0, 999]``.
    """
    tokens = []
    for box in boxes:
        bins = []
        for value, size in (
            (box.x0, width),
            (box.y0, height),
            (box.x1, width),
            (box.y1, height),
        ):
            normalized = max(0.0, min(1.0, value / size))
            bins.append(max(0, min(COORD_BINS - 1, int(normalized * (COORD_BINS - 1)))))
        tokens.append("".join(f"<{b}>" for b in bins))
    return tokens


def build_prompt(
    task: TaskType,
    categories: list[str] | None = None,
    keypoint_type: str | None = None,
    visual_prompt_boxes: list[Box] | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> str:
    """Build the user prompt for ``task``.

    Raises:
        ValueError: If an argument required by the task is missing or invalid.
    """
    template = PROMPT_TEMPLATES[task]

    if task is TaskType.VISUAL_PROMPTING:
        if not visual_prompt_boxes:
            raise ValueError("visual_prompt_boxes is required for visual prompting")
        if image_width is None or image_height is None:
            raise ValueError("image size is required for visual prompting")
        visual_prompt = json.dumps(
            {"object_1": boxes_to_bin_tokens(visual_prompt_boxes, image_width, image_height)}
        )
        return template.format(visual_prompt=visual_prompt)

    if task is TaskType.KEYPOINT:
        if not categories:
            raise ValueError("categories is required for the keypoint task")
        if keypoint_type is None:
            raise ValueError("keypoint_type is required for the keypoint task")
        keypoint_names = KEYPOINT_SETS.get(keypoint_type)
        if keypoint_names is None:
            raise ValueError(
                f"unknown keypoint_type {keypoint_type!r}; "
                f"expected one of {sorted(KEYPOINT_SETS)}"
            )
        return template.format(
            categories=", ".join(categories), keypoints=", ".join(keypoint_names)
        )

    if not categories:
        raise ValueError(f"categories is required for the {task.value} task")
    return template.format(categories=", ".join(categories))
