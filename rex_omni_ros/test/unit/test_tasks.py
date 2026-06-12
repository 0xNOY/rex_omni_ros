"""Prompt construction tests; expected strings mirror upstream Rex-Omni."""

import pytest

from rex_omni_ros.core import tasks
from rex_omni_ros.core.tasks import TaskType, build_prompt
from rex_omni_ros.core.types import Box


class TestStandardPrompts:
    def test_detection(self):
        assert build_prompt(TaskType.DETECTION, categories=["person", "dog"]) == (
            "Detect person, dog. Output the bounding box coordinates in "
            "[x0, y0, x1, y1] format."
        )

    def test_pointing(self):
        assert build_prompt(TaskType.POINTING, categories=["red cup"]) == (
            "Point to red cup."
        )

    def test_gui_grounding_keeps_upstream_doubled_quote(self):
        assert build_prompt(TaskType.GUI_GROUNDING, categories=["login button"]) == (
            'Detect element "login button"" in the image.'
        )

    def test_gui_pointing(self):
        assert build_prompt(TaskType.GUI_POINTING, categories=["search box"]) == (
            'Point to element "search box".'
        )

    def test_ocr_box(self):
        assert build_prompt(TaskType.OCR_BOX, categories=["word"]) == (
            "Detect all word and recognize them."
        )

    def test_ocr_polygon(self):
        assert build_prompt(TaskType.OCR_POLYGON, categories=["text line"]) == (
            "Can you detect all text line in this image in polygon format like "
            "[x0, y0, x1, y1, x2, y2 ...] and then recognize them?"
        )

    def test_categories_required(self):
        with pytest.raises(ValueError, match="categories"):
            build_prompt(TaskType.DETECTION)


class TestKeypointPrompt:
    def test_person(self):
        prompt = build_prompt(
            TaskType.KEYPOINT, categories=["person"], keypoint_type="person"
        )
        assert prompt.startswith("Can you detect each person in the image")
        assert "nose, left eye, right eye" in prompt
        assert "left ankle, right ankle" in prompt
        assert prompt.endswith("Output the answer in JSON format.")

    def test_animal_keypoint_names(self):
        prompt = build_prompt(
            TaskType.KEYPOINT, categories=["cat"], keypoint_type="animal"
        )
        assert "root of tail" in prompt

    def test_unknown_keypoint_type(self):
        with pytest.raises(ValueError, match="keypoint_type"):
            build_prompt(
                TaskType.KEYPOINT, categories=["person"], keypoint_type="robot"
            )

    def test_keypoint_type_required(self):
        with pytest.raises(ValueError, match="keypoint_type"):
            build_prompt(TaskType.KEYPOINT, categories=["person"])


class TestVisualPrompting:
    def test_boxes_rendered_as_bin_tokens(self):
        prompt = build_prompt(
            TaskType.VISUAL_PROMPTING,
            visual_prompt_boxes=[Box(100, 200, 300, 400)],
            image_width=1000,
            image_height=1000,
        )
        assert '{"object_1": ["<99><199><299><399>"]}' in prompt

    def test_full_image_box_clamps_to_max_bin(self):
        tokens = tasks.boxes_to_bin_tokens([Box(0, 0, 1000, 1000)], 1000, 1000)
        assert tokens == ["<0><0><999><999>"]

    def test_out_of_range_box_is_clamped(self):
        tokens = tasks.boxes_to_bin_tokens([Box(-50, -50, 2000, 2000)], 1000, 1000)
        assert tokens == ["<0><0><999><999>"]

    def test_boxes_required(self):
        with pytest.raises(ValueError, match="visual_prompt_boxes"):
            build_prompt(TaskType.VISUAL_PROMPTING, image_width=100, image_height=100)
