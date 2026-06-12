"""smart_resize tests; golden values follow qwen_vl_utils semantics."""

import pytest
from PIL import Image

from rex_omni_ros.core.preprocess import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    resize_for_model,
    smart_resize,
)


class TestSmartResize:
    def test_conformant_size_is_unchanged(self):
        assert smart_resize(560, 560) == (560, 560)

    def test_rounds_to_multiple_of_28(self):
        height, width = smart_resize(550, 570)
        assert height % 28 == 0 and width % 28 == 0

    def test_small_image_scaled_up_to_min_pixels(self):
        height, width = smart_resize(28, 28)
        assert (height, width) == (112, 112)
        assert height * width >= DEFAULT_MIN_PIXELS

    def test_large_image_scaled_down_to_max_pixels(self):
        height, width = smart_resize(10000, 10000)
        assert (height, width) == (1400, 1400)
        assert height * width <= DEFAULT_MAX_PIXELS

    def test_aspect_ratio_roughly_preserved_when_scaling_down(self):
        height, width = smart_resize(5000, 10000)
        assert height * width <= DEFAULT_MAX_PIXELS
        assert width / height == pytest.approx(2.0, rel=0.1)

    def test_extreme_aspect_ratio_rejected(self):
        with pytest.raises(ValueError, match="aspect ratio"):
            smart_resize(28, 10000)

    def test_custom_pixel_budget(self):
        height, width = smart_resize(1000, 1000, max_pixels=512 * 28 * 28)
        assert height * width <= 512 * 28 * 28


class TestResizeForModel:
    def test_no_resize_returns_same_object(self):
        image = Image.new("RGB", (560, 560))
        assert resize_for_model(image) is image

    def test_resized_dimensions_match_smart_resize(self):
        image = Image.new("RGB", (1234, 567))
        resized = resize_for_model(image)
        expected_height, expected_width = smart_resize(567, 1234)
        assert resized.size == (expected_width, expected_height)
