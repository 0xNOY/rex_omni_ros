"""Output parser tests.

Golden samples follow the format documented in the Rex-Omni paper and
implementation. Using 999x999 images makes pixel values equal bin values.
"""

import pytest

from rex_omni_ros.core.parser import (
    assign_confidences,
    parse_annotations,
    parse_keypoint_instances,
)
from rex_omni_ros.core.types import (
    Annotation,
    Box,
    Point,
    Polygon,
    coord_token_count,
)

W = H = 999


class TestParseAnnotations:
    def test_multiple_boxes_for_one_category(self):
        text = (
            "<|object_ref_start|>person<|object_ref_end|>"
            "<|box_start|><0><35><980><987>, <646><0><999><940><|box_end|>"
        )
        annotations = parse_annotations(text, W, H)
        assert [a.category for a in annotations] == ["person", "person"]
        assert annotations[0].shape == Box(0, 35, 980, 987)
        assert annotations[1].shape == Box(646, 0, 999, 940)

    def test_multiple_categories_preserve_order(self):
        text = (
            "<|object_ref_start|>cat<|object_ref_end|>"
            "<|box_start|><1><2><3><4><|box_end|>"
            "<|object_ref_start|>dog<|object_ref_end|>"
            "<|box_start|><5><6><7><8><|box_end|>"
        )
        annotations = parse_annotations(text, W, H)
        assert [a.category for a in annotations] == ["cat", "dog"]

    def test_point(self):
        text = (
            "<|object_ref_start|>cup<|object_ref_end|>"
            "<|box_start|><500><250><|box_end|>"
        )
        (annotation,) = parse_annotations(text, W, H)
        assert isinstance(annotation.shape, Point)
        assert annotation.shape.x == pytest.approx(500)
        assert annotation.shape.y == pytest.approx(250)

    def test_polygon(self):
        text = (
            "<|object_ref_start|>text line<|object_ref_end|>"
            "<|box_start|><0><0><100><0><100><50><0><50><|box_end|>"
        )
        (annotation,) = parse_annotations(text, W, H)
        assert annotation.shape == Polygon(
            [Point(0, 0), Point(100, 0), Point(100, 50), Point(0, 50)]
        )

    def test_bins_scale_to_pixels(self):
        text = (
            "<|object_ref_start|>person<|object_ref_end|>"
            "<|box_start|><0><999><999><999><|box_end|>"
        )
        (annotation,) = parse_annotations(text, 1998, 500)
        assert annotation.shape == Box(0, 500, 1998, 500)

    def test_truncated_output_without_box_end(self):
        text = (
            "<|object_ref_start|>person<|object_ref_end|>"
            "<|box_start|><1><2><3><4>"
        )
        assert len(parse_annotations(text, W, H)) == 1

    def test_im_end_marker_is_stripped(self):
        text = (
            "<|object_ref_start|>person<|object_ref_end|>"
            "<|box_start|><1><2><3><4><|box_end|><|im_end|>garbage"
        )
        assert len(parse_annotations(text, W, H)) == 1

    def test_garbage_yields_nothing(self):
        assert parse_annotations("hello world", W, H) == []

    def test_invalid_coordinate_group_is_skipped(self):
        text = (
            "<|object_ref_start|>cup<|object_ref_end|>"
            "<|box_start|><1><2><3>, <10><20><|box_end|>"
        )
        (annotation,) = parse_annotations(text, W, H)
        assert annotation.shape == Point(10, 20)

    def test_category_whitespace_is_stripped(self):
        text = (
            "<|object_ref_start|> red car <|object_ref_end|>"
            "<|box_start|><1><2><3><4><|box_end|>"
        )
        (annotation,) = parse_annotations(text, W, H)
        assert annotation.category == "red car"


KEYPOINT_JSON = """```json
{
    "person1": {
        "bbox": " <1> <36> <987> <984> ",
        "keypoints": {
            "nose": " <540> <351> ",
            "left eye": "unvisible",
            "right eye": " <500> <300> "
        }
    },
    "person2": {
        "bbox": " <10> <20> <30> <40> ",
        "keypoints": {
            "nose": null
        }
    }
}
```"""


class TestParseKeypointInstances:
    def test_golden_output(self):
        instances = parse_keypoint_instances(KEYPOINT_JSON, W, H)
        assert len(instances) == 2

        first = instances[0]
        assert first.category == "person"
        assert first.box == Box(1, 36, 987, 984)
        by_name = {kp.name: kp for kp in first.keypoints}
        assert by_name["nose"].position == Point(540, 351)
        assert by_name["left eye"].position is None
        assert by_name["right eye"].position.x == pytest.approx(500)
        assert by_name["right eye"].position.y == pytest.approx(300)

        assert instances[1].keypoints[0].position is None

    def test_unfenced_json(self):
        instances = parse_keypoint_instances(KEYPOINT_JSON.strip("`json\n"), W, H)
        assert len(instances) == 2

    def test_invalid_json(self):
        assert parse_keypoint_instances("```json\n{broken\n```", W, H) == []

    def test_no_json(self):
        assert parse_keypoint_instances("no json here", W, H) == []

    def test_instance_missing_bbox_is_skipped(self):
        text = '{"person1": {"keypoints": {"nose": " <1> <2> "}}}'
        assert parse_keypoint_instances(text, W, H) == []


class TestAssignConfidences:
    def test_mean_per_item(self):
        annotations = [
            Annotation("a", Box(0, 0, 1, 1)),
            Annotation("b", Point(0, 0)),
        ]
        assign_confidences(annotations, [1.0, 1.0, 1.0, 1.0, 0.5, 0.3])
        assert annotations[0].confidence == pytest.approx(1.0)
        assert annotations[1].confidence == pytest.approx(0.4)

    def test_unaligned_tail_keeps_zero_confidence(self):
        annotations = [
            Annotation("a", Box(0, 0, 1, 1)),
            Annotation("b", Point(0, 0)),
        ]
        assign_confidences(annotations, [1.0, 1.0, 1.0, 1.0, 0.5])
        assert annotations[0].confidence == pytest.approx(1.0)
        assert annotations[1].confidence == 0.0

    def test_keypoint_instance_token_count(self):
        instances = parse_keypoint_instances(KEYPOINT_JSON, W, H)
        # person1: bbox(4) + 2 visible keypoints(4); person2: bbox(4) only
        assert coord_token_count(instances[0]) == 8
        assert coord_token_count(instances[1]) == 4
        assign_confidences(instances, [0.8] * 8 + [0.4] * 4)
        assert instances[0].confidence == pytest.approx(0.8)
        assert instances[1].confidence == pytest.approx(0.4)
