"""Tests for uav_iqa.annotations utility functions."""

from uav_iqa.annotations import (
    build_sample_id,
    extract_uav_id_from_question_id,
    normalize_subtask_type,
)


class TestExtractUavIdFromQuestionId:
    def test_standard_pattern(self):
        assert extract_uav_id_from_question_id("Sim3_what2col_UAV2_1") == "UAV2"

    def test_mdmt_pattern(self):
        assert extract_uav_id_from_question_id("MDMT_OB_UAV2_001") == "UAV2"

    def test_qa_pattern(self):
        assert extract_uav_id_from_question_id("Sim3_QA_UAV1_1") == "UAV1"

    def test_trailing_uav(self):
        assert extract_uav_id_from_question_id("something_UAV3") == "UAV3"

    def test_no_uav_in_question_id(self):
        assert extract_uav_id_from_question_id("something_no_uav") == "UAV1"

    def test_no_uav_in_question_id_with_type_hint(self):
        assert (
            extract_uav_id_from_question_id("something_no_uav", "(UAV3)")
            == "UAV3"
        )

    def test_question_type_uav_in_parens(self):
        assert (
            extract_uav_id_from_question_id(
                "question_without_uav", "scene_understanding (UAV2)"
            )
            == "UAV2"
        )


class TestNormalizeSubtaskType:
    def test_scene_description(self):
        assert normalize_subtask_type("scene_description") == "scene_description"

    def test_from_code_1_1(self):
        assert normalize_subtask_type("1.1") == "scene_description"

    def test_from_code_2_2(self):
        assert normalize_subtask_type("2.2") == "object_counting"

    def test_from_code_1_2(self):
        assert normalize_subtask_type("1.2") == "scene_comparison"

    def test_keyword_scene_description(self):
        assert normalize_subtask_type("Scene Description with UAV1") == "scene_description"

    def test_keyword_object_counting(self):
        assert normalize_subtask_type("Object Counting in frame") == "object_counting"

    def test_code_takes_priority(self):
        assert normalize_subtask_type("2.2 Object Counting extra") == "object_counting"

    def test_fallback_empty(self):
        assert normalize_subtask_type("") == "scene_description"

    def test_fallback_snake_case(self):
        assert normalize_subtask_type("object_counting") == "scene_description"

    def test_fallback_unknown(self):
        assert normalize_subtask_type("some_unknown_type") == "scene_description"


class TestBuildSampleId:
    def test_old_backward_compat(self):
        sid = build_sample_id("Sim3", "scene_001", "gaussian_blur", 4)
        assert sid == "Sim3__scene_001__gaussian_blur_L04"

    def test_full_format(self):
        sid = build_sample_id(
            "Sim3",
            "scene_001_frame_001",
            "gaussian_blur",
            4,
            split="train",
            question_id="Sim3_QA_UAV1_1",
        )
        assert sid == "Sim3__train__scene_001_frame_001__Sim3_QA_UAV1_1__gaussian_blur_L04"

    def test_slash_sanitization(self):
        sid = build_sample_id(
            "Sim3",
            "scene/001/frame/001",
            "gaussian_blur",
            4,
            split="test",
            question_id="Q1",
        )
        assert sid == "Sim3__test__scene_001_frame_001__Q1__gaussian_blur_L04"
