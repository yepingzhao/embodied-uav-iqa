"""Tests for UAV distortion models."""

import numpy as np

from uav_iqa.distortion import (
    UAVDistortionPipeline,
    PropellerVibrationBlur,
    AtmosphericScatteringHaze,
    SixDoFViewpointBlur,
    CommunicationPacketLoss,
    LowResSuperResolution,
    PropellerShadow,
)


def _make_test_image(h=128, w=128):
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    img[32:64, 32:64, :] = 255
    return img


def test_pipeline_has_all_distortions():
    pipeline = UAVDistortionPipeline()
    all_names = pipeline.get_all_distortion_names()
    assert len(all_names) >= 24  # 6 UAV + 18+ generic from Embodied-IQA catalog
    uav_names = pipeline.get_uav_distortion_names()
    assert len(uav_names) == 6
    assert "propeller_vibration_blur" in uav_names
    assert "atmospheric_scattering_haze" in uav_names


def test_propeller_vibration_blur():
    img = _make_test_image()
    dist = PropellerVibrationBlur(seed=42)
    result = dist.apply(img, 0.5)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_atmospheric_scattering():
    img = _make_test_image()
    dist = AtmosphericScatteringHaze(seed=42)
    result = dist.apply(img, 0.5)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_six_dof_viewpoint_blur():
    img = _make_test_image()
    dist = SixDoFViewpointBlur(seed=42)
    result = dist.apply(img, 0.5)
    assert result.shape == img.shape


def test_communication_packet_loss():
    img = _make_test_image()
    dist = CommunicationPacketLoss(seed=42)
    result = dist.apply(img, 0.3)
    assert result.shape == img.shape


def test_low_res_super_resolution():
    img = _make_test_image()
    dist = LowResSuperResolution(seed=42)
    result = dist.apply(img, 0.5)
    assert result.shape == img.shape


def test_propeller_shadow():
    img = _make_test_image()
    dist = PropellerShadow(seed=42)
    result = dist.apply(img, 0.5)
    assert result.shape == img.shape


def test_intensity_range():
    pipeline = UAVDistortionPipeline(seed=42)
    img = _make_test_image()
    for name in pipeline.get_uav_distortion_names():
        for level in pipeline.INTENSITY_LEVELS:
            result = pipeline.apply_distortion(img, name, level)
            assert result.shape == img.shape
            assert result.dtype == np.uint8
            assert result.min() >= 0
            assert result.max() <= 255


def test_generate_all():
    pipeline = UAVDistortionPipeline(seed=42)
    img = _make_test_image()
    results = pipeline.generate_all(img, distortion_types=["propeller_vibration_blur"])
    assert len(results) == 5
    for key, val in results.items():
        assert val.shape == img.shape
        assert val.dtype == np.uint8


def test_deterministic():
    pipeline1 = UAVDistortionPipeline(seed=42)
    pipeline2 = UAVDistortionPipeline(seed=42)
    img = _make_test_image()
    r1 = pipeline1.apply_distortion(img, "propeller_vibration_blur", 0.5)
    r2 = pipeline2.apply_distortion(img, "propeller_vibration_blur", 0.5)
    assert np.array_equal(r1, r2)
