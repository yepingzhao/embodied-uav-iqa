"""Tests for UAV distortion models."""

import numpy as np

from uav_iqa.distortion import (
    AtmosphericScatteringHaze,
    CommunicationPacketLoss,
    GenericDistortions,
    LowResSuperResolution,
    PropellerShadow,
    PropellerVibrationBlur,
    SixDoFViewpointBlur,
    UAVDistortionPipeline,
)


def _make_test_image(h=128, w=128):
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    img[32:64, 32:64, :] = 255
    return img


def test_pipeline_has_all_distortions():
    pipeline = UAVDistortionPipeline()
    all_names = pipeline.get_all_distortion_names()
    assert len(all_names) >= 36  # 6 UAV + 30 generic (updated from 24)
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
    assert len(results) == 1
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


# --- GenericDistortions tests ---


def _test_generic_distortion_cat(name, category):
    """Helper: verify a generic distortion produces valid output."""
    img = _make_test_image()
    dist = GenericDistortions(name, seed=42)
    result = dist.apply(img, 0.5)
    assert result.shape == img.shape, f"{name}: shape mismatch"
    assert result.dtype == np.uint8, f"{name}: dtype mismatch"
    assert result.min() >= 0, f"{name}: min < 0"
    assert result.max() <= 255, f"{name}: max > 255"
    assert dist.name == name
    assert dist.CATEGORIES[name] == category


def test_generic_blur():
    for name in ["gaussian_blur", "lens_blur", "motion_blur"]:
        _test_generic_distortion_cat(name, "blur")


def test_generic_brightness():
    for name in [
        "brighten_max",
        "brighten_avg",
        "darken_max",
        "darken_min",
        "darken_avg",
    ]:
        _test_generic_distortion_cat(name, "brightness")


def test_generic_chromatic():
    for name in ["color_diffusion", "color_shift", "color_quantize"]:
        _test_generic_distortion_cat(name, "chromatic")


def test_generic_noise():
    for name in [
        "white_noise",
        "color_noise",
        "impulse_noise",
        "multiplicative_noise",
        "gaussian_denoise",
        "cnn_denoise",
    ]:
        _test_generic_distortion_cat(name, "noise")


def test_generic_compression():
    for name in ["jpeg_compression", "jp2k_compression", "webp_compression"]:
        _test_generic_distortion_cat(name, "compression")


def test_generic_spatial():
    for name in ["spatial_warp", "spatial_scale", "clock_jittering"]:
        _test_generic_distortion_cat(name, "spatial")


def test_generic_other():
    for name in ["resolution_limit", "grayscale", "sharpness", "contrast"]:
        _test_generic_distortion_cat(name, "other")


def test_generic_intensity_range():
    """Verify all generic distortions work across all intensity levels."""
    img = _make_test_image()
    for name in GenericDistortions.CATEGORIES:
        dist = GenericDistortions(name, seed=42)
        for level in [0.2, 0.4, 0.6, 0.8, 1.0]:
            result = dist.apply(img, level)
            assert result.shape == img.shape
            assert result.dtype == np.uint8


def test_generic_unknown_distortion_raises():
    import pytest

    with pytest.raises(ValueError):
        GenericDistortions("nonexistent_distortion")


def test_all_uav_distortions_are_deterministic_for_fixed_seed():
    """Two seeded pipelines produce equal outputs for every UAV distortion."""
    rows, columns = np.indices((64, 64), dtype=np.uint8)
    image = np.stack([rows, columns, rows + columns], axis=-1)
    first_pipeline = UAVDistortionPipeline(seed=17)
    second_pipeline = UAVDistortionPipeline(seed=17)

    assert first_pipeline.get_all_distortion_names() == second_pipeline.get_all_distortion_names()
    uav_names = first_pipeline.get_uav_distortion_names()
    assert uav_names == second_pipeline.get_uav_distortion_names()

    for name in uav_names:
        first_output = first_pipeline.apply_distortion(image, name, 0.4)
        second_output = second_pipeline.apply_distortion(image, name, 0.4)
        assert np.array_equal(first_output, second_output), name
