#!/usr/bin/env python3
"""M0-R002: Verify all 36 distortion types produce visually plausible outputs.

Loads 10 test images, applies each distortion at 5 intensity levels,
saves output grid for visual inspection.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from uav_iqa.distortion import UAVDistortionPipeline
from uav_iqa.utils import setup_logging

_log = setup_logging(__name__)


def create_test_image(size: int = 256) -> np.ndarray:
    """Create a synthetic test image with gradients and edges."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for c in range(3):
        for i in range(size):
            val = int(255 * i / size * ((c + 1) / 3))
            img[i, :, c] = val
    cv2.circle(img, (size // 2, size // 2), size // 4, (255, 255, 255), -1)
    cv2.rectangle(
        img, (size // 8, size // 8), (size * 3 // 8, size * 3 // 8), (0, 200, 0), -1
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "UAV-IQA", (size // 4, size - 20), font, 1, (255, 255, 255), 2)
    return img


def verify_distortions(
    output_dir: str, test_image_path: str = None, grid_size: int = 256
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = UAVDistortionPipeline(seed=42)

    if test_image_path and Path(test_image_path).exists():
        image = cv2.imread(test_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = create_test_image(grid_size)

    uav_dist_names = pipeline.get_uav_distortion_names()
    all_dist_names = UAVDistortionPipeline.get_all_distortion_names()
    generic_dist_names = [n for n in all_dist_names if n not in uav_dist_names]

    _log.info("=" * 60)
    _log.info("M0-R002: Distortion Verification")
    _log.info("=" * 60)
    _log.info("Test image shape: %s", image.shape)
    _log.info("UAV-specific distortions: %d", len(uav_dist_names))
    _log.info("Generic distortions: %d", len(generic_dist_names))
    _log.info("Total: %d", len(uav_dist_names) + len(generic_dist_names))
    _log.info("Intensity levels: %s", pipeline.INTENSITY_LEVELS)

    for category, names in [
        ("UAV-Specific", uav_dist_names),
        ("Generic", generic_dist_names),
    ]:
        _log.info("--- %s Distortions ---", category)
        for name in names:
            _log.info("  Applying %s...", name)
            for level in pipeline.INTENSITY_LEVELS:
                result = pipeline.apply_distortion(image, name, level)
                basename = f"{name}_L{int(level * 10):02d}.png"
                cv2.imwrite(
                    str(output_dir / basename), cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                )

    _log.info("All distortions saved to: %s", output_dir)
    _log.info("Verification complete.")


def main():
    parser = argparse.ArgumentParser(description="M0-R002: Distortion verification")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/m0_distortion_check",
        help="Directory to save distorted images",
    )
    parser.add_argument(
        "--test-image",
        type=str,
        default=None,
        help="Path to a test image (uses synthetic if not provided)",
    )
    parser.add_argument(
        "--grid-size", type=int, default=256, help="Size of the synthetic test image"
    )
    args = parser.parse_args()

    verify_distortions(
        output_dir=args.output_dir,
        test_image_path=args.test_image,
        grid_size=args.grid_size,
    )


if __name__ == "__main__":
    main()
