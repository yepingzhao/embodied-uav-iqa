from typing import Optional

import numpy as np

from .generic import GenericDistortions
from .uav import (
    AtmosphericScatteringHaze,
    CommunicationPacketLoss,
    LowResSuperResolution,
    PropellerShadow,
    PropellerVibrationBlur,
    SixDoFViewpointBlur,
)


class UAVDistortionPipeline:
    """Unified pipeline for applying all 36 distortion types at 1 randomly selected intensity level."""

    UAV_DISTORTIONS = {
        "propeller_vibration_blur": PropellerVibrationBlur,
        "atmospheric_scattering_haze": AtmosphericScatteringHaze,
        "six_dof_viewpoint_blur": SixDoFViewpointBlur,
        "communication_packet_loss": CommunicationPacketLoss,
        "low_res_super_resolution": LowResSuperResolution,
        "propeller_shadow": PropellerShadow,
    }

    GENERIC_DISTORTIONS = list(GenericDistortions.CATEGORIES.keys())

    INTENSITY_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._dist_cache = {}
        for name in self.get_all_distortion_names():
            if name in self.UAV_DISTORTIONS:
                self._dist_cache[name] = self.UAV_DISTORTIONS[name](seed=self.seed)
            else:
                self._dist_cache[name] = GenericDistortions(name, seed=self.seed)

    def apply_distortion(
        self, image: np.ndarray, distortion_name: str, intensity: float
    ) -> np.ndarray:
        if distortion_name not in self._dist_cache:
            raise ValueError(f"Unknown distortion: {distortion_name}")
        return self._dist_cache[distortion_name].apply(image, intensity)

    def generate_all(
        self,
        reference_image: np.ndarray,
        distortion_types: Optional[list] = None,
        intensity: Optional[float] = None,
    ) -> dict:
        if distortion_types is None:
            distortion_types = list(self.UAV_DISTORTIONS.keys()) + self.GENERIC_DISTORTIONS

        rng = np.random.RandomState(self.seed)
        results = {}
        for dist_name in distortion_types:
            level = intensity if intensity is not None else float(rng.choice(self.INTENSITY_LEVELS))
            key = f"{dist_name}_L{int(level * 10):02d}"
            results[key] = self.apply_distortion(reference_image, dist_name, level)
        return results

    @staticmethod
    def get_all_distortion_names() -> list:
        return list(UAVDistortionPipeline.UAV_DISTORTIONS.keys()) + list(
            UAVDistortionPipeline.GENERIC_DISTORTIONS
        )

    @staticmethod
    def get_uav_distortion_names() -> list:
        return list(UAVDistortionPipeline.UAV_DISTORTIONS.keys())

    @staticmethod
    def get_distortion_categories() -> dict[str, str]:
        """Return ``{distortion_name: category}`` for all 36 distortion types.

        UAV distortions have category ``"uav"``; generic distortions use
        their original category from ``GenericDistortions.CATEGORIES``
        (blur, brightness, chromatic, noise, compression, spatial, other,
        transmission).
        """
        cats = {name: "uav" for name in UAVDistortionPipeline.UAV_DISTORTIONS}
        cats.update(GenericDistortions.CATEGORIES)
        return cats


# Module-level constant — computed once, import everywhere
UAV_DISTORTION_NAMES = frozenset(UAVDistortionPipeline.get_uav_distortion_names())
