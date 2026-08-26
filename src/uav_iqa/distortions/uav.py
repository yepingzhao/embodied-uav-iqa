import math
from typing import Optional

import cv2
import numpy as np

from .base import BaseDistortion, _filter2d_wrap, _load_image, _motion_blur_kernel, _to_uint8


class PropellerVibrationBlur(BaseDistortion):
    """Propeller vibration blur: directional motion blur with periodic intensity modulation.

    f ∈ [80, 200] Hz, amplitude A ∈ [1, 8] pixels, θ ∼ Uniform(0, 2π).
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def apply(self, image: np.ndarray, intensity: float) -> np.ndarray:
        image = _load_image(image)
        params = self.get_param_range(intensity)
        h, w = image.shape[:2]
        rng = self.rng

        angle = rng.uniform(0, 360)
        kernel = _motion_blur_kernel(int(params["size"]), angle)

        result = _filter2d_wrap(image, kernel)

        f = params["freq"]
        period_px = max(1, w / (f / 60.0))
        t = rng.uniform(0, 2 * math.pi)
        modulation = 1.0 + params["amp"] * 0.15 * np.sin(
            2 * math.pi * np.arange(h)[:, None] / period_px + t
        )
        modulation = np.clip(modulation, 0.7, 1.3)
        result = np.clip(result * modulation[:, :, None], 0, 1)
        return _to_uint8(result)

    def get_param_range(self, intensity: float) -> dict:
        intensity = np.clip(intensity, 0.0, 1.0)
        return {
            "freq": 80 + intensity * 120,
            "amp": 1.0 + intensity * 7.0,
            "size": int(3 + intensity * 12),
        }


class AtmosphericScatteringHaze(BaseDistortion):
    """Atmospheric scattering/haze via Koschmieder model.

    β ∈ [0.5, 3.0], A_∞ is atmospheric light.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def apply(self, image: np.ndarray, intensity: float) -> np.ndarray:
        image = _load_image(image)
        params = self.get_param_range(intensity)
        h, w = image.shape[:2]

        depth = self._estimate_depth(image)
        depth_norm = depth / (depth.max() + 1e-8)

        beta = params["beta"]
        A_inf = params["A_inf"]

        transmission = np.exp(-beta * depth_norm)
        transmission = transmission[:, :, None] if image.ndim == 3 else transmission

        hazy = image * transmission + A_inf * (1 - transmission)
        return _to_uint8(np.clip(hazy, 0, 1))

    def _estimate_depth(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(_to_uint8(image), cv2.COLOR_RGB2GRAY)
        depth = 255.0 / (gray.astype(np.float32) + 1e-3)
        return cv2.GaussianBlur(depth, (15, 15), 10)

    def get_param_range(self, intensity: float) -> dict:
        intensity = np.clip(intensity, 0.0, 1.0)
        return {
            "beta": 0.5 + intensity * 2.5,
            "A_inf": 0.6 + self.rng.uniform(0, 0.3),
        }


class SixDoFViewpointBlur(BaseDistortion):
    """6DoF fast viewpoint change blur: motion vectors from MotionScape empirical flow.

    μ=36.63 px, σ=25.4 px optical flow distribution.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def apply(self, image: np.ndarray, intensity: float) -> np.ndarray:
        image = _load_image(image)
        params = self.get_param_range(intensity)

        magnitude = params["magnitude"]
        angle = self.rng.uniform(0, 360)

        size = max(3, min(31, int(magnitude)))
        if size % 2 == 0:
            size += 1

        kernel = _motion_blur_kernel(size, angle)

        result = _filter2d_wrap(image, kernel)

        return _to_uint8(np.clip(result, 0, 1))

    def get_param_range(self, intensity: float) -> dict:
        intensity = np.clip(intensity, 0.0, 1.0)
        magnitude = 10.0 + intensity * 60.0
        magnitude += self.rng.normal(0, 12.7)
        return {"magnitude": max(0, magnitude)}


class CommunicationPacketLoss(BaseDistortion):
    """Communication packet-loss block artifacts: random 16×16 macroblock replacement.

    Loss rate ∈ [1%, 30%].
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def apply(self, image: np.ndarray, intensity: float) -> np.ndarray:
        image = _load_image(image)
        params = self.get_param_range(intensity)
        h, w = image.shape[:2]
        block_size = 16

        h_blocks = math.ceil(h / block_size)
        w_blocks = math.ceil(w / block_size)

        mask = np.ones((h_blocks, w_blocks), dtype=bool)
        for bi in range(h_blocks):
            for bj in range(w_blocks):
                if self.rng.random() < params["loss_rate"]:
                    mask[bi, bj] = False

        result = image.copy()
        for bi in range(h_blocks):
            for bj in range(w_blocks):
                if mask[bi, bj]:
                    continue
                y1 = bi * block_size
                y2 = min((bi + 1) * block_size, h)
                x1 = bj * block_size
                x2 = min((bj + 1) * block_size, w)

                src = self._find_nearest_valid(mask, bi, bj, h_blocks, w_blocks)
                if src is not None:
                    sy1 = src[0] * block_size
                    sy2 = min((src[0] + 1) * block_size, h)
                    sx1 = src[1] * block_size
                    sx2 = min((src[1] + 1) * block_size, w)
                    block_h = min(y2 - y1, sy2 - sy1)
                    block_w = min(x2 - x1, sx2 - sx1)
                    result[y1 : y1 + block_h, x1 : x1 + block_w] = image[
                        sy1 : sy1 + block_h, sx1 : sx1 + block_w
                    ]
                else:
                    result[y1:y2, x1:x2] = 0

        return _to_uint8(result)

    def _find_nearest_valid(self, mask, bi, bj, h_blocks, w_blocks):
        best = None
        best_dist = float("inf")
        for si in range(h_blocks):
            for sj in range(w_blocks):
                if mask[si, sj]:
                    dist = (si - bi) ** 2 + (sj - bj) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best = (si, sj)
        return best

    def get_param_range(self, intensity: float) -> dict:
        intensity = np.clip(intensity, 0.0, 1.0)
        return {"loss_rate": 0.01 + intensity * 0.29}


class LowResSuperResolution(BaseDistortion):
    """Low-resolution + super-resolution artifacts.

    Bicubic downsample → Real-ESRGAN upscale, s ∈ [2, 8].
    """

    def __init__(self, seed: Optional[int] = None, model_path: Optional[str] = None):
        self.rng = np.random.RandomState(seed)
        self._sr_model = None
        self._model_path = model_path

    def _get_sr_model(self):
        if self._sr_model is None and self._model_path is not None:
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer

                model = RRDBNet(num_in_ch=3, num_out_ch=3)
                upsampler = RealESRGANer(
                    scale=4,
                    model_path=self._model_path,
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=False,
                )
                self._sr_model = upsampler
            except ImportError:
                pass
        return self._sr_model

    def apply(self, image: np.ndarray, intensity: float) -> np.ndarray:
        image_uint8 = _to_uint8(_load_image(image))
        params = self.get_param_range(intensity)
        h, w = image_uint8.shape[:2]
        scale = params["scale"]

        small_h = max(4, int(h / scale))
        small_w = max(4, int(w / scale))
        small = cv2.resize(image_uint8, (small_w, small_h), interpolation=cv2.INTER_CUBIC)

        sr_model = self._get_sr_model()
        if sr_model is not None:
            result, _ = sr_model.enhance(small, outscale=4)
            result = cv2.resize(result, (w, h))
        else:
            result = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
            ksize = int(1 + intensity * 3)
            if ksize % 2 == 0:
                ksize += 1
            result = cv2.GaussianBlur(result, (ksize, ksize), intensity * 2)
            kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 1.0
            result = cv2.filter2D(result, -1, kernel_sharpen * (1 + intensity))

        return _to_uint8(result)

    def get_param_range(self, intensity: float) -> dict:
        intensity = np.clip(intensity, 0.0, 1.0)
        return {"scale": 2.0 + intensity * 6.0}


class PropellerShadow(BaseDistortion):
    """Propeller shadow: periodic localized brightness modulation.

    α ∈ [0.05, 0.3], spatially localized depending on sun angle.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def apply(self, image: np.ndarray, intensity: float) -> np.ndarray:
        image = _load_image(image)
        params = self.get_param_range(intensity)
        h, w = image.shape[:2]
        alpha = params["alpha"]

        freq = 20 + self.rng.uniform(0, 40)
        t = self.rng.uniform(0, 2 * math.pi)
        x = np.arange(w) / w
        y = np.arange(h) / h
        xx, yy = np.meshgrid(x, y)

        sun_angle = self.rng.uniform(0, 2 * math.pi)
        proj = xx * math.cos(sun_angle) + yy * math.sin(sun_angle)
        pulse = 0.5 * (1 + np.sign(np.sin(2 * math.pi * freq * proj + t)))
        pulse = 1.0 - alpha * pulse

        cx = self.rng.uniform(0.2, 0.8)
        cy = self.rng.uniform(0.2, 0.8)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        sigma = 0.3 + self.rng.uniform(0, 0.2)
        spatial_mask = np.exp(-(dist**2) / (2 * sigma**2))

        shadow = 1.0 - spatial_mask * (1.0 - pulse)
        shadow = np.clip(shadow, 0.6, 1.0)

        result = image * shadow[:, :, None]
        return _to_uint8(np.clip(result, 0, 1))

    def get_param_range(self, intensity: float) -> dict:
        intensity = np.clip(intensity, 0.0, 1.0)
        return {"alpha": 0.05 + intensity * 0.25}
