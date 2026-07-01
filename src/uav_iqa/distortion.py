import math
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np


def _load_image(image, mode="float32"):
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if mode == "float32":
        image = image.astype(np.float32) / 255.0
    return image


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def _motion_blur_kernel(size: int, angle_deg: float) -> np.ndarray:
    kernel = np.zeros((size, size))
    angle_rad = math.radians(angle_deg)
    center = size // 2
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    for i in range(size):
        x = int(center + (i - center) * cos_a)
        y = int(center + (i - center) * sin_a)
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
    kernel /= kernel.sum()
    return kernel


def _filter2d_wrap(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """cv2.filter2D with wrap/circular boundary via manual padding."""
    kh, kw = kernel.shape[:2]
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw), (0, 0)), mode="wrap")
    result = cv2.filter2D(padded, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return result[ph : ph + image.shape[0], pw : pw + image.shape[1]]


class BaseDistortion(ABC):
    @abstractmethod
    def apply(self, image: np.ndarray, intensity: float) -> np.ndarray:
        pass

    @abstractmethod
    def get_param_range(self, intensity: float) -> dict:
        pass


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


class GenericDistortions:
    """30 generic distortion types from Embodied-IQA catalog, applied via Albumentations."""

    CATEGORIES = {
        "gaussian_blur": "blur",
        "lens_blur": "blur",
        "motion_blur": "blur",
        "brighten_max": "brightness",
        "brighten_avg": "brightness",
        "darken_max": "brightness",
        "darken_min": "brightness",
        "darken_avg": "brightness",
        "color_diffusion": "chromatic",
        "color_shift": "chromatic",
        "color_quantize": "chromatic",
        "white_noise": "noise",
        "color_noise": "noise",
        "impulse_noise": "noise",
        "multiplicative_noise": "noise",
        "gaussian_denoise": "noise",
        "cnn_denoise": "noise",
        "jpeg_compression": "compression",
        "jp2k_compression": "compression",
        "webp_compression": "compression",
        "spatial_warp": "spatial",
        "spatial_scale": "spatial",
        "clock_jittering": "spatial",
        "resolution_limit": "other",
        "grayscale": "other",
        "sharpness": "other",
        "contrast": "other",
        "block_lost": "transmission",
        "block_interpolation": "transmission",
        "block_exchange": "transmission",
    }

    def __init__(self, distortion_type: str, seed: Optional[int] = None):
        if distortion_type not in self.CATEGORIES:
            raise ValueError(f"Unknown distortion: {distortion_type}")
        self.name = distortion_type
        self.rng = np.random.RandomState(seed)

    def apply(self, image: np.ndarray, intensity: float) -> np.ndarray:
        try:
            import albumentations as A
        except ImportError:
            return _load_image(image)

        image = _load_image(image)
        intensity = np.clip(intensity, 0.0, 1.0)
        name = self.name

        if name == "gaussian_blur":
            blur_limit = int(3 + intensity * 15)
            if blur_limit % 2 == 0:
                blur_limit += 1
            transform = A.GaussianBlur(blur_limit=(blur_limit, blur_limit), p=1.0)
        elif name == "lens_blur":
            radius = 2 + int(intensity * 8)
            transform = A.Defocus(radius=(radius, radius), alias_blur=(0.1, 0.5), p=1.0)
        elif name == "motion_blur":
            blur_limit = 3 + int(intensity * 15)
            if blur_limit % 2 == 0:
                blur_limit += 1
            transform = A.MotionBlur(blur_limit=(blur_limit, blur_limit), p=1.0)
        elif name.startswith("brighten"):
            limit = 0.05 + intensity * 0.45
            transform = A.RandomBrightnessContrast(
                brightness_limit=(limit, limit), contrast_limit=0, p=1.0
            )
        elif name.startswith("darken"):
            limit = -(0.05 + intensity * 0.45)
            transform = A.RandomBrightnessContrast(
                brightness_limit=(limit, limit), contrast_limit=0, p=1.0
            )
        elif name == "color_diffusion":
            hue = int(intensity * 40)
            transform = A.HueSaturationValue(
                hue_shift_limit=(hue, hue), sat_shift_limit=0, val_shift_limit=0, p=1.0
            )
        elif name == "color_shift":
            sat = int(intensity * 60)
            transform = A.HueSaturationValue(
                hue_shift_limit=0, sat_shift_limit=(sat, sat), val_shift_limit=0, p=1.0
            )
        elif name == "color_quantize":
            num_colors = max(2, int(256 - intensity * 200))
            num_color_bits = max(1, int(math.log2(num_colors)))
            image = _to_uint8(image)
            for c in range(3):
                image[:, :, c] = (image[:, :, c] >> (8 - num_color_bits)) << (8 - num_color_bits)
            image = image.astype(np.float32) / 255.0
            transform = A.NoOp(p=1.0)
        elif name == "white_noise":
            transform = A.GaussNoise(
                std_range=(intensity * 0.05, intensity * 0.15),
                mean_range=(0, 0),
                per_channel=True,
                p=1.0,
            )
        elif name == "color_noise":
            transform = A.ISONoise(
                color_shift=(0.01, intensity * 0.1),
                intensity=(intensity * 0.1, intensity * 0.3),
                p=1.0,
            )
        elif name == "impulse_noise":
            amt = intensity * 0.1
            transform = A.SaltAndPepper(amount=(amt, amt + 0.02), salt_vs_pepper=(0.4, 0.6), p=1.0)
        elif name == "multiplicative_noise":
            multiplier = (1.0 - intensity * 0.3, 1.0 + intensity * 0.3)
            transform = A.MultiplicativeNoise(multiplier=multiplier, p=1.0)
        elif name == "gaussian_denoise":
            sigma = 1.0 + intensity * 5.0
            ksize = int(sigma * 2) | 1
            ksize = max(3, min(ksize, 31))
            image_uint8 = _to_uint8(image)
            image = cv2.GaussianBlur(image_uint8, (ksize, ksize), sigma).astype(np.float32) / 255.0
            transform = A.NoOp(p=1.0)
        elif name == "cnn_denoise":
            h_param = 3 + int(intensity * 30)
            template_size = 3 + int(intensity * 6) | 1
            search_size = min(15, template_size + 4) | 1
            image_uint8 = _to_uint8(image)
            image = (
                cv2.fastNlMeansDenoisingColored(
                    image_uint8, None, h_param, h_param, template_size, search_size
                ).astype(np.float32)
                / 255.0
            )
            transform = A.NoOp(p=1.0)
        elif name == "jpeg_compression":
            q = max(5, int(90 - intensity * 85))
            transform = A.ImageCompression(
                quality_range=(q, min(q + 10, 100)), compression_type="jpeg", p=1.0
            )
        elif name == "jp2k_compression":
            q = max(1, int(90 - intensity * 85))
            image_uint8 = _to_uint8(image)
            try:
                encode_param = [cv2.IMWRITE_JPEG2000_QUALITY, q]
                _, enc = cv2.imencode(
                    ".jp2", cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR), encode_param
                )
                decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            except AttributeError:
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, q]
                _, enc = cv2.imencode(
                    ".jpg", cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR), encode_param
                )
                decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            transform = A.NoOp(p=1.0)
        elif name == "webp_compression":
            q = max(5, int(90 - intensity * 85))
            transform = A.ImageCompression(quality_range=(q, q), compression_type="webp", p=1.0)
        elif name == "spatial_warp":
            transform = A.ElasticTransform(alpha=intensity * 200, sigma=intensity * 20 + 5, p=1.0)
        elif name == "spatial_scale":
            scale_limit = (1.0 - intensity * 0.3, 1.0 + intensity * 0.3)
            transform = A.Affine(scale=scale_limit, p=1.0)
        elif name == "clock_jittering":
            max_jitter = 1 + int(intensity * 30)
            h, w = image.shape[:2]
            image_uint8 = _to_uint8(image)
            result = np.zeros_like(image_uint8)
            for row in range(h):
                offset = self.rng.randint(-max_jitter, max_jitter + 1)
                if offset > 0:
                    result[row, offset:w] = image_uint8[row, : w - offset]
                    result[row, :offset] = image_uint8[row, :1]
                elif offset < 0:
                    offset = -offset
                    result[row, : w - offset] = image_uint8[row, offset:w]
                    result[row, w - offset :] = image_uint8[row, -1:]
                else:
                    result[row] = image_uint8[row]
            image = result.astype(np.float32) / 255.0
            transform = A.NoOp(p=1.0)
        elif name == "resolution_limit":
            scale = 1.0 - intensity * 0.8
            new_h = max(4, int(image.shape[0] * scale))
            new_w = max(4, int(image.shape[1] * scale))
            small = cv2.resize(_to_uint8(image), (new_w, new_h))
            image = cv2.resize(small, (image.shape[1], image.shape[0])).astype(np.float32) / 255.0
            transform = A.NoOp(p=1.0)
        elif name == "grayscale":
            gray = cv2.cvtColor(_to_uint8(image), cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
            transform = A.NoOp(p=1.0)
        elif name == "sharpness":
            limit = intensity * 1.0
            transform = A.Sharpen(alpha=(limit, limit), lightness=(0, 0), p=1.0)
        elif name == "contrast":
            limit = (1.0 - intensity * 0.5, 1.0 + intensity * 0.5)
            transform = A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=limit, p=1.0)
        elif name == "block_lost":
            loss_rate = 0.01 + intensity * 0.29
            h, w = image.shape[:2]
            bs = 16
            hb, wb = math.ceil(h / bs), math.ceil(w / bs)
            image_uint8 = _to_uint8(image)
            for bi in range(hb):
                for bj in range(wb):
                    if self.rng.random() < loss_rate:
                        y1, y2 = bi * bs, min((bi + 1) * bs, h)
                        x1, x2 = bj * bs, min((bj + 1) * bs, w)
                        image_uint8[y1:y2, x1:x2] = (128, 128, 128)
            image = image_uint8.astype(np.float32) / 255.0
            transform = A.NoOp(p=1.0)
        elif name == "block_interpolation":
            loss_rate = 0.01 + intensity * 0.29
            h, w = image.shape[:2]
            bs = 16
            hb, wb = math.ceil(h / bs), math.ceil(w / bs)
            image_uint8 = _to_uint8(image)
            for bi in range(hb):
                for bj in range(wb):
                    if self.rng.random() >= loss_rate:
                        continue
                    y1, y2 = bi * bs, min((bi + 1) * bs, h)
                    x1, x2 = bj * bs, min((bj + 1) * bs, w)
                    bh, bw = y2 - y1, x2 - x1
                    top = image_uint8[y1 - 1, x1:x2] if bi > 0 else image_uint8[y1, x1:x2]
                    bot = image_uint8[y2, x1:x2] if bi + 1 < hb else image_uint8[y2 - 1, x1:x2]
                    top = top.reshape(1, bw, 3).astype(np.float32)
                    bot = bot.reshape(1, bw, 3).astype(np.float32)
                    inter = np.linspace(0, 1, bh).reshape(-1, 1, 1)
                    filled = (1 - inter) * top + inter * bot
                    image_uint8[y1:y2, x1:x2] = np.clip(filled, 0, 255).astype(np.uint8)
            image = image_uint8.astype(np.float32) / 255.0
            transform = A.NoOp(p=1.0)
        elif name == "block_exchange":
            loss_rate = 0.01 + intensity * 0.29
            h, w = image.shape[:2]
            bs = 16
            hb, wb = math.ceil(h / bs), math.ceil(w / bs)
            image_uint8 = _to_uint8(image)
            total_blocks = hb * wb
            num_swaps = max(1, int(total_blocks * loss_rate / 2))
            block_indices = list(range(total_blocks))
            for _ in range(num_swaps):
                if len(block_indices) < 2:
                    break
                i = self.rng.randint(0, len(block_indices))
                a_idx = block_indices.pop(i)
                j = self.rng.randint(0, len(block_indices))
                b_idx = block_indices.pop(j)
                bi_a, bj_a = divmod(a_idx, wb)
                bi_b, bj_b = divmod(b_idx, wb)
                y1_a, y2_a = bi_a * bs, min((bi_a + 1) * bs, h)
                x1_a, x2_a = bj_a * bs, min((bj_a + 1) * bs, w)
                y1_b, y2_b = bi_b * bs, min((bi_b + 1) * bs, h)
                x1_b, x2_b = bj_b * bs, min((bj_b + 1) * bs, w)
                bh = min(y2_a - y1_a, y2_b - y1_b)
                bw = min(x2_a - x1_a, x2_b - x1_b)
                tmp = image_uint8[y1_a : y1_a + bh, x1_a : x1_a + bw].copy()
                image_uint8[y1_a : y1_a + bh, x1_a : x1_a + bw] = image_uint8[
                    y1_b : y1_b + bh, x1_b : x1_b + bw
                ]
                image_uint8[y1_b : y1_b + bh, x1_b : x1_b + bw] = tmp
            image = image_uint8.astype(np.float32) / 255.0
            transform = A.NoOp(p=1.0)
        else:
            transform = A.NoOp(p=1.0)

        result = transform(image=image)["image"]
        return _to_uint8(result)



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
