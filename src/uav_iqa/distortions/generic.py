import math
from typing import Optional

import cv2
import numpy as np

from .base import _load_image, _to_uint8


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
