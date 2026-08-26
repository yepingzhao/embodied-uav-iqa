import math
from abc import ABC, abstractmethod
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
