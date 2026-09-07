"""Dataset-specific reference-image discovery adapters."""

from abc import ABC, abstractmethod
from pathlib import Path

from uav_iqa.utils import find_images


class DatasetAdapter(ABC):
    """Registry-backed interface for dataset-specific image discovery."""

    _registry: dict[str, type["DatasetAdapter"]] = {}
    name: str

    @classmethod
    def register(cls, adapter_cls: type["DatasetAdapter"]) -> type["DatasetAdapter"]:
        cls._registry[adapter_cls.name] = adapter_cls
        return adapter_cls

    @classmethod
    def get(cls, name: str) -> type["DatasetAdapter"]:
        if name not in cls._registry:
            raise ValueError(f"Unknown dataset adapter: '{name}'. Available: {cls.list_adapters()}")
        return cls._registry[name]

    @classmethod
    def list_adapters(cls) -> list[str]:
        return sorted(cls._registry)

    @abstractmethod
    def get_exclude_dirs(self) -> set[str]:
        """Return directory names excluded during image discovery."""

    @abstractmethod
    def find_reference_images(self, input_root: Path) -> list[Path]:
        """Return reference images for this dataset layout."""


@DatasetAdapter.register
class AirCopBenchAdapter(DatasetAdapter):
    name = "aircopbench"
    EXCLUDE_DIRS = {
        "loss", "noise", "point_clouds", "Annotations", "original_json", "original_xml",
        "train", "test", ".git", "distorted",
    }

    def get_exclude_dirs(self) -> set[str]:
        return self.EXCLUDE_DIRS

    def find_reference_images(self, input_root: Path) -> list[Path]:
        return find_images(input_root, exclude_dirs=self.EXCLUDE_DIRS)


@DatasetAdapter.register
class ImageDirectoryAdapter(DatasetAdapter):
    name = "generic"

    def get_exclude_dirs(self) -> set[str]:
        return set()

    def find_reference_images(self, input_root: Path) -> list[Path]:
        return find_images(input_root)
