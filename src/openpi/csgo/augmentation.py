"""Train-only FPS occlusion transforms for the Seen-10 data adapter.

The old ``is_fps_dropout`` training path applied these three operations to
the raw FPV tensor, before resize: CoarseDropout, GridDropout, then
RandomErasing. This callable has the same per-sample placement and can be
composed with native OpenPI transforms. It touches only ``base_0_rgb`` and is
created only for the training split. Pi0.5 crop, rotation, and color jitter
remain the model's native ``preprocess_observation(train=True)`` behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import math

import numpy as np

from openpi.csgo import data as _data


@dataclasses.dataclass
class Seen10TrainFPVDropout:
    """Apply the historical CSGO FPS dropout sequence to one training sample.

    A local NumPy generator keeps augmentation independent of global random
    state. Recreating this transform with the same seed and processing the
    same sample order produces the same augmented images.
    """

    seed: int = 0
    coarse_p: float = 0.5
    max_holes: int = 8
    max_hole_height: int = 16
    max_hole_width: int = 16
    grid_p: float = 0.3
    grid_size: int = 4
    grid_cell_p: float = 0.5
    erase_p: float = 0.6
    erase_area: tuple[float, float] = (0.02, 0.4)
    erase_aspect_ratio: tuple[float, float] = (0.3, 1.0 / 0.3)
    fill_value: int | float = 0
    _rng: np.random.Generator = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.max_holes < 1 or self.max_hole_height < 1 or self.max_hole_width < 1:
            raise ValueError("CoarseDropout hole counts and sizes must be positive")
        if self.grid_size < 1:
            raise ValueError("grid_size must be positive")
        for name, probability in (
            ("coarse_p", self.coarse_p),
            ("grid_p", self.grid_p),
            ("grid_cell_p", self.grid_cell_p),
            ("erase_p", self.erase_p),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {probability}")
        area_low, area_high = self.erase_area
        ratio_low, ratio_high = self.erase_aspect_ratio
        if not 0.0 < area_low <= area_high <= 1.0:
            raise ValueError(f"erase_area must satisfy 0 < low <= high <= 1, got {self.erase_area}")
        if not 0.0 < ratio_low <= ratio_high:
            raise ValueError(f"erase_aspect_ratio must be positive and ordered, got {self.erase_aspect_ratio}")
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, sample: Mapping[str, object]) -> dict[str, object]:
        """Return a shallow-copied sample with only the valid FPV view changed."""

        if "image" not in sample or not isinstance(sample["image"], Mapping):
            raise ValueError("Expected a sample mapping with an 'image' mapping")
        images = sample["image"]
        if "base_0_rgb" not in images:
            raise ValueError("Expected the FPV image key 'base_0_rgb'")
        image_mask = sample.get("image_mask", {})
        if (
            isinstance(image_mask, Mapping)
            and "base_0_rgb" in image_mask
            and not bool(np.asarray(image_mask["base_0_rgb"]))
        ):
            return dict(sample)

        image = np.asarray(images["base_0_rgb"])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected an HWC RGB FPV image, got {image.shape}")
        image = np.array(image, copy=True)
        self._coarse_dropout(image)
        self._grid_dropout(image)
        self._random_erasing(image)

        result = dict(sample)
        result_images = dict(images)
        result_images["base_0_rgb"] = image
        result["image"] = result_images
        return result

    def _coarse_dropout(self, image: np.ndarray) -> None:
        if self._rng.random() >= self.coarse_p:
            return
        height, width = image.shape[:2]
        max_height = min(self.max_hole_height, height)
        max_width = min(self.max_hole_width, width)
        for _ in range(int(self._rng.integers(1, self.max_holes + 1))):
            hole_height = int(self._rng.integers(1, max_height + 1))
            hole_width = int(self._rng.integers(1, max_width + 1))
            top = int(self._rng.integers(0, height - hole_height + 1))
            left = int(self._rng.integers(0, width - hole_width + 1))
            image[top : top + hole_height, left : left + hole_width, :] = self.fill_value

    def _grid_dropout(self, image: np.ndarray) -> None:
        if self._rng.random() >= self.grid_p:
            return
        height, width = image.shape[:2]
        cell_height = height // self.grid_size
        cell_width = width // self.grid_size
        if cell_height == 0 or cell_width == 0:
            return
        for row in range(self.grid_size):
            for column in range(self.grid_size):
                if self._rng.random() < self.grid_cell_p:
                    top = row * cell_height
                    bottom = min((row + 1) * cell_height, height)
                    left = column * cell_width
                    right = min((column + 1) * cell_width, width)
                    image[top:bottom, left:right, :] = self.fill_value

    def _random_erasing(self, image: np.ndarray) -> None:
        if self._rng.random() >= self.erase_p:
            return
        height, width = image.shape[:2]
        image_area = height * width
        area_low, area_high = self.erase_area
        ratio_low, ratio_high = self.erase_aspect_ratio
        for _ in range(100):
            target_area = self._rng.uniform(area_low, area_high) * image_area
            aspect_ratio = self._rng.uniform(ratio_low, ratio_high)
            erase_height = round(math.sqrt(target_area * aspect_ratio))
            erase_width = round(math.sqrt(target_area / aspect_ratio))
            if 0 < erase_width < width and 0 < erase_height < height:
                top = int(self._rng.integers(0, height - erase_height + 1))
                left = int(self._rng.integers(0, width - erase_width + 1))
                image[top : top + erase_height, left : left + erase_width, :] = self.fill_value
                return


def make_seen10_fpv_dropout(split: str, *, seed: int = 0) -> Seen10TrainFPVDropout | None:
    """Return the seeded FPS dropout transform for train, and ``None`` for eval.

    Split aliases accepted by :class:`Seen10Dataset` are accepted here too.
    Returning ``None`` for validation/test means a shared transform pipeline
    can append the result conditionally without changing eval/inference images.
    """

    try:
        canonical_split = _data.SPLIT_ALIASES[split]
    except (KeyError, TypeError) as exc:
        choices = ", ".join(sorted(_data.SPLIT_ALIASES))
        raise ValueError(f"Unsupported CSGO Seen-10 split {split!r}; choose from {choices}") from exc
    if canonical_split != "seen_train":
        return None
    return Seen10TrainFPVDropout(seed=seed)


__all__ = ["Seen10TrainFPVDropout", "make_seen10_fpv_dropout"]
