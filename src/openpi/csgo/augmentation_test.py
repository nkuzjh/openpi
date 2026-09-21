"""Tests for deterministic Seen-10 training-only FPV dropout."""

from __future__ import annotations

import numpy as np
import pytest

from openpi.csgo.augmentation import Seen10TrainFPVDropout
from openpi.csgo.augmentation import make_seen10_fpv_dropout


def _sample() -> dict[str, object]:
    return {
        "image": {
            "base_0_rgb": np.full((128, 96, 3), 127, dtype=np.uint8),
            "left_wrist_0_rgb": np.full((128, 96, 3), 64, dtype=np.uint8),
            "right_wrist_0_rgb": np.zeros((128, 96, 3), dtype=np.uint8),
        },
        "image_mask": {
            "base_0_rgb": True,
            "left_wrist_0_rgb": True,
            "right_wrist_0_rgb": False,
        },
        "state": np.zeros((5,), dtype=np.float32),
        "prompt": "localize",
        "actions": np.zeros((1, 5), dtype=np.float32),
    }


def test_factory_enables_dropout_only_for_seen_train():
    train_transform = make_seen10_fpv_dropout("seen_train", seed=5)
    assert isinstance(train_transform, Seen10TrainFPVDropout)
    assert make_seen10_fpv_dropout("train", seed=5) is not None
    assert make_seen10_fpv_dropout("seen_validation", seed=5) is None
    assert make_seen10_fpv_dropout("seen_discrete_test", seed=5) is None
    with pytest.raises(ValueError, match="Unsupported"):
        make_seen10_fpv_dropout("unknown", seed=5)


def test_dropout_is_seed_reproducible_and_only_changes_fpv():
    sample = _sample()
    original_fpv = sample["image"]["base_0_rgb"].copy()
    original_radar = sample["image"]["left_wrist_0_rgb"].copy()
    first = make_seen10_fpv_dropout("seen_train", seed=0)(sample)
    second = make_seen10_fpv_dropout("seen_train", seed=0)(sample)

    np.testing.assert_array_equal(first["image"]["base_0_rgb"], second["image"]["base_0_rgb"])
    assert np.any(first["image"]["base_0_rgb"] != original_fpv)
    np.testing.assert_array_equal(first["image"]["left_wrist_0_rgb"], original_radar)
    np.testing.assert_array_equal(first["image"]["right_wrist_0_rgb"], sample["image"]["right_wrist_0_rgb"])
    # The transform returns copies and never edits the dataset sample in place.
    np.testing.assert_array_equal(sample["image"]["base_0_rgb"], original_fpv)


def test_masked_fpv_is_untouched_and_default_parameters_match_training_recipe():
    sample = _sample()
    sample["image_mask"]["base_0_rgb"] = False
    transform = Seen10TrainFPVDropout(seed=3)
    result = transform(sample)
    np.testing.assert_array_equal(result["image"]["base_0_rgb"], sample["image"]["base_0_rgb"])

    assert (transform.coarse_p, transform.max_holes, transform.max_hole_height, transform.max_hole_width) == (
        0.5,
        8,
        16,
        16,
    )
    assert (transform.grid_size, transform.grid_p, transform.grid_cell_p) == (4, 0.3, 0.5)
    assert (transform.erase_p, transform.erase_area, transform.erase_aspect_ratio, transform.fill_value) == (
        0.6,
        (0.02, 0.4),
        (0.3, 1.0 / 0.3),
        0,
    )
