"""Core CSGO Seen-10 data/visualization contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from openpi.csgo.data import (
    EXPECTED_PER_MAP_COUNTS,
    EXPECTED_SPLIT_COUNTS,
    SEEN_MAPS,
    Seen10Dataset,
    physical_pose,
    physical_to_normalized_pose,
)
from openpi.csgo.visualization import (
    fixed_sample_indices,
    map_pixel_from_normalized,
    render_prediction_maps,
)


DATA_ROOT = Path("/home/jiahao/task/UniLIP/data/csgo_benchmark_v2")
SHARED_EVAL_DIR = Path("/home/jiahao/task/csgo_benchmark_v2_eval_general")


def _release_available() -> bool:
    return (DATA_ROOT / "benchmark_manifest.json").is_file() and (SHARED_EVAL_DIR / "protocol.py").is_file()


pytestmark = pytest.mark.skipif(not _release_available(), reason="CSGO Benchmark v2 release is unavailable")


def _dataset(split: str, **kwargs) -> Seen10Dataset:
    return Seen10Dataset(DATA_ROOT, split, shared_eval_dir=SHARED_EVAL_DIR, **kwargs)


def test_real_split_counts_and_normalization_match_shared_protocol():
    """All formal split counts and one-to-one normalized row values are checked."""

    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        dataset = _dataset(split, include_actions=True)
        assert len(dataset) == expected_count
        assert dataset.counts == {
            map_name: EXPECTED_PER_MAP_COUNTS[split]
            for map_name in SEEN_MAPS
        }

        reference_rows = dataset.benchmark_data.rows(split, max_samples=3)
        for actual, reference in zip(dataset.rows[:3], reference_rows, strict=True):
            assert actual["sample_id"] == reference["sample_id"]
            assert actual["map_name"] == reference["map_name"]
            assert Path(actual["image_path"]).is_absolute()
            assert Path(actual["radar_path"]).is_absolute()
            raw = actual["pose_raw"]
            bounds = actual["z_calibration"]
            expected_pose = np.asarray(
                [
                    raw["x"] / 1024.0,
                    raw["y"] / 1024.0,
                    (raw["z"] - bounds["z_min"]) / (bounds["z_max"] - bounds["z_min"]),
                    raw["angle_v_rad"] / (2.0 * np.pi),
                    raw["angle_h_rad"] / (2.0 * np.pi),
                ],
                dtype=np.float64,
            )
            np.testing.assert_allclose(actual["pose"], expected_pose, rtol=0.0, atol=1e-12)
            assert actual["z_calibration"] == reference["z_calibration"]


def test_model_input_has_no_identity_or_ground_truth_and_action_contract():
    dataset = _dataset("seen_train", include_actions=True, limit=2)
    model_input = dataset.input_at(0)
    assert set(model_input) == {"image", "image_mask", "state", "prompt"}
    assert set(model_input["image"]) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }
    assert set(model_input["image_mask"]) == set(model_input["image"])
    assert model_input["image"]["base_0_rgb"].dtype == np.uint8
    assert model_input["image"]["left_wrist_0_rgb"].dtype == np.uint8
    assert model_input["image"]["base_0_rgb"].ndim == 3
    assert model_input["image"]["base_0_rgb"].shape[-1] == 3
    assert not bool(model_input["image_mask"]["right_wrist_0_rgb"])
    assert np.all(model_input["image"]["right_wrist_0_rgb"] == 0)
    assert model_input["state"].shape == (5,)
    assert model_input["state"].dtype == np.float32
    assert np.all(model_input["state"] == 0)
    assert "cs_agency" in model_input["prompt"]
    assert all(key not in model_input for key in ("sample_id", "map_name", "pose", "pose_raw", "z_calibration"))

    sample = dataset[0]
    assert sample["actions"].shape == (1, 5)
    assert sample["actions"].dtype == np.float32
    np.testing.assert_allclose(sample["actions"][0], dataset.rows[0]["pose"], rtol=0.0, atol=1e-7)

    # The radar cache is read-only internally and each public read is a copy.
    original_pixel = int(model_input["image"]["left_wrist_0_rgb"][0, 0, 0])
    model_input["image"]["left_wrist_0_rgb"][0, 0, 0] = (original_pixel + 1) % 256
    fresh = dataset.input_at(1)
    assert int(fresh["image"]["left_wrist_0_rgb"][0, 0, 0]) == original_pixel


def test_fixed_sample_selection_is_deterministic_and_per_map():
    dataset = _dataset("seen_validation", include_actions=False)
    first = fixed_sample_indices(dataset, seed=17, per_map=10)
    second = fixed_sample_indices(dataset, seed=17, per_map=10)
    assert first == second
    assert tuple(first) == SEEN_MAPS
    for map_name in SEEN_MAPS:
        assert len(first[map_name]) == 10
        assert len(set(first[map_name])) == 10
        assert all(dataset.rows[index]["map_name"] == map_name for index in first[map_name])


def test_visualization_clips_display_only_and_round_trips_physical_pose(tmp_path):
    dataset = _dataset("seen_validation", include_actions=False, limit=1)
    map_name = dataset.rows[0]["map_name"]
    normalized = (-0.25, 1.25, 1.4, 0.5, -0.25)
    physical = physical_pose(normalized, map_name, dataset.z_calibration)
    np.testing.assert_allclose(
        physical_to_normalized_pose(physical, map_name, dataset.z_calibration),
        normalized,
        rtol=0.0,
        atol=1e-12,
    )
    assert physical[0] < 0.0
    assert physical[1] > 1024.0
    assert map_pixel_from_normalized(-0.25, 1.25, 100, 80, margin=5) == (5, 74)

    sample_id = dataset.rows[0]["sample_id"]
    outputs = render_prediction_maps(
        dataset,
        {sample_id: normalized},
        tmp_path,
        seed=17,
        selection={map_name: [0]},
        radar_size=200,
        fpv_width=160,
    )
    assert [path.name for path in outputs] == [f"{map_name}.png"]
    selection_path = tmp_path / "selection.json"
    assert json.loads(selection_path.read_text(encoding="utf-8")) == [sample_id]
    output_path = tmp_path / f"{map_name}.png"
    assert output_path.is_file()
    original_bytes = output_path.read_bytes()
    # Existing PNGs are intentionally protected from accidental overwrite.
    render_prediction_maps(
        dataset,
        {sample_id: normalized},
        tmp_path,
        seed=999,
        selection=[sample_id],
        radar_size=200,
        fpv_width=160,
    )
    assert output_path.read_bytes() == original_bytes
