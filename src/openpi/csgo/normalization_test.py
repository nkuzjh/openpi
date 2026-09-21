"""Tests for Seen-10 train-only pose quantile statistics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from openpi.csgo import data as csgo_data
from openpi.csgo import normalization
from openpi.shared import normalize as normalize_lib


class _FakeSeen10Dataset:
    def __init__(self, poses: np.ndarray, *, split: str = "seen_train", limit: int | None = None):
        self.split = split
        self.limit = limit
        self.rows = [{"sample_id": f"seen10-{index:05d}", "pose": pose} for index, pose in enumerate(poses)]
        self.counts = dict.fromkeys(csgo_data.SEEN_MAPS, 5_000)
        self.expected_count = 50_000

    def __len__(self):
        return len(self.rows)


def _poses(count: int = 50_000, *, offset: float = 0.0) -> np.ndarray:
    ramp = np.linspace(-1.0 + offset, 1.0 + offset, count, dtype=np.float32)
    return np.stack([ramp + dimension for dimension in range(5)], axis=-1)


def test_compute_stats_uses_full_seen_train_rows_and_openpi_running_stats():
    poses = _poses()
    dataset = _FakeSeen10Dataset(poses)

    stats = normalization.compute_seen10_pose_norm_stats(dataset)
    running_stats = normalize_lib.RunningStats()
    running_stats.update(poses)
    expected = running_stats.get_statistics()

    assert set(stats) == {"actions"}
    result = stats["actions"]
    np.testing.assert_array_equal(result.mean, expected.mean)
    np.testing.assert_array_equal(result.std, expected.std)
    np.testing.assert_array_equal(result.q01, expected.q01)
    np.testing.assert_array_equal(result.q99, expected.q99)


@pytest.mark.parametrize(
    ("split", "limit", "count"),
    [("seen_validation", None, 50_000), ("seen_train", 10, 50_000), ("seen_train", None, 2)],
)
def test_stats_reject_non_train_or_incomplete_rows(split, limit, count):
    with pytest.raises(ValueError, match="seen_train|50,000|50000"):
        normalization.compute_seen10_pose_norm_stats(_FakeSeen10Dataset(_poses(count), split=split, limit=limit))


def test_cache_persists_openpi_stats_and_invalidates_on_pose_change(monkeypatch, tmp_path):
    dataset = _FakeSeen10Dataset(_poses())
    monkeypatch.setattr(csgo_data, "Seen10Dataset", lambda *args, **kwargs: dataset)

    first = normalization.get_seen10_pose_norm_stats(cache_dir=tmp_path)
    stats_path = tmp_path / "norm_stats.json"
    metadata_path = tmp_path / normalization.POSE_STATS_CACHE_METADATA
    assert stats_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["split"] == "seen_train"
    assert metadata["row_count"] == 50_000
    assert metadata["quantile_method"] == "openpi_running_stats_histogram"
    assert metadata["stats_digest"] == normalization.stats_digest(first)

    # A matching fingerprint returns persisted OpenPI NormStats.
    cached = normalization.get_seen10_pose_norm_stats(cache_dir=tmp_path)
    np.testing.assert_array_equal(cached["actions"].q01, first["actions"].q01)
    np.testing.assert_array_equal(cached["actions"].q99, first["actions"].q99)

    # Changing even one normalized pose invalidates the cache fingerprint.
    dataset = _FakeSeen10Dataset(_poses(offset=0.25))
    monkeypatch.setattr(csgo_data, "Seen10Dataset", lambda *args, **kwargs: dataset)
    changed = normalization.get_seen10_pose_norm_stats(cache_dir=tmp_path)
    assert not np.array_equal(changed["actions"].q01, first["actions"].q01)


def test_save_load_and_openpi_quantile_round_trip_preserve_five_dimensions(tmp_path):
    stats = normalization.compute_seen10_pose_norm_stats(_FakeSeen10Dataset(_poses()))
    normalization.save_seen10_pose_norm_stats(tmp_path, stats)
    loaded = normalization.load_seen10_pose_norm_stats(tmp_path)

    np.testing.assert_array_equal(loaded["actions"].q01, stats["actions"].q01)
    np.testing.assert_array_equal(loaded["actions"].q99, stats["actions"].q99)

    actions = np.stack([stats["actions"].q01, stats["actions"].q99]).reshape(2, 1, 5)
    normalized = normalization.normalize_seen10_actions(actions, loaded)
    assert normalized.shape == actions.shape
    expected = (actions - loaded["actions"].q01) / (loaded["actions"].q99 - loaded["actions"].q01 + 1e-6) * 2.0 - 1.0
    np.testing.assert_allclose(normalized, expected, rtol=0, atol=2e-7)

    restored = normalization.unnormalize_seen10_actions(normalized, loaded)
    assert restored.shape == (2, 1, 5)
    np.testing.assert_allclose(restored, actions, rtol=0, atol=2e-6)

    with pytest.raises(ValueError, match="ending in 5"):
        normalization.normalize_seen10_actions(np.zeros((1, 32), dtype=np.float32), loaded)
