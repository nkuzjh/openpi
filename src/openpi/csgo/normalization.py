"""Seen-10 train-only quantile statistics for the five-dimensional pose.

The public helpers in this module read pose values from the shared protocol
rows only. They never open FPV or radar images. Statistics are kept in the
normal OpenPI ``norm_stats.json`` format, under the ``actions`` key, so they
can be passed directly to ``openpi.transforms.Normalize`` and
``openpi.transforms.Unnormalize`` with quantile normalization enabled.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from openpi.csgo import data as _data
from openpi.shared import normalize as _normalize

POSE_STATS_KEY = "actions"
POSE_STATS_CACHE_VERSION = 2
POSE_STATS_CACHE_METADATA = "seen10_pose_quantile_cache.json"
POSE_DIM = 5


def _validate_stats(norm_stats: dict[str, _normalize.NormStats]) -> _normalize.NormStats:
    """Validate and return the Seen-10 action statistics entry."""

    if set(norm_stats) != {POSE_STATS_KEY}:
        raise ValueError(f"Seen-10 pose stats must contain only {POSE_STATS_KEY!r}, got {sorted(norm_stats)}")
    stats = norm_stats[POSE_STATS_KEY]
    for name in ("mean", "std", "q01", "q99"):
        value = getattr(stats, name)
        if value is None:
            raise ValueError(f"Seen-10 quantile stats are missing {name}")
        array = np.asarray(value)
        if array.shape != (POSE_DIM,):
            raise ValueError(f"Seen-10 {name} must have shape ({POSE_DIM},), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Seen-10 {name} contains non-finite values")
    return stats


def _pose_values(dataset: _data.Seen10Dataset) -> np.ndarray:
    """Read the full published seen_train pose matrix without touching images."""

    if dataset.split != "seen_train":
        raise ValueError(f"Seen-10 pose normalization may only use 'seen_train', got {dataset.split!r}")
    if dataset.limit is not None or len(dataset) != _data.EXPECTED_SPLIT_COUNTS["seen_train"]:
        raise ValueError(
            "Seen-10 pose normalization requires the full 50,000-row seen_train split "
            f"(got limit={dataset.limit!r}, rows={len(dataset)})"
        )
    expected_counts = dict.fromkeys(_data.SEEN_MAPS, _data.EXPECTED_PER_MAP_COUNTS["seen_train"])
    if dataset.counts != expected_counts:
        raise ValueError("Seen-10 pose normalization requires the published 5,000 seen_train rows per map")

    # ``rows`` are protocol metadata already loaded by Seen10Dataset. No call
    # to input_at/__getitem__ is made, so image files are never decoded.
    values = np.asarray([row["pose"] for row in dataset.rows], dtype=np.float32)
    if values.shape != (50_000, POSE_DIM):
        raise ValueError(f"Expected seen_train poses with shape (50000, {POSE_DIM}), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("seen_train pose rows contain non-finite values")
    return values


def compute_seen10_pose_norm_stats(dataset: _data.Seen10Dataset) -> dict[str, _normalize.NormStats]:
    """Compute q01/q99 pose stats from the full protocol seen_train split.

    The single batch update uses OpenPI's own deterministic histogram-based
    ``RunningStats`` quantile implementation, including its q01/q99 behavior.
    Training and inference should select OpenPI quantile normalization.
    """

    poses = _pose_values(dataset)
    result = _statistics_from_poses(poses)
    _validate_stats(result)
    return result


def _statistics_from_poses(poses: np.ndarray) -> dict[str, _normalize.NormStats]:
    running_stats = _normalize.RunningStats()
    running_stats.update(poses)
    return {POSE_STATS_KEY: running_stats.get_statistics()}


def _stats_fingerprint(dataset: _data.Seen10Dataset, poses: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(f"openpi-csgo-seen10-pose-q01-q99-v{POSE_STATS_CACHE_VERSION}\0".encode("ascii"))
    for row, pose in zip(dataset.rows, poses, strict=True):
        digest.update(str(row["sample_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(pose, dtype="<f4").tobytes())
    return digest.hexdigest()


def stats_digest(norm_stats: dict[str, _normalize.NormStats]) -> str:
    """Return a stable SHA-256 digest for an OpenPI Seen-10 stats mapping.

    The digest covers the algorithm version, field names, and each 5D array
    serialized as little-endian float32. It is suitable for recording in a
    checkpoint identity or experiment manifest.
    """

    _validate_stats(norm_stats)
    digest = hashlib.sha256()
    digest.update(f"openpi-csgo-seen10-pose-norm-v{POSE_STATS_CACHE_VERSION}\0".encode("ascii"))
    stats = norm_stats[POSE_STATS_KEY]
    for field_name in ("mean", "std", "q01", "q99"):
        digest.update(field_name.encode("ascii") + b"\0")
        digest.update(np.asarray(getattr(stats, field_name), dtype="<f4").tobytes())
    return digest.hexdigest()


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def save_seen10_pose_norm_stats(directory: str | os.PathLike[str], norm_stats: dict[str, _normalize.NormStats]) -> None:
    """Persist stats with OpenPI's standard ``norm_stats.json`` schema."""

    _validate_stats(norm_stats)
    _normalize.save(directory, norm_stats)


def load_seen10_pose_norm_stats(directory: str | os.PathLike[str]) -> dict[str, _normalize.NormStats]:
    """Load and validate OpenPI-format Seen-10 pose statistics."""

    result = _normalize.load(directory)
    _validate_stats(result)
    return result


def get_seen10_pose_norm_stats(
    data_root: str | os.PathLike[str] = _data.DEFAULT_DATA_ROOT,
    *,
    shared_eval_dir: str | os.PathLike[str] = _data.DEFAULT_SHARED_EVAL_DIR,
    cache_dir: str | os.PathLike[str] | None = None,
    force_recompute: bool = False,
) -> dict[str, _normalize.NormStats]:
    """Load cached stats or compute/persist them from the full seen_train rows.

    The cache contains OpenPI's ``norm_stats.json`` and a sidecar fingerprint
    for the exact protocol row IDs and pose values. A stale or unreadable
    cache is recomputed. Pass ``force_recompute=True`` to replace a valid
    cache explicitly. ``cache_dir=None`` computes without writing files.
    """

    dataset = _data.Seen10Dataset(
        data_root,
        split="seen_train",
        shared_eval_dir=shared_eval_dir,
        include_actions=False,
        limit=None,
    )
    poses = _pose_values(dataset)
    fingerprint = _stats_fingerprint(dataset, poses)

    if cache_dir is not None:
        cache_path = Path(cache_dir).expanduser()
        metadata_path = cache_path / POSE_STATS_CACHE_METADATA
        stats_path = cache_path / "norm_stats.json"
        if not force_recompute and metadata_path.is_file() and stats_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (
                    metadata.get("version") == POSE_STATS_CACHE_VERSION
                    and metadata.get("split") == "seen_train"
                    and metadata.get("row_count") == len(dataset)
                    and metadata.get("source_fingerprint") == fingerprint
                ):
                    cached_stats = load_seen10_pose_norm_stats(cache_path)
                    if metadata.get("stats_digest") == stats_digest(cached_stats):
                        return cached_stats
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                # An incomplete/corrupt cache is safe to replace from protocol rows.
                pass

    stats = _statistics_from_poses(poses)
    _validate_stats(stats)
    if cache_dir is not None:
        cache_path = Path(cache_dir).expanduser()
        save_seen10_pose_norm_stats(cache_path, stats)
        metadata = {
            "version": POSE_STATS_CACHE_VERSION,
            "split": "seen_train",
            "row_count": len(dataset),
            "quantile_method": "openpi_running_stats_histogram",
            "source_fingerprint": fingerprint,
            "stats_digest": stats_digest(stats),
        }
        _atomic_write(
            cache_path / POSE_STATS_CACHE_METADATA,
            json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        )
    return stats


def normalize_seen10_actions(
    actions: np.ndarray,
    norm_stats: dict[str, _normalize.NormStats],
) -> np.ndarray:
    """Apply OpenPI's q01/q99 normalization to 5D actions without padding."""

    from openpi import transforms as _transforms

    _validate_stats(norm_stats)
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim == 0 or values.shape[-1] != POSE_DIM:
        raise ValueError(f"Expected actions ending in {POSE_DIM} dimensions, got {values.shape}")
    result = _transforms.Normalize(norm_stats, use_quantiles=True, strict=True)({POSE_STATS_KEY: values})
    return np.asarray(result[POSE_STATS_KEY], dtype=np.float32)


def unnormalize_seen10_actions(
    actions: np.ndarray,
    norm_stats: dict[str, _normalize.NormStats],
) -> np.ndarray:
    """Invert OpenPI q01/q99 normalization for 5D model outputs."""

    from openpi import transforms as _transforms

    _validate_stats(norm_stats)
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim == 0 or values.shape[-1] != POSE_DIM:
        raise ValueError(f"Expected actions ending in {POSE_DIM} dimensions, got {values.shape}")
    result = _transforms.Unnormalize(norm_stats, use_quantiles=True)({POSE_STATS_KEY: values})
    return np.asarray(result[POSE_STATS_KEY], dtype=np.float32)


__all__ = [
    "POSE_DIM",
    "POSE_STATS_CACHE_METADATA",
    "POSE_STATS_CACHE_VERSION",
    "POSE_STATS_KEY",
    "compute_seen10_pose_norm_stats",
    "get_seen10_pose_norm_stats",
    "load_seen10_pose_norm_stats",
    "normalize_seen10_actions",
    "save_seen10_pose_norm_stats",
    "stats_digest",
    "unnormalize_seen10_actions",
]
