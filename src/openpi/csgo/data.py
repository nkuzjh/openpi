"""Manifest-driven CSGO Benchmark v2 Seen-10 data adapter.

The benchmark release is owned by the shared evaluator.  This module only
adapts its standard-library :class:`BenchmarkData` rows to the input shape
used by the native OpenPI transforms and models.  In particular, metadata
and ground-truth poses stay on the dataset side and are never included in
``input_at``.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import importlib.util
import math
import operator
import os
from pathlib import Path
import types
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_DATA_ROOT = "/home/jiahao/task/UniLIP/data/csgo_benchmark_v2"
DEFAULT_SHARED_EVAL_DIR = "/home/jiahao/task/csgo_benchmark_v2_eval_general"

SEEN_MAPS = (
    "cs_agency",
    "cs_italy",
    "de_ancient",
    "de_anubis",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
    "de_overpass",
    "de_train",
)

SPLIT_ALIASES = {
    "train": "seen_train",
    "seen_train": "seen_train",
    "validation": "seen_validation",
    "val": "seen_validation",
    "seen_validation": "seen_validation",
    "discrete_test": "seen_discrete_test",
    "test": "seen_discrete_test",
    "seen_discrete_test": "seen_discrete_test",
}

EXPECTED_SPLIT_COUNTS = {
    "seen_train": 50_000,
    "seen_validation": 5_000,
    "seen_discrete_test": 20_000,
}
EXPECTED_PER_MAP_COUNTS = {
    "seen_train": 5_000,
    "seen_validation": 500,
    "seen_discrete_test": 2_000,
}

# The native Pi0/Pi05 tokenizer receives this prompt through ModelTransform-
# Factory.  Keep the task wording fixed and put the map name in the prompt so
# no map identity has to be smuggled through metadata or state.
PROMPT_TEMPLATE = (
    "Localize the player in {map_name} using the first-person image and radar map. "
    "Predict the absolute normalized x, y, z, pitch, and yaw."
)

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _load_shared_protocol(shared_eval_dir: str | os.PathLike[str]) -> types.ModuleType:
    """Import the evaluator protocol by path without making it a package dependency."""

    protocol_path = Path(shared_eval_dir).expanduser().resolve() / "protocol.py"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Shared Benchmark v2 protocol not found: {protocol_path}")
    # A private module name prevents a model project from replacing another
    # project's protocol module in sys.modules.  protocol.py is standard
    # library only, so importlib is sufficient and avoids mutating SHARED.
    module_name = f"_openpi_csgo_protocol_{abs(hash(str(protocol_path))):x}"
    spec = importlib.util.spec_from_file_location(module_name, protocol_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared Benchmark v2 protocol: {protocol_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_index(index: int) -> int:
    try:
        return operator.index(index)
    except TypeError as exc:
        raise TypeError(f"Dataset index must be an integer, got {type(index).__name__}") from exc


def _z_bounds(z_calibration: Mapping[str, Any] | Mapping[str, float], map_name: str) -> tuple[float, float]:
    """Resolve a map's published ``(z_min, z_max)`` from common protocol shapes."""

    value: Any
    if hasattr(z_calibration, "z_calibration"):
        z_calibration = z_calibration.z_calibration
    if hasattr(z_calibration, "z_ranges"):
        z_calibration = z_calibration.z_ranges
    if map_name in z_calibration:
        value = z_calibration[map_name]
    else:
        value = z_calibration
    if isinstance(value, Mapping) and "z_min" in value and "z_max" in value:
        low, high = float(value["z_min"]), float(value["z_max"])
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        low, high = float(value[0]), float(value[1])
    else:
        raise ValueError(f"Missing z calibration for map {map_name!r}")
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError(f"Invalid z calibration for {map_name}: {(low, high)}")
    return low, high


def physical_pose(
    pose: Sequence[float] | np.ndarray,
    map_name: str,
    z_calibration: Mapping[str, Any] | Mapping[str, float],
) -> tuple[float, float, float, float, float]:
    """Convert normalized ``[x, y, z, pitch, yaw]`` to physical ``xyzhw``.

    X and Y use the benchmark's 1024 world scale, Z uses the released
    per-map extrema, and pitch/yaw are returned in degrees.  No clipping is
    applied here; out-of-range predictions must remain visible in labels and
    metrics even when their radar marker is clipped to the image edge.
    """

    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (5,):
        raise ValueError(f"Expected a normalized 5DoF pose with shape (5,), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Pose contains non-finite values: {pose!r}")
    z_min, z_max = _z_bounds(z_calibration, map_name)
    return (
        float(values[0] * 1024.0),
        float(values[1] * 1024.0),
        float(values[2] * (z_max - z_min) + z_min),
        float(values[3] * 360.0),
        float(values[4] * 360.0),
    )


def physical_to_normalized_pose(
    pose: Sequence[float] | np.ndarray,
    map_name: str,
    z_calibration: Mapping[str, Any] | Mapping[str, float],
) -> tuple[float, float, float, float, float]:
    """Convert physical ``xyzhw`` (degrees for pitch/yaw) to benchmark pose."""

    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (5,):
        raise ValueError(f"Expected a physical 5DoF pose with shape (5,), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Pose contains non-finite values: {pose!r}")
    z_min, z_max = _z_bounds(z_calibration, map_name)
    return (
        float(values[0] / 1024.0),
        float(values[1] / 1024.0),
        float((values[2] - z_min) / (z_max - z_min)),
        float(values[3] / 360.0),
        float(values[4] / 360.0),
    )


class Seen10Dataset:
    """Read one published localization split in manifest order.

    ``records``/``rows`` retain the complete shared-protocol row, including
    GT pose and calibration, for training labels and post-hoc evaluation.
    ``input_at`` and ``__getitem__`` construct a separate model-input mapping
    and therefore cannot accidentally pass sample IDs or GT to the model.

    ``limit`` is a global prefix after the protocol's fixed map ordering.  It
    is intended for explicit smoke runs; an unbounded dataset always validates
    the published per-map counts through ``BenchmarkData.rows``.
    """

    maps = SEEN_MAPS
    image_keys = IMAGE_KEYS

    def __init__(
        self,
        data_root: str | os.PathLike[str] = DEFAULT_DATA_ROOT,
        split: str = "seen_train",
        *,
        shared_eval_dir: str | os.PathLike[str] = DEFAULT_SHARED_EVAL_DIR,
        include_actions: bool = True,
        limit: int | None = None,
    ) -> None:
        try:
            canonical_split = SPLIT_ALIASES[split]
        except (KeyError, TypeError) as exc:
            choices = ", ".join(sorted(SPLIT_ALIASES))
            raise ValueError(f"Unsupported CSGO Seen-10 split {split!r}; choose from {choices}") from exc
        if limit is not None:
            try:
                limit = operator.index(limit)
            except TypeError as exc:
                raise TypeError(f"limit must be an integer or None, got {type(limit).__name__}") from exc
            if limit < 0:
                raise ValueError("limit must be non-negative")

        self.data_root = Path(data_root).expanduser().resolve()
        self.shared_eval_dir = Path(shared_eval_dir).expanduser().resolve()
        self.split = canonical_split
        self.include_actions = bool(include_actions)
        self.limit = limit

        protocol = _load_shared_protocol(self.shared_eval_dir)
        self.benchmark_data = protocol.BenchmarkData(self.data_root)
        if tuple(self.benchmark_data.maps) != SEEN_MAPS:
            raise ValueError("Shared protocol map order differs from Seen-10 contract")

        # Make a detached copy so a caller cannot alter protocol-owned
        # calibration state through this adapter.
        self.z_calibration = {
            map_name: {
                "z_min": float(bounds["z_min"]),
                "z_max": float(bounds["z_max"]),
            }
            for map_name, bounds in self.benchmark_data.z_ranges.items()
        }
        self.z_ranges = {
            map_name: (bounds["z_min"], bounds["z_max"])
            for map_name, bounds in self.z_calibration.items()
        }

        self.rows = list(self.benchmark_data.rows(canonical_split, max_samples=limit))
        self.records = self.rows
        self.counts = {map_name: 0 for map_name in SEEN_MAPS}
        for row in self.rows:
            self.counts[str(row["map_name"])] += 1
        self.expected_count = EXPECTED_SPLIT_COUNTS[canonical_split]
        self.expected_per_map = EXPECTED_PER_MAP_COUNTS[canonical_split]
        if limit is None and len(self.rows) != self.expected_count:
            raise ValueError(
                f"Published {canonical_split} count mismatch: expected {self.expected_count}, got {len(self.rows)}"
            )

        self._radar_cache: dict[str, np.ndarray] = {}

    def __getstate__(self) -> dict[str, Any]:
        """Make worker-process copies independent of the dynamic protocol module."""

        state = self.__dict__.copy()
        # ``BenchmarkData`` is defined in a module loaded by importlib and is
        # therefore not safely pickleable by spawn.  Rows and calibration are
        # already detached plain dictionaries; rebuild only this helper in the
        # worker.  Radar arrays are cheap to reload and should not be copied
        # through the worker payload either.
        state.pop("benchmark_data", None)
        state.pop("_protocol_module", None)
        state["_radar_cache"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        protocol = _load_shared_protocol(self.shared_eval_dir)
        self.benchmark_data = protocol.BenchmarkData(self.data_root)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def _resolve_index(self, index: int) -> int:
        index = _as_index(index)
        if index < 0:
            index += len(self.rows)
        if index < 0 or index >= len(self.rows):
            raise IndexError(f"Dataset index out of range: {index}")
        return index

    def row_at(self, index: int) -> dict[str, Any]:
        """Return the protocol row for metadata, labels, and evaluation joins."""

        return self.rows[self._resolve_index(index)]

    def _read_cached_rgb(self, path: str, cache: dict[str, np.ndarray]) -> np.ndarray:
        source = cache.get(path)
        if source is None:
            with Image.open(path) as image:
                # np.asarray can share a PIL buffer, so np.array(copy=True) is
                # deliberate.  The cache itself is read-only and each caller
                # receives a writable copy, preventing radar cross-sample
                # mutation while retaining decode caching.
                source = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
            source.setflags(write=False)
            cache[path] = source
        return source.copy()

    def input_at(self, index: int) -> dict[str, Any]:
        """Return exactly the model observation fields for one sample.

        The FPV and radar remain RGB uint8 at their release resolution.  The
        native ``ModelTransformFactory.ResizeImages224`` is responsible for
        resizing, and callers should apply it to each sample before stacking.
        """

        row = self.row_at(index)
        # FPV frames are all distinct and the formal split has 50k entries;
        # decode them per sample instead of retaining many gigabytes in RAM.
        with Image.open(str(row["image_path"])) as image:
            fpv = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        radar = self._read_cached_rgb(str(row["radar_path"]), self._radar_cache)
        return {
            "image": {
                "base_0_rgb": fpv,
                "left_wrist_0_rgb": radar,
                "right_wrist_0_rgb": np.zeros_like(fpv),
            },
            "image_mask": {
                "base_0_rgb": np.bool_(True),
                "left_wrist_0_rgb": np.bool_(True),
                "right_wrist_0_rgb": np.bool_(False),
            },
            "state": np.zeros((5,), dtype=np.float32),
            "prompt": PROMPT_TEMPLATE.format(map_name=str(row["map_name"])),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one model input, optionally adding a horizon-one action label."""

        resolved = self._resolve_index(index)
        sample = self.input_at(resolved)
        if self.include_actions:
            sample["actions"] = np.asarray([self.rows[resolved]["pose"]], dtype=np.float32)
        return sample

    def physical_pose(self, pose: Sequence[float] | np.ndarray, map_name: str):
        """Instance convenience wrapper around :func:`physical_pose`."""

        return physical_pose(pose, map_name, self.z_calibration)


def _stack_leaf(values: Sequence[Any]) -> Any:
    first = values[0]
    if isinstance(first, Mapping):
        keys = tuple(first)
        if any(tuple(value) != keys for value in values[1:]):
            raise ValueError("Cannot stack mappings with different keys")
        return {key: _stack_leaf([value[key] for value in values]) for key in keys}
    if isinstance(first, str):
        if any(not isinstance(value, str) for value in values):
            raise TypeError("Cannot stack strings with non-string values")
        return list(values)
    try:
        return np.stack([np.asarray(value) for value in values], axis=0)
    except ValueError as exc:
        shape_info = [getattr(value, "shape", None) for value in values]
        raise ValueError(
            f"Cannot stack sample values with shapes {shape_info}; apply the native per-sample transform first"
        ) from exc


def stack_inputs(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack model samples after per-sample transforms have made shapes equal."""

    if not samples:
        raise ValueError("Cannot stack an empty sample sequence")
    return _stack_leaf(samples)


def read_benchmark_rows(
    data_root: str | os.PathLike[str] = DEFAULT_DATA_ROOT,
    split: str = "seen_train",
    *,
    shared_eval_dir: str | os.PathLike[str] = DEFAULT_SHARED_EVAL_DIR,
    max_samples: int | None = None,
    require_images: bool = True,
) -> list[dict[str, Any]]:
    """Read protocol rows through :class:`Seen10Dataset` for runtime joins."""

    dataset = Seen10Dataset(
        data_root,
        split,
        shared_eval_dir=shared_eval_dir,
        include_actions=False,
        limit=max_samples,
    )
    rows = list(dataset.rows)
    if require_images:
        for row in rows:
            for key in ("image_path", "radar_path"):
                path = Path(row[key])
                if not path.is_file():
                    raise FileNotFoundError(f"Missing {key} for {row['sample_id']}: {path}")
    return rows


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_SHARED_EVAL_DIR",
    "EXPECTED_PER_MAP_COUNTS",
    "EXPECTED_SPLIT_COUNTS",
    "IMAGE_KEYS",
    "PROMPT_TEMPLATE",
    "read_benchmark_rows",
    "SEEN_MAPS",
    "SPLIT_ALIASES",
    "Seen10Dataset",
    "physical_pose",
    "physical_to_normalized_pose",
    "stack_inputs",
]
