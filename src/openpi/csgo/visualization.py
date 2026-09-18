"""Deterministic Seen-10 localization visualizations.

Each map panel puts its published radar on the left and the selected FPV
frames in a vertical column on the right.  Selection is persisted as a plain
JSON list of sample IDs so every checkpoint/evaluation reuses exactly the same
samples.  The renderer is intentionally independent of the model/runtime;
predictions are a mapping from sample ID to normalized five-dimensional pose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import random
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from openpi.csgo.data import (
    IMAGE_KEYS,
    SEEN_MAPS,
    Seen10Dataset,
    physical_pose,
)


SAMPLE_COLORS = (
    (255, 72, 72),
    (56, 220, 95),
    (72, 132, 255),
    (255, 218, 48),
    (255, 75, 220),
    (30, 225, 225),
    (255, 145, 40),
    (178, 105, 255),
    (245, 245, 245),
    (100, 255, 185),
)
POSE_FIELDS = ("pred_x", "pred_y", "pred_z", "pred_pitch", "pred_yaw")


def fixed_sample_indices(
    dataset: Seen10Dataset,
    seed: int,
    per_map: int = 10,
) -> dict[str, list[int]]:
    """Choose a deterministic sample-index subset independently for each map.

    The returned dictionary contains all Seen-10 map names in protocol order;
    a bounded smoke dataset can consequently have fewer than ten entries (or
    no entries) for a map, while a formal split always has ten.
    """

    if per_map <= 0:
        raise ValueError("per_map must be positive")
    if per_map > len(SAMPLE_COLORS):
        raise ValueError(f"per_map cannot exceed {len(SAMPLE_COLORS)} colors")
    records = getattr(dataset, "records", None)
    if records is None:
        raise TypeError("dataset must expose manifest rows through .records")
    grouped = {map_name: [] for map_name in SEEN_MAPS}
    for index, row in enumerate(records):
        map_name = str(row["map_name"])
        if map_name not in grouped:
            raise ValueError(f"Dataset row has a map outside Seen-10: {map_name!r}")
        grouped[map_name].append(index)

    rng = random.Random(int(seed))
    return {
        map_name: rng.sample(indices, min(per_map, len(indices)))
        for map_name, indices in grouped.items()
    }


def _selection_to_ids(
    dataset: Seen10Dataset,
    selection: Mapping[str, Sequence[int | str]] | Sequence[int | str],
) -> list[str]:
    """Normalize accepted selection forms to IDs in map/selection order."""

    records = dataset.records
    by_index = {index: str(row["sample_id"]) for index, row in enumerate(records)}
    by_id = {str(row["sample_id"]): index for index, row in enumerate(records)}
    selected: list[str] = []
    if isinstance(selection, Mapping):
        values_by_map = selection.items()
        for map_name, values in values_by_map:
            map_name = str(map_name)
            if map_name not in SEEN_MAPS:
                raise ValueError(f"Selection contains map outside Seen-10: {map_name!r}")
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                values = (values,)
            for value in values:
                selected.append(_selection_value_to_id(value, map_name, by_index, by_id, records))
    else:
        for value in selection:
            selected.append(_selection_value_to_id(value, None, by_index, by_id, records))
    if len(selected) != len(set(selected)):
        raise ValueError("Selection contains duplicate sample IDs")
    return selected


def _selection_value_to_id(
    value: int | str,
    expected_map: str | None,
    by_index: Mapping[int, str],
    by_id: Mapping[str, int],
    records: Sequence[Mapping[str, Any]],
) -> str:
    if isinstance(value, str):
        sample_id = value
        if sample_id not in by_id:
            raise ValueError(f"Selection sample ID is outside this dataset: {sample_id!r}")
        index = by_id[sample_id]
    else:
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Selection entry must be an index or sample ID, got {value!r}") from exc
        if index not in by_index:
            raise IndexError(f"Selection index is outside this dataset: {index}")
        sample_id = by_index[index]
    if expected_map is not None and str(records[index]["map_name"]) != expected_map:
        raise ValueError(
            f"Selection index/ID belongs to {records[index]['map_name']!r}, expected {expected_map!r}"
        )
    return sample_id


def _ordered_selection_ids(
    dataset: Seen10Dataset,
    seed: int,
    selection: Mapping[str, Sequence[int | str]] | Sequence[int | str] | None,
    selection_path: Path,
) -> list[str]:
    if selection_path.is_file():
        try:
            existing = json.loads(selection_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read existing selection file {selection_path}") from exc
        if not isinstance(existing, list) or not all(isinstance(value, str) for value in existing):
            raise ValueError("selection.json must contain only a JSON list of sample ID strings")
        existing_ids = _selection_to_ids(dataset, existing)
        if selection is not None:
            requested_ids = _selection_to_ids(dataset, selection)
            if requested_ids != existing_ids:
                raise ValueError(f"Existing selection differs from requested selection: {selection_path}")
        return existing_ids

    requested = fixed_sample_indices(dataset, seed, per_map=len(SAMPLE_COLORS)) if selection is None else selection
    return _selection_to_ids(dataset, requested)


def _group_ids(dataset: Seen10Dataset, ids: Sequence[str]) -> dict[str, list[str]]:
    record_by_id = {str(row["sample_id"]): row for row in dataset.records}
    grouped = {map_name: [] for map_name in SEEN_MAPS}
    for sample_id in ids:
        if sample_id not in record_by_id:
            raise ValueError(f"Selection sample ID is outside this dataset: {sample_id!r}")
        grouped[str(record_by_id[sample_id]["map_name"])].append(sample_id)
    return grouped


def _prediction_pose(predictions: Mapping[Any, Any], sample_id: str, map_name: str) -> tuple[float, ...]:
    """Read a normalized pose from common prediction mapping forms."""

    value: Any = None
    found = False
    for key in (sample_id, (map_name, sample_id)):
        try:
            if key in predictions:
                value = predictions[key]
                found = True
                break
        except TypeError:
            continue
    if not found:
        raise KeyError(f"Missing prediction for selected sample {sample_id!r}")
    if isinstance(value, Mapping):
        if "pose" in value:
            value = value["pose"]
        elif all(field in value for field in POSE_FIELDS):
            value = [value[field] for field in POSE_FIELDS]
        else:
            raise ValueError(f"Prediction for {sample_id!r} has no pose or pred_* fields")
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Prediction for {sample_id!r} is not a five-dimensional pose") from exc
    if len(values) != 5 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"Prediction for {sample_id!r} must contain five finite values")
    return values


def _font(size: int, *, bold: bool = False):
    filename = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / filename,
        Path("/usr/share/fonts/dejavu") / filename,
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def map_pixel_from_normalized(
    x: float,
    y: float,
    width: int,
    height: int,
    margin: int = 20,
) -> tuple[float, float]:
    """Map normalized XY to display pixels, clipping only for visualization."""

    if width <= 0 or height <= 0:
        raise ValueError("Visualization image dimensions must be positive")
    if margin < 0 or 2 * margin >= min(width, height):
        raise ValueError(f"Invalid marker margin {margin} for image size {(width, height)}")
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        raise ValueError(f"Normalized XY must be finite, got {(x, y)}")
    return (
        min(max(float(x) * width, margin), width - margin - 1),
        min(max(float(y) * height, margin), height - margin - 1),
    )


# A shorter name is convenient for callers and tests.
clip_map_pixel = map_pixel_from_normalized


def _fit_fpv(path: str | os.PathLike[str], size: tuple[int, int]) -> Image.Image:
    """Resize the complete FPV into a letterboxed cell without center-cropping."""

    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid FPV cell size {size}")
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        result = Image.new("RGB", size, "black")
        result.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        return result


def _draw_text(draw: ImageDraw.ImageDraw, xy, value: str, font, *, anchor: str = "la") -> None:
    draw.text(xy, value, font=font, fill="white", stroke_width=2, stroke_fill="black", anchor=anchor)


def render_prediction_maps(
    dataset: Seen10Dataset,
    predictions: Mapping[Any, Any],
    outdir: str | os.PathLike[str],
    seed: int,
    selection: Mapping[str, Sequence[int | str]] | Sequence[int | str] | None = None,
    *,
    radar_size: int = 1200,
    fpv_width: int = 420,
) -> list[Path]:
    """Render one non-overwriting PNG per selected map.

    ``predictions`` values are normalized ``[x, y, z, pitch, yaw]`` poses.
    Predictions outside the radar's normalized XY range are displayed at the
    nearest edge, while their physical labels are converted without clipping.
    ``selection.json`` is a JSON list containing sample IDs only.
    """

    if radar_size <= 0 or fpv_width <= 0:
        raise ValueError("radar_size and fpv_width must be positive")
    if not isinstance(predictions, Mapping):
        raise TypeError("predictions must be a mapping from sample ID to normalized pose")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    selection_path = outdir / "selection.json"
    selected_ids = _ordered_selection_ids(dataset, int(seed), selection, selection_path)

    # Persist selection exactly once.  The file deliberately contains no seed,
    # map metadata, GT, or predictions, only the identities needed to replay it.
    if not selection_path.exists():
        temporary = outdir / ".selection.json.tmp"
        temporary.write_text(json.dumps(selected_ids, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, selection_path)

    grouped = _group_ids(dataset, selected_ids)
    if getattr(dataset, "limit", None) is None:
        invalid = {map_name: len(values) for map_name, values in grouped.items() if len(values) != len(SAMPLE_COLORS)}
        if invalid:
            raise ValueError(
                "Formal Seen-10 visualization requires ten selected samples per map; "
                f"got {invalid}"
            )
    record_by_id = {str(row["sample_id"]): row for row in dataset.records}
    outputs: list[Path] = []
    body_font = _font(max(11, min(16, radar_size // 90)))
    marker_margin = max(8, min(24, radar_size // 40))

    for map_name in SEEN_MAPS:
        map_ids = grouped[map_name]
        if not map_ids:
            continue
        if len(map_ids) > len(SAMPLE_COLORS):
            raise ValueError(f"At most {len(SAMPLE_COLORS)} selected samples are supported per map")
        output_path = outdir / f"{map_name}.png"
        if output_path.exists():
            outputs.append(output_path)
            continue

        row_height = max(72, radar_size // max(len(SAMPLE_COLORS), len(map_ids)))
        radar_side = row_height * len(map_ids)
        panel_height = radar_side
        radar_path = Path(record_by_id[map_ids[0]]["radar_path"])
        with Image.open(radar_path) as source:
            radar = source.convert("RGB").resize((radar_side, radar_side), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (radar_side + fpv_width, panel_height), "black")
        canvas.paste(radar, (0, 0))
        radar_draw = ImageDraw.Draw(canvas)

        for index, sample_id in enumerate(map_ids):
            color = SAMPLE_COLORS[index]
            row = record_by_id[sample_id]
            gt = tuple(float(value) for value in row["pose"])
            pred = _prediction_pose(predictions, sample_id, map_name)
            if not all(math.isfinite(value) for value in gt):
                raise ValueError(f"Non-finite GT pose for {sample_id!r}")

            gt_xy = map_pixel_from_normalized(gt[0], gt[1], radar_side, radar_side, marker_margin)
            pred_xy = map_pixel_from_normalized(pred[0], pred[1], radar_side, radar_side, marker_margin)
            radar_draw.line((*gt_xy, *pred_xy), fill=color, width=max(2, radar_side // 500))

            gt_radius = max(5, radar_side // 90)
            radar_draw.ellipse(
                (gt_xy[0] - gt_radius, gt_xy[1] - gt_radius, gt_xy[0] + gt_radius, gt_xy[1] + gt_radius),
                fill=color,
                outline="black",
                width=max(1, radar_side // 600),
            )
            pred_radius = max(gt_radius + 3, radar_side // 65)
            radar_draw.ellipse(
                (
                    pred_xy[0] - pred_radius,
                    pred_xy[1] - pred_radius,
                    pred_xy[0] + pred_radius,
                    pred_xy[1] + pred_radius,
                ),
                outline="black",
                width=max(2, radar_side // 180),
            )
            radar_draw.ellipse(
                (
                    pred_xy[0] - pred_radius,
                    pred_xy[1] - pred_radius,
                    pred_xy[0] + pred_radius,
                    pred_xy[1] + pred_radius,
                ),
                outline=color,
                width=max(2, radar_side // 240),
            )

            fpv = _fit_fpv(row["image_path"], (fpv_width, row_height))
            fpv_draw = ImageDraw.Draw(fpv, "RGBA")
            text_height = max(44, min(row_height - 2, row_height // 3))
            fpv_draw.rectangle((0, 0, fpv_width, text_height), fill=(0, 0, 0, 175))
            tag_radius = max(5, min(11, row_height // 12))
            fpv_draw.ellipse(
                (7, 7, 7 + 2 * tag_radius, 7 + 2 * tag_radius),
                fill=color + (255,),
                outline=(0, 0, 0, 255),
                width=2,
            )
            gt_physical = physical_pose(gt, map_name, dataset.z_calibration)
            pred_physical = physical_pose(pred, map_name, dataset.z_calibration)
            gt_label = "gt_xyzhw=[" + ",".join(f"{value:.1f}" for value in gt_physical) + "]"
            pred_label = "pred_xyzhw=[" + ",".join(f"{value:.1f}" for value in pred_physical) + "]"
            center_x = fpv_width / 2
            # Keep the long physical labels inside the 420px default cell.
            label_font = body_font
            while getattr(label_font, "size", 8) > 6:
                gt_box = fpv_draw.textbbox((0, 0), gt_label, font=label_font)
                pred_box = fpv_draw.textbbox((0, 0), pred_label, font=label_font)
                if max(gt_box[2] - gt_box[0], pred_box[2] - pred_box[0]) <= fpv_width - 8:
                    break
                label_font = _font(getattr(label_font, "size", 9) - 1)
            _draw_text(fpv_draw, (center_x, 4), gt_label, label_font, anchor="mt")
            _draw_text(fpv_draw, (center_x, max(22, row_height // 7)), pred_label, label_font, anchor="mt")
            canvas.paste(fpv.convert("RGB"), (radar_side, index * row_height))

        # Preserve any existing artifact.  A temporary file prevents a partial
        # PNG from being observed if rendering is interrupted.
        temporary = outdir / f".{map_name}.png.tmp"
        canvas.save(temporary, format="PNG", optimize=True)
        if not output_path.exists():
            os.replace(temporary, output_path)
        else:
            temporary.unlink(missing_ok=True)
        outputs.append(output_path)
    return outputs


__all__ = [
    "IMAGE_KEYS",
    "POSE_FIELDS",
    "SAMPLE_COLORS",
    "SEEN_MAPS",
    "clip_map_pixel",
    "fixed_sample_indices",
    "map_pixel_from_normalized",
    "physical_pose",
    "render_prediction_maps",
]
