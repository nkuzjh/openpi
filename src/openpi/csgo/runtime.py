"""Training and inference runtime for native CSGO Seen-10 localization.

This module deliberately contains only orchestration.  Model math, optimizer
updates, sharding and Orbax state layout come from the native openpi modules;
the CSGO data module supplies manifest-driven rows and batches.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import functools
import hashlib
import importlib
import json
import logging
import math
import os
import pathlib
import time
from typing import Any

# Keep direct ``python -c 'from openpi.csgo.runtime import ...'`` usage from
# reserving most of the GPU before the caller can construct a small smoke run.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
os.environ.setdefault("OPENPI_DATA_HOME", str(_PROJECT_ROOT / ".cache" / "openpi"))
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(_PROJECT_ROOT / ".cache" / "jax"))

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms as _transforms
from openpi.csgo import data as _csgo_data
from openpi.csgo.model import CSGOPi0Config
from openpi.csgo.model import CSGOPi05WeightLoader
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.shared import nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import optimizer as _optimizer
from openpi.training import sharding as _sharding
from openpi.training import utils as _training_utils
from openpi.training import weight_loaders as _weight_loaders

logger = logging.getLogger("openpi.csgo")

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
TRAIN_SPLIT = "seen_train"
VALIDATION_SPLIT = "seen_validation"
TEST_SPLIT = "seen_discrete_test"
PROMPT_TEMPLATE = _csgo_data.PROMPT_TEMPLATE
PREDICTION_FIELDS = (
    "sample_id",
    "map_name",
    "pred_x",
    "pred_y",
    "pred_z",
    "pred_pitch",
    "pred_yaw",
)


@dataclasses.dataclass(frozen=True)
class Seen10RuntimeConfig:
    """Serializable runtime settings shared by the train and infer CLIs."""

    data_root: pathlib.Path
    run_dir: pathlib.Path
    seed: int = 0
    batch_size: int = 1
    num_workers: int = 0
    num_train_steps: int = 5_000
    learning_rate: float = 1.0e-5
    warmup_steps: int = 100
    decay_steps: int = 5_000
    decay_lr: float = 1.0e-6
    fsdp_devices: int = 1
    init_checkpoint: str | None = "gs://openpi-assets/checkpoints/pi05_base/params"
    resume: bool = False
    smoke_only: bool = False
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    model_dtype: str = "bfloat16"
    action_expert_variant: str = "gemma_300m_lora"
    paligemma_variant: str = "gemma_2b_lora"

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_train_steps <= 0 or self.num_train_steps % 5:
            raise ValueError("num_train_steps must be positive and divisible by five")
        if self.fsdp_devices < 1:
            raise ValueError("fsdp_devices must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.decay_steps <= self.warmup_steps:
            raise ValueError("decay_steps must be greater than warmup_steps")
        if self.max_train_samples is not None and self.max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive when set")
        if self.max_validation_samples is not None and self.max_validation_samples <= 0:
            raise ValueError("max_validation_samples must be positive when set")
        if not self.smoke_only and self.max_train_samples is not None:
            raise ValueError("Formal training cannot use max_train_samples")
        if not self.smoke_only and self.max_validation_samples is not None:
            raise ValueError("Formal training cannot use max_validation_samples")

    @property
    def interval(self) -> int:
        return self.num_train_steps // 5


def make_runtime_config(
    *,
    data_root: str | pathlib.Path,
    run_dir: str | pathlib.Path,
    seed: int = 0,
    batch_size: int = 1,
    num_train_steps: int = 5_000,
    num_workers: int = 0,
    learning_rate: float = 1.0e-5,
    warmup_steps: int = 100,
    decay_steps: int | None = None,
    decay_lr: float = 1.0e-6,
    fsdp_devices: int = 1,
    init_checkpoint: str | None = "gs://openpi-assets/checkpoints/pi05_base/params",
    resume: bool = False,
    smoke_only: bool = False,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    model_dtype: str = "bfloat16",
    action_expert_variant: str = "gemma_300m_lora",
    paligemma_variant: str = "gemma_2b_lora",
) -> Seen10RuntimeConfig:
    """Build and validate one reproducible runtime configuration."""

    if decay_steps is None:
        decay_steps = num_train_steps
    return Seen10RuntimeConfig(
        data_root=pathlib.Path(data_root).expanduser().resolve(),
        run_dir=pathlib.Path(run_dir).expanduser().resolve(),
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_lr=decay_lr,
        fsdp_devices=fsdp_devices,
        init_checkpoint=init_checkpoint,
        resume=resume,
        smoke_only=smoke_only,
        max_train_samples=max_train_samples,
        max_validation_samples=max_validation_samples,
        model_dtype=model_dtype,
        action_expert_variant=action_expert_variant,
        paligemma_variant=paligemma_variant,
    )


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def save_runtime_config(config: Seen10RuntimeConfig, *, model_config: CSGOPi0Config | None = None) -> pathlib.Path:
    """Persist the exact non-object configuration used by a run."""

    config.run_dir.mkdir(parents=True, exist_ok=True)
    path = config.run_dir / "run_config.json"
    payload = {
        "benchmark_id": "csgo_benchmark_v2",
        "model_name": "pi0.5",
        "task": "localization",
        "maps": list(SEEN_MAPS),
        "runtime": _jsonable(config),
        "model": _jsonable(model_config or CSGOPi0Config()),
        "action_contract": {"action_dim": 5, "action_horizon": 1, "pose_space": "normalized"},
        "smoke_only": bool(config.smoke_only),
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = dict(payload)
        existing_runtime = existing.get("runtime", {})
        comparable_runtime = dict(comparable["runtime"])
        # Resume changes only execution state; the requested target step count
        # is part of the immutable run contract because it determines the five
        # validation/save milestones.
        if isinstance(existing_runtime, Mapping):
            for key in ("resume",):
                if key in existing_runtime:
                    comparable_runtime[key] = existing_runtime[key]
        comparable["runtime"] = comparable_runtime
        if existing != comparable:
            raise RuntimeError(f"Existing run_config differs from requested configuration: {path}")
    else:
        _atomic_json(path, payload)
    return path


def build_native_train_config(config: Seen10RuntimeConfig) -> _config.TrainConfig:
    """Create the native ``TrainConfig`` consumed by ``scripts.train``."""

    model_config = CSGOPi0Config(
        dtype=config.model_dtype,
        paligemma_variant=config.paligemma_variant,  # type: ignore[arg-type]
        action_expert_variant=config.action_expert_variant,  # type: ignore[arg-type]
    )
    if config.init_checkpoint in (None, "", "none", "None"):
        weight_loader: _weight_loaders.WeightLoader = _weight_loaders.NoOpWeightLoader()
    else:
        weight_loader = CSGOPi05WeightLoader(str(config.init_checkpoint))
    # The data factory is intentionally not used by the native state builder;
    # the manifest loader below is passed directly to save_state instead.
    return _config.TrainConfig(
        name="csgo_seen10",
        exp_name=config.run_dir.name,
        model=model_config,
        weight_loader=weight_loader,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=config.warmup_steps,
            peak_lr=config.learning_rate,
            decay_steps=config.decay_steps,
            decay_lr=config.decay_lr,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=model_config.get_freeze_filter(),
        data=_config.FakeDataConfig(),
        checkpoint_base_dir=str(config.run_dir.parent),
        seed=config.seed,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        num_train_steps=config.num_train_steps,
        log_interval=1,
        save_interval=config.interval,
        keep_period=config.interval,
        overwrite=False,
        resume=config.resume,
        wandb_enabled=False,
        fsdp_devices=config.fsdp_devices,
    )


def _atomic_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class _TransformedDataset:
    """Apply native per-item transforms before the TorchDataLoader stacks rows."""

    def __init__(self, dataset: _csgo_data.Seen10Dataset, transform: Any):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.transform(self.dataset[index])


class _Seen10NativeLoader:
    """Native TorchDataLoader wrapper with the checkpoint asset hook."""

    def __init__(self, dataset: _csgo_data.Seen10Dataset, loader: Any):
        self.dataset = dataset
        self.loader = loader

    def __iter__(self):
        yield from self.loader

    def __len__(self) -> int:
        return len(self.dataset)

    def data_config(self) -> _config.DataConfig:
        # Manifest poses are already in benchmark normalized space.  An empty
        # stats mapping keeps native data-config consumers on the no-op path
        # without asking them to compute z-score statistics.
        return _config.DataConfig(repo_id="csgo_benchmark_v2_seen10", norm_stats={})


def _create_loader(
    config: Seen10RuntimeConfig,
    *,
    split: str,
    batch_size: int,
    shuffle: bool,
    max_samples: int | None,
    sharding: Any,
    num_batches: int | None = None,
    input_transform: Any | None = None,
) -> _Seen10NativeLoader:
    """Create the fixed manifest-driven loader used by native training."""

    dataset = _csgo_data.Seen10Dataset(
        config.data_root,
        split,
        shared_eval_dir=os.environ.get("CSGO_SHARED_EVAL_DIR", _csgo_data.DEFAULT_SHARED_EVAL_DIR),
        include_actions=True,
        limit=max_samples,
    )
    if split == TRAIN_SPLIT and len(dataset) % batch_size:
        raise ValueError(
            "train split length must be divisible by batch_size because the native TorchDataLoader uses drop_last=True; "
            f"got {len(dataset)} samples and batch_size={batch_size}"
        )
    model_config = CSGOPi0Config(
        dtype=config.model_dtype,
        paligemma_variant=config.paligemma_variant,  # type: ignore[arg-type]
        action_expert_variant=config.action_expert_variant,  # type: ignore[arg-type]
    )
    transforms = _native_input_transform(model_config) if input_transform is None else input_transform
    transformed = _TransformedDataset(dataset, transforms)
    native_loader = _data_loader.TorchDataLoader(
        transformed,
        local_batch_size=batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        framework="jax",
    )
    return _Seen10NativeLoader(dataset, native_loader)


def _native_input_transform(model_config: CSGOPi0Config) -> Any:
    """Create one reusable native per-sample resize/tokenizer transform."""

    return _transforms.CompositeTransform(
        (
            _transforms.ResizeImages(224, 224),
            _transforms.TokenizePrompt(
                _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                discrete_state_input=model_config.discrete_state_input,
            ),
            _transforms.PadStatesAndActions(5),
        )
    )


def _coerce_batch(batch: Any) -> tuple[_model.Observation, _model.Actions]:
    """Convert one native transformed mapping to the model tree."""

    if not isinstance(batch, Mapping):
        raise TypeError(f"Expected the native CSGO mapping, got {type(batch).__name__}")
    required = {"image", "image_mask", "state", "actions"}
    missing = required - set(batch)
    if missing:
        raise ValueError(f"Native CSGO batch is missing fields: {sorted(missing)}")
    # Observation.from_dict converts uint8 images in place.  Copy the two
    # nested mappings so a loader-owned batch is never changed by that step.
    mapping = dict(batch)
    mapping["image"] = dict(batch["image"])
    mapping["image_mask"] = dict(batch["image_mask"])
    actions = jnp.asarray(mapping.pop("actions"))
    if actions.ndim == 2:
        actions = actions[:, None, :]
    if actions.ndim != 3 or actions.shape[-2:] != (1, 5):
        raise ValueError(f"CSGO actions must have shape [batch, 1, 5], got {actions.shape}")
    observation = _model.Observation.from_dict(mapping)
    if observation.state.ndim == 1:
        observation = dataclasses.replace(observation, state=observation.state[None, :])
    return observation, actions


def _step_checkpoint_path(checkpoint_dir: pathlib.Path, step: int) -> pathlib.Path:
    return checkpoint_dir / str(step)


def _checkpoint_identity(checkpoint_path: pathlib.Path) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    identity_path = checkpoint_path / "checkpoint_identity.json"
    if identity_path.is_file():
        payload = json.loads(identity_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("checkpoint_path") == str(checkpoint_path):
            return payload
    digest = hashlib.sha256()
    entries = []
    if checkpoint_path.is_dir():
        for path in sorted(checkpoint_path.rglob("*")):
            if path.is_file():
                stat = path.stat()
                relative = path.relative_to(checkpoint_path).as_posix()
                entries.append((relative, stat.st_size, stat.st_mtime_ns))
    digest.update(json.dumps(entries, separators=(",", ":")).encode())
    return {"checkpoint_path": str(checkpoint_path), "fingerprint": digest.hexdigest()}


def _write_checkpoint_identity(
    checkpoint_path: pathlib.Path,
    *,
    step: int,
    config: Seen10RuntimeConfig,
    model_config: CSGOPi0Config,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    payload = {
        "benchmark_id": "csgo_benchmark_v2",
        "model_name": "pi0.5",
        "task": "localization",
        "checkpoint_path": str(checkpoint_path),
        "step": int(step),
        "seed": int(config.seed),
        "smoke_only": bool(config.smoke_only),
        "model": _jsonable(model_config),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    payload["fingerprint"] = digest
    _atomic_json(checkpoint_path / "checkpoint_identity.json", payload)
    return payload


def _atomic_symlink(target: pathlib.Path, link: pathlib.Path) -> None:
    if link.exists() and not link.is_symlink():
        raise FileExistsError(f"Cannot replace non-symlink checkpoint alias: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.tmp-{os.getpid()}-{time.time_ns()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    os.symlink(os.path.relpath(target, link.parent), temporary)
    os.replace(temporary, link)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row at {path}:{line_no} must be an object")
            rows.append(value)
    return rows


def _write_jsonl_append(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _loss_plot(loss_path: pathlib.Path, plot_path: pathlib.Path) -> None:
    values = _read_jsonl(loss_path)
    if not values:
        return
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to write the CSGO loss plot") from exc
    steps = [int(row["step"]) for row in values if "step" in row and "loss" in row]
    losses = [float(row["loss"]) for row in values if "step" in row and "loss" in row]
    if not steps:
        return
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(steps, losses, linewidth=1.2)
    axis.set(xlabel="completed step", ylabel="flow matching loss", title="pi0.5 CSGO Seen-10 training")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def _validation_loss(
    state: _training_utils.TrainState,
    loader: Iterable[Any],
    *,
    eval_fn: Any,
    eval_seed: int,
    max_batches: int | None,
) -> float:
    total = 0.0
    count = 0
    for index, raw_batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        observation, actions = _coerce_batch(raw_batch)
        rng = jax.random.fold_in(jax.random.key(eval_seed), index)
        losses = eval_fn(state.params, rng, observation, actions)
        array = np.asarray(losses, dtype=np.float64)
        if not np.isfinite(array).all():
            raise FloatingPointError(f"Non-finite validation loss at batch {index}")
        total += float(array.sum())
        count += int(array.size)
    if count == 0:
        raise RuntimeError("Validation split yielded no batches")
    return total / count


def _make_eval_fn(model_def: Any) -> Any:
    """Compile one parameter-explicit eval function for all checkpoints."""

    def eval_fn(params: Any, rng: Any, observation: _model.Observation, actions: _model.Actions):
        model = nnx.merge(model_def, params)
        model.eval()
        return model.compute_loss(rng, observation, actions, train=False)

    return jax.jit(eval_fn)


def _make_sample_fn(model_def: Any) -> Any:
    """Compile one parameter-explicit sampler for all checkpoint visualizations."""

    def sample_fn(params: Any, rng: Any, observation: _model.Observation):
        model = nnx.merge(model_def, params)
        model.eval()
        return model.sample_actions(rng, observation, num_steps=10)

    return jax.jit(sample_fn)


def _read_existing_best(metrics_path: pathlib.Path) -> tuple[float, int | None]:
    values = _read_jsonl(metrics_path)
    best: tuple[float, int | None] = (math.inf, None)
    for row in values:
        if row.get("validation_loss") is None:
            continue
        loss = float(row["validation_loss"])
        if loss < best[0]:
            best = (loss, int(row["step"]))
    return best


def _call_visualization(
    *,
    config: Seen10RuntimeConfig,
    split: str,
    step: int,
    params: Any,
    sample_fn: Any,
    input_transform: Any,
    max_samples_per_map: int = 10,
) -> None:
    """Render the fixed ten-sample-per-map validation panels."""

    module = importlib.import_module("openpi.csgo.visualization")
    output_dir = config.run_dir / "visualizations" / split / f"step_{step}"
    if output_dir.exists() and any(output_dir.iterdir()):
        return
    dataset = _csgo_data.Seen10Dataset(
        config.data_root,
        split,
        shared_eval_dir=os.environ.get("CSGO_SHARED_EVAL_DIR", _csgo_data.DEFAULT_SHARED_EVAL_DIR),
        include_actions=False,
        limit=config.max_validation_samples if config.smoke_only else None,
    )
    selection = module.fixed_sample_indices(dataset, config.seed, per_map=max_samples_per_map)
    selected_indices = [index for values in selection.values() for index in values]
    predictions = {}
    for index in selected_indices:
        row = dataset.row_at(index)
        observation = _mapping_to_observation(dataset.input_at(index), input_transform)
        observation = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], observation)
        sampled = sample_fn(params, jax.random.fold_in(jax.random.key(config.seed), index), observation)
        predictions[str(row["sample_id"])] = np.asarray(sampled)[0, 0]
    module.render_prediction_maps(dataset, predictions, output_dir, config.seed, selection=selection)


def _render_inference_visualization(
    *,
    config: Seen10RuntimeConfig,
    split: str,
    dataset: _csgo_data.Seen10Dataset,
    predictions: Mapping[str, Mapping[str, Any]],
) -> pathlib.Path:
    """Render fixed samples from the prediction file after inference."""

    module = importlib.import_module("openpi.csgo.visualization")
    selection = module.fixed_sample_indices(dataset, config.seed, per_map=10)
    poses = {
        sample_id: tuple(float(row[field]) for field in PREDICTION_FIELDS[2:]) for sample_id, row in predictions.items()
    }
    output_dir = config.run_dir / "visualizations" / split / "inference"
    module.render_prediction_maps(dataset, poses, output_dir, config.seed, selection=selection)
    return output_dir


def run_training(config: Seen10RuntimeConfig) -> dict[str, Any]:
    """Run native training with exactly five validation/save milestones."""

    model_config = CSGOPi0Config(
        dtype=config.model_dtype,
        paligemma_variant=config.paligemma_variant,  # type: ignore[arg-type]
        action_expert_variant=config.action_expert_variant,  # type: ignore[arg-type]
    )
    if config.resume and not (config.run_dir / "run_config.json").is_file():
        raise FileNotFoundError(f"Cannot resume without run_config.json: {config.run_dir}")
    if config.run_dir.exists() and any(config.run_dir.iterdir()) and not config.resume:
        raise FileExistsError(
            f"Run directory already contains files: {config.run_dir}; choose a new run or pass --resume"
        )
    if not config.smoke_only and config.init_checkpoint in (None, "", "none", "None"):
        raise ValueError("Formal training requires a pi05 pretrained initialization checkpoint")
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"batch_size ({config.batch_size}) must be divisible by the number of JAX devices ({jax.device_count()})"
        )
    save_runtime_config(config, model_config=model_config)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.run_dir / "checkpoints"
    metrics_path = config.run_dir / "train_metrics.jsonl"
    loss_path = config.run_dir / "loss.jsonl"

    train_config = build_native_train_config(config)
    mesh = _sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(_sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    input_transform = _native_input_transform(model_config)
    train_loader = _create_loader(
        config,
        split=TRAIN_SPLIT,
        batch_size=config.batch_size,
        shuffle=True,
        max_samples=config.max_train_samples,
        sharding=data_sharding,
        input_transform=input_transform,
    )
    validation_limit = config.max_validation_samples
    val_loader = _create_loader(
        config,
        split=VALIDATION_SPLIT,
        batch_size=1,
        shuffle=False,
        max_samples=validation_limit,
        # Batch-one validation is replicated so it remains valid when the
        # training mesh has more devices than the fixed validation batch.
        sharding=replicated_sharding,
        num_batches=(validation_limit if validation_limit is not None else 5_000),
        input_transform=input_transform,
    )
    train_iter = iter(train_loader)
    first_batch = _coerce_batch(next(train_iter))
    logger.info("Initialized CSGO data batch: %s", _training_utils.array_tree_to_info(first_batch))

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        checkpoint_dir,
        keep_period=config.interval,
        overwrite=False,
        resume=config.resume,
    )
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    train_state, train_state_sharding = _native_init_train_state(train_config, init_rng, mesh, resume=resuming)
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, train_loader)
    jax.block_until_ready(train_state)
    eval_fn = _make_eval_fn(train_state.model_def)
    sample_fn = _make_sample_fn(train_state.model_def)

    # The first batch was read before state initialization to validate the data
    # contract. It is the first actual batch, so use it as the initial step.
    ptrain_step = jax.jit(
        functools.partial(_native_train_step, train_config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    start_step = int(train_state.step)
    # Native checkpoint restore includes optimizer state.  Keep the iterator
    # created for the contract check: constructing a second TorchDataLoader
    # iterator advances its shuffle generator and would change the resume
    # batch order.  The already consumed first batch is index zero, so a
    # resumed state at step N skips N-1 further batches before taking index N.
    if start_step == 0:
        data_iter = train_iter
    else:
        data_iter = train_iter
        for _ in range(start_step - 1):
            next(data_iter)
    raw_existing = {int(row["step"]) for row in _read_jsonl(loss_path) if "step" in row}
    best_loss, best_step = _read_existing_best(metrics_path)
    completed_step = start_step
    interval = config.interval
    started_at = time.monotonic()

    # The first batch was already consumed above on a fresh run; resumed runs
    # seek the same deterministic iterator to their next unprocessed batch.
    batch = first_batch if start_step == 0 else _coerce_batch(next(data_iter))

    while completed_step < config.num_train_steps:
        with _sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        completed_step = int(train_state.step)
        info_values = {key: float(np.asarray(value)) for key, value in info.items()}
        non_finite = {key: value for key, value in info_values.items() if not math.isfinite(value)}
        if non_finite:
            raise FloatingPointError(f"Non-finite training metrics at step {completed_step}: {non_finite}")
        if completed_step not in raw_existing:
            _write_jsonl_append(
                loss_path,
                [{"step": completed_step, "loss": info_values["loss"], "grad_norm": info_values.get("grad_norm")}],
            )
            raw_existing.add(completed_step)

        if completed_step % interval == 0 or completed_step == config.num_train_steps:
            validation_loss = _validation_loss(
                train_state,
                val_loader,
                eval_fn=eval_fn,
                eval_seed=config.seed + 1_000_003,
                max_batches=validation_limit,
            )
            should_be_best = validation_loss < best_loss
            if should_be_best:
                best_loss, best_step = validation_loss, completed_step
            _checkpoints.save_state(checkpoint_manager, train_state, train_loader, completed_step)
            checkpoint_manager.wait_until_finished()
            step_path = _step_checkpoint_path(checkpoint_dir, completed_step)
            _write_checkpoint_identity(step_path, step=completed_step, config=config, model_config=model_config)
            # Keep aliases usable if the process is interrupted after this
            # checkpoint.  The final pass below repairs aliases on a resumed
            # run, while these updates make every completed save discoverable.
            _atomic_symlink(step_path, checkpoint_dir / "late")
            metric_row = {
                "step": completed_step,
                "validation_loss": validation_loss,
                "is_best": bool(should_be_best),
                "checkpoint": str(step_path.resolve()),
                "seed": config.seed,
                "smoke_only": config.smoke_only,
            }
            existing_metrics_steps = {int(row["step"]) for row in _read_jsonl(metrics_path) if "step" in row}
            if completed_step not in existing_metrics_steps:
                _write_jsonl_append(metrics_path, [metric_row])
            if should_be_best:
                _atomic_symlink(step_path, checkpoint_dir / "best")
            _call_visualization(
                config=config,
                split=VALIDATION_SPLIT,
                step=completed_step,
                params=train_state.params,
                sample_fn=sample_fn,
                input_transform=input_transform,
            )
            elapsed = max(time.monotonic() - started_at, 1e-6)
            rate = max(completed_step - start_step, 1) / elapsed
            remaining = max(config.num_train_steps - completed_step, 0)
            logger.info(
                "progress=%d/%d (%.0f%%) rate=%.3f step/s eta=%.1fs loss=%.6f val=%.6f",
                completed_step,
                config.num_train_steps,
                100.0 * completed_step / config.num_train_steps,
                rate,
                remaining / rate,
                info_values["loss"],
                validation_loss,
            )

        if completed_step < config.num_train_steps:
            batch = _coerce_batch(next(data_iter))

    checkpoint_manager.wait_until_finished()
    steps = sorted(int(step) for step in checkpoint_manager.all_steps())
    if not steps:
        raise RuntimeError("Training finished without a checkpoint")
    late_step = max(steps)
    if best_step is None or best_step not in steps:
        best_step = late_step
    _atomic_symlink(_step_checkpoint_path(checkpoint_dir, late_step), checkpoint_dir / "late")
    _atomic_symlink(_step_checkpoint_path(checkpoint_dir, best_step), checkpoint_dir / "best")
    _loss_plot(loss_path, config.run_dir / "loss.png")
    return {
        "run_dir": str(config.run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "late_step": late_step,
        "best_step": best_step,
        "validation_loss": best_loss,
        "smoke_only": config.smoke_only,
    }


def _native_init_train_state(config: _config.TrainConfig, init_rng: Any, mesh: Any, *, resume: bool):
    # Import lazily so importing the CSGO data/inference module does not import
    # the CLI's wandb initialization or execute a training side effect.
    train_script = importlib.import_module("scripts.train")
    return train_script.init_train_state(config, init_rng, mesh, resume=resume)


def _native_train_step(config: _config.TrainConfig, rng: Any, state: Any, batch: Any):
    train_script = importlib.import_module("scripts.train")
    return train_script.train_step(config, rng, state, batch)


def _mapping_to_observation(mapping: Mapping[str, Any], input_transform: Any) -> _model.Observation:
    """Apply the shared native transform to one data adapter input mapping."""

    mapping = dict(mapping)
    mapping["image"] = dict(mapping["image"])
    # Dataset samples use scalar ``np.bool_`` masks, while Observation's
    # jaxtyping contract requires array leaves.  The batched Torch loader
    # naturally stacks these into arrays; visualization/inference handles one
    # sample directly, so normalize the leaves at this boundary as well.
    mapping["image_mask"] = {
        key: np.asarray(value, dtype=np.bool_) for key, value in mapping["image_mask"].items()
    }
    mapping = input_transform(mapping)
    return _model.Observation.from_dict(mapping)


def _ensure_inference_manifest(
    path: pathlib.Path,
    *,
    config: Seen10RuntimeConfig,
    split: str,
    checkpoint_identity: Mapping[str, Any],
    sample_count: int,
) -> None:
    expected = {
        "benchmark_id": "csgo_benchmark_v2",
        "model_name": "pi0.5",
        "task": "localization",
        "split": split,
        "data_root": str(config.data_root),
        "seed": config.seed,
        "checkpoint_identity": dict(checkpoint_identity),
        "sample_count_requested": sample_count,
        "smoke_only": config.smoke_only,
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "benchmark_id",
            "model_name",
            "task",
            "split",
            "data_root",
            "seed",
            "checkpoint_identity",
            "smoke_only",
        ):
            if existing.get(key) != expected[key]:
                raise RuntimeError(
                    f"Existing inference manifest differs for {key}; refusing to mix checkpoints/configs"
                )
        if existing.get("sample_count_requested") != sample_count:
            raise RuntimeError("Existing inference manifest selected a different row count")
        return
    _atomic_json(path, expected)


def run_inference(
    config: Seen10RuntimeConfig,
    *,
    split: str = TEST_SPLIT,
    checkpoint: str | pathlib.Path | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Run resumable localization inference and emit the strict JSONL contract."""

    if split not in (VALIDATION_SPLIT, TEST_SPLIT):
        raise ValueError(f"Localization inference supports {VALIDATION_SPLIT!r} and {TEST_SPLIT!r}")
    if not config.smoke_only and max_samples is not None:
        raise ValueError("Partial inference is allowed only for smoke runs")
    if config.smoke_only and max_samples is None:
        max_samples = 1
    dataset = _csgo_data.Seen10Dataset(
        config.data_root,
        split,
        shared_eval_dir=os.environ.get("CSGO_SHARED_EVAL_DIR", _csgo_data.DEFAULT_SHARED_EVAL_DIR),
        include_actions=False,
        limit=max_samples,
    )
    rows = list(dataset.rows)
    if not rows:
        raise RuntimeError(f"No rows found for {split}")
    if not config.smoke_only and split == TEST_SPLIT and len(rows) != 20_000:
        raise ValueError(f"Formal Seen-10 inference requires exactly 20,000 rows, got {len(rows)}")

    checkpoint_path = pathlib.Path(checkpoint or config.run_dir / "checkpoints" / "best").expanduser().resolve()
    params_path = checkpoint_path / "params" if (checkpoint_path / "params").is_dir() else checkpoint_path
    if not params_path.is_dir():
        raise FileNotFoundError(f"Checkpoint params directory not found: {params_path}")
    checkpoint_identity = _checkpoint_identity(checkpoint_path)
    if not config.smoke_only and checkpoint_identity.get("smoke_only"):
        raise ValueError("Formal inference cannot use a smoke-only checkpoint")
    output_root = config.run_dir / "localization"
    output_path = output_root / "predictions.jsonl"
    manifest_path = output_root / "inference_manifest.json"
    if output_path.is_file() and not manifest_path.is_file():
        raise RuntimeError(f"Existing predictions have no provenance manifest: {output_path}")
    _ensure_inference_manifest(
        manifest_path,
        config=config,
        split=split,
        checkpoint_identity=checkpoint_identity,
        sample_count=len(rows),
    )

    model_config = CSGOPi0Config(
        dtype=config.model_dtype,
        paligemma_variant=config.paligemma_variant,  # type: ignore[arg-type]
        action_expert_variant=config.action_expert_variant,  # type: ignore[arg-type]
    )
    model = model_config.load(_model.restore_params(params_path, dtype=jnp.bfloat16))
    model.eval()
    sample_fn = nnx_utils.module_jit(model.sample_actions)
    input_transform = _native_input_transform(model_config)
    existing = _read_jsonl(output_path)
    by_sample: dict[str, dict[str, Any]] = {}
    expected_maps = {str(row["sample_id"]): str(row["map_name"]) for row in rows}
    for row in existing:
        if set(row.keys()) != set(PREDICTION_FIELDS):
            raise ValueError(f"Existing prediction row has non-standard fields: {row.keys()}")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in by_sample:
            raise ValueError(f"Duplicate or invalid existing sample_id: {sample_id!r}")
        if sample_id not in expected_maps or row.get("map_name") != expected_maps[sample_id]:
            raise ValueError(f"Existing prediction is outside the requested split: {sample_id!r}")
        for field in PREDICTION_FIELDS[2:]:
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Existing prediction {sample_id!r} has invalid {field}: {value!r}")
        by_sample[sample_id] = row

    base_rng = jax.random.key(config.seed)
    pending: list[dict[str, Any]] = []
    generated = 0
    started_at = time.monotonic()
    progress_interval = max(1, math.ceil(len(rows) / 5))
    next_progress = progress_interval
    for index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        if sample_id in by_sample:
            pass
        else:
            observation = _mapping_to_observation(dataset.input_at(index), input_transform)
            batch_observation = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], observation)
            sample_rng = jax.random.fold_in(base_rng, index)
            actions = sample_fn(sample_rng, batch_observation, num_steps=10)
            prediction = np.asarray(actions)[0, 0]
            if prediction.shape != (5,) or not np.isfinite(prediction).all():
                raise FloatingPointError(f"Invalid prediction for {sample_id}: {prediction}")
            payload = {
                "sample_id": sample_id,
                "map_name": str(row["map_name"]),
                "pred_x": float(prediction[0]),
                "pred_y": float(prediction[1]),
                "pred_z": float(prediction[2]),
                "pred_pitch": float(prediction[3]),
                "pred_yaw": float(prediction[4]),
            }
            pending.append(payload)
            by_sample[sample_id] = payload
            generated += 1
        if len(pending) >= 32:
            _write_jsonl_append(output_path, pending)
            pending.clear()
        completed_rows = index + 1
        if completed_rows >= next_progress or completed_rows == len(rows):
            elapsed = max(time.monotonic() - started_at, 1e-6)
            rate = completed_rows / elapsed
            remaining = max(len(rows) - completed_rows, 0)
            logger.info(
                "inference progress=%d/%d (%.0f%%) generated=%d rate=%.3f row/s eta=%.1fs",
                completed_rows,
                len(rows),
                100.0 * completed_rows / len(rows),
                generated,
                rate,
                remaining / rate,
            )
            while next_progress <= completed_rows:
                next_progress += progress_interval
    _write_jsonl_append(output_path, pending)

    expected_ids = {str(row["sample_id"]) for row in rows}
    missing = sorted(expected_ids - set(by_sample))
    if missing:
        raise RuntimeError(f"Inference left {len(missing)} selected samples missing; first={missing[:5]}")
    visualization_path = _render_inference_visualization(
        config=config,
        split=split,
        dataset=dataset,
        predictions=by_sample,
    )
    return {
        "split": split,
        "sample_count": len(rows),
        "generated": generated,
        "existing": len(rows) - generated,
        "predictions": str(output_path),
        "visualizations": str(visualization_path),
        "checkpoint": checkpoint_identity,
        "smoke_only": config.smoke_only,
    }


__all__ = [
    "SEEN_MAPS",
    "TEST_SPLIT",
    "TRAIN_SPLIT",
    "VALIDATION_SPLIT",
    "Seen10RuntimeConfig",
    "build_native_train_config",
    "make_runtime_config",
    "run_inference",
    "run_training",
    "save_runtime_config",
]
