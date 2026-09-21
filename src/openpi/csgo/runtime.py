"""Training and inference runtime for native CSGO Seen-10 localization.

This module deliberately contains only orchestration.  Model math, optimizer
updates, sharding and Orbax state layout come from the native openpi modules;
the CSGO data module supplies manifest-driven rows and batches.
"""

# ruff: noqa: E402 -- set data/cache environment defaults before importing JAX and OpenPI modules.

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
import optax

from openpi import transforms as _transforms
from openpi.csgo import data as _csgo_data
from openpi.csgo.model import CSGOPi0Config
from openpi.csgo.model import CSGOPi05WeightLoader
from openpi.csgo.profiles import ExperimentProfile
from openpi.csgo.profiles import get_profile
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
    max_token_len: int = 200
    pytorch_compile_mode: str | None = "max-autotune"
    experiment_profile: str = "v2_5k"
    gradient_accumulation_steps: int = 1
    effective_batch_size: int | None = None
    internal_action_dim: int = 5
    state_dim: int = 5
    normalization_mode: str = "benchmark_manifest"
    use_quantile_norm: bool = False
    use_augmentation: bool = False
    discrete_state_input: bool = True
    norm_stats_digest: str | None = None

    def __post_init__(self) -> None:
        profile = get_profile(self.experiment_profile)
        if self.effective_batch_size is None:
            object.__setattr__(self, "effective_batch_size", self.batch_size * self.gradient_accumulation_steps)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.effective_batch_size != self.batch_size * self.gradient_accumulation_steps:
            raise ValueError(
                "effective_batch_size must equal batch_size * gradient_accumulation_steps; "
                f"got {self.effective_batch_size} != {self.batch_size} * {self.gradient_accumulation_steps}"
            )
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
        if self.internal_action_dim != profile.internal_action_dim:
            raise ValueError(
                f"Profile {profile.name!r} requires internal_action_dim={profile.internal_action_dim}, "
                f"got {self.internal_action_dim}"
            )
        if self.state_dim != profile.state_dim:
            raise ValueError(f"Profile {profile.name!r} requires state_dim={profile.state_dim}, got {self.state_dim}")
        if self.normalization_mode != profile.normalization_mode:
            raise ValueError(
                f"Profile {profile.name!r} requires normalization_mode={profile.normalization_mode!r}, "
                f"got {self.normalization_mode!r}"
            )
        if self.use_quantile_norm != profile.use_quantile_norm:
            raise ValueError(
                f"Profile {profile.name!r} requires use_quantile_norm={profile.use_quantile_norm}, "
                f"got {self.use_quantile_norm}"
            )
        if self.use_augmentation != profile.use_augmentation:
            raise ValueError(
                f"Profile {profile.name!r} requires use_augmentation={profile.use_augmentation}, "
                f"got {self.use_augmentation}"
            )
        if self.discrete_state_input != profile.discrete_state_input:
            raise ValueError(
                f"Profile {profile.name!r} requires discrete_state_input={profile.discrete_state_input}, "
                f"got {self.discrete_state_input}"
            )
        if profile.name == "exp32_loc_main" and not self.smoke_only:
            if self.num_train_steps != profile.num_train_steps:
                raise ValueError(f"Formal exp32_loc_main requires {profile.num_train_steps} optimizer updates")
            if self.effective_batch_size != profile.effective_batch_size:
                raise ValueError(f"Formal exp32_loc_main requires effective_batch_size={profile.effective_batch_size}")
            if (self.learning_rate, self.warmup_steps, self.decay_steps, self.decay_lr) != (
                profile.learning_rate,
                profile.warmup_steps,
                profile.decay_steps,
                profile.decay_lr,
            ):
                raise ValueError("Formal exp32_loc_main learning-rate settings are fixed by the profile")
            if (self.paligemma_variant, self.action_expert_variant) != (
                profile.paligemma_variant,
                profile.action_expert_variant,
            ):
                raise ValueError("Formal exp32_loc_main model variants are fixed by the profile")

    @property
    def interval(self) -> int:
        return self.num_train_steps // 5


def make_runtime_config(
    *,
    data_root: str | pathlib.Path,
    run_dir: str | pathlib.Path,
    seed: int = 0,
    batch_size: int | None = None,
    num_train_steps: int | None = None,
    num_workers: int = 0,
    learning_rate: float | None = None,
    warmup_steps: int | None = None,
    decay_steps: int | None = None,
    decay_lr: float | None = None,
    fsdp_devices: int = 1,
    init_checkpoint: str | None = "gs://openpi-assets/checkpoints/pi05_base/params",
    resume: bool = False,
    smoke_only: bool = False,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    model_dtype: str | None = None,
    action_expert_variant: str | None = None,
    paligemma_variant: str | None = None,
    max_token_len: int = 200,
    pytorch_compile_mode: str | None = "max-autotune",
    experiment_profile: str = "v2_5k",
    gradient_accumulation_steps: int | None = None,
    effective_batch_size: int | None = None,
    internal_action_dim: int | None = None,
    state_dim: int | None = None,
    normalization_mode: str | None = None,
    use_quantile_norm: bool | None = None,
    use_augmentation: bool | None = None,
    discrete_state_input: bool | None = None,
    norm_stats_digest: str | None = None,
) -> Seen10RuntimeConfig:
    """Build and validate one reproducible runtime configuration."""

    profile = get_profile(experiment_profile)
    if batch_size is None:
        if profile.effective_batch_size is not None:
            if (
                gradient_accumulation_steps is not None
                and profile.effective_batch_size % gradient_accumulation_steps == 0
            ):
                batch_size = profile.effective_batch_size // gradient_accumulation_steps
            else:
                device_count = jax.device_count()
                if profile.effective_batch_size % device_count == 0:
                    batch_size = device_count
                else:
                    batch_size = profile.default_microbatch_size
        else:
            batch_size = profile.default_microbatch_size
    if action_expert_variant is None:
        action_expert_variant = profile.action_expert_variant
    if paligemma_variant is None:
        paligemma_variant = profile.paligemma_variant
    if model_dtype is None:
        model_dtype = "bfloat16"
    if gradient_accumulation_steps is None:
        if smoke_only or profile.effective_batch_size is None:
            gradient_accumulation_steps = 1
        else:
            if profile.effective_batch_size % batch_size:
                raise ValueError(
                    f"microbatch size {batch_size} must divide profile effective batch {profile.effective_batch_size}"
                )
            gradient_accumulation_steps = profile.effective_batch_size // batch_size
    if num_train_steps is None:
        num_train_steps = profile.num_train_steps
    if learning_rate is None:
        learning_rate = profile.learning_rate
    if warmup_steps is None:
        warmup_steps = profile.warmup_steps
    if decay_steps is None:
        decay_steps = profile.decay_steps if profile.name != "v2_5k" else num_train_steps
    if decay_lr is None:
        decay_lr = profile.decay_lr
    if effective_batch_size is None:
        effective_batch_size = batch_size * gradient_accumulation_steps
    if internal_action_dim is None:
        internal_action_dim = profile.internal_action_dim
    if state_dim is None:
        state_dim = profile.state_dim
    if normalization_mode is None:
        normalization_mode = profile.normalization_mode
    if use_quantile_norm is None:
        use_quantile_norm = profile.use_quantile_norm
    if use_augmentation is None:
        use_augmentation = profile.use_augmentation
    if discrete_state_input is None:
        discrete_state_input = profile.discrete_state_input
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
        max_token_len=max_token_len,
        pytorch_compile_mode=pytorch_compile_mode,
        experiment_profile=profile.name,
        gradient_accumulation_steps=gradient_accumulation_steps,
        effective_batch_size=effective_batch_size,
        internal_action_dim=internal_action_dim,
        state_dim=state_dim,
        normalization_mode=normalization_mode,
        use_quantile_norm=use_quantile_norm,
        use_augmentation=use_augmentation,
        discrete_state_input=discrete_state_input,
        norm_stats_digest=norm_stats_digest,
    )


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
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
        "experiment_profile": config.experiment_profile,
        "maps": list(SEEN_MAPS),
        "runtime": _jsonable(config),
        "model": _jsonable(model_config or _model_config_from_runtime(config)),
        "action_contract": {
            "action_dim": 5,
            "internal_action_dim": config.internal_action_dim,
            "state_dim": config.state_dim,
            "action_horizon": 1,
            "pose_space": "normalized",
        },
        "normalization": _normalization_summary(config),
        "smoke_only": bool(config.smoke_only),
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = dict(payload)
        existing = _canonicalize_run_config(existing)
        comparable = _canonicalize_run_config(comparable)
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


def _model_config_from_runtime(config: Seen10RuntimeConfig) -> CSGOPi0Config:
    """Build the profile-specific model contract from serialized runtime settings."""

    return CSGOPi0Config(
        profile=config.experiment_profile,  # type: ignore[arg-type]
        dtype=config.model_dtype,
        action_dim=config.internal_action_dim,
        action_horizon=1,
        max_token_len=config.max_token_len,
        paligemma_variant=config.paligemma_variant,  # type: ignore[arg-type]
        action_expert_variant=config.action_expert_variant,  # type: ignore[arg-type]
        discrete_state_input=config.discrete_state_input,
        pytorch_compile_mode=config.pytorch_compile_mode,
    )


def _normalization_summary(config: Seen10RuntimeConfig) -> dict[str, Any]:
    """Return a stable summary of normalization state for run/checkpoint identity.

    The legacy profile consumes already-normalized manifest poses and has no
    learned dataset statistics.  The experimental profile records the digest
    of its train-only q01/q99 action statistics.
    """

    if config.experiment_profile == "v2_5k":
        digest = hashlib.sha256(b"{}").hexdigest()
        return {
            "mode": "benchmark_manifest",
            "use_quantile_norm": False,
            "action_dim": 5,
            "stats_digest": config.norm_stats_digest or digest,
        }
    return {
        "mode": config.normalization_mode,
        "use_quantile_norm": bool(config.use_quantile_norm),
        "action_dim": 5,
        "stats_digest": config.norm_stats_digest,
    }


def _normalization_asset_id(config: Seen10RuntimeConfig) -> str:
    return f"csgo_seen10_{config.experiment_profile}"


def _load_profile_norm_stats(
    config: Seen10RuntimeConfig,
    *,
    checkpoint_path: pathlib.Path | None = None,
) -> Mapping[str, Any] | None:
    """Load or compute the profile's train-only normalization statistics."""

    if not config.use_quantile_norm:
        return None
    module = importlib.import_module("openpi.csgo.normalization")
    asset_id = _normalization_asset_id(config)
    if checkpoint_path is not None:
        checkpoint_stats = checkpoint_path / "assets" / asset_id
        if (checkpoint_stats / "norm_stats.json").is_file():
            stats = module.load_seen10_pose_norm_stats(checkpoint_stats)
            digest = module.stats_digest(stats)
            if config.norm_stats_digest is not None and digest != config.norm_stats_digest:
                raise RuntimeError("Checkpoint normalization stats do not match run_config.json")
            return stats
    cache_dir = config.run_dir / "normalization"
    stats = module.get_seen10_pose_norm_stats(
        config.data_root,
        shared_eval_dir=os.environ.get("CSGO_SHARED_EVAL_DIR", _csgo_data.DEFAULT_SHARED_EVAL_DIR),
        cache_dir=cache_dir,
    )
    digest = module.stats_digest(stats)
    if config.norm_stats_digest is not None and digest != config.norm_stats_digest:
        raise RuntimeError(
            "Current seen_train normalization stats do not match the saved run/checkpoint identity; "
            f"expected {config.norm_stats_digest}, got {digest}"
        )
    return stats


def _canonicalize_run_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize current and historical run_config.json shapes for comparison."""

    normalized = dict(payload)
    runtime = dict(normalized.get("runtime", {}))
    model = dict(normalized.get("model", {}))
    profile_name = runtime.get("experiment_profile", normalized.get("experiment_profile", "v2_5k"))
    profile = get_profile(profile_name)
    normalized["experiment_profile"] = profile.name

    # Historical v2_5k run_config files serialized the model's complete
    # config but omitted several overlapping runtime fields. Resolve them
    # from the saved model first, then from the legacy defaults so resuming an
    # old run compares the same configuration as a newly serialized run.
    model.setdefault("dtype", runtime.get("model_dtype", "bfloat16"))
    model.setdefault("action_dim", runtime.get("internal_action_dim", profile.internal_action_dim))
    model.setdefault("action_horizon", 1)
    model.setdefault("max_token_len", runtime.get("max_token_len", 200))
    model.setdefault("paligemma_variant", runtime.get("paligemma_variant", profile.paligemma_variant))
    model.setdefault("action_expert_variant", runtime.get("action_expert_variant", profile.action_expert_variant))
    model.setdefault("discrete_state_input", runtime.get("discrete_state_input", profile.discrete_state_input))
    model.setdefault("pytorch_compile_mode", runtime.get("pytorch_compile_mode", "max-autotune"))
    model.setdefault("pi05", True)
    normalized["model"] = model

    runtime.setdefault("experiment_profile", profile.name)
    runtime.setdefault("gradient_accumulation_steps", 1)
    runtime.setdefault("effective_batch_size", int(runtime.get("batch_size", 1)))
    runtime.setdefault("internal_action_dim", int(model.get("action_dim", profile.internal_action_dim)))
    runtime.setdefault("state_dim", profile.state_dim)
    runtime.setdefault("normalization_mode", profile.normalization_mode)
    runtime.setdefault("use_quantile_norm", profile.use_quantile_norm)
    runtime.setdefault("use_augmentation", profile.use_augmentation)
    runtime.setdefault("discrete_state_input", model.get("discrete_state_input", profile.discrete_state_input))
    runtime.setdefault("norm_stats_digest", None)
    runtime.setdefault("model_dtype", model.get("dtype", "bfloat16"))
    runtime.setdefault("max_token_len", model.get("max_token_len", 200))
    runtime.setdefault("pytorch_compile_mode", model.get("pytorch_compile_mode", "max-autotune"))
    runtime.setdefault("paligemma_variant", model.get("paligemma_variant", profile.paligemma_variant))
    runtime.setdefault("action_expert_variant", model.get("action_expert_variant", profile.action_expert_variant))
    normalized["runtime"] = runtime
    action_contract = dict(normalized.get("action_contract", {}))
    action_contract.setdefault("action_dim", 5)
    action_contract.setdefault("internal_action_dim", int(model.get("action_dim", profile.internal_action_dim)))
    action_contract.setdefault("state_dim", profile.state_dim)
    action_contract.setdefault("action_horizon", 1)
    action_contract.setdefault("pose_space", "normalized")
    normalized["action_contract"] = action_contract
    normalized.setdefault("normalization", _normalization_summary_from_payload(normalized, profile))
    return normalized


def _normalization_summary_from_payload(payload: Mapping[str, Any], profile: ExperimentProfile) -> dict[str, Any]:
    if "normalization" in payload and isinstance(payload["normalization"], Mapping):
        return dict(payload["normalization"])
    runtime = payload.get("runtime", {})
    if profile.name == "v2_5k":
        return {
            "mode": profile.normalization_mode,
            "use_quantile_norm": False,
            "action_dim": 5,
            "stats_digest": hashlib.sha256(b"{}").hexdigest(),
        }
    return {
        "mode": runtime.get("normalization_mode", profile.normalization_mode),
        "use_quantile_norm": bool(runtime.get("use_quantile_norm", profile.use_quantile_norm)),
        "action_dim": 5,
        "stats_digest": runtime.get("norm_stats_digest"),
    }


def load_run_config(run_dir: str | pathlib.Path) -> dict[str, Any]:
    """Read a run config and fill fields omitted by historical v2_5k runs."""

    path = pathlib.Path(run_dir).expanduser().resolve() / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run configuration not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Run configuration must be a JSON object: {path}")
    return _canonicalize_run_config(payload)


def _restore_model_settings(
    config: Seen10RuntimeConfig,
    *,
    payload: Mapping[str, Any] | None = None,
    checkpoint_identity: Mapping[str, Any] | None = None,
) -> Seen10RuntimeConfig:
    """Restore profile/model/norm choices from the run directory, if present."""

    if payload is None:
        path = config.run_dir / "run_config.json"
        if path.is_file():
            payload = _canonicalize_run_config(json.loads(path.read_text(encoding="utf-8")))
        elif checkpoint_identity is not None and isinstance(checkpoint_identity.get("model"), Mapping):
            model = checkpoint_identity["model"]
            profile_name = checkpoint_identity.get("experiment_profile")
            if profile_name is None:
                profile_name = "v2_5k" if int(model.get("action_dim", 5)) == 5 else "exp32_loc_main"
            profile = get_profile(str(profile_name))
            restored = make_runtime_config(
                data_root=config.data_root,
                run_dir=config.run_dir,
                seed=config.seed,
                experiment_profile=profile.name,
                smoke_only=config.smoke_only,
                max_train_samples=config.max_train_samples,
                max_validation_samples=config.max_validation_samples,
                model_dtype=model.get("dtype", config.model_dtype),
                paligemma_variant=model.get("paligemma_variant", profile.paligemma_variant),
                action_expert_variant=model.get("action_expert_variant", profile.action_expert_variant),
                max_token_len=model.get("max_token_len", config.max_token_len),
                pytorch_compile_mode=model.get("pytorch_compile_mode", config.pytorch_compile_mode),
                internal_action_dim=model.get("action_dim", profile.internal_action_dim),
                discrete_state_input=model.get("discrete_state_input", profile.discrete_state_input),
                norm_stats_digest=(checkpoint_identity.get("normalization") or {}).get("stats_digest")
                if isinstance(checkpoint_identity.get("normalization"), Mapping)
                else None,
            )
            return dataclasses.replace(
                restored,
                num_train_steps=config.num_train_steps if config.smoke_only else restored.num_train_steps,
                smoke_only=config.smoke_only,
            )
        else:
            return config
    runtime = dict(payload.get("runtime", {}))
    model = dict(payload.get("model", {}))
    model_to_runtime = {
        "dtype": "model_dtype",
        "action_dim": "internal_action_dim",
        "max_token_len": "max_token_len",
        "paligemma_variant": "paligemma_variant",
        "action_expert_variant": "action_expert_variant",
        "discrete_state_input": "discrete_state_input",
        "pytorch_compile_mode": "pytorch_compile_mode",
    }
    for model_key, runtime_key in model_to_runtime.items():
        if model_key in model:
            runtime[runtime_key] = model[model_key]
    runtime.setdefault("experiment_profile", payload.get("experiment_profile", "v2_5k"))
    profile = get_profile(runtime["experiment_profile"])
    runtime.setdefault("internal_action_dim", profile.internal_action_dim)
    runtime.setdefault("state_dim", profile.state_dim)
    runtime.setdefault("normalization_mode", profile.normalization_mode)
    runtime.setdefault("use_quantile_norm", profile.use_quantile_norm)
    runtime.setdefault("use_augmentation", profile.use_augmentation)
    runtime.setdefault("discrete_state_input", profile.discrete_state_input)
    runtime.setdefault("norm_stats_digest", None)
    normalization = payload.get("normalization")
    if isinstance(normalization, Mapping) and "stats_digest" in normalization:
        runtime["norm_stats_digest"] = normalization["stats_digest"]
    fields = {field.name for field in dataclasses.fields(Seen10RuntimeConfig)}
    values = {key: value for key, value in runtime.items() if key in fields}
    # Paths and inference execution bounds belong to the current invocation.
    # All profile and training/model settings come from the saved run.
    for key in ("data_root", "run_dir", "smoke_only", "resume"):
        values.pop(key, None)
    return dataclasses.replace(config, **values)


def _restore_training_config(config: Seen10RuntimeConfig) -> Seen10RuntimeConfig:
    """Use the original run contract when resuming an interrupted run."""

    payload = load_run_config(config.run_dir)
    saved_data_root = pathlib.Path(payload.get("runtime", {}).get("data_root", config.data_root)).expanduser().resolve()
    if saved_data_root != config.data_root:
        raise RuntimeError(f"Resume data root differs from run_config.json: {config.data_root} != {saved_data_root}")
    restored = _restore_model_settings(config, payload=payload)
    runtime = dict(payload.get("runtime", {}))
    fields = {field.name for field in dataclasses.fields(Seen10RuntimeConfig)}
    values = {key: value for key, value in runtime.items() if key in fields}
    values["data_root"] = config.data_root
    values["run_dir"] = config.run_dir
    values["resume"] = True
    values["smoke_only"] = bool(payload.get("smoke_only", runtime.get("smoke_only", False)))
    model = dict(payload.get("model", {}))
    for model_key, runtime_key in {
        "dtype": "model_dtype",
        "action_dim": "internal_action_dim",
        "max_token_len": "max_token_len",
        "paligemma_variant": "paligemma_variant",
        "action_expert_variant": "action_expert_variant",
        "discrete_state_input": "discrete_state_input",
        "pytorch_compile_mode": "pytorch_compile_mode",
    }.items():
        if model_key in model:
            values[runtime_key] = model[model_key]
    return dataclasses.replace(restored, **values)


def _initial_microbatches(
    train_iter: Iterable[Any],
    first_batch: tuple[_model.Observation, _model.Actions],
    *,
    start_step: int,
    accumulation_steps: int,
    action_dim: int,
) -> tuple[Iterable[Any], list[Any]]:
    """Seek the deterministic loader to the next update's first microbatch."""

    data_iter = iter(train_iter)
    # ``first_batch`` was already consumed for data-contract validation. On a
    # resume at optimizer step N, exactly N * K microbatches belong to the
    # restored state, including the one already consumed above.
    skip_count = max(start_step * accumulation_steps - 1, 0)
    for _ in range(skip_count):
        next(data_iter)
    if start_step == 0:
        microbatches: list[Any] = [first_batch]
        remaining = accumulation_steps - 1
    else:
        microbatches = [_coerce_batch(next(data_iter), action_dim=action_dim)]
        remaining = accumulation_steps - 1
    for _ in range(remaining):
        microbatches.append(_coerce_batch(next(data_iter), action_dim=action_dim))
    return data_iter, microbatches


def build_native_train_config(config: Seen10RuntimeConfig) -> _config.TrainConfig:
    """Create the native ``TrainConfig`` consumed by ``scripts.train``."""

    profile = get_profile(config.experiment_profile)
    model_config = _model_config_from_runtime(config)
    if config.init_checkpoint in (None, "", "none", "None"):
        weight_loader: _weight_loaders.WeightLoader = _weight_loaders.NoOpWeightLoader()
    else:
        weight_loader = CSGOPi05WeightLoader(
            str(config.init_checkpoint),
            action_dim=config.internal_action_dim,
            profile=config.experiment_profile,  # type: ignore[arg-type]
        )
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
        optimizer=_optimizer.AdamW(
            b1=profile.adam_b1,
            b2=profile.adam_b2,
            eps=profile.adam_eps,
            weight_decay=profile.weight_decay,
            clip_gradient_norm=profile.clip_gradient_norm,
        ),
        ema_decay=profile.ema_decay,
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

    def __init__(
        self,
        dataset: _csgo_data.Seen10Dataset,
        loader: Any,
        *,
        norm_stats: Mapping[str, Any] | None = None,
        asset_id: str | None = None,
    ):
        self.dataset = dataset
        self.loader = loader
        self.norm_stats = None if norm_stats is None else dict(norm_stats)
        self.asset_id = asset_id

    def __iter__(self):
        yield from self.loader

    def __len__(self) -> int:
        return len(self.dataset)

    def data_config(self) -> _config.DataConfig:
        # Profile quantile stats are stored as a regular native checkpoint
        # asset; the legacy profile keeps its manifest-normalized no-op path.
        return _config.DataConfig(
            repo_id="csgo_benchmark_v2_seen10",
            asset_id=self.asset_id,
            norm_stats=self.norm_stats if self.norm_stats is not None else {},
        )


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
    norm_stats: Mapping[str, Any] | None = None,
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
    model_config = _model_config_from_runtime(config)
    transforms = _native_input_transform(model_config) if input_transform is None else input_transform
    sample_transform = _profile_sample_transform(
        config,
        transforms,
        split=split,
        norm_stats=norm_stats,
    )
    transformed = _TransformedDataset(dataset, sample_transform)
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
    return _Seen10NativeLoader(
        dataset,
        native_loader,
        norm_stats=norm_stats,
        asset_id=_normalization_asset_id(config) if norm_stats is not None else None,
    )


def _profile_sample_transform(
    config: Seen10RuntimeConfig,
    native_transform: Any,
    *,
    split: str,
    norm_stats: Mapping[str, Any] | None,
) -> Any:
    """Apply profile-specific sample transforms before native tokenization/padding."""

    augmentation = None
    if config.use_augmentation:
        module = importlib.import_module("openpi.csgo.augmentation")
        augmentation = module.make_seen10_fpv_dropout(split, seed=config.seed)

    normalizer = None
    if norm_stats is not None:
        module = importlib.import_module("openpi.csgo.normalization")
        normalizer = module.normalize_seen10_actions
    transforms = tuple(getattr(native_transform, "transforms", ()))
    if not transforms or not isinstance(transforms[-1], _transforms.PadStatesAndActions):
        raise TypeError("The native CSGO transform must end with PadStatesAndActions")
    prepad_transform = _transforms.CompositeTransform(transforms[:-1])
    padding_transform = transforms[-1]

    def transform(sample: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(sample)
        if augmentation is not None:
            value = augmentation(value)
        value = prepad_transform(value)
        if normalizer is not None and "actions" in value:
            value["actions"] = normalizer(np.asarray(value["actions"], dtype=np.float32), dict(norm_stats))
        return padding_transform(value)

    return transform


def _native_input_transform(model_config: CSGOPi0Config) -> Any:
    """Create one reusable native per-sample resize/tokenizer transform."""

    return _transforms.CompositeTransform(
        (
            _transforms.ResizeImages(224, 224),
            _transforms.TokenizePrompt(
                _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                discrete_state_input=model_config.discrete_state_input,
            ),
            _transforms.PadStatesAndActions(model_config.action_dim),
        )
    )


def _coerce_batch(batch: Any, *, action_dim: int = 5) -> tuple[_model.Observation, _model.Actions]:
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
    if actions.ndim != 3 or actions.shape[-2:] != (1, action_dim):
        raise ValueError(f"CSGO actions must have shape [batch, 1, {action_dim}], got {actions.shape}")
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


def _validate_checkpoint_profile(config: Seen10RuntimeConfig, identity: Mapping[str, Any]) -> None:
    """Reject profile/action/norm mismatches while accepting metadata-free legacy checkpoints."""

    identity_profile = identity.get("experiment_profile", identity.get("profile"))
    if identity_profile is not None and identity_profile != config.experiment_profile:
        raise ValueError(
            f"Checkpoint profile {identity_profile!r} does not match run profile {config.experiment_profile!r}"
        )
    model = identity.get("model", {})
    identity_action_dim = identity.get("action_dim")
    if identity_action_dim is None and isinstance(model, Mapping):
        identity_action_dim = model.get("action_dim")
    if identity_action_dim is not None and int(identity_action_dim) != config.internal_action_dim:
        raise ValueError(
            f"Checkpoint action_dim={identity_action_dim} does not match run action_dim={config.internal_action_dim}"
        )
    state = identity.get("state")
    if isinstance(state, Mapping) and state.get("dim") is not None and int(state["dim"]) != config.state_dim:
        raise ValueError(f"Checkpoint state_dim={state['dim']} does not match run state_dim={config.state_dim}")
    normalization = identity.get("normalization")
    if isinstance(normalization, Mapping):
        identity_digest = normalization.get("stats_digest")
        if (
            config.norm_stats_digest is not None
            and identity_digest is not None
            and identity_digest != config.norm_stats_digest
        ):
            raise ValueError("Checkpoint normalization stats digest does not match run_config.json")


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
        "experiment_profile": config.experiment_profile,
        "action_dim": int(config.internal_action_dim),
        "state": {
            "dim": int(config.state_dim),
            "representation": "discrete_tokens" if config.discrete_state_input else "disabled",
            "input": "zero",
        },
        "normalization": _normalization_summary(config),
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
    action_dim: int = 5,
) -> float:
    total = 0.0
    count = 0
    for index, raw_batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        observation, actions = _coerce_batch(raw_batch, action_dim=action_dim)
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


def _read_existing_best(
    metrics_path: pathlib.Path, *, available_steps: set[int] | None = None
) -> tuple[float, int | None]:
    values = _read_jsonl(metrics_path)
    best: tuple[float, int | None] = (math.inf, None)
    for row in values:
        if row.get("validation_loss") is None:
            continue
        step = int(row["step"])
        if available_steps is not None and step not in available_steps:
            continue
        loss = float(row["validation_loss"])
        if loss < best[0]:
            best = (loss, step)
    return best


def _reconcile_resumed_checkpoints(
    *,
    config: Seen10RuntimeConfig,
    model_config: CSGOPi0Config,
    checkpoint_manager: Any,
    train_state: _training_utils.TrainState,
    train_loader: Iterable[Any],
    val_loader: Iterable[Any],
    eval_fn: Any,
) -> tuple[float, int | None]:
    """Repair metadata/aliases when a process stopped after checkpoint commit.

    A saved TrainState can be durable before validation metrics and aliases are
    appended. On resume, validate any committed milestone missing from the
    metrics file before taking another update, then reconstruct both aliases.
    """

    steps = sorted(int(step) for step in checkpoint_manager.all_steps())
    available_steps = set(steps)
    best_loss, best_step = _read_existing_best(config.run_dir / "train_metrics.jsonl", available_steps=available_steps)
    metric_steps = {
        int(row["step"])
        for row in _read_jsonl(config.run_dir / "train_metrics.jsonl")
        if "step" in row and row.get("validation_loss") is not None
    }

    for step in steps:
        step_path = _step_checkpoint_path(config.run_dir / "checkpoints", step)
        identity_path = step_path / "checkpoint_identity.json"
        identity_is_valid = False
        if identity_path.is_file():
            try:
                identity = json.loads(identity_path.read_text(encoding="utf-8"))
                identity_is_valid = (
                    isinstance(identity, dict)
                    and identity.get("checkpoint_path") == str(step_path.resolve())
                    and identity.get("experiment_profile") is not None
                )
            except (OSError, json.JSONDecodeError):
                identity_is_valid = False
            if identity_is_valid:
                _validate_checkpoint_profile(config, identity)
        if not identity_is_valid:
            _write_checkpoint_identity(step_path, step=step, config=config, model_config=model_config)

    for step in steps:
        is_milestone = step > 0 and (step % config.interval == 0 or step == config.num_train_steps)
        if not is_milestone or step in metric_steps:
            continue
        if step == int(train_state.step):
            milestone_state = train_state
        else:
            milestone_state = _checkpoints.restore_state(checkpoint_manager, train_state, train_loader, step=step)
        validation_loss = _validation_loss(
            milestone_state,
            val_loader,
            eval_fn=eval_fn,
            eval_seed=config.seed + 1_000_003,
            max_batches=config.max_validation_samples,
            action_dim=config.internal_action_dim,
        )
        is_best = validation_loss < best_loss
        if is_best:
            best_loss, best_step = validation_loss, step
        _write_jsonl_append(
            config.run_dir / "train_metrics.jsonl",
            [
                {
                    "step": step,
                    "validation_loss": validation_loss,
                    "is_best": bool(is_best),
                    "checkpoint": str(_step_checkpoint_path(config.run_dir / "checkpoints", step).resolve()),
                    "seed": config.seed,
                    "smoke_only": config.smoke_only,
                }
            ],
        )
        metric_steps.add(step)

    if not steps:
        return best_loss, best_step
    latest_step = max(steps)
    if best_step is None or best_step not in available_steps:
        best_step = latest_step
    checkpoint_dir = config.run_dir / "checkpoints"
    _atomic_symlink(_step_checkpoint_path(checkpoint_dir, latest_step), checkpoint_dir / "late")
    _atomic_symlink(_step_checkpoint_path(checkpoint_dir, best_step), checkpoint_dir / "best")
    return best_loss, best_step


def _call_visualization(
    *,
    config: Seen10RuntimeConfig,
    split: str,
    step: int,
    params: Any,
    sample_fn: Any,
    input_transform: Any,
    norm_stats: Mapping[str, Any] | None = None,
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
        predictions[str(row["sample_id"])] = _prediction_to_pose(
            config,
            np.asarray(sampled)[0, 0],
            norm_stats=norm_stats,
        )
    module.render_prediction_maps(dataset, predictions, output_dir, config.seed, selection=selection)


def _prediction_to_pose(
    config: Seen10RuntimeConfig,
    prediction: Any,
    *,
    norm_stats: Mapping[str, Any] | None,
) -> np.ndarray:
    """Convert the model's internal action back to benchmark normalized 5DoF."""

    values = np.asarray(prediction, dtype=np.float32)
    if values.ndim == 0 or values.shape[-1] < 5:
        raise ValueError(f"Expected a model action with at least five dimensions, got {values.shape}")
    pose = values[..., :5]
    if config.use_quantile_norm:
        if norm_stats is None:
            raise RuntimeError("The selected profile requires train-only quantile statistics for inference")
        module = importlib.import_module("openpi.csgo.normalization")
        pose = module.unnormalize_seen10_actions(pose, dict(norm_stats))
    return np.asarray(pose, dtype=np.float32)


def _render_inference_visualization(
    *,
    config: Seen10RuntimeConfig,
    split: str,
    dataset: _csgo_data.Seen10Dataset,
    predictions: Mapping[str, Mapping[str, Any]],
    output_name: str,
) -> pathlib.Path:
    """Render fixed samples from the prediction file after inference."""

    module = importlib.import_module("openpi.csgo.visualization")
    selection = module.fixed_sample_indices(dataset, config.seed, per_map=10)
    poses = {
        sample_id: tuple(float(row[field]) for field in PREDICTION_FIELDS[2:]) for sample_id, row in predictions.items()
    }
    visualization_name = (
        "inference" if output_name == "localization" else f"inference_{output_name.removeprefix('localization_')}"
    )
    output_dir = config.run_dir / "visualizations" / split / visualization_name
    module.render_prediction_maps(dataset, poses, output_dir, config.seed, selection=selection)
    return output_dir


def run_training(config: Seen10RuntimeConfig) -> dict[str, Any]:
    """Run native training with exactly five validation/save milestones."""

    if config.resume and (config.run_dir / "run_config.json").is_file():
        config = _restore_training_config(config)
    model_config = _model_config_from_runtime(config)
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
    norm_stats = _load_profile_norm_stats(config)
    if norm_stats is not None:
        stats_module = importlib.import_module("openpi.csgo.normalization")
        stats_digest = stats_module.stats_digest(dict(norm_stats))
        if config.norm_stats_digest is not None and config.norm_stats_digest != stats_digest:
            raise RuntimeError("Loaded normalization stats differ from the saved run configuration")
        config = dataclasses.replace(config, norm_stats_digest=stats_digest)
        model_config = _model_config_from_runtime(config)
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
        norm_stats=norm_stats,
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
        norm_stats=norm_stats,
    )
    train_iter = iter(train_loader)
    first_batch = _coerce_batch(next(train_iter), action_dim=config.internal_action_dim)
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

    start_step = int(train_state.step)
    accumulation_steps = config.gradient_accumulation_steps
    saved_steps = {int(step) for step in checkpoint_manager.all_steps()}
    best_loss, best_step = _read_existing_best(metrics_path, available_steps=saved_steps)
    if resuming and saved_steps:
        best_loss, best_step = _reconcile_resumed_checkpoints(
            config=config,
            model_config=model_config,
            checkpoint_manager=checkpoint_manager,
            train_state=train_state,
            train_loader=train_loader,
            val_loader=val_loader,
            eval_fn=eval_fn,
        )
    if accumulation_steps == 1:
        ptrain_step = jax.jit(
            functools.partial(_native_train_step, train_config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
    else:
        # The added leading accumulation dimension remains replicated; the
        # existing per-microbatch data axes still shard across devices.
        accumulated_data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(None, _sharding.DATA_AXIS)
        )
        ptrain_step = jax.jit(
            functools.partial(
                _native_accumulated_train_step,
                train_config,
                accumulation_steps=accumulation_steps,
            ),
            in_shardings=(replicated_sharding, train_state_sharding, accumulated_data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
    data_iter, microbatches = _initial_microbatches(
        train_iter,
        first_batch,
        start_step=start_step,
        accumulation_steps=accumulation_steps,
        action_dim=config.internal_action_dim,
    )
    raw_existing = {int(row["step"]) for row in _read_jsonl(loss_path) if "step" in row}
    completed_step = start_step
    interval = config.interval
    started_at = time.monotonic()

    while completed_step < config.num_train_steps:
        train_batch = microbatches[0] if accumulation_steps == 1 else _stack_micro_batches(microbatches)
        with _sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, train_batch)
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
                action_dim=config.internal_action_dim,
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
                norm_stats=norm_stats,
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
            microbatches = [
                _coerce_batch(next(data_iter), action_dim=config.internal_action_dim) for _ in range(accumulation_steps)
            ]

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
        "experiment_profile": config.experiment_profile,
        "effective_batch_size": config.effective_batch_size,
    }


def _native_init_train_state(config: _config.TrainConfig, init_rng: Any, mesh: Any, *, resume: bool):
    # Import lazily so importing the CSGO data/inference module does not import
    # the CLI's wandb initialization or execute a training side effect.
    train_script = importlib.import_module("scripts.train")
    return train_script.init_train_state(config, init_rng, mesh, resume=resume)


def _native_train_step(config: _config.TrainConfig, rng: Any, state: Any, batch: Any):
    train_script = importlib.import_module("scripts.train")
    return train_script.train_step(config, rng, state, batch)


def _stack_micro_batches(micro_batches: Sequence[Any]) -> Any:
    """Stack same-shaped microbatches on an accumulation axis."""

    if not micro_batches:
        raise ValueError("At least one microbatch is required")
    return jax.tree.map(lambda *values: jnp.stack(values, axis=0), *micro_batches)


def _native_accumulated_train_step(
    config: _config.TrainConfig,
    rng: Any,
    state: _training_utils.TrainState,
    stacked_micro_batches: Any,
    *,
    accumulation_steps: int,
):
    """Apply one optimizer update from the mean gradient of several microbatches.

    The optimizer, clipping transform, TrainState counter, and EMA (if enabled)
    each run exactly once per effective batch.  Every microbatch gets an RNG
    key derived independently from the update key and its accumulation index.
    """

    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model: _model.BaseModel, micro_rng: Any, observation: _model.Observation, actions: _model.Actions):
        return jnp.mean(model.compute_loss(micro_rng, observation, actions, train=True))

    update_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    trainable_params = state.params.filter(config.trainable_filter)
    zero_grads = jax.tree.map(jnp.zeros_like, trainable_params)

    def accumulate(carry: tuple[Any, Any], values: tuple[Any, Any]):
        grads_sum, loss_sum = carry
        micro_index, micro_batch = values
        observation, actions = micro_batch
        micro_rng = jax.random.fold_in(update_rng, micro_index)
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, micro_rng, observation, actions)
        grads_sum = jax.tree.map(lambda total, grad: total + grad, grads_sum, grads)
        return (grads_sum, loss_sum + loss), None

    (grads_sum, loss_sum), _ = jax.lax.scan(
        accumulate,
        (zero_grads, jnp.asarray(0.0, dtype=jnp.float32)),
        (jnp.arange(accumulation_steps, dtype=jnp.int32), stacked_micro_batches),
    )
    grads = jax.tree.map(lambda value: value / accumulation_steps, grads_sum)

    updates, new_opt_state = state.tx.update(grads, state.opt_state, trainable_params)
    new_trainable_params = optax.apply_updates(trainable_params, updates)
    nnx.update(model, new_trainable_params)
    new_params = nnx.state(model)
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, value: value.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss_sum / accumulation_steps,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def _mapping_to_observation(mapping: Mapping[str, Any], input_transform: Any) -> _model.Observation:
    """Apply the shared native transform to one data adapter input mapping."""

    mapping = dict(mapping)
    mapping["image"] = dict(mapping["image"])
    # Dataset samples use scalar ``np.bool_`` masks, while Observation's
    # jaxtyping contract requires array leaves.  The batched Torch loader
    # naturally stacks these into arrays; visualization/inference handles one
    # sample directly, so normalize the leaves at this boundary as well.
    mapping["image_mask"] = {key: np.asarray(value, dtype=np.bool_) for key, value in mapping["image_mask"].items()}
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


def _default_inference_output_name(checkpoint: str | pathlib.Path, output_name: str | None) -> str:
    """Return a safe output directory name for one selected checkpoint."""

    if output_name is None:
        output_name = "localization_late" if pathlib.Path(checkpoint).name == "late" else "localization"
    if not output_name or pathlib.Path(output_name).name != output_name or output_name in (".", ".."):
        raise ValueError("output_name must be a single directory name")
    return output_name


def run_inference(
    config: Seen10RuntimeConfig,
    *,
    split: str = TEST_SPLIT,
    checkpoint: str | pathlib.Path | None = None,
    max_samples: int | None = None,
    output_name: str | None = None,
) -> dict[str, Any]:
    """Run resumable localization inference and emit the strict JSONL contract."""

    requested_checkpoint = pathlib.Path(checkpoint or config.run_dir / "checkpoints" / "best").expanduser()
    output_name = _default_inference_output_name(requested_checkpoint, output_name)
    checkpoint_path = requested_checkpoint.resolve()
    params_path = checkpoint_path / "params" if (checkpoint_path / "params").is_dir() else checkpoint_path
    if not params_path.is_dir():
        raise FileNotFoundError(f"Checkpoint params directory not found: {params_path}")
    checkpoint_identity = _checkpoint_identity(checkpoint_path)
    config = _restore_model_settings(config, checkpoint_identity=checkpoint_identity)
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

    if not config.smoke_only and checkpoint_identity.get("smoke_only"):
        raise ValueError("Formal inference cannot use a smoke-only checkpoint")
    _validate_checkpoint_profile(config, checkpoint_identity)
    norm_stats = _load_profile_norm_stats(config, checkpoint_path=checkpoint_path)
    output_root = config.run_dir / output_name
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

    model_config = _model_config_from_runtime(config)
    restore_dtype = jnp.bfloat16 if config.model_dtype == "bfloat16" else jnp.float32
    model = model_config.load(_model.restore_params(params_path, dtype=restore_dtype))
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
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
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
            prediction = _prediction_to_pose(config, np.asarray(actions)[0, 0], norm_stats=norm_stats)
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
        output_name=output_name,
    )
    return {
        "split": split,
        "sample_count": len(rows),
        "generated": generated,
        "existing": len(rows) - generated,
        "output_name": output_name,
        "predictions": str(output_path),
        "visualizations": str(visualization_path),
        "checkpoint": checkpoint_identity,
        "smoke_only": config.smoke_only,
        "experiment_profile": config.experiment_profile,
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
