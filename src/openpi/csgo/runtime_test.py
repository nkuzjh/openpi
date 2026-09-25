"""Focused tests for the native CSGO runtime profile and checkpoint contracts."""

# ruff: noqa: SLF001 -- the tests verify runtime-private transformation and recovery contracts.

from __future__ import annotations

import dataclasses
import importlib
import os
import pathlib
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi import transforms as _transforms
from openpi.csgo import runtime
from openpi.csgo.profiles import EXP32_LOC_MAIN
from openpi.csgo.profiles import EXP32_LOC_MAIN_FROZEN_VL
from openpi.models import model as _model
from openpi.shared.normalize import NormStats
from openpi.training import utils as _training_utils


def _exp32_smoke_config(tmp_path: pathlib.Path) -> runtime.Seen10RuntimeConfig:
    return runtime.make_runtime_config(
        data_root=tmp_path,
        run_dir=tmp_path / "exp32",
        experiment_profile="exp32_loc_main",
        batch_size=1,
        gradient_accumulation_steps=1,
        effective_batch_size=1,
        num_train_steps=5,
        warmup_steps=1,
        decay_steps=5,
        smoke_only=True,
        max_train_samples=1,
        max_validation_samples=1,
    )


def test_profile_defaults_and_formal_contract(tmp_path: pathlib.Path):
    legacy = runtime.make_runtime_config(data_root=tmp_path, run_dir=tmp_path / "legacy")
    assert legacy.experiment_profile == "v2_5k"
    assert legacy.num_train_steps == 5_000
    assert legacy.batch_size * legacy.gradient_accumulation_steps == 1
    assert legacy.internal_action_dim == 5
    assert not legacy.use_quantile_norm
    assert not legacy.use_augmentation

    new = runtime.make_runtime_config(
        data_root=tmp_path, run_dir=tmp_path / "formal", experiment_profile="exp32_loc_main"
    )
    assert new.num_train_steps == 19_500
    assert new.effective_batch_size == 128
    assert new.batch_size * new.gradient_accumulation_steps == 128
    assert new.interval == 3_900
    assert new.learning_rate == 5e-5
    assert new.warmup_steps == 1_000
    assert new.decay_steps == 100_000
    assert new.decay_lr == 5e-5
    assert (new.internal_action_dim, new.state_dim) == (32, 32)
    assert new.use_quantile_norm
    assert new.normalization_mode == "train_quantile"
    assert new.use_augmentation
    assert not new.discrete_state_input
    assert (new.paligemma_variant, new.action_expert_variant) == (
        "gemma_2b_lora_r32",
        "gemma_300m_lora_r32",
    )

    train_config = runtime.build_native_train_config(new)
    assert train_config.optimizer.b1 == 0.9
    assert train_config.optimizer.b2 == 0.95
    assert train_config.optimizer.eps == 1e-8
    assert train_config.optimizer.weight_decay == 1e-10
    assert train_config.optimizer.clip_gradient_norm == 1.0
    assert train_config.ema_decay is None

    with pytest.raises(ValueError, match="learning-rate settings are fixed"):
        runtime.make_runtime_config(
            data_root=tmp_path,
            run_dir=tmp_path / "conflict",
            experiment_profile="exp32_loc_main",
            learning_rate=1e-4,
        )
    with pytest.raises(ValueError, match="requires effective_batch_size=128"):
        runtime.make_runtime_config(
            data_root=tmp_path,
            run_dir=tmp_path / "small-formal",
            experiment_profile="exp32_loc_main",
            batch_size=1,
            gradient_accumulation_steps=1,
            effective_batch_size=1,
        )


def test_frozen_vl_inherits_exp32_contract_and_formal_enforcement(tmp_path: pathlib.Path):
    assert dataclasses.replace(
        EXP32_LOC_MAIN, name=EXP32_LOC_MAIN_FROZEN_VL.name, use_augmentation=False
    ) == EXP32_LOC_MAIN_FROZEN_VL
    main = runtime.make_runtime_config(
        data_root=tmp_path, run_dir=tmp_path / "main", experiment_profile="exp32_loc_main"
    )
    frozen = runtime.make_runtime_config(
        data_root=tmp_path, run_dir=tmp_path / "frozen", experiment_profile="exp32_loc_main_frozen_vl"
    )
    assert dataclasses.replace(
        main, run_dir=frozen.run_dir, experiment_profile=frozen.experiment_profile, use_augmentation=False
    ) == frozen
    assert main.use_augmentation
    assert not frozen.use_augmentation
    assert frozen.effective_batch_size == 128
    assert frozen.num_train_steps == 19_500
    assert frozen.interval == 4_000
    assert [step for step in range(1, frozen.num_train_steps + 1) if frozen.is_checkpoint_step(step)] == [
        4_000,
        8_000,
        12_000,
        16_000,
        19_500,
    ]
    assert main.interval == 3_900
    frozen_smoke = dataclasses.replace(frozen, smoke_only=True, num_train_steps=10)
    assert frozen_smoke.interval == 2
    assert [step for step in range(1, 11) if frozen_smoke.is_checkpoint_step(step)] == [2, 4, 6, 8, 10]
    for override, pattern in (
        ({"num_train_steps": 5}, "optimizer updates"),
        ({"batch_size": 1, "gradient_accumulation_steps": 1}, "effective_batch_size=128"),
        ({"learning_rate": 1e-4}, "learning-rate settings"),
        ({"paligemma_variant": "gemma_2b_lora"}, "model variants"),
    ):
        with pytest.raises(ValueError, match=pattern):
            runtime.make_runtime_config(
                data_root=tmp_path,
                run_dir=tmp_path / "invalid",
                experiment_profile="exp32_loc_main_frozen_vl",
                **override,
            )


@pytest.mark.parametrize("profile", ["exp32_loc_main", "exp32_loc_main_frozen_vl"])
@pytest.mark.parametrize(("batch_size", "accumulation_steps"), [(32, 4), (64, 2), (128, 1)])
def test_formal_profiles_accept_aligned_microbatches(
    tmp_path: pathlib.Path, profile: str, batch_size: int, accumulation_steps: int
):
    config = runtime.make_runtime_config(
        data_root=tmp_path,
        run_dir=tmp_path / "formal",
        experiment_profile=profile,
        batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
    )
    assert config.effective_batch_size == 128
    assert (config.batch_size, config.gradient_accumulation_steps) == (batch_size, accumulation_steps)


@pytest.mark.parametrize("profile", ["exp32_loc_main", "exp32_loc_main_frozen_vl"])
def test_formal_profiles_reject_other_effective_batch(tmp_path: pathlib.Path, profile: str):
    with pytest.raises(ValueError, match="requires effective_batch_size=128"):
        runtime.make_runtime_config(
            data_root=tmp_path,
            run_dir=tmp_path / "invalid",
            experiment_profile=profile,
            batch_size=32,
            gradient_accumulation_steps=2,
        )


def test_legacy_profile_keeps_its_batch_contract(tmp_path: pathlib.Path):
    default = runtime.make_runtime_config(data_root=tmp_path, run_dir=tmp_path / "default")
    assert (default.batch_size, default.gradient_accumulation_steps, default.effective_batch_size) == (1, 1, 1)
    larger = runtime.make_runtime_config(
        data_root=tmp_path,
        run_dir=tmp_path / "larger",
        batch_size=32,
        gradient_accumulation_steps=4,
    )
    assert larger.effective_batch_size == 128


def test_train_loader_drops_partial_tail_and_resume_replays_full_batches(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    class FakeSeen10Dataset:
        def __init__(self, data_root, split, *, shared_eval_dir, include_actions, limit):
            del data_root, shared_eval_dir
            assert split == runtime.TRAIN_SPLIT
            assert include_actions
            self.size = 130 if limit is None else min(130, limit)

        def __len__(self):
            return self.size

        def __getitem__(self, index):
            return {
                "index": np.asarray(index, dtype=np.int32),
                "state": np.zeros(5, dtype=np.float32),
                "actions": np.zeros(5, dtype=np.float32),
            }

    monkeypatch.setattr(runtime._csgo_data, "Seen10Dataset", FakeSeen10Dataset)
    monkeypatch.setattr(runtime, "_coerce_batch", lambda value, *, action_dim: value)
    config = runtime.make_runtime_config(data_root=tmp_path, run_dir=tmp_path / "loader", seed=17)
    transform = _transforms.CompositeTransform((lambda sample: sample, _transforms.PadStatesAndActions(5)))

    def make_loader():
        return runtime._create_loader(
            config,
            split=runtime.TRAIN_SPLIT,
            batch_size=128,
            shuffle=True,
            max_samples=None,
            sharding=None,
            input_transform=transform,
        )

    expected_iter = iter(make_loader())
    expected = [np.asarray(next(expected_iter)["index"]) for _ in range(4)]
    assert all(batch.shape == (128,) and np.unique(batch).size == 128 for batch in expected)
    assert not np.array_equal(expected[0], expected[1])

    replay_iter = iter(make_loader())
    first_batch = next(replay_iter)
    iterator, resumed = runtime._initial_microbatches(
        replay_iter,
        first_batch,
        start_step=2,
        accumulation_steps=1,
        action_dim=5,
    )
    np.testing.assert_array_equal(resumed[0]["index"], expected[2])
    np.testing.assert_array_equal(next(iterator)["index"], expected[3])


def test_native_transform_and_profile_pipeline_pad_after_normalization(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    config = _exp32_smoke_config(tmp_path)
    model_config = runtime._model_config_from_runtime(config)
    monkeypatch.setattr(runtime._tokenizer, "PaligemmaTokenizer", lambda max_len: object())
    native_transform = runtime._native_input_transform(model_config)
    padding = native_transform.transforms[-1]
    assert isinstance(padding, _transforms.PadStatesAndActions)
    assert padding.model_action_dim == 32

    events: list[str] = []

    def augment(sample: dict[str, Any]) -> dict[str, Any]:
        events.append("raw_fpv_dropout")
        return dict(sample)

    def make_dropout(split: str, *, seed: int):
        assert split == "seen_train"
        assert seed == config.seed
        return augment

    def resize_and_tokenize(sample: dict[str, Any]) -> dict[str, Any]:
        events.append("resize_tokenize")
        return dict(sample)

    def normalize(actions: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
        assert stats == {"stats": "train-only"}
        assert actions.shape == (5,)
        events.append("quantile_normalize")
        return actions + 10.0

    augmentation_module = SimpleNamespace(make_seen10_fpv_dropout=make_dropout)
    normalization_module = SimpleNamespace(normalize_seen10_actions=normalize)
    real_import_module = importlib.import_module

    def import_module(name: str):
        if name == "openpi.csgo.augmentation":
            return augmentation_module
        if name == "openpi.csgo.normalization":
            return normalization_module
        return real_import_module(name)

    monkeypatch.setattr(runtime, "importlib", SimpleNamespace(import_module=import_module))
    wrapped = runtime._profile_sample_transform(
        config,
        _transforms.CompositeTransform((resize_and_tokenize, padding)),
        split="seen_train",
        norm_stats={"stats": "train-only"},
    )
    sample = {
        "image": {"base_0_rgb": np.ones((8, 8, 3), dtype=np.uint8)},
        "image_mask": {"base_0_rgb": np.asarray(1, dtype=np.bool_)},
        "state": np.arange(5, dtype=np.float32),
        "actions": np.arange(5, dtype=np.float32),
    }
    result = wrapped(sample)
    assert events == ["raw_fpv_dropout", "resize_tokenize", "quantile_normalize"]
    np.testing.assert_array_equal(result["actions"][:5], np.arange(5, dtype=np.float32) + 10.0)
    np.testing.assert_array_equal(result["actions"][5:], np.zeros(27, dtype=np.float32))
    assert result["state"].shape == (32,)


def test_frozen_vl_skips_custom_fpv_dropout(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    config = runtime.make_runtime_config(
        data_root=tmp_path, run_dir=tmp_path / "frozen", experiment_profile="exp32_loc_main_frozen_vl"
    )

    def unexpected_import(name: str):
        raise AssertionError(f"Unexpected augmentation import: {name}")

    monkeypatch.setattr(runtime, "importlib", SimpleNamespace(import_module=unexpected_import))
    native_transform = _transforms.CompositeTransform(
        (lambda sample: dict(sample), _transforms.PadStatesAndActions(32))
    )
    wrapped = runtime._profile_sample_transform(config, native_transform, split="seen_train", norm_stats=None)
    result = wrapped({"state": np.arange(5, dtype=np.float32), "actions": np.arange(5, dtype=np.float32)})
    assert result["state"].shape == (32,)
    np.testing.assert_array_equal(result["actions"][:5], np.arange(5, dtype=np.float32))


def test_prediction_uses_first_five_dimensions_then_inverse_normalizes(tmp_path: pathlib.Path):
    config = _exp32_smoke_config(tmp_path)
    stats = {
        "actions": NormStats(
            mean=np.zeros(5, dtype=np.float32),
            std=np.ones(5, dtype=np.float32),
            q01=np.full(5, -2.0, dtype=np.float32),
            q99=np.full(5, 6.0, dtype=np.float32),
        )
    }
    prediction = np.concatenate((np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32), np.full(27, 999.0)))
    actual = runtime._prediction_to_pose(config, prediction, norm_stats=stats)
    expected = importlib.import_module("openpi.csgo.normalization").unnormalize_seen10_actions(prediction[:5], stats)
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (5,)


@dataclasses.dataclass
class _ToyModel(_model.BaseModel):
    weight: nnx.Param
    rng_scaled: bool = False

    def __init__(self, *, rng_scaled: bool = False):
        super().__init__(action_dim=1, action_horizon=1, max_token_len=1)
        self.weight = nnx.Param(jnp.asarray(0.0))
        self.rng_scaled = rng_scaled

    def compute_loss(
        self,
        rng: Any,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        del observation, train
        scale = jax.random.uniform(rng, ()) if self.rng_scaled else 1.0
        error = self.weight.value * scale - actions[:, 0, 0]
        return error**2

    def sample_actions(self, rng: Any, observation: _model.Observation, **kwargs):
        del rng, kwargs
        return jnp.zeros((observation.state.shape[0], 1, 1))


def _toy_train_state(optimizer: optax.GradientTransformation, *, rng_scaled: bool):
    toy_model = _ToyModel(rng_scaled=rng_scaled)
    graph_def, params = nnx.split(toy_model)
    trainable_filter = nnx.All(nnx.Param)
    state = _training_utils.TrainState(
        step=jnp.asarray(0, dtype=jnp.int32),
        params=params,
        model_def=graph_def,
        opt_state=optimizer.init(params.filter(trainable_filter)),
        tx=optimizer,
        ema_decay=None,
        ema_params=None,
    )
    return state, trainable_filter


def _toy_microbatch(target: float):
    observation = _model.Observation(images={}, image_masks={}, state=jnp.ones((1, 1)))
    actions = jnp.asarray([[[target]]], dtype=jnp.float32)
    return observation, actions


def test_accumulated_step_averages_gradients_with_independent_rng_and_steps_once():
    optimizer = optax.sgd(learning_rate=0.05)
    state, trainable_filter = _toy_train_state(optimizer, rng_scaled=True)
    batch = runtime._stack_micro_batches((_toy_microbatch(1.0), _toy_microbatch(3.0)))
    rng = jax.random.key(17)
    next_state, info = runtime._native_accumulated_train_step(
        SimpleNamespace(trainable_filter=trainable_filter),
        rng,
        state,
        batch,
        accumulation_steps=2,
    )

    update_rng = jax.random.fold_in(rng, state.step)
    scales = [float(jax.random.uniform(jax.random.fold_in(update_rng, index), ())) for index in range(2)]
    mean_gradient = np.mean([-2.0 * target * scale for target, scale in zip((1.0, 3.0), scales, strict=True)])
    expected_weight = -0.05 * mean_gradient
    assert int(next_state.step) == 1
    np.testing.assert_allclose(next_state.params["weight"].value, expected_weight, rtol=1e-6)
    np.testing.assert_allclose(info["loss"], 5.0)


def test_accumulated_step_clips_once_after_gradient_mean():
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(learning_rate=1.0))
    state, trainable_filter = _toy_train_state(optimizer, rng_scaled=False)
    batch = runtime._stack_micro_batches((_toy_microbatch(2.0), _toy_microbatch(-1.0)))
    next_state, _ = runtime._native_accumulated_train_step(
        SimpleNamespace(trainable_filter=trainable_filter),
        jax.random.key(4),
        state,
        batch,
        accumulation_steps=2,
    )

    # Per-microbatch gradients are -4 and +2, whose mean is -1. Clipping the
    # mean once leaves -1 and advances w to 1; clipping them independently
    # before averaging would cancel to zero.
    assert int(next_state.step) == 1
    np.testing.assert_allclose(next_state.params["weight"].value, 1.0, rtol=1e-6)


def test_resume_seeks_one_effective_batch_per_restored_update(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(runtime, "_coerce_batch", lambda value, *, action_dim: value)
    batches = iter(range(1, 20))  # batch 0 was consumed for preflight validation
    iterator, microbatches = runtime._initial_microbatches(
        batches,
        0,
        start_step=2,
        accumulation_steps=3,
        action_dim=32,
    )
    assert microbatches == [6, 7, 8]
    assert next(iterator) == 9


def test_current_legacy_seed0_run_config_resumes_without_rewrite():
    run_dir = runtime._PROJECT_ROOT / "outputs" / "csgo_benchmark_v2_seen10" / "pi0.5" / "seed_0"
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        pytest.skip("the workspace's historical seed_0 run_config is not present")
    original_bytes = config_path.read_bytes()
    saved = runtime.load_run_config(run_dir)
    assert saved["runtime"]["experiment_profile"] == "v2_5k"
    assert saved["runtime"]["max_token_len"] == saved["model"]["max_token_len"] == 200
    assert saved["runtime"]["pytorch_compile_mode"] == saved["model"]["pytorch_compile_mode"] == "max-autotune"

    invocation = runtime.make_runtime_config(
        data_root=saved["runtime"]["data_root"],
        run_dir=run_dir,
        experiment_profile="v2_5k",
        resume=True,
    )
    restored = runtime._restore_training_config(invocation)
    assert restored.max_token_len == 200
    assert restored.pytorch_compile_mode == "max-autotune"
    runtime.save_runtime_config(restored)
    assert config_path.read_bytes() == original_bytes


def test_frozen_vl_restores_from_checkpoint_identity_and_rejects_cross_profile(tmp_path: pathlib.Path):
    profile = "exp32_loc_main_frozen_vl"
    checkpoint = tmp_path / "checkpoints" / "1000"
    (checkpoint / "params").mkdir(parents=True)
    frozen = runtime.make_runtime_config(data_root=tmp_path, run_dir=tmp_path / "run", experiment_profile=profile)
    identity = runtime._write_checkpoint_identity(
        checkpoint, step=1000, config=frozen, model_config=runtime._model_config_from_runtime(frozen)
    )
    assert identity["experiment_profile"] == profile
    restored = runtime._restore_model_settings(
        runtime.make_runtime_config(data_root=tmp_path, run_dir=tmp_path / "no_run_config"),
        checkpoint_identity=identity,
    )
    assert restored.experiment_profile == profile
    assert restored.internal_action_dim == 32

    with pytest.raises(ValueError, match="does not match requested profile"):
        runtime.run_inference(frozen, checkpoint=checkpoint, requested_profile="exp32_loc_main")
    main = runtime.make_runtime_config(
        data_root=tmp_path, run_dir=tmp_path / "main", experiment_profile="exp32_loc_main"
    )
    runtime.save_runtime_config(main)
    with pytest.raises(ValueError, match="does not match run profile"):
        runtime.run_inference(main, checkpoint=checkpoint)

    runtime.save_runtime_config(frozen)
    restored_run = runtime._restore_training_config(dataclasses.replace(frozen, resume=True))
    assert restored_run.experiment_profile == profile
    runtime.save_runtime_config(restored_run)


def test_resume_recovers_metrics_and_aliases_after_checkpoint_commit(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "crash_window"
    checkpoint_dir = run_dir / "checkpoints"
    saved_checkpoint = checkpoint_dir / "1000"
    saved_checkpoint.mkdir(parents=True)
    config = runtime.make_runtime_config(
        data_root=tmp_path,
        run_dir=run_dir,
        experiment_profile="v2_5k",
        resume=True,
    )
    model_config = runtime._model_config_from_runtime(config)

    class CheckpointManager:
        def all_steps(self):
            return (1000,)

    state = SimpleNamespace(step=jnp.asarray(1000, dtype=jnp.int32))
    monkeypatch.setattr(runtime, "_validation_loss", lambda *args, **kwargs: 0.25)
    best_loss, best_step = runtime._reconcile_resumed_checkpoints(
        config=config,
        model_config=model_config,
        checkpoint_manager=CheckpointManager(),
        train_state=state,
        train_loader=(),
        val_loader=(),
        eval_fn=object(),
    )

    metrics = runtime._read_jsonl(run_dir / "train_metrics.jsonl")
    assert len(metrics) == 1
    assert metrics[0]["step"] == 1000
    assert metrics[0]["validation_loss"] == 0.25
    assert (saved_checkpoint / "checkpoint_identity.json").is_file()
    assert (checkpoint_dir / "late").resolve() == saved_checkpoint.resolve()
    assert (checkpoint_dir / "best").resolve() == saved_checkpoint.resolve()
    assert (best_loss, best_step) == (0.25, 1000)


def test_frozen_vl_resume_validates_final_checkpoint_and_preserves_best_alias(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "frozen"
    checkpoint_dir = run_dir / "checkpoints"
    steps = (4_000, 8_000, 12_000, 16_000, 19_500)
    for step in steps:
        (checkpoint_dir / str(step)).mkdir(parents=True)
    config = runtime.make_runtime_config(
        data_root=tmp_path,
        run_dir=run_dir,
        experiment_profile="exp32_loc_main_frozen_vl",
        resume=True,
    )
    runtime._write_jsonl_append(
        run_dir / "train_metrics.jsonl",
        [{"step": step, "validation_loss": loss} for step, loss in zip(steps[:-1], (0.4, 0.2, 0.3, 0.5), strict=True)],
    )

    class CheckpointManager:
        def all_steps(self):
            return steps

    monkeypatch.setattr(runtime, "_validation_loss", lambda *args, **kwargs: 0.35)
    best_loss, best_step = runtime._reconcile_resumed_checkpoints(
        config=config,
        model_config=runtime._model_config_from_runtime(config),
        checkpoint_manager=CheckpointManager(),
        train_state=SimpleNamespace(step=jnp.asarray(19_500, dtype=jnp.int32)),
        train_loader=(),
        val_loader=(),
        eval_fn=object(),
    )

    assert [row["step"] for row in runtime._read_jsonl(run_dir / "train_metrics.jsonl")] == list(steps)
    assert (best_loss, best_step) == (0.2, 8_000)
    assert (checkpoint_dir / "best").resolve() == (checkpoint_dir / "8000").resolve()
    assert (checkpoint_dir / "late").resolve() == (checkpoint_dir / "19500").resolve()


@pytest.mark.parametrize(
    ("checkpoint", "requested", "expected"),
    [
        ("/tmp/run/checkpoints/best", None, "localization"),
        ("/tmp/run/checkpoints/late", None, "localization_late"),
        ("/tmp/run/checkpoints/late", "custom_outputs", "custom_outputs"),
    ],
)
def test_best_and_late_have_independent_default_output_names(checkpoint: str, requested: str | None, expected: str):
    assert runtime._default_inference_output_name(checkpoint, requested) == expected


def test_inference_output_name_cannot_escape_run_dir():
    with pytest.raises(ValueError, match="single directory name"):
        runtime._default_inference_output_name("/tmp/checkpoints/best", "../outside")
