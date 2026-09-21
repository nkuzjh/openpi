"""CLI default and path tests for the two Seen-10 profiles."""

# ruff: noqa: SLF001 -- private CLI helpers expose the profile defaults under test.

from __future__ import annotations

import pathlib
import sys

import pytest

from openpi.csgo.runtime import make_runtime_config
from scripts import infer_seen10
from scripts import train_seen10


def test_profile_omission_keeps_v2_defaults_and_paths(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["train_seen10.py"])
    args = train_seen10.parse_args()
    assert args.experiment_profile is None
    config = make_runtime_config(data_root=tmp_path, run_dir=tmp_path / "default")
    assert config.experiment_profile == "v2_5k"
    assert config.num_train_steps == 5_000
    assert train_seen10._default_run_dir(0, profile="v2_5k", smoke=False) == train_seen10.DEFAULT_OUTPUT_BASE / "seed_0"
    assert infer_seen10._default_run_dir(0, profile="v2_5k", smoke=False) == infer_seen10.DEFAULT_OUTPUT_BASE / "seed_0"


def test_exp32_smoke_defaults_to_small_batch_and_profile_run_dir(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(sys, "argv", ["train_seen10.py", "--profile", "exp32_loc_main", "--smoke"])
    args = train_seen10.parse_args()
    train_seen10._apply_smoke_defaults(args, profile_name=args.experiment_profile)
    config = make_runtime_config(
        data_root=tmp_path,
        run_dir=tmp_path / "smoke",
        experiment_profile=args.experiment_profile,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_batch_size=args.effective_batch_size,
        num_train_steps=args.num_train_steps,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        smoke_only=True,
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
    )
    assert args.batch_size == max(1, train_seen10.jax.device_count())
    assert args.gradient_accumulation_steps == 1
    assert args.max_train_samples == args.batch_size
    assert config.effective_batch_size == max(1, train_seen10.jax.device_count())
    assert config.num_train_steps == 5
    assert config.internal_action_dim == 32
    assert train_seen10._default_run_dir(0, profile="exp32_loc_main", smoke=False).parts[-2:] == (
        "pi0.5_exp32_loc_main",
        "seed_0",
    )
    assert infer_seen10._default_run_dir(0, profile="exp32_loc_main", smoke=False).parts[-2:] == (
        "pi0.5_exp32_loc_main",
        "seed_0",
    )


def test_infer_cli_late_tag_selects_separate_prediction_directory(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["infer_seen10.py", "--checkpoint-tag", "late"])
    args = infer_seen10.parse_args()
    assert args.experiment_profile is None
    assert infer_seen10._default_output_name(args) == "localization_late"

    monkeypatch.setattr(sys, "argv", ["infer_seen10.py", "--output-name", "localization_review"])
    args = infer_seen10.parse_args()
    assert infer_seen10._default_output_name(args) == "localization_review"
