#!/usr/bin/env python3
"""Train native Pi0.5 on CSGO Benchmark v2 Seen-10 localization."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OPENPI_DATA_HOME", str(ROOT / ".cache" / "openpi"))
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(ROOT / ".cache" / "jax"))
os.environ.setdefault(
    "CSGO_SHARED_EVAL_DIR",
    os.environ.get("SHARED_EVAL_DIR", "/home/jiahao/task/csgo_benchmark_v2_eval_general"),
)

import jax  # noqa: E402

from openpi.csgo.profiles import profile_names  # noqa: E402
from openpi.csgo.runtime import load_run_config  # noqa: E402
from openpi.csgo.runtime import make_runtime_config  # noqa: E402
from openpi.csgo.runtime import run_training  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
DEFAULT_DATA_ROOT = os.environ.get("CSGO_BENCHMARK_V2_DATA", "/home/jiahao/task/UniLIP/data/csgo_benchmark_v2")
DEFAULT_OUTPUT_BASE = ROOT / "outputs" / "csgo_benchmark_v2_seen10" / "pi0.5"


def _default_run_dir(seed: int, *, profile: str, smoke: bool) -> Path:
    profile_path = "pi0.5" if profile == "v2_5k" else f"pi0.5_{profile}"
    if smoke:
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S_%f")
        return ROOT / "outputs" / "csgo_benchmark_v2_smoke" / profile_path / f"seed_{seed}" / stamp
    if profile == "v2_5k":
        return DEFAULT_OUTPUT_BASE / f"seed_{seed}"
    return ROOT / "outputs" / "csgo_benchmark_v2_seen10" / profile_path / f"seed_{seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--profile", "--experiment-profile", dest="experiment_profile", choices=profile_names())
    parser.add_argument("--batch-size", type=int, help="global microbatch size across JAX devices")
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--effective-batch-size", type=int)
    parser.add_argument("--num-train-steps", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--decay-steps", type=int)
    parser.add_argument("--decay-lr", type=float)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument(
        "--init-checkpoint",
        default=os.environ.get("CSGO_PI05_BASE", "gs://openpi-assets/checkpoints/pi05_base/params"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--model-dtype", choices=("bfloat16", "float32"))
    parser.add_argument("--paligemma-variant")
    parser.add_argument("--action-expert-variant")
    parser.add_argument("--use-quantile-norm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-augmentation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--discrete-state-input", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def _resume_profile(run_dir: Path, requested_profile: str | None) -> tuple[str, dict | None]:
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        return requested_profile or "v2_5k", None
    saved = load_run_config(run_dir)
    saved_profile = saved["runtime"]["experiment_profile"]
    if requested_profile is not None and requested_profile != saved_profile:
        raise SystemExit(f"--profile {requested_profile} conflicts with run_config profile {saved_profile}")
    return saved_profile, saved


def _check_resume_overrides(args: argparse.Namespace, saved: dict) -> None:
    """Reject explicit contract changes while allowing defaults to restore from run_config."""

    runtime = saved["runtime"]
    model = saved["model"]
    pairs = {
        "batch_size": runtime.get("batch_size"),
        "gradient_accumulation_steps": runtime.get("gradient_accumulation_steps", 1),
        "effective_batch_size": runtime.get("effective_batch_size", runtime.get("batch_size", 1)),
        "num_train_steps": runtime.get("num_train_steps"),
        "learning_rate": runtime.get("learning_rate"),
        "warmup_steps": runtime.get("warmup_steps"),
        "decay_steps": runtime.get("decay_steps"),
        "decay_lr": runtime.get("decay_lr"),
        "model_dtype": model.get("dtype"),
        "paligemma_variant": model.get("paligemma_variant"),
        "action_expert_variant": model.get("action_expert_variant"),
        "use_quantile_norm": runtime.get("use_quantile_norm"),
        "use_augmentation": runtime.get("use_augmentation"),
        "discrete_state_input": runtime.get("discrete_state_input", model.get("discrete_state_input")),
    }
    for arg_name, saved_value in pairs.items():
        requested = getattr(args, arg_name, None)
        if requested is not None and requested != saved_value:
            raise SystemExit(f"--{arg_name.replace('_', '-')}={requested} conflicts with saved run value {saved_value}")


def _apply_smoke_defaults(args: argparse.Namespace, *, profile_name: str) -> None:
    """Fill bounded defaults without overriding explicit small smoke values."""

    if args.num_train_steps is None:
        args.num_train_steps = 5
    if args.batch_size is None:
        # The 32D profiles' formal batch is 128. A one-sample microbatch and
        # one update are the safe smoke defaults on a single accelerator. A
        # multi-device loader needs at least one sample for every device.
        args.batch_size = max(1, jax.device_count())
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = 1
    if args.warmup_steps is None:
        args.warmup_steps = 1
    if args.decay_steps is None:
        args.decay_steps = args.num_train_steps
    if args.max_train_samples is None:
        args.max_train_samples = args.batch_size
    elif args.max_train_samples % args.batch_size:
        raise SystemExit(
            "--max-train-samples must be divisible by --batch-size because the smoke train loader drops incomplete batches"
        )
    if args.max_validation_samples is None:
        args.max_validation_samples = 1
    # Preserve a user-specified effective size when it matches the explicit
    # microbatch/accumulation pair; otherwise make_runtime_config reports the
    # exact conflict instead of silently changing it.
    if args.effective_batch_size is None and profile_name == "v2_5k":
        args.effective_batch_size = args.batch_size * args.gradient_accumulation_steps


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    saved_config = None
    if args.resume and run_dir is not None:
        profile_name, saved_config = _resume_profile(run_dir, args.experiment_profile)
        if saved_config is not None:
            _check_resume_overrides(args, saved_config)
        smoke_only = (
            bool(saved_config.get("smoke_only", saved_config["runtime"].get("smoke_only", False)))
            if saved_config
            else args.smoke
        )
    else:
        profile_name = args.experiment_profile or "v2_5k"
        smoke_only = args.smoke
    if not smoke_only and os.environ.get("RUN_FULL", "0") != "1":
        raise SystemExit("RUN_FULL=0; pass --smoke for the bounded run or set RUN_FULL=1 for formal training")
    if run_dir is None:
        run_dir = _default_run_dir(args.seed, profile=profile_name, smoke=smoke_only).expanduser().resolve()
    if args.resume and saved_config is None and (run_dir / "run_config.json").is_file():
        profile_name, saved_config = _resume_profile(run_dir, args.experiment_profile)
        if saved_config is not None:
            _check_resume_overrides(args, saved_config)
            smoke_only = bool(saved_config.get("smoke_only", saved_config["runtime"].get("smoke_only", False)))
        if not smoke_only and os.environ.get("RUN_FULL", "0") != "1":
            raise SystemExit("RUN_FULL=0; set RUN_FULL=1 to resume formal training")
    if smoke_only and not args.resume:
        _apply_smoke_defaults(args, profile_name=profile_name)
    config = make_runtime_config(
        data_root=args.data_root,
        run_dir=run_dir,
        seed=args.seed,
        experiment_profile=profile_name,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_batch_size=args.effective_batch_size,
        num_train_steps=args.num_train_steps,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        decay_lr=args.decay_lr,
        fsdp_devices=args.fsdp_devices,
        init_checkpoint=args.init_checkpoint,
        resume=args.resume,
        smoke_only=smoke_only,
        max_train_samples=getattr(args, "max_train_samples", None),
        max_validation_samples=getattr(args, "max_validation_samples", None),
        model_dtype=args.model_dtype,
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
        use_quantile_norm=args.use_quantile_norm,
        use_augmentation=args.use_augmentation,
        discrete_state_input=args.discrete_state_input,
    )
    result = run_training(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
