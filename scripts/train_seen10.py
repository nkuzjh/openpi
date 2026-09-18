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

from openpi.csgo.runtime import make_runtime_config  # noqa: E402, I001
from openpi.csgo.runtime import run_training  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
DEFAULT_DATA_ROOT = os.environ.get("CSGO_BENCHMARK_V2_DATA", "/home/jiahao/task/UniLIP/data/csgo_benchmark_v2")
DEFAULT_OUTPUT_BASE = ROOT / "outputs" / "csgo_benchmark_v2_seen10" / "pi0.5"


def _default_run_dir(seed: int, *, smoke: bool) -> Path:
    if smoke:
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S_%f")
        return ROOT / "outputs" / "csgo_benchmark_v2_smoke" / "pi0.5" / f"seed_{seed}" / stamp
    return DEFAULT_OUTPUT_BASE / f"seed_{seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-train-steps", type=int, default=5_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--decay-steps", type=int)
    parser.add_argument("--decay-lr", type=float, default=1.0e-6)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument(
        "--init-checkpoint",
        default=os.environ.get("CSGO_PI05_BASE", "gs://openpi-assets/checkpoints/pi05_base/params"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--paligemma-variant", default="gemma_2b_lora")
    parser.add_argument("--action-expert-variant", default="gemma_300m_lora")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.smoke and os.environ.get("RUN_FULL", "0") != "1":
        raise SystemExit("RUN_FULL=0; pass --smoke for the bounded run or set RUN_FULL=1 for formal training")
    if args.smoke:
        args.num_train_steps = 5
        args.batch_size = 1
        args.num_workers = 0
        args.warmup_steps = 1
        args.decay_steps = 5
        args.max_train_samples = 1
        args.max_validation_samples = 1
    run_dir = (args.run_dir or _default_run_dir(args.seed, smoke=args.smoke)).expanduser().resolve()
    config = make_runtime_config(
        data_root=args.data_root,
        run_dir=run_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        decay_lr=args.decay_lr,
        fsdp_devices=args.fsdp_devices,
        init_checkpoint=args.init_checkpoint,
        resume=args.resume,
        smoke_only=args.smoke,
        max_train_samples=getattr(args, "max_train_samples", None),
        max_validation_samples=getattr(args, "max_validation_samples", None),
        model_dtype=args.model_dtype,
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
    )
    result = run_training(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
