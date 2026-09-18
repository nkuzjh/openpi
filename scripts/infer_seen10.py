#!/usr/bin/env python3
"""Run native Pi0.5 localization inference on Seen-10 rows."""

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

from openpi.csgo.runtime import TEST_SPLIT  # noqa: E402, I001
from openpi.csgo.runtime import make_runtime_config  # noqa: E402
from openpi.csgo.runtime import run_inference  # noqa: E402


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
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--split", choices=("seen_validation", TEST_SPLIT), default=TEST_SPLIT)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--paligemma-variant", default="gemma_2b_lora")
    parser.add_argument("--action-expert-variant", default="gemma_300m_lora")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.smoke and os.environ.get("RUN_FULL", "0") != "1":
        raise SystemExit("RUN_FULL=0; pass --smoke for the bounded run or set RUN_FULL=1 for formal inference")
    if not args.smoke and args.max_samples is not None:
        raise SystemExit("--max-samples is only valid with --smoke")
    if args.smoke and args.max_samples is None:
        args.max_samples = 1
    run_dir = (args.run_dir or _default_run_dir(args.seed, smoke=args.smoke)).expanduser().resolve()
    checkpoint = args.checkpoint or run_dir / "checkpoints" / "best"
    config = make_runtime_config(
        data_root=args.data_root,
        run_dir=run_dir,
        seed=args.seed,
        num_train_steps=5,
        warmup_steps=1,
        decay_steps=5,
        init_checkpoint=os.environ.get("CSGO_PI05_BASE", "gs://openpi-assets/checkpoints/pi05_base/params"),
        smoke_only=args.smoke,
        model_dtype=args.model_dtype,
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
    )
    result = run_inference(config, split=args.split, checkpoint=checkpoint, max_samples=args.max_samples)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
