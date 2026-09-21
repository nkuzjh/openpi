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
from openpi.csgo.profiles import profile_names  # noqa: E402
from openpi.csgo.runtime import load_run_config  # noqa: E402
from openpi.csgo.runtime import make_runtime_config  # noqa: E402
from openpi.csgo.runtime import run_inference  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
DEFAULT_DATA_ROOT = os.environ.get("CSGO_BENCHMARK_V2_DATA", "/home/jiahao/task/UniLIP/data/csgo_benchmark_v2")
DEFAULT_OUTPUT_BASE = ROOT / "outputs" / "csgo_benchmark_v2_seen10" / "pi0.5"


def _default_run_dir(seed: int, *, profile: str, smoke: bool) -> Path:
    profile_path = "pi0.5" if profile == "v2_5k" else "pi0.5_exp32_loc_main"
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
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-name", help="prediction directory under run_dir (default: localization for best)")
    parser.add_argument("--checkpoint-tag", choices=("best", "late"), help="select the checkpoint alias and output")
    parser.add_argument("--seed", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--profile", "--experiment-profile", dest="experiment_profile", choices=profile_names())
    parser.add_argument("--split", choices=("seen_validation", TEST_SPLIT), default=TEST_SPLIT)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-dtype", choices=("bfloat16", "float32"))
    parser.add_argument("--paligemma-variant")
    parser.add_argument("--action-expert-variant")
    return parser.parse_args()


def _inference_profile(run_dir: Path | None, requested_profile: str | None) -> tuple[str, dict | None]:
    if run_dir is None or not (run_dir / "run_config.json").is_file():
        return requested_profile or "v2_5k", None
    saved = load_run_config(run_dir)
    saved_profile = saved["runtime"]["experiment_profile"]
    if requested_profile is not None and requested_profile != saved_profile:
        raise SystemExit(f"--profile {requested_profile} conflicts with run_config profile {saved_profile}")
    return saved_profile, saved


def _check_model_overrides(args: argparse.Namespace, saved: dict | None) -> None:
    if saved is None:
        return
    model = saved["model"]
    for arg_name, model_key in (
        ("model_dtype", "dtype"),
        ("paligemma_variant", "paligemma_variant"),
        ("action_expert_variant", "action_expert_variant"),
    ):
        requested = getattr(args, arg_name)
        if requested is not None and requested != model.get(model_key):
            raise SystemExit(
                f"--{arg_name.replace('_', '-')}={requested} conflicts with saved run value {model.get(model_key)}"
            )


def _default_output_name(args: argparse.Namespace) -> str | None:
    if args.output_name is not None:
        return args.output_name
    if args.checkpoint_tag == "late":
        return "localization_late"
    return None


def main() -> int:
    args = parse_args()
    if args.checkpoint is not None and args.checkpoint_tag is not None:
        raise SystemExit("Pass either --checkpoint or --checkpoint-tag, not both")
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    profile_name, saved_config = _inference_profile(run_dir, args.experiment_profile)
    _check_model_overrides(args, saved_config)
    if not args.smoke and os.environ.get("RUN_FULL", "0") != "1":
        raise SystemExit("RUN_FULL=0; pass --smoke for the bounded run or set RUN_FULL=1 for formal inference")
    if not args.smoke and args.max_samples is not None:
        raise SystemExit("--max-samples is only valid with --smoke")
    if args.smoke and args.max_samples is None:
        args.max_samples = 1
    if run_dir is None:
        run_dir = _default_run_dir(args.seed, profile=profile_name, smoke=args.smoke).expanduser().resolve()
    checkpoint = args.checkpoint or run_dir / "checkpoints" / (args.checkpoint_tag or "best")
    output_name = _default_output_name(args)
    config = make_runtime_config(
        data_root=args.data_root,
        run_dir=run_dir,
        seed=args.seed,
        experiment_profile=profile_name,
        init_checkpoint=os.environ.get("CSGO_PI05_BASE", "gs://openpi-assets/checkpoints/pi05_base/params"),
        smoke_only=args.smoke,
        model_dtype=args.model_dtype,
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
    )
    result = run_inference(
        config,
        split=args.split,
        checkpoint=checkpoint,
        max_samples=args.max_samples,
        output_name=output_name,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
