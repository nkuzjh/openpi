#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${OPENPI_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
UNILIP_PYTHON="${UNILIP_PYTHON:-/home/jiahao/miniconda3/envs/UniLIP/bin/python}"
SHARED_EVAL_DIR="${SHARED_EVAL_DIR:-/home/jiahao/task/csgo_benchmark_v2_eval_general}"
EVALUATOR="${SHARED_EVAL_DIR}/run_eval.py"
DATA_ROOT="${CSGO_BENCHMARK_V2_DATA:-/home/jiahao/task/UniLIP/data/csgo_benchmark_v2}"
OUTPUT_BASE="${PROJECT_ROOT}/outputs/csgo_benchmark_v2_seen10/pi0.5"
RUN_FULL="${RUN_FULL:-0}"
export CSGO_SHARED_EVAL_DIR="$SHARED_EVAL_DIR"
export SHARED_EVAL_DIR
export RUN_FULL
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${PROJECT_ROOT}/.cache/openpi}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PROJECT_ROOT}/.cache/jax}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {train|infer|eval|smoke|all} [--seed N] [options...]" >&2
  exit 2
fi

ACTION="$1"
shift
SEED=0
CHECKPOINT=""
SPLIT="seen_discrete_test"
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      [[ $# -ge 2 ]] || { echo "--seed needs a value" >&2; exit 2; }
      SEED="$2"
      shift 2
      ;;
    --seed=*)
      SEED="${1#*=}"
      shift
      ;;
    --checkpoint)
      [[ $# -ge 2 ]] || { echo "--checkpoint needs a value" >&2; exit 2; }
      CHECKPOINT="$2"
      shift 2
      ;;
    --checkpoint=*)
      CHECKPOINT="${1#*=}"
      shift
      ;;
    --split)
      [[ $# -ge 2 ]] || { echo "--split needs a value" >&2; exit 2; }
      SPLIT="$2"
      shift 2
      ;;
    --split=*)
      SPLIT="${1#*=}"
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

RUN_ROOT="${OUTPUT_BASE}/seed_${SEED}"

require_python() {
  [[ -x "$PYTHON" ]] || { echo "OpenPI Python environment is missing: $PYTHON" >&2; exit 1; }
}

require_evaluator() {
  [[ -f "$EVALUATOR" ]] || { echo "Shared evaluator is missing: $EVALUATOR" >&2; exit 1; }
  [[ -x "$UNILIP_PYTHON" ]] || { echo "UNILIP_PYTHON is missing: $UNILIP_PYTHON" >&2; exit 1; }
}

run_train() {
  require_python
  "$PYTHON" "$PROJECT_ROOT/scripts/train_seen10.py" \
    --seed "$SEED" --data-root "$DATA_ROOT" --run-dir "$1" "${@:2}"
}

run_infer() {
  require_python
  local run_dir="$1"
  shift
  local args=(--seed "$SEED" --data-root "$DATA_ROOT" --run-dir "$run_dir" --split "$SPLIT")
  if [[ -n "$CHECKPOINT" ]]; then
    args+=(--checkpoint "$CHECKPOINT")
  fi
  "$PYTHON" "$PROJECT_ROOT/scripts/infer_seen10.py" "${args[@]}" "$@"
}

run_eval() {
  require_evaluator
  local run_dir="$1"
  local output="$run_dir/evaluation/localization"
  "$UNILIP_PYTHON" "$EVALUATOR" localization \
    --pred-root "$run_dir/localization" --data-root "$DATA_ROOT" --output "$output"
}

case "$ACTION" in
  smoke)
    require_evaluator
    require_python
    # The shared localization evaluator's smoke protocol uses the test split;
    # keep this bounded end-to-end action on that same contract.
    SPLIT="seen_discrete_test"
    stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
    smoke_root="${CSGO_SMOKE_ROOT:-${PROJECT_ROOT}/outputs/csgo_benchmark_v2_smoke/pi0.5/seed_${SEED}/${stamp}}"
    run_train "$smoke_root" --smoke "${FORWARD_ARGS[@]}"
    run_infer "$smoke_root" --smoke --checkpoint "$smoke_root/checkpoints/best"
    "$UNILIP_PYTHON" "$EVALUATOR" smoke localization \
      --pred-root "$smoke_root/localization" --data-root "$DATA_ROOT" --limit 1 \
      | tee "$smoke_root/smoke_eval.json"
    ;;
  train)
    [[ "$RUN_FULL" == "1" ]] || { echo "RUN_FULL=0; use '$0 smoke' or set RUN_FULL=1" >&2; exit 2; }
    run_train "$RUN_ROOT" "${FORWARD_ARGS[@]}"
    ;;
  infer)
    [[ "$RUN_FULL" == "1" ]] || { echo "RUN_FULL=0; use '$0 smoke' or set RUN_FULL=1" >&2; exit 2; }
    run_infer "$RUN_ROOT" "${FORWARD_ARGS[@]}"
    ;;
  eval)
    run_eval "$RUN_ROOT"
    ;;
  all)
    if [[ "$RUN_FULL" == "1" ]]; then
      run_train "$RUN_ROOT" "${FORWARD_ARGS[@]}"
      run_infer "$RUN_ROOT"
      run_eval "$RUN_ROOT"
    else
      exec "$0" smoke --seed "$SEED" "${FORWARD_ARGS[@]}"
    fi
    ;;
  *)
    echo "Unknown action: $ACTION (expected train, infer, eval, smoke, or all)" >&2
    exit 2
    ;;
esac
