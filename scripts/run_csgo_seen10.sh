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
  echo "Usage: $0 {train|infer|eval|smoke|all} [--seed N] [--profile v2_5k|exp32_loc_main|exp32_loc_main_frozen_vl] [options...]" >&2
  exit 2
fi

ACTION="$1"
shift
SEED=0
PROFILE=""
RUN_DIR_OVERRIDE=""
CHECKPOINT=""
CHECKPOINT_TAG=""
OUTPUT_NAME=""
SPLIT="seen_discrete_test"
SMOKE_REQUESTED=0
TRAIN_ARGS=()
INFER_ARGS=()
COMMON_ARGS=()

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
    --profile|--experiment-profile)
      [[ $# -ge 2 ]] || { echo "$1 needs a value" >&2; exit 2; }
      PROFILE="$2"
      shift 2
      ;;
    --profile=*|--experiment-profile=*)
      PROFILE="${1#*=}"
      shift
      ;;
    --run-dir)
      [[ $# -ge 2 ]] || { echo "--run-dir needs a value" >&2; exit 2; }
      RUN_DIR_OVERRIDE="$2"
      shift 2
      ;;
    --run-dir=*)
      RUN_DIR_OVERRIDE="${1#*=}"
      shift
      ;;
    --data-root)
      [[ $# -ge 2 ]] || { echo "--data-root needs a value" >&2; exit 2; }
      DATA_ROOT="$2"
      shift 2
      ;;
    --data-root=*)
      DATA_ROOT="${1#*=}"
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
    --checkpoint-tag)
      [[ $# -ge 2 ]] || { echo "--checkpoint-tag needs a value" >&2; exit 2; }
      CHECKPOINT_TAG="$2"
      shift 2
      ;;
    --checkpoint-tag=*)
      CHECKPOINT_TAG="${1#*=}"
      shift
      ;;
    --output-name)
      [[ $# -ge 2 ]] || { echo "--output-name needs a value" >&2; exit 2; }
      OUTPUT_NAME="$2"
      shift 2
      ;;
    --output-name=*)
      OUTPUT_NAME="${1#*=}"
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
    --smoke)
      SMOKE_REQUESTED=1
      shift
      ;;
    --batch-size|--gradient-accumulation-steps|--effective-batch-size|--num-train-steps|--num-workers|--learning-rate|--warmup-steps|--decay-steps|--decay-lr|--fsdp-devices|--init-checkpoint|--max-train-samples|--max-validation-samples)
      [[ $# -ge 2 ]] || { echo "$1 needs a value" >&2; exit 2; }
      TRAIN_ARGS+=("$1" "$2")
      shift 2
      ;;
    --batch-size=*|--gradient-accumulation-steps=*|--effective-batch-size=*|--num-train-steps=*|--num-workers=*|--learning-rate=*|--warmup-steps=*|--decay-steps=*|--decay-lr=*|--fsdp-devices=*|--init-checkpoint=*|--max-train-samples=*|--max-validation-samples=*)
      TRAIN_ARGS+=("$1")
      shift
      ;;
    --max-samples)
      [[ $# -ge 2 ]] || { echo "--max-samples needs a value" >&2; exit 2; }
      INFER_ARGS+=("$1" "$2")
      shift 2
      ;;
    --max-samples=*)
      INFER_ARGS+=("$1")
      shift
      ;;
    --model-dtype|--paligemma-variant|--action-expert-variant)
      [[ $# -ge 2 ]] || { echo "$1 needs a value" >&2; exit 2; }
      TRAIN_ARGS+=("$1" "$2")
      INFER_ARGS+=("$1" "$2")
      shift 2
      ;;
    --model-dtype=*|--paligemma-variant=*|--action-expert-variant=*)
      TRAIN_ARGS+=("$1")
      INFER_ARGS+=("$1")
      shift
      ;;
    --use-quantile-norm|--no-use-quantile-norm|--use-augmentation|--no-use-augmentation|--discrete-state-input|--no-discrete-state-input)
      TRAIN_ARGS+=("$1")
      shift
      ;;
    --resume)
      TRAIN_ARGS+=("$1")
      shift
      ;;
    *)
      # Unknown options can still be used with the matching single phase.
      # They are rejected for multi-phase commands so a train-only value is
      # never accidentally handed to the inference parser (or vice versa).
      COMMON_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$PROFILE" && "$PROFILE" != "v2_5k" && "$PROFILE" != "exp32_loc_main" && "$PROFILE" != "exp32_loc_main_frozen_vl" ]]; then
  echo "Unknown profile: $PROFILE" >&2
  exit 2
fi
if [[ -n "$CHECKPOINT_TAG" && "$CHECKPOINT_TAG" != "best" && "$CHECKPOINT_TAG" != "late" ]]; then
  echo "--checkpoint-tag must be best or late" >&2
  exit 2
fi
if [[ -n "$CHECKPOINT" && -n "$CHECKPOINT_TAG" ]]; then
  echo "Pass either --checkpoint or --checkpoint-tag, not both" >&2
  exit 2
fi

if [[ -z "$PROFILE" || "$PROFILE" == "v2_5k" ]]; then
  PROFILE_DIR="pi0.5"
else
  PROFILE_DIR="pi0.5_${PROFILE}"
fi
if [[ -n "$RUN_DIR_OVERRIDE" ]]; then
  RUN_ROOT="$RUN_DIR_OVERRIDE"
else
  RUN_ROOT="${PROJECT_ROOT}/outputs/csgo_benchmark_v2_seen10/${PROFILE_DIR}/seed_${SEED}"
fi

require_python() {
  [[ -x "$PYTHON" ]] || { echo "OpenPI Python environment is missing: $PYTHON" >&2; exit 1; }
}

require_evaluator() {
  [[ -f "$EVALUATOR" ]] || { echo "Shared evaluator is missing: $EVALUATOR" >&2; exit 1; }
  [[ -x "$UNILIP_PYTHON" ]] || { echo "UNILIP_PYTHON is missing: $UNILIP_PYTHON" >&2; exit 1; }
}

run_train() {
  local run_dir="$1"
  shift
  require_python
  local args=(--seed "$SEED" --data-root "$DATA_ROOT" --run-dir "$run_dir")
  if [[ -n "$PROFILE" ]]; then args+=(--profile "$PROFILE"); fi
  "$PYTHON" "$PROJECT_ROOT/scripts/train_seen10.py" "${args[@]}" "$@"
}

run_infer() {
  local run_dir="$1"
  local checkpoint_override="$2"
  local output_override="$3"
  shift 3
  require_python
  local args=(--seed "$SEED" --data-root "$DATA_ROOT" --run-dir "$run_dir" --split "$SPLIT")
  if [[ -n "$PROFILE" ]]; then args+=(--profile "$PROFILE"); fi
  if [[ -n "$checkpoint_override" ]]; then
    args+=(--checkpoint "$checkpoint_override")
  elif [[ -n "$CHECKPOINT" ]]; then
    args+=(--checkpoint "$CHECKPOINT")
  elif [[ -n "$CHECKPOINT_TAG" ]]; then
    args+=(--checkpoint-tag "$CHECKPOINT_TAG")
  fi
  if [[ -n "$output_override" ]]; then
    args+=(--output-name "$output_override")
  elif [[ -n "$OUTPUT_NAME" ]]; then
    args+=(--output-name "$OUTPUT_NAME")
  fi
  "$PYTHON" "$PROJECT_ROOT/scripts/infer_seen10.py" "${args[@]}" "$@"
}

run_eval() {
  local run_dir="$1"
  local output_name="$2"
  require_evaluator
  local output="$run_dir/evaluation/$output_name"
  "$UNILIP_PYTHON" "$EVALUATOR" localization \
    --pred-root "$run_dir/$output_name" --data-root "$DATA_ROOT" --output "$output"
}

run_smoke() {
  require_evaluator
  require_python
  # The evaluator smoke protocol uses the test split.
  SPLIT="seen_discrete_test"
  local stamp smoke_root
  stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
  if [[ -n "$RUN_DIR_OVERRIDE" ]]; then
    smoke_root="$RUN_DIR_OVERRIDE"
  else
    smoke_root="${CSGO_SMOKE_ROOT:-${PROJECT_ROOT}/outputs/csgo_benchmark_v2_smoke/${PROFILE_DIR}/seed_${SEED}/${stamp}}"
  fi
  local smoke_args=(--smoke)
  run_train "$smoke_root" "${smoke_args[@]}" "${TRAIN_ARGS[@]}"
  local smoke_infer_args=(--smoke)
  run_infer "$smoke_root" "$smoke_root/checkpoints/best" "${OUTPUT_NAME:-localization}" \
    "${smoke_infer_args[@]}" "${INFER_ARGS[@]}"
  local smoke_output="${OUTPUT_NAME:-localization}"
  "$UNILIP_PYTHON" "$EVALUATOR" smoke localization \
    --pred-root "$smoke_root/$smoke_output" --data-root "$DATA_ROOT" --limit 1 \
    | tee "$smoke_root/smoke_eval.json"
}

case "$ACTION" in
  smoke)
    [[ ${#COMMON_ARGS[@]} -eq 0 ]] || { echo "Unsupported smoke option: ${COMMON_ARGS[*]}" >&2; exit 2; }
    run_smoke
    ;;
  train)
    [[ ${#COMMON_ARGS[@]} -eq 0 ]] || TRAIN_ARGS+=("${COMMON_ARGS[@]}")
    [[ "$RUN_FULL" == "1" || "$SMOKE_REQUESTED" == "1" ]] || {
      echo "RUN_FULL=0; use '$0 smoke' or set RUN_FULL=1" >&2; exit 2;
    }
    train_smoke_args=()
    if [[ "$SMOKE_REQUESTED" == "1" ]]; then train_smoke_args+=(--smoke); fi
    run_train "$RUN_ROOT" "${TRAIN_ARGS[@]}" "${train_smoke_args[@]}"
    ;;
  infer)
    [[ ${#COMMON_ARGS[@]} -eq 0 ]] || INFER_ARGS+=("${COMMON_ARGS[@]}")
    [[ "$RUN_FULL" == "1" || "$SMOKE_REQUESTED" == "1" ]] || {
      echo "RUN_FULL=0; use '$0 smoke' or set RUN_FULL=1" >&2; exit 2;
    }
    infer_smoke_args=()
    if [[ "$SMOKE_REQUESTED" == "1" ]]; then infer_smoke_args+=(--smoke); fi
    run_infer "$RUN_ROOT" "" "" "${infer_smoke_args[@]}" "${INFER_ARGS[@]}"
    ;;
  eval)
    [[ ${#COMMON_ARGS[@]} -eq 0 ]] || { echo "Unsupported eval option: ${COMMON_ARGS[*]}" >&2; exit 2; }
    if [[ -n "$OUTPUT_NAME" ]]; then
      eval_name="$OUTPUT_NAME"
    elif [[ "$CHECKPOINT_TAG" == "late" ]]; then
      eval_name="localization_late"
    else
      eval_name="localization"
    fi
    run_eval "$RUN_ROOT" "$eval_name"
    ;;
  all)
    [[ ${#COMMON_ARGS[@]} -eq 0 ]] || { echo "Unsupported all option: ${COMMON_ARGS[*]}" >&2; exit 2; }
    if [[ "$RUN_FULL" != "1" ]]; then
      run_smoke
    else
      run_train "$RUN_ROOT" "${TRAIN_ARGS[@]}"
      best_output="${OUTPUT_NAME:-localization}"
      if [[ "$best_output" == "localization" ]]; then
        late_output="localization_late"
      else
        late_output="${best_output}_late"
      fi
      run_infer "$RUN_ROOT" "$RUN_ROOT/checkpoints/best" "$best_output" "${INFER_ARGS[@]}"
      run_eval "$RUN_ROOT" "$best_output"
      run_infer "$RUN_ROOT" "$RUN_ROOT/checkpoints/late" "$late_output" "${INFER_ARGS[@]}"
      run_eval "$RUN_ROOT" "$late_output"
    fi
    ;;
  *)
    echo "Unknown action: $ACTION (expected train, infer, eval, smoke, or all)" >&2
    exit 2
    ;;
esac
