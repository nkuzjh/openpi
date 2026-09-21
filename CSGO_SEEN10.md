# pi0.5 CSGO Benchmark v2 Seen-10 localization

This adapter uses the native JAX/NNX Pi0.5 flow-matching path. The manifest
adapter in `src/openpi/csgo/data.py` reads `seen_train`, `seen_validation`, and
`seen_discrete_test` in published row order, maps FPV and radar to
`base_0_rgb` and `left_wrist_0_rgb`, masks `right_wrist_0_rgb`, and keeps
ground-truth pose out of model inputs. The model predicts a horizon-one
absolute normalized action `[x, y, z, pitch, yaw]`.

The legacy `v2_5k` adaptation uses native JAX LoRA together with trainable
action input/output projections and the Pi0.5 time MLP. SigLIP is frozen. Its
input preprocessing is resize/tokenization/padding only; the profile has no
coordinate crop augmentation.

## Environment

Run commands from `/home/jiahao/task/openpi`. The prepared project environment
is `.venv` with managed CPython 3.11.16. A locally installed `.tools/uv`
executable reports uv 0.12.15. If that executable is absent, install the
versioned uv release into this checkout without changing `PATH`:

```bash
cd /home/jiahao/task/openpi
curl -LsSf https://astral.sh/uv/0.12.15/install.sh | \
  UV_INSTALL_DIR="$PWD/.tools" UV_NO_MODIFY_PATH=1 sh
```

To recreate or verify the locked environment:

```bash
cd /home/jiahao/task/openpi
GIT_LFS_SKIP_SMUDGE=1 \
UV_CACHE_DIR="$PWD/.cache/uv" \
UV_PYTHON_INSTALL_DIR="$PWD/.cache/uv/python" \
./.tools/uv sync --locked --python 3.11 --managed-python
```

The locked environment already provides JAX 0.5.3, Flax 0.10.2, and Torch
2.7.1+cu126; this integration does not change project dependencies.

The OpenPI and UniLIP evaluator environments are separate. These paths match
the current checkout and evaluator installation:

```bash
export OPENPI_PYTHON=/home/jiahao/task/openpi/.venv/bin/python
export CSGO_BENCHMARK_V2_DATA=/home/jiahao/task/UniLIP/data/csgo_benchmark_v2
export UNILIP_PYTHON=/home/jiahao/miniconda3/envs/UniLIP/bin/python
export SHARED_EVAL_DIR=/home/jiahao/task/csgo_benchmark_v2_eval_general
export OPENPI_DATA_HOME=/home/jiahao/task/openpi/.cache/openpi
export JAX_COMPILATION_CACHE_DIR=/home/jiahao/task/openpi/.cache/jax
```

The prepared local Pi0.5 weights are at
`/home/jiahao/task/openpi/.cache/openpi/openpi-assets/checkpoints/pi05_base/params`.
Use the local path to avoid a download:

```bash
export CSGO_PI05_BASE=/home/jiahao/task/openpi/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
```

If `CSGO_PI05_BASE` is unset, the train and infer entry points fall back to
`gs://openpi-assets/checkpoints/pi05_base/params`, cached below
`OPENPI_DATA_HOME`. The native tokenizer is likewise expected at
`/home/jiahao/task/openpi/.cache/openpi/big_vision/paligemma_tokenizer.model`.

## Commands

`RUN_FULL=0` is the default. The bounded smoke command uses a fresh UTC
timestamped directory, one train sample, five one-step updates, a checkpoint
restore, one prediction row, and the shared evaluator smoke read. It can use
the configured JAX accelerator, but is not a formal benchmark run:

```bash
RUN_FULL=0 scripts/run_csgo_seen10.sh smoke --seed 0
```

Formal training, test inference, and shared evaluation are opt-in. Formal
inference uses all 20,000 `seen_discrete_test` rows and does not accept a
partial sample limit:

```bash
RUN_FULL=1 scripts/run_csgo_seen10.sh train --seed 0
RUN_FULL=1 scripts/run_csgo_seen10.sh infer --seed 0
RUN_FULL=1 scripts/run_csgo_seen10.sh eval --seed 0
```

The wrapper accepts seeds `0`, `1`, and `2`; repeat the three commands with
`--seed 1` or `--seed 2` as needed. `all` runs the corresponding sequence:

```bash
RUN_FULL=1 scripts/run_csgo_seen10.sh all --seed 0
```

To continue an interrupted formal training run, keep the same run directory
and configuration and pass `--resume`:

```bash
RUN_FULL=1 scripts/run_csgo_seen10.sh train --seed 0 --resume
```

Inference is resumable by rerunning the same command: existing validated rows
are retained and missing rows are appended. A validation pass can be selected
explicitly with `--split seen_validation`.

## Profiles

Omitting `--profile` selects the original `v2_5k` run and keeps its existing
`outputs/csgo_benchmark_v2_seen10/pi0.5/seed_<seed>/` path. The approved
`exp32_loc_main` profile uses 32D internal actions, train-only q01/q99 pose
normalization, train-only FPV dropout, and a 128-sample effective batch. It
defaults to a one-sample microbatch accumulated 128 times, 19,500 optimizer
updates, and a `5e-5` peak and ending learning rate:

```bash
RUN_FULL=1 scripts/run_csgo_seen10.sh train --profile exp32_loc_main --seed 0
RUN_FULL=1 scripts/run_csgo_seen10.sh infer --profile exp32_loc_main --seed 0 --checkpoint-tag best
RUN_FULL=1 scripts/run_csgo_seen10.sh eval --profile exp32_loc_main --seed 0 --checkpoint-tag best
RUN_FULL=1 scripts/run_csgo_seen10.sh all --profile exp32_loc_main --seed 0
RUN_FULL=0 scripts/run_csgo_seen10.sh smoke --profile exp32_loc_main --seed 0
```

The new formal run is isolated under
`outputs/csgo_benchmark_v2_seen10/pi0.5_exp32_loc_main/seed_<seed>/`. Smoke
defaults to a one-sample microbatch on one device (one sample per device on a
multi-device host) and one update per effective batch; pass `--batch-size` and
`--gradient-accumulation-steps` for another bounded setup.
Training saves at completed steps 3,900, 7,800, 11,700, 15,600, and 19,500.
Inference from `best` writes `localization/predictions.jsonl`; inference from
`late` writes `localization_late/predictions.jsonl`, with separate manifests
and evaluator reports. `all` runs inference and evaluation for both aliases.
The default `best` output remains `localization` for existing commands.

## Outputs

Formal `v2_5k` runs use
`outputs/csgo_benchmark_v2_seen10/pi0.5/seed_<seed>`; `exp32_loc_main` uses
the separate profile path above. Checkpoints are saved at five equal
completed-step intervals under `checkpoints/<step>/`, with
`checkpoints/late` and `checkpoints/best` symlinks. Training writes
`run_config.json`, `loss.jsonl`, `train_metrics.jsonl`, `loss.png`, and fixed
validation visualizations.

Predictions are written to `localization/predictions.jsonl`; each row contains
only `sample_id`, `map_name`, `pred_x`, `pred_y`, `pred_z`, `pred_pitch`, and
`pred_yaw` in benchmark normalized pose space. The evaluator writes its formal
report under `evaluation/localization`. Smoke output is isolated under
`outputs/csgo_benchmark_v2_smoke/pi0.5/seed_<seed>/<UTC timestamp>_<pid>/` and writes
`smoke_eval.json` instead of a formal report. Training and inference images are
under `visualizations/<split>/step_<completed_step>/` and
`visualizations/<split>/inference/`; each map uses the same seeded ten-sample
selection.

## Recorded smoke evidence

On 2026-09-17, independent `.venv` data contract tests and GPU matrix
multiplication and convolution
checks passed. The complete `RUN_FULL=0` wrapper smoke run succeeded under
`outputs/csgo_benchmark_v2_smoke/pi0.5/seed_0/runtime_finish_20260917_091204/`:
steps 1 through 5 completed, validation ran five times, and five checkpoints
were saved. `checkpoints/late` and `checkpoints/best` both resolve to step 5,
which had the minimum fixed validation flow-matching loss. The run produced
`loss.png`, five fixed validation visualizations,
`localization/predictions.jsonl`, and the wrapper-generated `smoke_eval.json`.

The main run's `smoke_eval.json` flags are `smoke_only=true`, `formal=false`,
and `official_output_written=false`; `RUN_FULL=0` therefore produces no formal
Table 1 result. A separate read-only native optimizer restore passed with
evidence at
`outputs/csgo_benchmark_v2_smoke/pi0.5/seed_0/native_resume_acceptance/result.json`;
the Adam and schedule counters were both at step 1. Independent repeated
inference evidence remains at
`outputs/csgo_benchmark_v2_smoke/pi0.5/seed_0/infer_acceptance_checkpoint1/`:
it generated 0 new rows, retained 1 existing row, and left the file unchanged.

No code or dependency blocker remains. Shared-disk checkpoint writes made this
five-step smoke take about 25 minutes, so long save periods can be expected.
The complete wrapper session 49244 exited 0; its `train --smoke --resume`
session 16149 exited 0, restored checkpoint 5, created no step 6 or additional
save, and left `late=best=5`.
