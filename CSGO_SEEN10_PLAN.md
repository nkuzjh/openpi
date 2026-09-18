# pi0.5 CSGO Benchmark v2 Seen-10 localization plan

## Scope and acceptance boundary

Implement the smallest native JAX/NNX adapter for the Table 1 localization
task. The adapter consumes only the manifest-selected Benchmark v2
`seen_train`, `seen_validation`, and `seen_discrete_test` rows, and exposes the
repository's native `TrainState`, `train_step`, optimizer, sharding, and Orbax
checkpoint formats. The default command path is a bounded accelerator smoke
run (`RUN_FULL=0`) that exercises the native Pi0.5 forward/backward and
checkpoint path; this documentation change starts no formal/full run.

The model task contract is fixed to a one-step absolute normalized 5DoF action
`[x, y, z, pitch, yaw]`, a zero state vector, a fixed task/map prompt, and two
image views (`base_0_rgb` and `left_wrist_0_rgb`) with the third native view
(`right_wrist_0_rgb`) masked. Labels and predictions use the released
manifest/calibration normalization, including the frozen per-map Z extrema.

## Files and responsibilities

- `src/openpi/csgo/model.py`: `CSGOPi0Config` and `CSGOPi0`, a thin Pi0/Pi05
  specialization with action dimension 5, horizon 1, zero-state inputs, and
  `compute_loss(..., train=False)` so coordinate crop augmentation cannot move
  the localization labels. Native image/language/action-expert execution
  remains in `openpi.models.pi0.Pi0`.
- `src/openpi/csgo/runtime.py`: config serialization, native train-state
  initialization/training loop, fixed five-way eval/save schedule, JSONL loss
  logging/plotting, checkpoint aliases, native resume, deterministic eval,
  prediction writing, and checkpoint identity/output-coverage guards. The
  runtime is intentionally a light wrapper around `scripts.train`
  primitives rather than a second trainer.
- `scripts/train_seen10.py`: CLI for native smoke/full/resume training by
  seed, with positive `num_train_steps` divisible by five and interval equal to
  `num_train_steps // 5`.
- `scripts/infer_seen10.py`: CLI for validation/test localization inference,
  fixed per-map visual sample selection, and resumable standards-compliant
  prediction JSONL (`sample_id`, `map_name`, and `pred_*` fields only).
- `scripts/run_csgo_seen10.sh`: `train`, `infer`, `eval`, `smoke`, and `all`
  actions plus `--seed`; `RUN_FULL=0` never launches a full run.
- `src/openpi/csgo/data.py`: manifest-driven split loading, released pose
  normalization/calibration, and the model-input boundary that excludes
  identity and ground truth.
- `src/openpi/csgo/visualization.py`: deterministic per-map sample selection
  and radar/FPV prediction rendering for validation and inference.
- `src/openpi/csgo/data_test.py`: data contract, normalization, input isolation,
  and visualization contract tests against the published release.
- `CSGO_SEEN10.md`: environment, commands, checkpoint locations, output
  contract, and smoke/full evidence boundary.

The data/visualization boundary is implemented by the three `src/openpi/csgo/`
modules above. It is manifest-driven, preserves published row order, and does
not scan image directories or re-partition rows. Plain mappings are converted
only to native `Observation`/`Actions` at the runtime boundary.

## Model and checkpoint design

`CSGOPi0Config` subclasses `Pi0Config` with `pi05=True` by default,
`action_dim=5`, `action_horizon=1`, and the two requested LoRA variants
(`gemma_2b_lora` and `gemma_300m_lora`). `CSGOPi0.compute_loss` delegates to
the native Pi05 flow-matching implementation with `train=False`; sampling uses
the unmodified native integrator. SigLIP is frozen by the CSGO filter in
addition to the native freeze filter. The adaptation trainable set retains
the native action input/output projections and Pi05 time MLP alongside LoRA;
EMA is disabled. Input preprocessing uses native resize, tokenization, and
padding only, with no coordinate crop augmentation.

When initializing from `pi05_base`, the loader validates the source model
structure and slices only the action projections: `action_in_proj.kernel`
uses `[:5, :]`, `action_out_proj.kernel` uses `[:, :5]`, and the output bias
uses `[:5]`. All other parameters retain native shapes/dtypes. The adaptation
filter leaves the requested LoRA parameters plus native action input/output
projections and Pi05 time MLP trainable, while preserving native optimizer
state on resume.

## Training/evaluation protocol

The wrapper builds an independent timestamped run directory for smoke runs;
smoke artifacts cannot be mistaken for formal results. Full runs use
`outputs/csgo_benchmark_v2_seen10/pi0.5/seed_<seed>/`, save exactly at the
five equal intervals, preserve five checkpoints (`keep_period=interval`), and
maintain `checkpoints/late` and `checkpoints/best` links. Validation uses one
fixed RNG/sample set across checkpoints; the minimum validation flow-matching
loss selects `best`. Every step writes a JSONL record containing the primary
loss, and training completion writes a loss plot and a reproducible config.

Inference is resumable and output-protecting: existing rows are retained,
missing sample IDs are appended only after validating the requested checkpoint
identity, and a complete formal `seen_discrete_test` invocation requires all
20,000 rows. Smoke uses an isolated timestamped directory and an explicitly
bounded row count. Standard prediction rows contain only `sample_id`,
`map_name`, `pred_x`, `pred_y`, `pred_z`, `pred_pitch`, and `pred_yaw`.
Inference also renders the fixed ten-sample-per-map visualization set from the
completed prediction file.

`eval` invokes the already-synchronized shared evaluator through
`UNILIP_PYTHON` and never reimplements metrics. `all` runs smoke or full stages
according to `RUN_FULL`; no command implicitly starts full training.

The Python entry points default `OPENPI_DATA_HOME` to `.cache/openpi` and
`JAX_COMPILATION_CACHE_DIR` to `.cache/jax` within this checkout, while still
allowing explicit environment overrides.

## Validation evidence

Static/import checks and the data/visualization contract tests can run without
training. The designated smoke path must exercise one manifest batch, one real
Pi05 forward/backward, one native checkpoint save/restore, one standard
prediction JSONL, and one shared-evaluator smoke read. No smoke result is
claimed by this plan. A complete benchmark result is claimed only when
`RUN_FULL=1` has produced the full train/validation/test artifacts and the
shared evaluator has consumed complete coverage.
