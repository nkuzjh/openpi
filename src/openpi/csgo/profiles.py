"""Named training profiles for the native CSGO localization runtime.

Profiles keep benchmark contracts in one place.  ``v2_5k`` mirrors the
original Seen-10 runtime defaults; the ``exp32_loc_main`` profiles record the long-run
32D-action experiment settings.  Runtime and CLI overrides are validated in
``runtime.py`` so smoke runs can shorten a profile without changing its
identity.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses


@dataclasses.dataclass(frozen=True)
class ExperimentProfile:
    name: str
    num_train_steps: int
    default_microbatch_size: int
    effective_batch_size: int | None
    learning_rate: float
    warmup_steps: int
    decay_steps: int
    decay_lr: float
    internal_action_dim: int
    state_dim: int
    paligemma_variant: str
    action_expert_variant: str
    normalization_mode: str
    use_quantile_norm: bool
    use_augmentation: bool
    discrete_state_input: bool
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    adam_eps: float = 1.0e-8
    weight_decay: float = 1.0e-10
    clip_gradient_norm: float = 1.0
    ema_decay: float | None = None

    @property
    def external_action_dim(self) -> int:
        return 5


V2_5K = ExperimentProfile(
    name="v2_5k",
    num_train_steps=5_000,
    default_microbatch_size=1,
    # v2_5k keeps the historical direct-batch behavior.  Its effective batch
    # size is resolved from the requested batch size by make_runtime_config.
    effective_batch_size=None,
    learning_rate=1.0e-5,
    warmup_steps=100,
    decay_steps=5_000,
    decay_lr=1.0e-6,
    internal_action_dim=5,
    state_dim=5,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
    normalization_mode="benchmark_manifest",
    use_quantile_norm=False,
    use_augmentation=False,
    discrete_state_input=True,
)

EXP32_LOC_MAIN = ExperimentProfile(
    name="exp32_loc_main",
    num_train_steps=19_500,
    # A small physical microbatch is robust for the 32D model.  The global
    # batch of 128 is reached through gradient accumulation.
    default_microbatch_size=1,
    effective_batch_size=128,
    learning_rate=5.0e-5,
    warmup_steps=1_000,
    decay_steps=100_000,
    decay_lr=5.0e-5,
    internal_action_dim=32,
    state_dim=32,
    paligemma_variant="gemma_2b_lora_r32",
    action_expert_variant="gemma_300m_lora_r32",
    normalization_mode="train_quantile",
    use_quantile_norm=True,
    use_augmentation=True,
    discrete_state_input=False,
)

EXP32_LOC_MAIN_FROZEN_VL = dataclasses.replace(
    EXP32_LOC_MAIN, name="exp32_loc_main_frozen_vl", use_augmentation=False
)

PROFILES: Mapping[str, ExperimentProfile] = {
    V2_5K.name: V2_5K,
    EXP32_LOC_MAIN.name: EXP32_LOC_MAIN,
    EXP32_LOC_MAIN_FROZEN_VL.name: EXP32_LOC_MAIN_FROZEN_VL,
}


def get_profile(name: str | None) -> ExperimentProfile:
    """Return a profile, treating an absent name as the legacy default."""

    if name is None:
        name = V2_5K.name
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(PROFILES)
        raise ValueError(f"Unknown CSGO experiment profile {name!r}; choose from {choices}") from exc


def profile_names() -> tuple[str, ...]:
    return tuple(PROFILES)


__all__ = [
    "EXP32_LOC_MAIN",
    "EXP32_LOC_MAIN_FROZEN_VL",
    "PROFILES",
    "V2_5K",
    "ExperimentProfile",
    "get_profile",
    "profile_names",
]
