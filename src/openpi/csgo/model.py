"""Pi0.5 model adapter for CSGO Seen-10 localization profiles.

The legacy profile uses a one-step, five-dimensional action. The experiment
profile keeps all 32 native Pi0.5 action dimensions while adapting the training
and freeze configuration around the native model graph.
"""

from __future__ import annotations

import dataclasses
import os
import re
from typing import Literal

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import numpy as np
from typing_extensions import override

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config as _pi0_config
from openpi.shared import array_typing as at
from openpi.shared import download
from openpi.shared import nnx_utils

CSGOProfile = Literal["v2_5k", "exp32_loc_main", "exp32_loc_main_frozen_vl"]
EXP32_PROFILES = ("exp32_loc_main", "exp32_loc_main_frozen_vl")


@dataclasses.dataclass(frozen=True)
class CSGOPi0Config(_pi0_config.Pi0Config):
    """Configuration for a CSGO localization Pi0.5 profile.

    ``v2_5k`` is the legacy five-dimensional profile. Both 32D profiles use
    the full Pi0.5 action projection and PEFT-style rank-32 adapters.
    ``profile`` is an InitVar so it does not change the serialized
    legacy model-config keys used by existing run metadata.
    """

    profile: dataclasses.InitVar[CSGOProfile] = "v2_5k"
    action_dim: int | None = None
    action_horizon: int = 1
    pi05: bool = True
    paligemma_variant: _gemma.Variant | None = None
    action_expert_variant: _gemma.Variant | None = None

    def __post_init__(self, profile: CSGOProfile) -> None:
        if profile not in ("v2_5k", *EXP32_PROFILES):
            raise ValueError(f"Unknown CSGO profile {profile!r}; choose 'v2_5k' or one of {EXP32_PROFILES!r}.")
        action_dim = 5 if profile == "v2_5k" else 32
        if self.action_dim is not None and self.action_dim != action_dim:
            raise ValueError(f"CSGO profile {profile!r} requires action_dim={action_dim}.")
        object.__setattr__(self, "action_dim", action_dim)

        if not self.pi05:
            raise ValueError("CSGO Seen-10 requires the Pi0.5 model (pi05=True).")
        if self.action_horizon != 1:
            raise ValueError("CSGO localization requires action_horizon=1.")

        if profile in EXP32_PROFILES:
            if self.discrete_state_input is True:
                raise ValueError(f"The {profile} profile requires discrete_state_input=False.")
            required_variants = ("gemma_2b_lora_r32", "gemma_300m_lora_r32")
            if self.paligemma_variant not in (None, required_variants[0]):
                raise ValueError(f"{profile} requires paligemma_variant={required_variants[0]!r}.")
            if self.action_expert_variant not in (None, required_variants[1]):
                raise ValueError(f"{profile} requires action_expert_variant={required_variants[1]!r}.")
            object.__setattr__(self, "paligemma_variant", required_variants[0])
            object.__setattr__(self, "action_expert_variant", required_variants[1])
        else:
            object.__setattr__(self, "paligemma_variant", self.paligemma_variant or "gemma_2b_lora")
            object.__setattr__(self, "action_expert_variant", self.action_expert_variant or "gemma_300m_lora")

        super().__post_init__()
        if profile in EXP32_PROFILES:
            object.__setattr__(self, "discrete_state_input", False)
        object.__setattr__(self, "profile", profile)

    @override
    def create(self, rng: at.KeyArrayLike) -> CSGOPi0:
        return CSGOPi0(self, rngs=nnx.Rngs(rng))

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze base language weights and the SigLIP encoder.

        ``Pi0Config.get_freeze_filter`` already freezes the non-LoRA Gemma
        weights. The legacy and frozen-VL profiles freeze the full image
        module. In ``exp32_loc_main``, the SigLIP encoder is frozen while
        ``PaliGemma/img/head`` remains trainable. Action projections and the
        Pi0.5 time MLP remain trainable in both 32D profiles.
        """

        native_filter = super().get_freeze_filter()
        if self.profile in ("v2_5k", "exp32_loc_main_frozen_vl"):
            image_filter = nnx_utils.PathRegex(".*img.*")
        else:
            image_filter = nnx_utils.PathRegex(r".*PaliGemma/img/(?!head(?:/|$)).*")
        return nnx.Any(native_filter, image_filter)


class CSGOPi0(_pi0.Pi0):
    """Native Pi0.5 with label-safe CSGO loss preprocessing."""

    def __init__(self, config: CSGOPi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.profile = config.profile

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        # Preserve the historical label-safe inference preprocessing in the
        # v2_5k profile. The frozen-VL experiment keeps only native color jitter.
        native_train = train if self.profile in EXP32_PROFILES else False
        return super().compute_loss(
            rng,
            observation,
            actions,
            train=native_train,
            geometric_augmentation=self.profile != "exp32_loc_main_frozen_vl",
        )


@dataclasses.dataclass(frozen=True)
class CSGOPi05WeightLoader:
    """Load ``pi05_base`` while adapting its native 32DoF action projections.

    All non-action weights are loaded by exact key. Missing LoRA leaves are
    retained from the freshly initialized target graph, as in the native
    ``CheckpointWeightLoader``. A five-dimensional target receives the legacy
    leading-five slice; a 32-dimensional target receives every action weight.
    """

    params_path: str = "gs://openpi-assets/checkpoints/pi05_base/params"
    action_dim: int | None = None
    profile: dataclasses.InitVar[CSGOProfile | None] = None

    def __post_init__(self, profile: CSGOProfile | None) -> None:
        if profile is not None and profile not in ("v2_5k", *EXP32_PROFILES):
            raise ValueError(f"Unknown CSGO profile {profile!r}.")
        expected_action_dim = None if profile is None else (5 if profile == "v2_5k" else 32)
        if self.action_dim is not None and self.action_dim not in (5, 32):
            raise ValueError(f"CSGO weight loading supports action_dim=5 or action_dim=32, got {self.action_dim}.")
        if expected_action_dim is not None:
            if self.action_dim is not None and self.action_dim != expected_action_dim:
                raise ValueError(f"Profile {profile!r} requires action_dim={expected_action_dim}.")
            object.__setattr__(self, "action_dim", expected_action_dim)
        object.__setattr__(self, "profile", profile)

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat_ref = traverse_util.flatten_dict(params, sep="/")
        flat_loaded = traverse_util.flatten_dict(loaded_params, sep="/")
        result: dict[str, object] = {}

        for key, value in flat_loaded.items():
            if key not in flat_ref:
                continue
            reference = flat_ref[key]
            array = np.asarray(value)
            if key.endswith("action_in_proj/kernel"):
                target_action_dim = reference.shape[0]
                if self.action_dim is not None and target_action_dim != self.action_dim:
                    raise ValueError(
                        f"Configured action_dim={self.action_dim} does not match target {key} shape {reference.shape}."
                    )
                if target_action_dim not in (5, 32) or array.ndim != 2 or array.shape[0] != 32:
                    raise ValueError(f"Unexpected {key} source shape {array.shape}; target shape is {reference.shape}")
                if array.shape[1] != reference.shape[1]:
                    raise ValueError(f"Unexpected {key} source shape {array.shape}; target shape is {reference.shape}")
                array = array[:target_action_dim, :]
            elif key.endswith("action_out_proj/kernel"):
                target_action_dim = reference.shape[1]
                if self.action_dim is not None and target_action_dim != self.action_dim:
                    raise ValueError(
                        f"Configured action_dim={self.action_dim} does not match target {key} shape {reference.shape}."
                    )
                if (
                    target_action_dim not in (5, 32)
                    or array.ndim != 2
                    or array.shape[0] != reference.shape[0]
                    or array.shape[1] != 32
                ):
                    raise ValueError(f"Unexpected {key} source shape {array.shape}; target shape is {reference.shape}")
                array = array[:, :target_action_dim]
            elif key.endswith("action_out_proj/bias"):
                target_action_dim = reference.shape[0]
                if self.action_dim is not None and target_action_dim != self.action_dim:
                    raise ValueError(
                        f"Configured action_dim={self.action_dim} does not match target {key} shape {reference.shape}."
                    )
                if target_action_dim not in (5, 32) or array.ndim != 1 or array.shape[0] != 32:
                    raise ValueError(f"Unexpected {key} source shape {array.shape}; target shape is {reference.shape}")
                array = array[:target_action_dim]

            if array.shape != reference.shape:
                raise ValueError(f"Shape mismatch at {key}: source {array.shape}, target {reference.shape}.")
            result[key] = array.astype(reference.dtype, copy=False)

        # Released Pi05 weights do not contain LoRA leaves.  Keep initialized
        # target LoRA leaves so the native initializer can train them.
        lora_pattern = re.compile(r".*lora.*")
        for key, reference in flat_ref.items():
            if key not in result and lora_pattern.fullmatch(key):
                result[key] = reference

        if not result:
            raise ValueError(f"No compatible parameters found in {self.params_path!r}.")
        return traverse_util.unflatten_dict(result, sep="/")


__all__ = ["CSGOPi0", "CSGOPi0Config", "CSGOPi05WeightLoader", "CSGOProfile"]
