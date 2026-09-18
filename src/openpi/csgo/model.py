"""Pi0.5 model adapter for CSGO Seen-10 localization.

The benchmark uses the native Pi0.5 flow matching model with a one-step,
five-dimensional action.  Keeping the adapter as a thin subclass is useful:
the native NNX graph, image encoder, tokenizer, action expert and sampler stay
bit-for-bit compatible with the released checkpoint format.
"""

from __future__ import annotations

import dataclasses
import os
import re

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


@dataclasses.dataclass(frozen=True)
class CSGOPi0Config(_pi0_config.Pi0Config):
    """Configuration for the Seen-10 five-DoF Pi0.5 localization adapter."""

    action_dim: int = 5
    action_horizon: int = 1
    pi05: bool = True
    paligemma_variant: _gemma.Variant = "gemma_2b_lora"
    action_expert_variant: _gemma.Variant = "gemma_300m_lora"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.pi05:
            raise ValueError("CSGO Seen-10 requires the Pi0.5 model (pi05=True).")
        if self.action_dim != 5 or self.action_horizon != 1:
            raise ValueError("CSGO Seen-10 requires action_dim=5 and action_horizon=1.")

    @override
    def create(self, rng: at.KeyArrayLike) -> CSGOPi0:
        return CSGOPi0(self, rngs=nnx.Rngs(rng))

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze native language weights and SigLIP, preserving adapter heads.

        ``Pi0Config.get_freeze_filter`` already freezes the non-LoRA Gemma
        weights.  SigLIP is selected explicitly here because the native filter
        intentionally leaves the vision module trainable for some fine-tunes.
        Action projections and Pi0.5 time MLP are intentionally not selected by
        this filter and therefore remain trainable with the LoRA parameters.
        """

        native_filter = super().get_freeze_filter()
        image_filter = nnx_utils.PathRegex(".*img.*")
        return nnx.Any(native_filter, image_filter)


class CSGOPi0(_pi0.Pi0):
    """Native Pi0.5 with label-safe CSGO loss preprocessing."""

    def __init__(self, config: CSGOPi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        # Camera rotation/crop augmentation changes the camera pose label.  The
        # native Pi0 implementation applies those transforms when train=True,
        # so always use the native flow loss in eval preprocessing mode here.
        del train
        return super().compute_loss(rng, observation, actions, train=False)


@dataclasses.dataclass(frozen=True)
class CSGOPi05WeightLoader:
    """Load ``pi05_base`` while adapting its native 32DoF action projections.

    All non-action weights are loaded by exact key.  Missing LoRA leaves are
    retained from the freshly initialized target graph, as in the native
    ``CheckpointWeightLoader``.  The action projection slices are the only
    shape adaptation performed here.
    """

    params_path: str = "gs://openpi-assets/checkpoints/pi05_base/params"

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
                if array.ndim != 2 or array.shape[0] < 5 or array.shape[1] != reference.shape[1]:
                    raise ValueError(f"Unexpected {key} source shape {array.shape}; target shape is {reference.shape}")
                array = array[:5, :]
            elif key.endswith("action_out_proj/kernel"):
                if array.ndim != 2 or array.shape[0] != reference.shape[0] or array.shape[1] < 5:
                    raise ValueError(f"Unexpected {key} source shape {array.shape}; target shape is {reference.shape}")
                array = array[:, :5]
            elif key.endswith("action_out_proj/bias"):
                if array.ndim != 1 or array.shape[0] < 5:
                    raise ValueError(f"Unexpected {key} source shape {array.shape}; target shape is {reference.shape}")
                array = array[:5]

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


__all__ = ["CSGOPi0", "CSGOPi0Config", "CSGOPi05WeightLoader"]
