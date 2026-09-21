import math
import re

import flax.linen as nn
import flax.struct as struct
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at


def peft_lora_a_init(key, shape, dtype=jnp.float32):
    """Initializes LoRA A like PEFT's default linear-layer initialization.

    LoRA weights may have leading expert/head dimensions. Treat those as batch
    axes so every matrix gets the same Kaiming-uniform fan-in calculation as an
    ordinary two-dimensional linear weight.
    """
    if len(shape) < 2:
        raise ValueError(f"LoRA A weights must be matrices, got shape {shape}.")
    batch_axis = tuple(range(len(shape) - 2))
    initializer = nn.initializers.variance_scaling(
        scale=1.0 / 3.0,
        mode="fan_in",
        distribution="uniform",
        batch_axis=batch_axis,
    )
    return initializer(key, shape, dtype)


def _apply_lora_dropout(x: at.Array, rate: float, deterministic, rng) -> at.Array:
    keep = jax.random.bernoulli(rng, p=1.0 - rate, shape=x.shape)
    dropped = jnp.where(keep, x / (1.0 - rate), jnp.zeros_like(x))
    return jnp.where(deterministic, x, dropped)


@struct.dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    # LoRA rank.
    rank: int
    # LoRA scaling factor.
    alpha: float = 1.0
    # Initialization function for LoRA parameters.
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732
    rslora: bool = False
    # Optional PEFT-style separate initializers. When unset, legacy LoRA uses
    # init_fn for both matrices as before.
    a_init_fn: nn.initializers.Initializer | None = None
    b_init_fn: nn.initializers.Initializer | None = None
    # Applied to adapter inputs only. Base model activations are never dropped.
    dropout: float = 0.0
    # Axes in the weight to apply LoRA to. Should typically be the last two axes.
    axes: tuple[int, int] = (-2, -1)
    # Axis label which is used by LoRA in einsum equations. Must not be present in the original equation.
    label: str = "L"

    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank

    @property
    def a_initializer(self) -> nn.initializers.Initializer:
        return self.init_fn if self.a_init_fn is None else self.a_init_fn

    @property
    def b_initializer(self) -> nn.initializers.Initializer:
        return self.init_fn if self.b_init_fn is None else self.b_init_fn


class Einsum(nn.Module):
    """Einsum with LoRA support. Can be used as a drop-in replacement for the Gemma Einsum."""

    # Shape of the weight.
    shape: tuple[int, ...]
    # Initialization function for the weight.
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w = self.param("w", self.init_fn, self.shape)

        if config := self.lora_config:
            # Setup LoRA parameters.
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            self.w_a = self.param("lora_a", config.a_initializer, shape_a)
            self.w_b = self.param("lora_b", config.b_initializer, shape_b)

    @nn.compact
    def __call__(self, eqn: str, x, *, deterministic: bool = True):
        dtype = x.dtype  # original dtype, could be half-precision
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if config := self.lora_config:
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora_x = x
            if config.dropout:
                lora_x = _apply_lora_dropout(x, config.dropout, deterministic, self.make_rng("dropout"))
            lora = jnp.einsum(eqn_a, lora_x, self.w_a.astype(dtype))
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            result = result + lora * config.scaling_value

        return result

    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()

        assert self.lora_config is not None
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)
        label = self.lora_config.label

        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b


class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.lora_config:
            # Setup LoRA parameters.
            # TODO: follow up with a simplified init_fn api.
            self.w_gating_lora = (
                self.param(
                    "gating_einsum_lora_a",
                    self.lora_config.a_initializer,
                    (2, self.features, self.lora_config.rank),
                ),
                self.param(
                    "gating_einsum_lora_b",
                    self.lora_config.b_initializer,
                    (2, self.lora_config.rank, self.hidden_dim),
                ),
            )
            self.w_linear_lora = (
                self.param("linear_lora_a", self.lora_config.a_initializer, (self.hidden_dim, self.lora_config.rank)),
                self.param("linear_lora_b", self.lora_config.b_initializer, (self.lora_config.rank, self.features)),
            )

    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        dtype = x.dtype  # original dtype, could be half-precision
        gating_lora_x = x
        linear_lora_x = None
        if self.lora_config is not None and self.lora_config.dropout:
            gating_lora_x = _apply_lora_dropout(x, self.lora_config.dropout, deterministic, self.make_rng("dropout"))

        ff_gate = self._dot(
            x,
            self.w_gating[0],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
            lora_input=gating_lora_x,
        )
        gate_value = nn.gelu(ff_gate)

        ff1 = self._dot(
            x,
            self.w_gating[1],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
            lora_input=gating_lora_x,
        )
        activations = gate_value * ff1

        linear_lora_x = activations
        if self.lora_config is not None and self.lora_config.dropout:
            linear_lora_x = _apply_lora_dropout(
                activations, self.lora_config.dropout, deterministic, self.make_rng("dropout")
            )
        outputs = self._dot(activations, self.w_linear, self.w_linear_lora, lora_input=linear_lora_x)
        assert outputs.dtype == dtype
        return outputs

    def _dot(
        self,
        x: at.Array,
        w: at.Array,
        lora_weights: tuple[at.Array, at.Array] | None,
        *,
        lora_input: at.Array | None = None,
    ) -> at.Array:
        base = jnp.dot(x, w.astype(x.dtype))
        if lora_weights is None:
            return base
        if self.lora_config is None:
            raise AssertionError("LoRA weights require a LoRA configuration.")
        if lora_input is None:
            lora_input = x
        delta = jnp.dot(jnp.dot(lora_input, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))
        return base + delta * self.lora_config.scaling_value
