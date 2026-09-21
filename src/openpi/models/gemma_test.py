import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import gemma
from openpi.models import lora


def test_exp32_gemma_variants_use_peft_lora_config():
    for variant in ("gemma_2b_lora_r32", "gemma_300m_lora_r32"):
        config = gemma.get_config(variant)
        assert set(config.lora_configs) == {"attn", "ffn"}
        for lora_config in config.lora_configs.values():
            assert lora_config.rank == 32
            assert lora_config.alpha == 64.0
            assert lora_config.scaling_value == 2.0
            assert lora_config.dropout == 0.05
            assert lora_config.rslora is False
            assert lora_config.a_init_fn is lora.peft_lora_a_init
            assert lora_config.b_init_fn is nn.initializers.zeros


def test_gemma_lora_dropout_uses_explicit_bridge_rng_and_eval_is_deterministic():
    lora_config = lora.LoRAConfig(
        rank=2,
        alpha=2.0,
        init_fn=nn.initializers.ones,
        dropout=0.5,
    )
    config = gemma.Config(
        width=8,
        depth=1,
        mlp_dim=16,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        lora_configs={"attn": lora_config, "ffn": lora_config},
    )
    module = nnx_bridge.ToNNX(gemma.Module(configs=[config], embed_dtype="float32"))
    init_key = jax.random.key(0)
    module.lazy_init(
        rngs=nnx.Rngs(params=init_key, dropout=jax.random.fold_in(init_key, 1)),
        method="init",
        use_adarms=[False],
    )

    embedded = [jnp.ones((2, 3, config.width), dtype=jnp.float32)]
    positions = jnp.tile(jnp.arange(3, dtype=jnp.int32), (2, 1))
    mask = jnp.ones((2, 3, 3), dtype=jnp.bool_)

    eval_rngs = nnx.Rngs(dropout=jax.random.key(0))
    eval_a, _ = module(embedded, positions, mask, deterministic=True, rngs=eval_rngs)
    eval_b, _ = module(embedded, positions, mask, deterministic=True, rngs=eval_rngs)
    assert jnp.allclose(eval_a[0], eval_b[0])

    train_a, _ = module(
        embedded,
        positions,
        mask,
        deterministic=False,
        rngs=nnx.Rngs(dropout=jax.random.key(1)),
    )
    train_b, _ = module(
        embedded,
        positions,
        mask,
        deterministic=False,
        rngs=nnx.Rngs(dropout=jax.random.key(2)),
    )
    assert not jnp.allclose(train_a[0], train_b[0])
