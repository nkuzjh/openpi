import dataclasses

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import numpy as np
import pytest

from openpi.csgo import model as csgo_model
from openpi.models import model as _model
from openpi.models import pi0 as _pi0


def test_profile_defaults_preserve_legacy_serialization():
    config = csgo_model.CSGOPi0Config()
    values = dataclasses.asdict(config)

    assert config.profile == "v2_5k"
    assert config.action_dim == 5
    assert config.action_horizon == 1
    assert config.discrete_state_input is True
    assert config.paligemma_variant == "gemma_2b_lora"
    assert config.action_expert_variant == "gemma_300m_lora"
    assert values == {
        "dtype": "bfloat16",
        "paligemma_variant": "gemma_2b_lora",
        "action_expert_variant": "gemma_300m_lora",
        "action_dim": 5,
        "action_horizon": 1,
        "max_token_len": 200,
        "pi05": True,
        "discrete_state_input": True,
        "pytorch_compile_mode": "max-autotune",
    }


def test_exp32_profile_resolves_native_32d_and_transform_contract():
    config = csgo_model.CSGOPi0Config(profile="exp32_loc_main")

    assert config.profile == "exp32_loc_main"
    assert config.action_dim == 32
    assert config.action_horizon == 1
    assert config.discrete_state_input is False
    assert config.model_type == _model.ModelType.PI05
    assert config.paligemma_variant == "gemma_2b_lora_r32"
    assert config.action_expert_variant == "gemma_300m_lora_r32"
    assert config.inputs_spec(batch_size=2)[0].state.shape == (2, 32)
    assert config.inputs_spec(batch_size=2)[1].shape == (2, 1, 32)

    with pytest.raises(ValueError, match="discrete_state_input=False"):
        csgo_model.CSGOPi0Config(profile="exp32_loc_main", discrete_state_input=True)
    with pytest.raises(ValueError, match="action_dim=32"):
        csgo_model.CSGOPi0Config(profile="exp32_loc_main", action_dim=5)


def test_exp32_freeze_filter_leaves_image_connector_and_action_heads_trainable():
    config = csgo_model.CSGOPi0Config(profile="exp32_loc_main")
    model = nnx.eval_shape(config.create, jax.random.key(0))
    all_params = nnx.state(model, nnx.Param).flat_state()
    frozen = nnx.state(model, nnx.All(nnx.Param, config.get_freeze_filter())).flat_state()
    paths = {"/".join(path) if isinstance(path, tuple) else str(path) for path in all_params}
    frozen_paths = {"/".join(path) if isinstance(path, tuple) else str(path) for path in frozen}
    params_by_path = {
        "/".join(path) if isinstance(path, tuple) else str(path): value for path, value in all_params.items()
    }

    siglip_encoder_paths = {path for path in paths if "PaliGemma/img/" in path and "/head/" not in path}
    connector_paths = {path for path in paths if "PaliGemma/img/head/" in path}
    base_llm_paths = {path for path in paths if "PaliGemma/llm/" in path and "lora" not in path}
    lora_paths = {path for path in paths if "lora" in path}

    assert siglip_encoder_paths
    assert siglip_encoder_paths <= frozen_paths
    assert connector_paths
    assert connector_paths.isdisjoint(frozen_paths)
    assert base_llm_paths
    assert base_llm_paths <= frozen_paths
    assert lora_paths
    assert lora_paths.isdisjoint(frozen_paths)

    attention_projections = ("q_einsum", "kv_einsum", "attn_vec_einsum")
    feed_forward_projections = (
        "gating_einsum_lora_a",
        "gating_einsum_lora_b",
        "linear_lora_a",
        "linear_lora_b",
    )
    for projection in attention_projections:
        for expert_suffix in ("", "_1"):
            for leaf in ("lora_a", "lora_b"):
                path = f"PaliGemma/llm/layers/attn/{projection}{expert_suffix}/{leaf}"
                assert path in paths
                assert path not in frozen_paths
    for module_name in ("mlp", "mlp_1"):
        for leaf in feed_forward_projections:
            path = f"PaliGemma/llm/layers/{module_name}/{leaf}"
            assert path in paths
            assert path not in frozen_paths

    # The scanned layer axis comes first; the next axis is size two because
    # Gemma fuses K/V and gate/up projections.
    for path, value in params_by_path.items():
        if path.endswith(("/kv_einsum/lora_a", "/kv_einsum_1/lora_a")):
            assert value.value.shape[1] == 2
        if path.endswith("/gating_einsum_lora_a"):
            assert value.value.shape[1] == 2

    for module_name in ("action_in_proj", "time_mlp_in", "time_mlp_out", "action_out_proj"):
        module_paths = {path for path in paths if path.startswith(f"{module_name}/")}
        assert module_paths
        assert module_paths.isdisjoint(frozen_paths)


def test_frozen_vl_changes_only_image_connector_trainability():
    main = csgo_model.CSGOPi0Config(profile="exp32_loc_main")
    frozen_vl = csgo_model.CSGOPi0Config(profile="exp32_loc_main_frozen_vl")
    assert dataclasses.asdict(main) == dataclasses.asdict(frozen_vl)

    model = nnx.eval_shape(main.create, jax.random.key(0))
    all_params = nnx.state(model, nnx.Param).flat_state()
    all_paths = {"/".join(path) for path in all_params}
    main_frozen = {
        "/".join(path) for path in nnx.state(model, nnx.All(nnx.Param, main.get_freeze_filter())).flat_state()
    }
    frozen_vl_frozen = {
        "/".join(path) for path in nnx.state(model, nnx.All(nnx.Param, frozen_vl.get_freeze_filter())).flat_state()
    }
    connector = {path for path in all_paths if path.startswith("PaliGemma/img/head/")}
    assert connector == {"PaliGemma/img/head/kernel", "PaliGemma/img/head/bias"}
    assert frozen_vl_frozen - main_frozen == connector
    assert main_frozen - frozen_vl_frozen == set()


@pytest.mark.parametrize(
    ("profile", "expected_train"),
    [("v2_5k", False), ("exp32_loc_main", True), ("exp32_loc_main_frozen_vl", True)],
)
def test_native_training_augmentation_flag_is_profile_specific(monkeypatch, profile, expected_train):
    calls = []

    def parent_compute_loss(self, rng, observation, actions, *, train=False):
        del self, rng, observation, actions
        calls.append(train)

    monkeypatch.setattr(_pi0.Pi0, "compute_loss", parent_compute_loss)
    model = object.__new__(csgo_model.CSGOPi0)
    object.__setattr__(model, "profile", profile)
    model.compute_loss(None, None, None, train=True)
    model.compute_loss(None, None, None, train=False)
    assert calls == [expected_train, False]


def test_weight_loader_slices_legacy_and_loads_all_32_actions(monkeypatch):
    source_values = {
        "action_in_proj/kernel": np.arange(32 * 4, dtype=np.float32).reshape(32, 4),
        "action_in_proj/bias": np.arange(4, dtype=np.float32),
        "action_out_proj/kernel": np.arange(4 * 32, dtype=np.float32).reshape(4, 32),
        "action_out_proj/bias": np.arange(32, dtype=np.float32),
    }
    loaded = traverse_util.unflatten_dict(source_values, sep="/")
    monkeypatch.setattr(csgo_model.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(_model, "restore_params", lambda path, restore_type: loaded)

    for action_dim, profile in ((5, "v2_5k"), (32, "exp32_loc_main"), (32, "exp32_loc_main_frozen_vl")):
        target_values = {
            "action_in_proj/kernel": np.zeros((action_dim, 4), dtype=np.float32),
            "action_in_proj/bias": np.zeros((4,), dtype=np.float32),
            "action_out_proj/kernel": np.zeros((4, action_dim), dtype=np.float32),
            "action_out_proj/bias": np.zeros((action_dim,), dtype=np.float32),
            "PaliGemma/llm/lora_test/lora_a": np.full((2,), 7.0, dtype=np.float32),
        }
        params = traverse_util.unflatten_dict(target_values, sep="/")
        loader = csgo_model.CSGOPi05WeightLoader("test-checkpoint", profile=profile)
        result = traverse_util.flatten_dict(loader.load(params), sep="/")

        np.testing.assert_array_equal(
            result["action_in_proj/kernel"], source_values["action_in_proj/kernel"][:action_dim]
        )
        np.testing.assert_array_equal(
            result["action_out_proj/kernel"], source_values["action_out_proj/kernel"][:, :action_dim]
        )
        np.testing.assert_array_equal(
            result["action_out_proj/bias"], source_values["action_out_proj/bias"][:action_dim]
        )
        np.testing.assert_array_equal(
            result["PaliGemma/llm/lora_test/lora_a"], target_values["PaliGemma/llm/lora_test/lora_a"]
        )
