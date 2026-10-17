"""FabricPC local predictor: formula, feedforward-only graph, gradient routing."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from frozenllm.substrate import FrozenSubstrate
from residual import ResidualAdapter, ResidualConfig
from residual.local_predictor import (
    LocalPredictorBank,
    PredictorConfig,
    merge_params,
    predictor_key,
    split_params,
)
from tests.frozenllm.conftest import BATCH, SEQ, make_substrate


@pytest.fixture(scope="module")
def substrate() -> FrozenSubstrate:
    sub: FrozenSubstrate = make_substrate("gpt2")[1]
    return sub


@pytest.fixture(scope="module")
def adapter(substrate: FrozenSubstrate) -> ResidualAdapter:
    return ResidualAdapter(ResidualConfig(rank=4), substrate.architecture)


def _bank(substrate: FrozenSubstrate, adapter: ResidualAdapter, **kw: object) -> LocalPredictorBank:
    cfg = PredictorConfig(**{"ff_dim": 16, **kw})  # type: ignore[arg-type]
    return LocalPredictorBank(cfg, substrate.architecture, SEQ, adapter.layers)


def _u(d: int, seed: int = 1) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(seed), (BATCH, SEQ, d))


def test_sites_default_to_adapted_layers(substrate, adapter) -> None:
    bank = _bank(substrate, adapter)
    assert bank.layers == adapter.layers
    assert set(bank.init_params()) == {predictor_key(i) for i in adapter.layers}


def test_graph_is_feedforward_without_inference(substrate, adapter) -> None:
    bank = _bank(substrate, adapter)
    assert bank.structure.config["inference"] is None
    assert list(bank.structure.node_order) == ["pred_input", "pred_hidden", "pred_output"]
    kinds = {n: v.node_info.node_type for n, v in bank.structure.nodes.items()}
    assert kinds == {
        "pred_input": "IdentityNode",
        "pred_hidden": "LnMlp1Node",
        "pred_output": "Linear",
    }


def test_matches_ln_gelu_mlp_formula(substrate, adapter) -> None:
    bank = _bank(substrate, adapter)
    params = bank.init_params()
    # Non-trivial biases/gain so the check is not satisfied by zeros/ones.
    q = {k: v + 0.1 * jax.random.normal(jax.random.PRNGKey(i), v.shape)
         for i, (k, v) in enumerate(params["pred_layer_7"].items())}
    params = {**params, "pred_layer_7": q}
    u = _u(substrate.architecture.hidden_size)
    mu, var = u.mean(-1, keepdims=True), u.var(-1, keepdims=True)
    xn = (u - mu) / jnp.sqrt(var + 1e-5) * q["pred_hidden.w.ln_gamma"] + q["pred_hidden.b.ln_beta"]
    hid = jax.nn.gelu(xn @ q["pred_hidden.w.W_ff1"] + q["pred_hidden.b.b_ff1"])
    expected = hid @ q["pred_output.w.W"] + q["pred_output.b.b"]
    got = bank.predict(params, 7, u)
    assert got.shape == (BATCH, SEQ, substrate.architecture.hidden_size)
    assert jnp.allclose(got, expected, atol=1e-5)


def test_predict_delta_adds_the_skip(substrate, adapter) -> None:
    plain, delta = _bank(substrate, adapter), _bank(substrate, adapter, predict_delta=True)
    params = plain.init_params()
    u = _u(substrate.architecture.hidden_size)
    assert jnp.allclose(delta.predict(params, 8, u), u + plain.predict(params, 8, u), atol=1e-6)


def test_gradient_routing_detach_input(substrate, adapter) -> None:
    d = substrate.architecture.hidden_size
    u = _u(d)
    for detach in (True, False):
        bank = _bank(substrate, adapter, detach_input=detach)
        params = bank.init_params()
        g_params, g_u = jax.grad(lambda p, x: jnp.sum(bank.predict(p, 7, x) ** 2), argnums=(0, 1))(params, u)
        assert float(sum(jnp.abs(x).sum() for x in jax.tree_util.tree_leaves(g_params))) > 0
        assert (float(jnp.abs(g_u).sum()) == 0.0) is detach


def test_rejects_wrong_input_shape_and_missing_layer(substrate, adapter) -> None:
    bank = _bank(substrate, adapter)
    params = bank.init_params()
    d = substrate.architecture.hidden_size
    with pytest.raises(ValueError, match="predictor input"):
        bank.predict(params, 7, jnp.zeros((BATCH, SEQ + 1, d)))
    with pytest.raises(KeyError):
        bank.predict(params, 2, jnp.zeros((BATCH, SEQ, d)))


def test_last_block_cannot_have_a_predictor(substrate, adapter) -> None:
    last = substrate.architecture.num_layers - 1
    with pytest.raises(ValueError, match="following block"):
        LocalPredictorBank(PredictorConfig(layers=[last]), substrate.architecture, SEQ, adapter.layers)


def test_split_and_merge_roundtrip(substrate, adapter) -> None:
    bank = _bank(substrate, adapter)
    a, p = adapter.init_params(), bank.init_params()
    merged = merge_params(a, p)
    a2, p2 = split_params(merged)
    assert set(a2) == set(a) and set(p2) == set(p)
    with pytest.raises(ValueError, match="overlap"):
        merge_params(a, a)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="ff_dim"):
        PredictorConfig(ff_dim=0)
    with pytest.raises(ValueError, match="Unknown PredictorConfig"):
        PredictorConfig.from_dict({"bogus": 1})
