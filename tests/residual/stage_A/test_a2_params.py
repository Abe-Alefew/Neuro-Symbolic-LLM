"""Stage A2 parameter tree, predictor bank, and checkpointing tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from frozenllm.substrate import FrozenSubstrate
from residual import ResidualAdapter, ResidualConfig, layer_key
from residual.cpc_heads import a_key, q_key
from residual.hep import hep_key
from residual.local_predictor import (
    LocalPredictorBank,
    PredictorConfig,
    merge_params,
    predictor_key,
    split_params,
)
from stages.results import RunWriter, load_params
from stages.stage_A.a2 import (
    A2Config,
    CPCConfig,
    HiddenConfig,
    adapter_params,
    init_a2_params,
    param_counts,
    predictor_bank,
    predictor_params,
)
from tests.residual.stage_A.conftest import (
    BATCH,
    FF,
    LAYERS,
    SEQ,
    make_cfg,
    run_objective,
    with_random_b,
)


def _bank(
    substrate: FrozenSubstrate, adapter: ResidualAdapter, **kw: object
) -> LocalPredictorBank:
    cfg = PredictorConfig(**{"ff_dim": FF, **kw})  # type: ignore[arg-type]
    return LocalPredictorBank(cfg, substrate.architecture, SEQ, adapter.layers)


def _u(d: int, seed: int = 1) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(seed), (BATCH, SEQ, d))


def test_with_every_aux_term_off_the_tree_is_the_adapter_tree(
    substrate: FrozenSubstrate, adapter: ResidualAdapter
) -> None:
    params = init_a2_params(make_cfg(kl=0.1, wd=0.1), adapter)
    expected = adapter.init_params()
    assert set(params) == set(expected)
    for k in expected:
        for name in expected[k]:
            np.testing.assert_array_equal(params[k][name], expected[k][name])


def test_cpc_heads_are_added_per_site_and_offset(
    substrate: FrozenSubstrate, adapter: ResidualAdapter
) -> None:
    cfg = A2Config(cpc=CPCConfig(weight=0.1, dim=8, offsets=(1, 2)))
    params = init_a2_params(cfg, adapter)
    expected = {layer_key(layer) for layer in LAYERS}
    for layer in LAYERS:
        expected |= {q_key(layer), a_key(layer, 1), a_key(layer, 2)}
    assert set(params) == expected
    d = substrate.architecture.hidden_size
    assert params[q_key(5)]["W"].shape == (d, 8)
    assert params[a_key(5, 1)]["W"].shape == (d, 8)


@pytest.mark.parametrize(
    "kind,leaves",
    [
        ("lowrank", {"U", "V"}),
        ("dense", {"W"}),
        ("gain", {"g"}),
    ],
)
def test_trainable_highways_are_added_and_fixed_ones_are_not(
    substrate: FrozenSubstrate, adapter: ResidualAdapter, kind: str, leaves: set[str]
) -> None:
    cfg = make_cfg(stab=0.1, hep=kind)
    params = init_a2_params(cfg, adapter)
    for layer in LAYERS:
        assert set(params[hep_key(layer)]) == leaves
    fixed = init_a2_params(make_cfg(stab=0.1, hep="fixed_random"), adapter)
    assert not any(k.startswith("hep_") for k in fixed)
    ident = init_a2_params(make_cfg(stab=0.1, hep="identity"), adapter)
    assert not any(k.startswith("hep_") for k in ident)


def test_predictors_are_added_per_site_only_when_enabled(
    substrate: FrozenSubstrate, adapter: ResidualAdapter
) -> None:
    off = init_a2_params(make_cfg(), adapter)
    assert set(off) == set(adapter.init_params())
    on = init_a2_params(make_cfg(hid=0.1), adapter)
    assert set(on) - set(off) == {predictor_key(layer) for layer in LAYERS}
    assert set(predictor_params(on)) == {predictor_key(layer) for layer in LAYERS}


def test_predictor_sites_default_to_the_adapted_layers_and_can_be_restricted(
    substrate: FrozenSubstrate, adapter: ResidualAdapter
) -> None:
    cfg = replace(
        make_cfg(hid=0.1),
        hid=HiddenConfig(weight=0.1, predictor=PredictorConfig(ff_dim=FF, layers=[5])),
    )
    assert set(predictor_params(init_a2_params(cfg, adapter))) == {predictor_key(5)}


def test_predictor_sites_must_be_adapted_layers(
    substrate: FrozenSubstrate, adapter: ResidualAdapter
) -> None:
    cfg = replace(
        make_cfg(hid=0.1),
        hid=HiddenConfig(weight=0.1, predictor=PredictorConfig(ff_dim=FF, layers=[2])),
    )
    with pytest.raises(ValueError, match="adapted layers"):
        init_a2_params(cfg, adapter)


def test_bank_last_block_cannot_have_a_predictor(substrate: FrozenSubstrate) -> None:
    last = substrate.architecture.num_layers - 1
    tmp_adapter = ResidualAdapter(
        ResidualConfig(rank=4, layers=[LAYERS[0], last]), substrate.architecture
    )
    with pytest.raises(ValueError, match="last block"):
        _bank(substrate, tmp_adapter)


def test_predictors_are_independent_of_the_sequence_length(
    substrate: FrozenSubstrate, adapter: ResidualAdapter
) -> None:
    params = init_a2_params(make_cfg(hid=0.1), adapter)
    for key, sub in predictor_params(params).items():
        assert all(SEQ not in v.shape for v in sub.values()), key
    cfg = make_cfg(hid=0.1)
    a = init_a2_params(cfg, adapter)
    b = init_a2_params(cfg, adapter, seq_len=SEQ)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for x, y in zip(
        jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True
    ):
        np.testing.assert_array_equal(x, y)


def test_adapter_params_selects_only_the_adapter(
    substrate: FrozenSubstrate, adapter: ResidualAdapter
) -> None:
    params = init_a2_params(
        make_cfg(cpc=0.1, stab=0.1, hep="lowrank", hid=0.1), adapter
    )
    assert set(adapter_params(params)) == {layer_key(layer) for layer in LAYERS}


def test_param_counts(substrate: FrozenSubstrate, adapter: ResidualAdapter) -> None:
    cfg = make_cfg(cpc=0.1, stab=0.1, hep="lowrank", hid=0.1)
    params = init_a2_params(cfg, adapter)
    counts = param_counts(params)
    d = substrate.architecture.hidden_size
    assert counts["adapter"] == ResidualAdapter.num_params(adapter.init_params())
    assert counts["cpc"] == len(LAYERS) * 2 * (d * 8)
    assert counts["hep"] == len(LAYERS) * (d * 4 + 4 * d)
    per_pred_site = d * FF + FF + 2 * d + FF * d + d
    assert counts["pred"] == len(LAYERS) * per_pred_site
    assert counts["total"] == sum(counts[g] for g in ("adapter", "cpc", "hep", "pred"))


@pytest.mark.parametrize("group", ["cpc_heads", "predictors", "objective_seed"])
def test_seed_controls_isolated_parameter_groups(
    substrate: FrozenSubstrate, adapter: ResidualAdapter, group: str
) -> None:
    pred_leaf = "pred_hidden.w.W_ff1"
    if group == "cpc_heads":
        a = init_a2_params(make_cfg(cpc=0.1, seed=0), adapter)
        b = init_a2_params(make_cfg(cpc=0.1, seed=0), adapter)
        c = init_a2_params(make_cfg(cpc=0.1, seed=1), adapter)
        np.testing.assert_array_equal(a[q_key(5)]["W"], b[q_key(5)]["W"])
        assert not np.allclose(a[q_key(5)]["W"], c[q_key(5)]["W"])
        np.testing.assert_array_equal(a[layer_key(5)]["A"], c[layer_key(5)]["A"])
    elif group == "predictors":

        def cfg(seed: int) -> A2Config:
            return replace(
                make_cfg(hid=0.1),
                hid=HiddenConfig(
                    weight=0.1, predictor=PredictorConfig(ff_dim=FF, seed=seed)
                ),
            )

        a, b, c = (init_a2_params(cfg(s), adapter) for s in (0, 0, 1))
        np.testing.assert_array_equal(
            a[predictor_key(5)][pred_leaf], b[predictor_key(5)][pred_leaf]
        )
        assert not np.allclose(
            a[predictor_key(5)][pred_leaf], c[predictor_key(5)][pred_leaf]
        )
        np.testing.assert_array_equal(a[layer_key(5)]["A"], c[layer_key(5)]["A"])
    elif group == "objective_seed":
        a = init_a2_params(make_cfg(hid=0.1, seed=0), adapter)
        b = init_a2_params(make_cfg(hid=0.1, seed=3), adapter)
        np.testing.assert_array_equal(
            a[predictor_key(5)][pred_leaf], b[predictor_key(5)][pred_leaf]
        )


def test_the_seed_reaches_the_fixed_random_highway(
    substrate: FrozenSubstrate, adapter: ResidualAdapter, input_ids: jax.Array
) -> None:
    def nudge(seed: int) -> float:
        cfg = make_cfg(stab=0.1, hep="fixed_random", seed=seed)
        params = with_random_b(init_a2_params(cfg, adapter))
        return float(
            run_objective(substrate, adapter, cfg, params, input_ids)[1][
                "stab/hep_nudge"
            ]
        )

    assert nudge(0) != nudge(1)


def test_bank_split_and_merge_roundtrip(
    substrate: FrozenSubstrate, adapter: ResidualAdapter
) -> None:
    bank = _bank(substrate, adapter)
    a, p = adapter.init_params(), bank.init_params()
    merged = merge_params(a, p)
    a2, p2 = split_params(merged)
    assert set(a2) == set(a) and set(p2) == set(p)
    for k in a:
        for leaf in a[k]:
            np.testing.assert_array_equal(a2[k][leaf], a[k][leaf])
    for k in p:
        for leaf in p[k]:
            np.testing.assert_array_equal(p2[k][leaf], p[k][leaf])


@pytest.mark.parametrize(
    "cfg",
    [
        make_cfg(hid=0.1, cpc=0.1, stab=0.1, hep="lowrank", w_hep=0.1),
        make_cfg(),
    ],
)
def test_tree_is_flat_two_level_and_checkpointable(
    substrate: FrozenSubstrate, adapter: ResidualAdapter, tmp_path: Path, cfg: A2Config
) -> None:
    params = init_a2_params(cfg, adapter)
    assert all(isinstance(v, dict) for v in params.values())
    assert all(isinstance(x, jax.Array) for v in params.values() for x in v.values())
    with RunWriter(tmp_path, "run") as writer:
        path = writer.save_params("t", params)
    saved = load_params(path)
    assert set(saved) == set(params)
    for k in params:
        assert set(saved[k]) == set(params[k])
        for name in params[k]:
            np.testing.assert_array_equal(saved[k][name], np.asarray(params[k][name]))


def test_bank_graph_is_feedforward_without_inference(
    substrate: FrozenSubstrate, adapter: ResidualAdapter
) -> None:
    bank = _bank(substrate, adapter)
    assert bank.structure.config["inference"] is None
    assert list(bank.structure.node_order) == [
        "pred_input",
        "pred_hidden",
        "pred_output",
    ]


def test_bank_matches_ln_gelu_mlp_formula(
    substrate: FrozenSubstrate, adapter: ResidualAdapter
) -> None:
    bank = _bank(substrate, adapter)
    params = bank.init_params()
    layer = LAYERS[0]
    pkey = predictor_key(layer)
    q = {
        k: v + 0.1 * jax.random.normal(jax.random.PRNGKey(i), v.shape)
        for i, (k, v) in enumerate(params[pkey].items())
    }
    params = {**params, pkey: q}
    u = _u(substrate.architecture.hidden_size)
    mu, var = u.mean(-1, keepdims=True), u.var(-1, keepdims=True)
    xn = (u - mu) / jnp.sqrt(var + 1e-5) * q["pred_hidden.w.ln_gamma"] + q[
        "pred_hidden.b.ln_beta"
    ]
    hid = jax.nn.gelu(xn @ q["pred_hidden.w.W_ff1"] + q["pred_hidden.b.b_ff1"])
    expected = hid @ q["pred_output.w.W"] + q["pred_output.b.b"]
    got = bank.predict(params, layer, u)
    assert got.shape == (BATCH, SEQ, substrate.architecture.hidden_size)
    assert jnp.allclose(got, expected, atol=1e-5)


def test_bank_predict_delta_adds_the_skip(
    substrate: FrozenSubstrate, adapter: ResidualAdapter
) -> None:
    layer = LAYERS[0]
    plain = _bank(substrate, adapter)
    delta = _bank(substrate, adapter, predict_delta=True)
    params = plain.init_params()
    u = _u(substrate.architecture.hidden_size)
    assert jnp.allclose(
        delta.predict(params, layer, u),
        u + plain.predict(params, layer, u),
        atol=1e-6,
    )


def test_bank_gradient_routing_detach_input(
    substrate: FrozenSubstrate, adapter: ResidualAdapter
) -> None:
    d = substrate.architecture.hidden_size
    layer = LAYERS[0]
    u = _u(d)
    for detach in (True, False):
        bank = _bank(substrate, adapter, detach_input=detach)
        params = bank.init_params()
        g_params, g_u = jax.grad(
            lambda p, x, bank=bank: jnp.sum(bank.predict(p, layer, x) ** 2),
            argnums=(0, 1),
        )(params, u)
        assert (
            float(sum(jnp.abs(x).sum() for x in jax.tree_util.tree_leaves(g_params)))
            > 0
        )
        assert (float(jnp.abs(g_u).sum()) == 0.0) is detach


def test_bank_rejects_wrong_input_shape_and_missing_layer(
    substrate: FrozenSubstrate, adapter: ResidualAdapter
) -> None:
    bank = _bank(substrate, adapter)
    params = bank.init_params()
    d = substrate.architecture.hidden_size
    layer = LAYERS[0]
    with pytest.raises(ValueError, match="predictor input"):
        bank.predict(params, layer, jnp.zeros((BATCH, SEQ + 1, d)))
    with pytest.raises(KeyError, match="layer 999"):
        bank.predict(params, 999, _u(d))


def test_the_bank_is_built_once_per_configuration(
    substrate: FrozenSubstrate, adapter: ResidualAdapter
) -> None:
    a = predictor_bank(make_cfg(hid=1.0), adapter, SEQ)
    b = predictor_bank(make_cfg(hid=2.0), adapter, SEQ)
    c = predictor_bank(
        replace(
            make_cfg(hid=1.0),
            hid=HiddenConfig(weight=1.0, predictor=PredictorConfig(ff_dim=FF + 1)),
        ),
        adapter,
        SEQ,
    )
    assert a is b
    assert a is not c
