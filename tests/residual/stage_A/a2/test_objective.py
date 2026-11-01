"""Stage A2 objective through-model contract tests.

Covers:
- Finiteness and term presence across configurations
- Total loss decomposition (weighted sum)
- Parity with A1 when auxiliary terms are disabled or monitor alone is active
- Weight decay exclusion of auxiliary parameters
- Ignored labels and padding handling through the full model
- Numerical equivalence between JIT and eager execution across ablation arms
- Cross-architecture compatibility (GPT-2 and Pythia/NeoX)
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest

from frozenllm.substrate import FrozenSubstrate, identity_modify
from residual import ResidualAdapter, ResidualConfig
from stages.stage_A.a1 import A1Config, a1_objective
from stages.stage_A.a2 import (
    A2Config,
    adapter_params,
    apply_arm,
    cpc_term,
    init_a2_params,
    stability_term,
    token_mask,
)
from stages.stage_base import base_logits
from tests.residual.conftest import (
    FAMILIES,
    LAYERS,
    SEQ,
    make_cfg,
    run_objective,
    with_random_b,
    with_random_predictors,
)


@pytest.mark.parametrize("kl,wd", [(0.0, 0.0), (0.2, 0.01)])
def test_equals_a1_when_the_aux_terms_are_off(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
    kl: float,
    wd: float,
) -> None:
    cfg = make_cfg(kl=kl, wd=wd)
    params = with_random_b(init_a2_params(cfg, adapter))

    def a1_total(p: dict):
        return a1_objective(
            p,
            substrate=substrate,
            adapter=adapter,
            config=A1Config(lambda_kl=kl, lambda_wd=wd),
            input_ids=input_ids,
        )

    def a2_total(p: dict):
        return run_objective(substrate, adapter, cfg, p, input_ids)

    a1_val, a1_terms = a1_total(params)
    a2_val, a2_terms = a2_total(params)
    assert set(a2_terms) == set(a1_terms)
    np.testing.assert_allclose(float(a2_val), float(a1_val), rtol=1e-5)
    for k in a1_terms:
        np.testing.assert_allclose(
            float(a2_terms[k]), float(a1_terms[k]), rtol=1e-5, atol=1e-7
        )

    a1_g = jax.grad(lambda p: a1_total(p)[0])(params)
    a2_g = jax.grad(lambda p: a2_total(p)[0])(params)
    for k in params:
        for leaf in params[k]:
            np.testing.assert_allclose(
                np.asarray(a2_g[k][leaf]),
                np.asarray(a1_g[k][leaf]),
                rtol=1e-5,
                atol=1e-7,
            )


def test_monitor_alone_equals_a1(
    substrate: FrozenSubstrate, adapter: ResidualAdapter, input_ids: jax.Array
) -> None:
    cfg = make_cfg(monitor="diagnostic")
    params = with_random_b(init_a2_params(cfg, adapter))
    a1_val = a1_objective(
        params,
        substrate=substrate,
        adapter=adapter,
        config=A1Config(),
        input_ids=input_ids,
    )[0]
    a2_val = run_objective(substrate, adapter, cfg, params, input_ids)[0]
    np.testing.assert_allclose(float(a2_val), float(a1_val), rtol=1e-5)


@pytest.mark.parametrize(
    "cfg,present,absent,weighted_check",
    [
        (make_cfg(), {"task", "total"}, {"cpc", "stab", "hid"}, False),
        (make_cfg(cpc=0.2), {"task", "total", "cpc"}, {"stab", "hid"}, False),
        (
            make_cfg(stab=0.2, hep="fixed_random"),
            {"task", "total", "stab"},
            {"cpc", "hid"},
            False,
        ),
        (make_cfg(hid=0.2), {"task", "total", "hid"}, {"cpc", "stab"}, False),
        (
            make_cfg(
                kl=0.3,
                wd=0.02,
                cpc=0.4,
                stab=0.7,
                hep="fixed_random",
                w_smooth=0.1,
                hid=0.3,
            ),
            {"task", "total", "kl", "wd", "cpc", "stab", "hid"},
            set(),
            True,
        ),
    ],
)
def test_contract_terms_presence_and_weighted_sum(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
    cfg: A2Config,
    present: set[str],
    absent: set[str],
    weighted_check: bool,
) -> None:
    params = with_random_predictors(with_random_b(init_a2_params(cfg, adapter)))
    total, terms = run_objective(substrate, adapter, cfg, params, input_ids)
    assert np.isfinite(float(total))
    for k, v in terms.items():
        assert np.isfinite(float(v)), f"{k} is not finite"
    assert present <= set(terms)
    assert not (absent & set(terms))
    if weighted_check:
        expected = (
            float(terms["task"])
            + 0.3 * float(terms["kl"])
            + 0.02 * float(terms["wd"])
            + 0.4 * float(terms["cpc"])
            + 0.7 * float(terms["stab"])
            + 0.3 * float(terms["hid"])
        )
        assert float(total) == pytest.approx(expected, rel=1e-5)


def test_terms_match_an_independent_path_through_the_model(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
) -> None:
    cfg = make_cfg(cpc=0.5, stab=0.5, hep="fixed_random", positive="base")
    params = with_random_b(init_a2_params(cfg, adapter))
    _, terms = run_objective(substrate, adapter, cfg, params, input_ids)

    ap = adapter_params(params)
    run = substrate.run_with_interception(
        input_ids,
        modify_fn=adapter.modify_fn(ap),
        intercept_layers=adapter.layers,
    )
    corrected = {k: adapter.apply(ap, run.hidden_state(k), k) for k in adapter.layers}
    base_run = substrate.run_with_interception(
        input_ids, modify_fn=identity_modify, intercept_layers=adapter.layers
    )
    base = {k: base_run.hidden_state(k) for k in adapter.layers}

    l_cpc, m_cpc = cpc_term(params, config=cfg.cpc, corrected=corrected, base=base)
    l_stab, m_stab = stability_term(
        params,
        config=cfg.stab,
        corrected=corrected,
        base=base,
        logits=run.logits,
        labels=input_ids,
        seed=cfg.seed,
    )
    assert float(terms["cpc"]) == pytest.approx(float(l_cpc), rel=1e-5)
    assert float(terms["stab"]) == pytest.approx(float(l_stab), rel=1e-5)
    for name, value in {**m_cpc, **m_stab}.items():
        np.testing.assert_allclose(float(terms[name]), float(value), rtol=1e-5)


def test_precomputed_base_logits_give_the_same_result(
    substrate: FrozenSubstrate, adapter: ResidualAdapter, input_ids: jax.Array
) -> None:
    cfg = make_cfg(kl=0.5, wd=0.01, cpc=0.2, stab=0.2, hid=0.1)
    params = with_random_predictors(with_random_b(init_a2_params(cfg, adapter)))
    t0, terms0 = run_objective(substrate, adapter, cfg, params, input_ids)
    base = base_logits(substrate, input_ids)
    t1, terms1 = run_objective(
        substrate, adapter, cfg, params, input_ids, precomputed_base_logits=base
    )
    np.testing.assert_allclose(float(t0), float(t1), rtol=1e-5)
    for k in terms0:
        np.testing.assert_allclose(float(terms0[k]), float(terms1[k]), rtol=1e-5)


def test_weight_decay_covers_the_adapter_and_not_the_heads_or_predictors(
    substrate: FrozenSubstrate, adapter: ResidualAdapter, input_ids: jax.Array
) -> None:
    cfg = make_cfg(wd=0.5, cpc=0.1, stab=0.1, hep="lowrank", w_hep=0.1, hid=0.1)
    params = with_random_predictors(with_random_b(init_a2_params(cfg, adapter)))
    _, terms = run_objective(substrate, adapter, cfg, params, input_ids)
    expected = ResidualAdapter.l2_norm_sq(adapter_params(params))
    np.testing.assert_allclose(float(terms["wd"]), float(expected), rtol=1e-6)
    everything = ResidualAdapter.l2_norm_sq(params)
    assert float(everything) > 1.01 * float(expected)


def test_kl_is_skipped_without_a_weight_and_without_base(
    substrate: FrozenSubstrate, adapter: ResidualAdapter, input_ids: jax.Array
) -> None:
    cfg = make_cfg(cpc=0.1, positive="corrected")
    _, terms = run_objective(
        substrate, adapter, cfg, init_a2_params(cfg, adapter), input_ids
    )
    assert "kl" not in terms


def test_the_last_layer_can_be_adapted(
    substrate: FrozenSubstrate, input_ids: jax.Array
) -> None:
    last = substrate.architecture.num_layers - 1
    ad = ResidualAdapter(
        ResidualConfig(rank=4, layers=[last - 1, last]), substrate.architecture
    )
    cfg = make_cfg(cpc=0.1, stab=0.1, hep="fixed_random")
    params = with_random_b(init_a2_params(cfg, ad))
    total, terms = run_objective(substrate, ad, cfg, params, input_ids)
    assert np.isfinite(float(total))
    assert "cpc" in terms and "stab" in terms


def test_all_valid_labels_equal_no_labels(
    substrate: FrozenSubstrate, adapter: ResidualAdapter, input_ids: jax.Array
) -> None:
    cfg = make_cfg(cpc=0.3, stab=0.3, hep="fixed_random")
    params = with_random_b(init_a2_params(cfg, adapter))
    t0, terms0 = run_objective(substrate, adapter, cfg, params, input_ids)
    t1, terms1 = run_objective(
        substrate, adapter, cfg, params, input_ids, labels=input_ids
    )
    np.testing.assert_allclose(float(t0), float(t1), rtol=1e-5)
    for k in terms0:
        np.testing.assert_allclose(float(terms0[k]), float(terms1[k]), rtol=1e-5)


def test_ignored_labels_leave_the_auxiliary_terms_finite(
    substrate: FrozenSubstrate, adapter: ResidualAdapter, input_ids: jax.Array
) -> None:
    cfg = make_cfg(
        cpc=0.3,
        stab=0.3,
        hep="fixed_random",
        w_smooth=0.1,
        hid=0.2,
        detach=False,
    )
    params = with_random_predictors(with_random_b(init_a2_params(cfg, adapter)))
    labels = input_ids.at[:, SEQ // 2 :].set(-100)
    mask = token_mask(labels)
    assert mask is not None and not bool(mask[:, SEQ // 2 :].any())
    total, terms = run_objective(
        substrate, adapter, cfg, params, input_ids, labels=labels
    )
    assert np.isfinite(float(total))
    for k in ("cpc", "stab", "stab_anch", "stab_smooth", "hid"):
        assert np.isfinite(float(terms[k]))


@pytest.mark.parametrize(
    "arm",
    [
        "a1_mode",
        "cpc_base",
        "full",
    ],
)
def test_jit_matches_eager(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
    arm: str,
) -> None:
    cfg = apply_arm(make_cfg(kl=0.1, wd=0.01), arm)
    params = with_random_b(init_a2_params(cfg, adapter))
    if cfg.hid.enabled:
        params = with_random_predictors(params)

    def f(p: dict, ids: jax.Array) -> Any:
        return run_objective(substrate, adapter, cfg, p, ids)

    eager_total, eager_terms = f(params, input_ids)
    jit_f = jax.jit(f)
    jit_total, jit_terms = jit_f(params, input_ids)

    np.testing.assert_allclose(
        float(eager_total), float(jit_total), rtol=1e-5, atol=1e-6
    )
    for k in eager_terms:
        np.testing.assert_allclose(
            float(eager_terms[k]), float(jit_terms[k]), rtol=1e-5, atol=1e-6
        )

    eager_g = jax.grad(lambda p: f(p, input_ids)[0])(params)
    jit_g = jax.grad(lambda p: jit_f(p, input_ids)[0])(params)
    for k in params:
        for leaf in params[k]:
            np.testing.assert_allclose(
                np.asarray(eager_g[k][leaf]),
                np.asarray(jit_g[k][leaf]),
                rtol=1e-5,
                atol=1e-6,
            )


@pytest.mark.parametrize("family", FAMILIES)
def test_interception_and_norm_across_architectures(
    family: str, input_ids: jax.Array
) -> None:
    from tests.frozenllm.conftest import make_substrate

    sub: FrozenSubstrate = make_substrate(family)[1]
    ad = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), sub.architecture)
    cfg = make_cfg(cpc=0.1, stab=0.1, hid=0.1)
    params = with_random_predictors(with_random_b(init_a2_params(cfg, ad)))
    total, terms = run_objective(sub, ad, cfg, params, input_ids)
    assert np.isfinite(float(total))
    assert {"task", "cpc", "stab", "hid"} <= set(terms)
