"""Stage A2 gradient routing, linearity, and descent tests.

Matrix of gradient routing across all auxiliary terms, linearity of gradients
in term weights, and verification that gradient descent steps lower each term.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from frozenllm.substrate import FrozenSubstrate
from losses.a2_loss import hidden_prediction_loss
from residual import ResidualAdapter, ResidualConfig
from stages.stage_A.a2 import (
    A2Config,
    adapter_params,
    init_a2_params,
    predictor_bank,
    predictor_params,
)
from tests.residual.stage_A.a2.conftest import (
    LAYERS,
    SEQ,
    cpc_head_params,
    full_cfg,
    hep_params,
    make_cfg,
    run_objective,
    with_random_b,
    with_random_predictors,
)


def max_abs(tree: Any) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return 0.0
    return float(max(float(jnp.max(jnp.abs(x))) for x in leaves))


def tree_diff(a: Any, b: Any) -> Any:
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


ROUTING_CASES = [
    ("task", {}, {"adapter"}, {"cpc", "hep", "pred"}),
    ("cpc", {"cpc": 1.0}, {"adapter", "cpc"}, {"hep", "pred"}),
    (
        "stab_anch",
        {"stab": 1.0, "w_anch": 1.0},
        {"adapter"},
        {"cpc", "hep", "pred"},
    ),
    (
        "stab_hep",
        {"stab": 1.0, "hep": "lowrank", "w_hep": 1.0, "w_anch": 0.0},
        {"hep"},
        {"adapter", "cpc", "pred"},
    ),
    (
        "stab_smooth",
        {"stab": 1.0, "w_smooth": 1.0, "w_anch": 0.0},
        {"adapter"},
        {"cpc", "hep", "pred"},
    ),
    (
        "stab_step",
        {"stab": 1.0, "w_step": 1.0, "w_anch": 0.0},
        set(),
        {"adapter", "cpc", "hep", "pred"},
    ),
    (
        "hid_detached",
        {"hid": 1.0, "detach": True},
        {"pred"},
        {"adapter", "cpc", "hep"},
    ),
    (
        "hid_coupled",
        {"hid": 1.0, "detach": False},
        {"adapter", "pred"},
        {"cpc", "hep"},
    ),
]


@pytest.mark.parametrize(
    "term,cfg_kw,active_groups,zero_groups",
    ROUTING_CASES,
)
def test_gradient_routing_matrix(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
    term: str,
    cfg_kw: dict[str, Any],
    active_groups: set[str],
    zero_groups: set[str],
) -> None:
    """Verify that each loss term routes gradients to its designated parameter groups."""
    cfg = make_cfg(**cfg_kw)
    super_cfg = full_cfg(detach=False)
    params = with_random_predictors(with_random_b(init_a2_params(super_cfg, adapter)))

    def term_loss(p: dict) -> jax.Array:
        val, metrics = run_objective(substrate, adapter, cfg, p, input_ids)
        if term == "task":
            return metrics["task"]
        if term == "cpc":
            return metrics["cpc"]
        if term == "stab_anch":
            return metrics["stab_anch"]
        if term == "stab_hep":
            return metrics["stab_hep"]
        if term == "stab_smooth":
            return metrics["stab_smooth"]
        if term == "stab_step":
            return metrics["stab_step"]
        if term in ("hid_detached", "hid_coupled"):
            return metrics["hid"]
        return val

    g = jax.grad(term_loss)(params)

    group_map = {
        "adapter": adapter_params(g),
        "cpc": cpc_head_params(g),
        "hep": hep_params(g),
        "pred": predictor_params(g),
    }

    for grp in active_groups:
        assert (
            max_abs(group_map[grp]) > 0.0
        ), f"Expected gradient for active group {grp} on term {term}"

    for grp in zero_groups:
        assert (
            max_abs(group_map[grp]) == 0.0
        ), f"Expected zero gradient for inactive group {grp} on term {term}"


def test_coupled_gradient_matches_a_reference_with_a_stopped_target(
    gpt2: FrozenSubstrate, input_ids: jax.Array
) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    cfg = make_cfg(hid=1.0, detach=False)
    params = with_random_predictors(with_random_b(init_a2_params(cfg, adapter)))
    bank = predictor_bank(cfg, adapter, SEQ)
    every = tuple(range(gpt2.architecture.num_layers))

    def reference(p: dict) -> jax.Array:
        ap = adapter_params(p)
        run = gpt2.run_with_interception(
            input_ids, modify_fn=adapter.modify_fn(ap), intercept_layers=every
        )
        losses = []
        for i in LAYERS:
            source = adapter.apply(ap, run.hidden_state(i), i)
            target = jax.lax.stop_gradient(run.hidden_state(i + 1))
            pred = bank.predict(p, i, source)
            losses.append(hidden_prediction_loss(pred, target, source, cfg.hid)["loss"])
        return jnp.mean(jnp.stack(losses))

    want = jax.grad(reference)(params)
    got = jax.grad(lambda p: run_objective(gpt2, adapter, cfg, p, input_ids)[1]["hid"])(
        params
    )
    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=2e-3, atol=1e-6),
        got,
        want,
    )


@pytest.mark.parametrize(
    "term_name,build_cfg",
    [
        ("cpc", lambda w: make_cfg(cpc=w)),
        ("stab", lambda w: make_cfg(stab=w, hep="fixed_random")),
        ("hid", lambda w: make_cfg(hid=w, detach=False)),
        ("w_smooth", lambda w: make_cfg(stab=1.0, w_anch=0.0, w_smooth=w)),
    ],
)
def test_a_term_enters_the_gradient_linearly_with_its_weight(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
    term_name: str,
    build_cfg: Callable[[float], A2Config],
) -> None:
    params = with_random_predictors(
        with_random_b(init_a2_params(full_cfg(detach=False), adapter))
    )

    def grad_at(w: float):
        c = build_cfg(w)
        return jax.grad(
            lambda p: run_objective(substrate, adapter, c, p, input_ids)[0]
        )(params)

    g0 = grad_at(0.0)
    g1 = grad_at(1.0)
    g3 = grad_at(3.0)

    d1 = tree_diff(g1, g0)
    d3 = tree_diff(g3, g0)

    for k in params:
        for leaf in params[k]:
            np.testing.assert_allclose(
                np.asarray(d3[k][leaf]),
                3.0 * np.asarray(d1[k][leaf]),
                rtol=1e-5,
                atol=1e-6,
            )


@pytest.mark.parametrize(
    "term_key,cfg",
    [
        ("cpc", make_cfg(cpc=1.0)),
        (
            "stab_hep",
            make_cfg(stab=1.0, hep="lowrank", w_hep=1.0, w_anch=0.0),
        ),
        ("stab_smooth", make_cfg(stab=1.0, w_smooth=1.0, w_anch=0.0)),
        ("hid", make_cfg(hid=1.0, detach=True)),
        ("hid", make_cfg(hid=1.0, detach=False)),
        (
            "total",
            make_cfg(
                cpc=0.5, stab=0.5, hep="lowrank", w_hep=0.5, hid=0.5, detach=False
            ),
        ),
    ],
)
def test_optimising_the_objective_lowers_target_term(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
    term_key: str,
    cfg: A2Config,
) -> None:
    params = with_random_predictors(
        with_random_b(init_a2_params(cfg, adapter), scale=0.1)
    )
    opt = optax.adam(1e-2)
    state = opt.init(params)

    def loss_fn(p: dict) -> Any:
        return run_objective(substrate, adapter, cfg, p, input_ids)

    initial_total, initial_terms = loss_fn(params)
    initial_target = (
        float(initial_total) if term_key == "total" else float(initial_terms[term_key])
    )

    p = params
    for _ in range(5):
        g = jax.grad(lambda q: loss_fn(q)[0])(p)
        updates, state = opt.update(g, state, p)
        p = optax.apply_updates(p, updates)

    final_total, final_terms = loss_fn(p)
    final_target = (
        float(final_total) if term_key == "total" else float(final_terms[term_key])
    )

    assert final_target < initial_target
