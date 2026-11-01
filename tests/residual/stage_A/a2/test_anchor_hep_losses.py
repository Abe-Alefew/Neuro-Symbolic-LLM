"""Unit tests for anchor loss and HEP-nudge loss (pure math and FabricPC energy nodes).

Consolidates the former test_stability_losses.py and test_stability_fabricpc.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fabricpc.nodes import IdentityNode

from losses.a2_loss import (
    anchor_loss,
    hep_nudge,
    hep_nudge_loss,
    output_error,
)
from residual import fpc_energy
from residual.fpc_energy import (
    BACKENDS,
    anchor_sq_distance,
    highway_sq_norm,
)
from residual.hep import init_hep

D, DY = 8, 20
R = 4


def randn(*shape: int, seed: int = 0) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(seed), shape)


def _x(*shape: int, seed: int = 0) -> jax.Array:
    return jnp.asarray(np.random.default_rng(seed).normal(size=shape), jnp.float32)


# --- Anchor Loss Math -------------------------------------------------------------------------


def test_anchor_loss_zero_at_the_anchor() -> None:
    a = randn(3, 5, D)
    loss, m = anchor_loss(a, a)
    assert float(loss) == 0.0
    assert float(m["anch_ratio"]) == 0.0 and float(m["anch_dist"]) == pytest.approx(
        0, abs=1e-3
    )


def test_anchor_loss_known_value() -> None:
    a = randn(3, 5, D, seed=1)
    delta = jnp.arange(D, dtype=jnp.float32) / 4
    z = a + delta
    d2 = float(jnp.sum(delta**2))
    none, m = anchor_loss(z, a, normalize="none")
    np.testing.assert_allclose(none, d2, rtol=1e-5)
    np.testing.assert_allclose(m["anch_dist"], np.sqrt(d2), rtol=1e-4)
    rel, mr = anchor_loss(z, a)
    expected = d2 / float(jnp.mean(jnp.sum(a**2, -1)))
    np.testing.assert_allclose(rel, expected, rtol=1e-5)
    np.testing.assert_allclose(mr["anch_ratio"], np.sqrt(expected), rtol=1e-4)


def test_relative_loss_is_scale_invariant_absolute_is_not() -> None:
    a, z = randn(3, 5, D, seed=2), randn(3, 5, D, seed=3)
    r1, _ = anchor_loss(z, a)
    r2, _ = anchor_loss(10 * z, 10 * a)
    np.testing.assert_allclose(r1, r2, rtol=1e-5)
    n1, _ = anchor_loss(z, a, normalize="none")
    n2, _ = anchor_loss(10 * z, 10 * a, normalize="none")
    np.testing.assert_allclose(100 * n1, n2, rtol=1e-4)


def test_anchor_gets_no_gradient_state_does() -> None:
    a, z = randn(3, 5, D, seed=4), randn(3, 5, D, seed=5)
    gz, ga = jax.grad(lambda z, a: anchor_loss(z, a)[0], argnums=(0, 1))(z, a)
    assert not np.any(ga), "the anchor is a stop-gradient target"
    assert np.abs(gz).max() > 0 and np.isfinite(gz).all()
    gn = jax.grad(lambda z: anchor_loss(z, a, normalize="none")[0])(z)
    np.testing.assert_allclose(gn, 2 * (z - a) / (3 * 5), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize(
    "z_shape, a_shape, kwargs, match",
    [
        ((3, 5, D), (3, 5, D + 1), {}, "z_anchor"),
        ((3, 5, D), (2, 3, 5, D), {}, "z_anchor"),
        ((3, 5, D), (4, 5, D), {}, "z_anchor"),
        ((5, D), (5, D), {}, "expected"),
        ((3, 5, D), (3, 5, D), {"normalize": "l2"}, "normalize"),
    ],
)
def test_anchor_loss_validation(
    z_shape: tuple[int, ...],
    a_shape: tuple[int, ...],
    kwargs: dict,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        anchor_loss(randn(*z_shape), randn(*a_shape), **kwargs)


# --- Output Error Math ------------------------------------------------------------------------


def test_output_error_is_softmax_minus_onehot() -> None:
    logits = randn(2, 4, DY, seed=14)
    labels = jnp.array([[2, 5, 0, 19], [1, 1, 1, -100]])
    eps, mask = output_error(logits, labels)
    p = jax.nn.softmax(logits, axis=-1)
    onehot = jax.nn.one_hot(labels, DY)
    np.testing.assert_allclose(eps[mask], (p - onehot)[mask], rtol=1e-5, atol=1e-5)
    assert not np.any(eps[~mask])


def test_output_error_equals_the_cross_entropy_gradient() -> None:
    logits = randn(2, 3, DY, seed=15)
    labels = jnp.array([[1, 4, 18], [0, 2, 7]])

    def ce(logits_val: jax.Array) -> jax.Array:
        logp = jax.nn.log_softmax(logits_val, axis=-1)
        oh = jax.nn.one_hot(labels, DY)
        return -jnp.sum(logp * oh)

    g = jax.grad(ce)(logits)
    eps, mask = output_error(logits, labels)
    assert mask.all()
    np.testing.assert_allclose(eps, g, rtol=1e-5, atol=1e-5)


def test_output_error_ignores_masked_labels_and_stops_gradient() -> None:
    logits = randn(2, 4, DY, seed=16)
    labels = jnp.array([[0, -100, 2, -100], [-100, -100, -100, -100]])
    eps, mask = output_error(logits, labels)
    np.testing.assert_array_equal(
        mask, [[True, False, True, False], [False, False, False, False]]
    )
    assert not np.any(
        jax.grad(lambda logits_in: output_error(logits_in, labels)[0].sum())(logits)
    )


def test_output_error_validation() -> None:
    with pytest.raises(ValueError, match="logits"):
        output_error(randn(4, DY), jnp.zeros((4,), jnp.int32))
    with pytest.raises(ValueError, match="labels shape"):
        output_error(randn(2, 4, DY), jnp.zeros((2, 5), jnp.int32))


# --- HEP Nudge & Loss Math --------------------------------------------------------------------


def test_hep_nudge_dense_lowrank_and_gain() -> None:
    eps = randn(2, 5, DY, seed=17)
    u, w = randn(DY, 3, seed=18), randn(3, D, seed=19)
    v = u @ w
    np.testing.assert_allclose(hep_nudge(eps, v, 1.0), eps @ v, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        hep_nudge(eps, (u, w), 1.0), eps @ v, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        hep_nudge(eps, v, 2.5), 2.5 * (eps @ v), rtol=1e-4, atol=1e-4
    )
    assert hep_nudge(eps, v).shape == (2, 5, D)


@pytest.mark.parametrize("backend", BACKENDS)
def test_hep_nudge_stops_gradient_to_eps(backend: str) -> None:
    eps, v = randn(2, 5, DY, seed=20), randn(DY, D, seed=21)
    g = jax.grad(lambda e: hep_nudge_loss(e, v, backend=backend)[0])(eps)
    assert not np.any(g)


def test_hep_loss_known_value() -> None:
    eps, v = randn(2, 5, DY, seed=22), randn(DY, D, seed=23)
    loss, m = hep_nudge_loss(eps, v, alpha=0.5)
    nudge = 0.5 * np.asarray(eps) @ np.asarray(v)
    np.testing.assert_allclose(loss, np.mean(np.sum(nudge**2, -1)), rtol=1e-4)
    np.testing.assert_allclose(
        m["hep_nudge_norm"], np.mean(np.linalg.norm(nudge, axis=-1)), rtol=1e-4
    )
    assert "hep_nudge_ratio" not in m


def test_hep_loss_scales_with_alpha_squared_and_vanishes_when_off() -> None:
    eps, v = randn(2, 5, DY, seed=24), randn(DY, D, seed=25)
    l1, _ = hep_nudge_loss(eps, v, alpha=1.0)
    l3, _ = hep_nudge_loss(eps, v, alpha=3.0)
    np.testing.assert_allclose(l3, 9 * l1, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_alpha_and_gain_give_zero_penalty(backend: str) -> None:
    eps = _x(3, 6, DY, seed=19)
    v = _x(DY, D, seed=20)
    loss_gain, m_gain = hep_nudge_loss(eps, v, 0.0, backend=backend)
    assert float(loss_gain) == 0.0 and float(m_gain["hep_nudge_norm"]) < 1e-3
    loss_zero_v, _ = hep_nudge_loss(eps, jnp.zeros_like(v), backend=backend)
    assert float(loss_zero_v) == 0.0


def test_hep_loss_relative_to_the_reference_state() -> None:
    eps, v = randn(2, 5, DY, seed=26), randn(DY, D, seed=27)
    ref = randn(2, 5, D, seed=28)
    absolute, _ = hep_nudge_loss(eps, v)
    rel, m = hep_nudge_loss(eps, v, reference=ref)
    np.testing.assert_allclose(rel, absolute / jnp.mean(jnp.sum(ref**2, -1)), rtol=1e-4)
    np.testing.assert_allclose(m["hep_nudge_ratio"], np.sqrt(float(rel)), rtol=1e-4)
    rel2, _ = hep_nudge_loss(eps, v, reference=2 * ref)
    np.testing.assert_allclose(rel2, rel / 4, rtol=1e-4)


def test_hep_loss_gradient_audit() -> None:
    eps, ref = randn(2, 5, DY, seed=31), randn(2, 5, D, seed=32)
    v = randn(DY, D, seed=33)
    g_eps, g_ref, g_v = jax.grad(
        lambda e, r, v: hep_nudge_loss(e, v, 1.0, reference=r)[0],
        argnums=(0, 1, 2),
    )(eps, ref, v)
    assert not np.any(g_eps) and not np.any(g_ref)
    assert np.abs(g_v).max() > 0 and np.isfinite(g_v).all()

    u, w = randn(DY, 3, seed=34), randn(3, D, seed=35)
    g_u, g_w = jax.grad(
        lambda u, w: hep_nudge_loss(eps, (u, w), 1.0)[0], argnums=(0, 1)
    )(u, w)
    assert np.abs(g_u).max() > 0 and np.abs(g_w).max() > 0


def test_hep_loss_gradient_descent_shrinks_the_highway() -> None:
    eps = randn(4, 6, DY, seed=36)
    v = randn(DY, D, seed=37)
    f = jax.jit(jax.value_and_grad(lambda v: hep_nudge_loss(eps, v, 1.0)[0]))
    first, _ = f(v)
    for _ in range(50):
        _, g = f(v)
        v = v - 0.05 * g
    last, _ = f(v)
    assert float(last) < 0.1 * float(first)


# --- Parametrized Cross-Function Tests (JIT, Dtype, Step, Mask) --------------------------------


@pytest.mark.parametrize("fn", ["anchor", "hep"])
def test_stability_losses_jit_matches_eager(fn: str) -> None:
    if fn == "anchor":
        a, z = randn(3, 5, D, seed=12), randn(2, 3, 5, D, seed=13)
        eager = jax.value_and_grad(lambda z: anchor_loss(z, a)[0])(z)
        jitted = jax.jit(jax.value_and_grad(lambda z: anchor_loss(z, a)[0]))(z)
        np.testing.assert_allclose(eager[0], jitted[0], rtol=1e-5)
        np.testing.assert_allclose(eager[1], jitted[1], rtol=1e-4, atol=1e-7)
    else:
        eps, v = randn(2, 5, DY, seed=38), randn(DY, D, seed=39)
        eager = hep_nudge_loss(eps, v, 1.0)[0]
        jitted = jax.jit(lambda e, v: hep_nudge_loss(e, v, 1.0)[0])(eps, v)
        np.testing.assert_allclose(eager, jitted, rtol=1e-5)


@pytest.mark.parametrize("kind", ["anchor", "hep"])
def test_stability_losses_computed_in_float32(kind: str) -> None:
    if kind == "anchor":
        z, a = randn(3, 5, D, seed=10), randn(3, 5, D, seed=11)
        f32, _ = anchor_loss(z, a)
        bf16, _ = anchor_loss(z.astype(jnp.bfloat16), a.astype(jnp.bfloat16))
        assert bf16.dtype == jnp.float32
        np.testing.assert_allclose(bf16, f32, rtol=5e-2)
        assert np.isfinite(anchor_loss(z, jnp.zeros_like(a))[0])
    else:
        eps, v = randn(2, 5, DY, seed=38), randn(DY, D, seed=39)
        f32, _ = hep_nudge_loss(eps, v, 1.0)
        bf16, _ = hep_nudge_loss(eps.astype(jnp.bfloat16), v, 1.0)
        assert bf16.dtype == jnp.float32
        np.testing.assert_allclose(bf16, f32, rtol=5e-2)


@pytest.mark.parametrize("fn", ["anchor", "hep"])
def test_stability_losses_with_leading_step_axis(fn: str) -> None:
    if fn == "anchor":
        a = randn(3, 5, D, seed=6)
        z = randn(4, 3, 5, D, seed=7)
        stacked, _ = anchor_loss(z, a)
        per_step = [anchor_loss(z[s], a)[0] for s in range(4)]
        np.testing.assert_allclose(
            stacked, np.mean([float(x) for x in per_step]), rtol=1e-5
        )
    else:
        eps, v = randn(3, 2, 5, DY, seed=40), randn(DY, D, seed=41)
        stacked, _ = hep_nudge_loss(eps, v)
        per = [float(hep_nudge_loss(eps[s], v)[0]) for s in range(3)]
        np.testing.assert_allclose(stacked, np.mean(per), rtol=1e-5)


@pytest.mark.parametrize("fn", ["anchor", "hep"])
def test_stability_losses_mask_excludes_positions(fn: str) -> None:
    mask = jnp.array([[1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]], bool)
    if fn == "anchor":
        a, z = randn(2, 6, D, seed=8), randn(2, 6, D, seed=9)
        clean, _ = anchor_loss(z, a, mask=mask)
        corrupted = z.at[:, 3:].set(100.0).at[1, 1:].set(100.0)
        dirty, _ = anchor_loss(corrupted, a, mask=mask)
        np.testing.assert_allclose(clean, dirty, rtol=1e-6)
    else:
        eps, v = randn(2, 6, DY, seed=29), randn(DY, D, seed=30)
        clean, _ = hep_nudge_loss(eps, v, mask=mask)
        corrupted = eps.at[:, 3:].set(50.0).at[1, 1:].set(50.0)
        dirty, _ = hep_nudge_loss(corrupted, v, mask=mask)
        np.testing.assert_allclose(clean, dirty, rtol=1e-6)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"alpha": -1.0}, "alpha"),
        ({"reference": jnp.zeros((2, 4, D))}, "reference"),
    ],
)
def test_hep_loss_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        hep_nudge_loss(randn(2, 5, DY), randn(DY, D), **kwargs)
    with pytest.raises(ValueError, match="expected eps"):
        hep_nudge_loss(randn(5, DY), randn(DY, D))


# --- FabricPC Graph & Energy Representation Tests ---------------------------------------------


@pytest.mark.parametrize(
    "z_shape, a_shape",
    [
        ((3, 5, D), (3, 5, D)),
        ((4, 3, 5, D), (3, 5, D)),
    ],
)
def test_anchor_energy_is_the_squared_distance(
    z_shape: tuple[int, ...], a_shape: tuple[int, ...]
) -> None:
    z, a = _x(*z_shape, seed=1), _x(*a_shape, seed=2)
    out = anchor_sq_distance(z, a)
    assert out.shape == z_shape[:-1] and out.dtype == jnp.float32
    np.testing.assert_allclose(out, jnp.sum((z - a) ** 2, axis=-1), rtol=1e-6)


@pytest.mark.parametrize("kind", ["dense", "lowrank"])
def test_highway_energy_is_the_squared_norm_of_the_nudge(kind: str) -> None:
    eps = _x(3, 5, DY, seed=5)
    p = init_hep(kind, 0, [3], DY, D, rank=R)["hep_3"]
    v = p["V"] if kind == "dense" else (p["U"], p["W"])
    ref = (eps @ p["V"]) if kind == "dense" else (eps @ p["U"] @ p["W"])
    out = highway_sq_norm(eps, v)
    assert out.shape == (3, 5) and out.dtype == jnp.float32
    np.testing.assert_allclose(out, jnp.sum(ref**2, axis=-1), rtol=1e-5)


def test_the_low_rank_highway_is_never_materialised() -> None:
    p = init_hep("lowrank", 0, [3], DY, D, rank=R)["hep_3"]
    eps = _x(2, 3, DY, seed=6)
    jaxpr = jax.make_jaxpr(lambda eps: highway_sq_norm(eps, (p["U"], p["W"])))(eps)
    shapes = {tuple(var.aval.shape) for eqn in jaxpr.eqns for var in eqn.outvars}
    assert (DY, D) not in shapes


def test_the_graphs_are_fabricpc_nodes_with_the_expected_parameters() -> None:
    from fabricpc.graph import initialize_params

    anchor = fpc_energy._anchor_graph(D)
    assert isinstance(anchor.nodes["anch"], IdentityNode)
    params = initialize_params(anchor, jax.random.PRNGKey(10))
    assert set(params.nodes) == {"z", "anch"} and params.nodes["anch"].weights == {}

    lowrank = fpc_energy._highway_graph(DY, D, R)
    assert list(lowrank.node_order) == ["eps", "mid", "hep"]
    params = initialize_params(lowrank, jax.random.PRNGKey(11))
    assert set(params.nodes) == {"eps", "mid", "hep"}
    assert params.nodes["mid"].weights["eps->mid:in"].shape == (DY, R)
    assert params.nodes["hep"].weights["mid->hep:in"].shape == (R, D)


def test_malformed_highways_are_rejected() -> None:
    eps = _x(2, 3, DY, seed=12)
    with pytest.raises(ValueError, match="matrix"):
        highway_sq_norm(eps, jnp.ones((DY, D, 1)))
    with pytest.raises(ValueError, match="low-rank"):
        highway_sq_norm(eps, (jnp.ones((DY, R)), jnp.ones((R + 1, D))))


def test_output_error_feeds_the_highway_energy_like_the_reference() -> None:
    logits = _x(3, 6, DY, seed=21)
    labels = (
        jax.random.randint(jax.random.PRNGKey(22), (3, 6), 0, DY).at[1, 3:].set(-100)
    )
    eps, mask = output_error(logits, labels)
    v = _x(DY, D, seed=23)
    ref = _x(3, 6, D, seed=24)
    a = hep_nudge_loss(eps, v, reference=ref, mask=mask)[0]
    b = hep_nudge_loss(eps, v, reference=ref, mask=mask, backend="jax")[0]
    np.testing.assert_allclose(a, b, rtol=1e-5)
