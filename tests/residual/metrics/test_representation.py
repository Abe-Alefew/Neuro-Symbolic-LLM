"""Unit tests for the representation metrics. Only the last test needs a model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from metrics.performance import cross_entropy_loss
from metrics.representation import (
    compressibility_curve,
    direction_redundancy,
    gradient_conflict,
    loss_improvement,
    residual_compressibility,
    residual_norm_ratio,
    residual_redundancy,
)
from residual import ResidualAdapter, ResidualConfig
from tests.frozenllm.conftest import make_substrate


def test_loss_improvement_sign() -> None:
    assert float(loss_improvement(3.0, 2.5)) == pytest.approx(0.5)
    assert float(loss_improvement(3.0, 3.2)) == pytest.approx(-0.2)


def test_residual_compressibility() -> None:
    assert float(residual_compressibility(0.4, 0.5)) == pytest.approx(0.8)
    # eps keeps a zero denominator finite.
    assert float(residual_compressibility(0.0, 0.0, eps=1e-3)) == 0.0


def test_negative_eps_raises() -> None:
    with pytest.raises(ValueError, match="eps"):
        residual_compressibility(0.1, 0.2, eps=-1.0)


def test_compressibility_curve_is_relative_to_largest_rank() -> None:
    curve = compressibility_curve({64: 0.50, 1: 0.10, 16: 0.45}, eps=0.0)
    assert list(curve) == [1, 16, 64]
    assert float(curve[1]) == pytest.approx(0.2)
    assert float(curve[16]) == pytest.approx(0.9)
    assert float(curve[64]) == pytest.approx(1.0)


@pytest.mark.parametrize("deltas", [{}, {0: 0.1, 4: 0.2}])
def test_compressibility_curve_rejects_bad_input(deltas: dict[int, float]) -> None:
    with pytest.raises(ValueError):
        compressibility_curve(deltas)


# --- residual_norm_ratio -----------------------------------------------------------------


def test_norm_ratio_known_value() -> None:
    h = jnp.array([[[3.0, 4.0], [0.0, 2.0]]])  # norms 5 and 2
    r = jnp.array([[[0.0, 5.0], [0.0, 1.0]]])  # norms 5 and 1
    assert float(residual_norm_ratio(h, h + r, eps=0.0)) == pytest.approx(
        (1.0 + 0.5) / 2
    )


def test_norm_ratio_is_zero_without_a_residual_and_scale_invariant() -> None:
    rng = np.random.default_rng(0)
    h = jnp.asarray(rng.normal(size=(2, 5, 8)), jnp.float32)
    r = jnp.asarray(rng.normal(size=(2, 5, 8)), jnp.float32)
    assert float(residual_norm_ratio(h, h)) == 0.0
    base = float(residual_norm_ratio(h, h + r, eps=0.0))
    scaled = float(residual_norm_ratio(7.0 * h, 7.0 * (h + r), eps=0.0))
    assert scaled == pytest.approx(base, rel=1e-5)
    assert float(residual_norm_ratio(h, h + 2 * r, eps=0.0)) == pytest.approx(
        2 * base, rel=1e-5
    )


def test_norm_ratio_mask_leaves_tokens_out() -> None:
    h = jnp.ones((1, 4, 3))
    r = jnp.zeros((1, 4, 3)).at[0, 2:].set(10.0)
    mask = jnp.array([[True, True, False, False]])
    assert float(residual_norm_ratio(h, h + r, mask, eps=0.0)) == 0.0
    assert float(residual_norm_ratio(h, h + r, ~mask, eps=0.0)) > 1.0
    # A fully masked batch is 0, not NaN.
    assert float(residual_norm_ratio(h, h + r, jnp.zeros((1, 4), bool))) == 0.0


def test_norm_ratio_rejects_bad_input() -> None:
    h = jnp.ones((1, 4, 3))
    with pytest.raises(ValueError, match="shape"):
        residual_norm_ratio(h, jnp.ones((1, 4, 2)))
    with pytest.raises(ValueError, match="mask"):
        residual_norm_ratio(h, h, jnp.ones((1, 3), bool))
    with pytest.raises(ValueError, match="eps"):
        residual_norm_ratio(h, h, eps=-1.0)


# --- redundancy ----------------------------------------------------------------------------


def test_redundancy_extremes() -> None:
    assert float(direction_redundancy(jnp.eye(4))) == pytest.approx(0.0)
    parallel = jnp.array([[1.0, 2.0], [-2.0, -4.0], [0.5, 1.0]])
    assert float(direction_redundancy(parallel)) == pytest.approx(1.0, rel=1e-5)


def test_redundancy_known_value_and_invariances() -> None:
    v = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 3.0]])
    cos01, cos02, cos12 = 1 / np.sqrt(2), 0.0, 1 / np.sqrt(2)
    expected = (cos01 + cos02 + cos12) / 3  # mean over the 3 unordered pairs
    assert float(direction_redundancy(v)) == pytest.approx(expected, rel=1e-5)
    # Invariant to row scale, sign and order.
    scaled = v * jnp.array([[5.0], [-0.1], [2.0]])
    assert float(direction_redundancy(scaled)) == pytest.approx(expected, rel=1e-5)
    assert float(direction_redundancy(v[::-1])) == pytest.approx(expected, rel=1e-5)


def test_redundancy_matches_a_pairwise_loop() -> None:
    rng = np.random.default_rng(1)
    v = rng.normal(size=(6, 9))
    pairs = [
        abs(v[i] @ v[j]) / (np.linalg.norm(v[i]) * np.linalg.norm(v[j]))
        for i in range(6)
        for j in range(i + 1, 6)
    ]
    assert float(direction_redundancy(jnp.asarray(v))) == pytest.approx(
        np.mean(pairs), rel=1e-4
    )


def test_redundancy_edge_cases() -> None:
    assert float(direction_redundancy(jnp.ones((1, 5)))) == 0.0  # one direction
    assert float(direction_redundancy(jnp.zeros((3, 5)))) == 0.0  # all zero: finite
    mixed = jnp.array([[1.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    assert float(direction_redundancy(mixed)) == pytest.approx(2 / 6)  # one live pair
    with pytest.raises(ValueError, match="2-D"):
        direction_redundancy(jnp.ones((2, 3, 4)))


def test_residual_redundancy_reads_columns_of_a_and_rows_of_b() -> None:
    d, r = 6, 3
    a = jnp.stack([jnp.ones(d)] * r, axis=1)  # parallel columns -> read = 1
    b = jnp.eye(r, d)  # orthogonal rows -> write = 0
    out = residual_redundancy(a, b)
    assert float(out["read"]) == pytest.approx(1.0, rel=1e-5)
    assert float(out["write"]) == pytest.approx(0.0, abs=1e-6)
    # B = 0 (the initial state) has no write redundancy.
    assert float(residual_redundancy(a, jnp.zeros((r, d)))["write"]) == 0.0
    with pytest.raises(ValueError, match="A \\[d, r\\]"):
        residual_redundancy(jnp.ones((d, r)), jnp.ones((d, r)))


# --- gradient conflict ---------------------------------------------------------------------


def test_gradient_conflict_known_values() -> None:
    g = {"w": jnp.array([1.0, 0.0]), "b": {"c": jnp.array([[0.0, 2.0]])}}
    neg = jax.tree_util.tree_map(lambda x: -x, g)
    assert float(gradient_conflict(g, neg)) == pytest.approx(1.0, rel=1e-6)
    assert float(gradient_conflict(g, g)) == 0.0
    orthogonal = {"w": jnp.array([0.0, 1.0]), "b": {"c": jnp.array([[0.0, 0.0]])}}
    assert float(gradient_conflict(g, orthogonal)) == pytest.approx(0.0, abs=1e-7)
    # 60 degrees apart on the flattened vector: cos = 0.5 -> agreement -> 0; 120 -> 0.5.
    u = {"w": jnp.array([1.0, 0.0])}
    v120 = {"w": jnp.array([-0.5, np.sqrt(3) / 2])}
    assert float(gradient_conflict(u, v120)) == pytest.approx(0.5, rel=1e-5)


def test_gradient_conflict_is_symmetric_scale_free_and_handles_zero() -> None:
    rng = np.random.default_rng(2)
    a = {"x": jnp.asarray(rng.normal(size=(4, 3)), jnp.float32)}
    b = {"x": jnp.asarray(rng.normal(size=(4, 3)), jnp.float32)}
    ab = float(gradient_conflict(a, b))
    assert ab == pytest.approx(float(gradient_conflict(b, a)), rel=1e-6)
    scaled = jax.tree_util.tree_map(lambda x: 100.0 * x, b)
    assert float(gradient_conflict(a, scaled)) == pytest.approx(ab, rel=1e-5)
    zero = jax.tree_util.tree_map(jnp.zeros_like, a)
    assert float(gradient_conflict(a, zero)) == 0.0


def test_gradient_conflict_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="structure"):
        gradient_conflict({"a": jnp.ones(2)}, {"b": jnp.ones(2)})
    with pytest.raises(ValueError, match="shape"):
        gradient_conflict({"a": jnp.ones(2)}, {"a": jnp.ones(3)})
    with pytest.raises(ValueError, match="empty"):
        gradient_conflict({}, {})
    with pytest.raises(ValueError, match="eps"):
        gradient_conflict({"a": jnp.ones(2)}, {"a": jnp.ones(2)}, eps=-1.0)


def test_metrics_are_jittable() -> None:
    h = jnp.ones((1, 3, 4))
    g = {"w": jnp.arange(4.0)}
    assert float(jax.jit(residual_norm_ratio)(h, 2 * h)) == pytest.approx(1.0, rel=1e-5)
    assert float(jax.jit(gradient_conflict)(g, g)) == 0.0
    assert float(jax.jit(direction_redundancy)(jnp.eye(3))) == pytest.approx(0.0)


# --- with the real adapter ----------------------------------------------------------------


def test_metrics_on_the_adapter_and_two_task_gradients() -> None:
    substrate = make_substrate("gpt2")[1]
    adapter = ResidualAdapter(
        ResidualConfig(rank=4, layers=[5, 6]), substrate.architecture
    )
    params = adapter.init_params()
    rng = np.random.default_rng(0)
    ids = jnp.asarray(rng.integers(0, 64, (2, 10)), dtype=jnp.int32)

    def ratio(p) -> float:
        run = substrate.run_with_interception(
            ids, modify_fn=adapter.modify_fn(p), intercept_layers=adapter.layers
        )
        h = run.hidden_state(5)
        return float(residual_norm_ratio(h, adapter.apply(p, h, 5)))

    assert ratio(params) == 0.0  # B = 0: adapted = base
    moved = {
        k: {
            "A": v["A"],
            "B": jnp.asarray(0.05 * rng.normal(size=v["B"].shape), jnp.float32),
        }
        for k, v in params.items()
    }
    assert ratio(moved) > 0.0
    # Redundancy of the initial adapter: B = 0 writes nothing.
    red = residual_redundancy(params["layer_5"]["A"], params["layer_5"]["B"])
    assert 0.0 <= float(red["read"]) <= 1.0 and float(red["write"]) == 0.0

    def grad_on(batch):
        def loss(p):
            logits = substrate.run_with_interception(
                batch, modify_fn=adapter.modify_fn(p), intercept_layers=adapter.layers
            ).logits
            return cross_entropy_loss(logits, batch)

        return jax.grad(loss)(moved)

    other = jnp.asarray(rng.integers(0, 64, (2, 10)), dtype=jnp.int32)
    conflict = float(gradient_conflict(grad_on(ids), grad_on(other)))
    assert 0.0 <= conflict <= 1.0
    assert float(gradient_conflict(grad_on(ids), grad_on(ids))) == 0.0