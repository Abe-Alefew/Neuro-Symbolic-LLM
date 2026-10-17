"""Stability diagnostics: residual smoothness and step metrics (metrics only)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from frozenllm.substrate import FrozenSubstrate
from residual import ResidualAdapter, ResidualConfig
from residual.stability import (
    StabilityConfig,
    StabilityMonitor,
    residual_smoothness,
    step_metrics,
)
from tests.frozenllm.conftest import BATCH, SEQ, make_substrate


@pytest.fixture(scope="module")
def substrate() -> FrozenSubstrate:
    sub: FrozenSubstrate = make_substrate("gpt2")[1]
    return sub


@pytest.fixture(scope="module")
def adapter(substrate: FrozenSubstrate) -> ResidualAdapter:
    return ResidualAdapter(ResidualConfig(rank=4), substrate.architecture)


# ── residual smoothness ──────────────────────────────────────────────────────


def test_smoothness_matches_the_formula() -> None:
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.normal(size=(2, 5, 3)), jnp.float32)
    h = jnp.asarray(rng.normal(size=(2, 5, 3)), jnp.float32)
    out = residual_smoothness(r, h)
    expected_r = np.mean([np.sum((np.asarray(r)[b, t] - np.asarray(r)[b, t - 1]) ** 2)
                          for b in range(2) for t in range(1, 5)])
    expected_h = np.mean([np.sum((np.asarray(h)[b, t] - np.asarray(h)[b, t - 1]) ** 2)
                          for b in range(2) for t in range(1, 5)])
    assert float(out["res_smooth"]) == pytest.approx(expected_r, rel=1e-5)
    assert float(out["input_smooth"]) == pytest.approx(expected_h, rel=1e-5)
    assert float(out["res_vs_input"]) == pytest.approx(expected_r / expected_h, rel=1e-4)


def test_constant_correction_is_perfectly_smooth() -> None:
    h = jax.random.normal(jax.random.PRNGKey(0), (2, 6, 4))
    const = jnp.broadcast_to(jnp.arange(4.0), (2, 6, 4))
    out = residual_smoothness(const, h)
    assert float(out["res_smooth"]) == 0.0 and float(out["input_smooth"]) > 0


def test_single_token_sequence_gives_zeros() -> None:
    x = jnp.ones((2, 1, 4))
    out = residual_smoothness(x, x)
    assert all(float(v) == 0.0 for v in out.values())


# ── step metrics ─────────────────────────────────────────────────────────────


def test_no_settling_means_zero_step_metrics() -> None:
    z = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 5, 4))  # S = 0: one state
    out = step_metrics(z)
    assert set(out) == {"step_loss", "step_noncontract", "step_last_norm"}
    assert all(float(v) == 0.0 for v in out.values())


def test_step_metrics_on_a_known_trajectory() -> None:
    # One token, D=1: z = 0, 1, 3, 4  ->  d = 1, 2, 1
    z = jnp.asarray([0.0, 1.0, 3.0, 4.0]).reshape(4, 1, 1, 1)
    out = step_metrics(z, delta=0.0)
    assert float(out["step_loss"]) == pytest.approx(1 + 4 + 1)
    assert float(out["step_last_norm"]) == pytest.approx(1.0)
    # ratios: 2/1 - 1 = 1 (counts), 1/2 - 1 < 0 (clipped to 0)
    assert float(out["step_noncontract"]) == pytest.approx(1.0)


def test_one_step_has_no_noncontractivity() -> None:
    z = jnp.asarray([0.0, 2.0]).reshape(2, 1, 1, 1)
    out = step_metrics(z)
    assert float(out["step_loss"]) == pytest.approx(4.0)
    assert float(out["step_noncontract"]) == 0.0


def test_step_metrics_reject_bad_rank() -> None:
    with pytest.raises(ValueError, match=r"\[S\+1, B, T, D\]"):
        step_metrics(jnp.zeros((3, 4)))


def test_metrics_carry_no_gradient() -> None:
    z = jax.random.normal(jax.random.PRNGKey(0), (4, 2, 3, 5))
    g = jax.grad(lambda x: step_metrics(x)["step_loss"])(z)
    assert float(jnp.abs(g).sum()) == 0.0
    r = jax.random.normal(jax.random.PRNGKey(1), (2, 6, 4))
    g2 = jax.grad(lambda x: residual_smoothness(x, r)["res_smooth"])(r)
    assert float(jnp.abs(g2).sum()) == 0.0


# ── monitor ──────────────────────────────────────────────────────────────────


def test_monitor_metrics_on_the_adapted_model(substrate, adapter) -> None:
    rng = np.random.default_rng(1)
    params = {k: {"A": v["A"], "B": jnp.asarray(0.05 * rng.normal(size=v["B"].shape), jnp.float32)}
              for k, v in adapter.init_params().items()}
    d = substrate.architecture.hidden_size
    h = {i: jax.random.normal(jax.random.PRNGKey(i), (BATCH, SEQ, d)) for i in adapter.layers}
    mon = StabilityMonitor(StabilityConfig(mode="diagnostic"), adapter.layers)
    m = mon.metrics(adapter, params, h)
    for key in ("res_smooth", "input_smooth", "res_vs_input", "step_loss", "step_noncontract", "step_last_norm"):
        assert key in m and bool(jnp.isfinite(m[key]))
    assert all(f"res_smooth_l{i}" in m for i in adapter.layers)
    assert float(m["res_smooth"]) > 0
    assert float(m["step_loss"]) == float(m["step_noncontract"]) == float(m["step_last_norm"]) == 0.0


def test_zero_adapter_has_zero_residual_smoothness(substrate, adapter) -> None:
    d = substrate.architecture.hidden_size
    h = {i: jax.random.normal(jax.random.PRNGKey(i), (BATCH, SEQ, d)) for i in adapter.layers}
    mon = StabilityMonitor(StabilityConfig(mode="diagnostic"), adapter.layers)
    m = mon.metrics(adapter, adapter.init_params(), h)  # B = 0  =>  R = 0
    assert float(m["res_smooth"]) == 0.0 and float(m["input_smooth"]) > 0


def test_sites_must_be_adapted(adapter) -> None:
    with pytest.raises(ValueError, match="adapted sites"):
        StabilityMonitor(StabilityConfig(mode="diagnostic", layers=[2]), adapter.layers)


def test_config_validation_and_yaml_off() -> None:
    import yaml

    with pytest.raises(ValueError, match="stability mode"):
        StabilityConfig(mode="loss")
    with pytest.raises(ValueError, match="Unknown StabilityConfig"):
        StabilityConfig.from_dict({"bogus": 1})
    assert yaml.safe_load("mode: off")["mode"] is False
    assert StabilityConfig.from_dict({"mode": False}).mode == "off"
