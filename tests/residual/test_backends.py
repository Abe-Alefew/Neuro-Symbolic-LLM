"""Consolidated backend agreement, selection, and validation tests.

Verifies that FabricPC and pure JAX backends produce identical numerical values
and gradients across all primitive loss functions and through-the-model executions.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from frozenllm.substrate import FrozenSubstrate
from losses.a2_loss import (
    anchor_loss,
    check_backend,
    hep_nudge_loss,
)
from residual import ResidualAdapter, ResidualConfig
from residual.cpc_heads import project
from residual.fabricpc import fpc_energy
from residual.fabricpc.fpc_energy import BACKENDS
from residual.hep import init_hep
from stages.stage_A.a2 import (
    A2Config,
    init_a2_params,
    residual_smoothness_loss,
    step_loss,
)
from tests.frozenllm.conftest import BATCH, NUM_TOKENS
from tests.residual.stage_A.a2.conftest import (
    LAYERS,
    SEQ,
    make_cfg,
    run_objective,
    with_random_b,
)

D = 16
V = 32
R = 4


def _x(*shape: int, seed: int = 0) -> jax.Array:
    return jnp.asarray(np.random.default_rng(seed).normal(size=shape), jnp.float32)


def _mask(b: int = 3, s: int = 6) -> jax.Array:
    return jnp.asarray(
        [
            [True, True, True, False, False, False],
            [True] * s,
            [True] * (s - 1) + [False],
        ][:b]
    )


def test_backends_are_listed_and_checked() -> None:
    assert BACKENDS == ("fabricpc", "jax")
    for bad in ("torch", "", None, "invalid"):
        with pytest.raises(ValueError, match="backend"):
            check_backend(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fn_name",
    ["anchor", "hep", "smoothness", "step", "cpc_project"],
)
def test_unknown_backend_is_rejected(fn_name: str) -> None:
    bad = "torch"
    with pytest.raises(ValueError, match="backend"):
        if fn_name == "anchor":
            anchor_loss(_x(3, 6, D), _x(3, 6, D, seed=1), backend=bad)
        elif fn_name == "hep":
            hep_nudge_loss(_x(3, 6, V), _x(V, D, seed=1), backend=bad)
        elif fn_name == "smoothness":
            residual_smoothness_loss(_x(3, 6, D), _x(3, 6, D, seed=1), backend=bad)
        elif fn_name == "step":
            step_loss(_x(2, 3, 6, D), backend=bad)
        elif fn_name == "cpc_project":
            project(_x(3, 6, D), {"W": _x(D, 8)}, backend=bad)


# --- Primitive Agreement ----------------------------------------------------------------------


@pytest.mark.parametrize("normalize", ["none", "relative"])
@pytest.mark.parametrize("masked", [False, True])
def test_anchor_loss_backends_agree(normalize: str, masked: bool) -> None:
    z = _x(3, 6, D, seed=13)
    anchor = _x(3, 6, D, seed=14)
    mask = _mask() if masked else None

    def run(backend: str) -> tuple[float, float]:
        val, grad = jax.value_and_grad(
            lambda x: anchor_loss(
                x, anchor, normalize=normalize, mask=mask, backend=backend
            )[0]
        )(z)
        return float(val), float(jnp.max(jnp.abs(grad)))

    fpc_v, fpc_g = run("fabricpc")
    jax_v, jax_g = run("jax")
    assert fpc_v == pytest.approx(jax_v, rel=1e-5)
    assert fpc_g == pytest.approx(jax_g, rel=1e-5)


@pytest.mark.parametrize("kind", ["dense", "lowrank", "gain"])
@pytest.mark.parametrize("masked", [False, True])
def test_hep_nudge_loss_backends_agree(kind: str, masked: bool) -> None:
    eps = _x(3, 6, V, seed=15)
    ref = _x(3, 6, D, seed=16)
    p = init_hep(kind, 0, [3], V, D, rank=R)["hep_3"]
    mask = _mask() if masked else None

    def run(backend: str) -> tuple[float, float]:
        def f(param: dict, e: jax.Array) -> jax.Array:
            if kind == "dense":
                h = param["W"]
            elif kind == "lowrank":
                h = (param["U"], param["V"])
            else:
                h = param["g"]
            return hep_nudge_loss(
                e, h, alpha=1.0, reference=ref, mask=mask, backend=backend
            )[0]

        val, (gp, ge) = jax.value_and_grad(f, argnums=(0, 1))(p, eps)
        g_norm = sum(float(jnp.max(jnp.abs(x))) for x in gp.values()) + float(
            jnp.max(jnp.abs(ge))
        )
        return float(val), g_norm

    fpc_v, fpc_g = run("fabricpc")
    jax_v, jax_g = run("jax")
    assert fpc_v == pytest.approx(jax_v, rel=1e-5)
    assert fpc_g == pytest.approx(jax_g, rel=1e-5)


@pytest.mark.parametrize("masked", [False, True])
def test_residual_smoothness_loss_backends_agree(masked: bool) -> None:
    r = _x(3, 6, D, seed=17)
    h = _x(3, 6, D, seed=18)
    mask = _mask() if masked else None

    def run(backend: str) -> tuple[float, float]:
        val, grad = jax.value_and_grad(
            lambda x: residual_smoothness_loss(
                x, h, mask=mask, normalize="input", backend=backend
            )[0]
        )(r)
        return float(val), float(jnp.max(jnp.abs(grad)))

    fpc_v, fpc_g = run("fabricpc")
    jax_v, jax_g = run("jax")
    assert fpc_v == pytest.approx(jax_v, rel=1e-5)
    assert fpc_g == pytest.approx(jax_g, rel=1e-5)


def test_step_loss_backends_agree() -> None:
    z = _x(4, 3, 6, D, seed=19)

    def run(backend: str) -> tuple[float, float]:
        val, grad = jax.value_and_grad(
            lambda x: step_loss(x, normalize="initial", backend=backend)[0]
        )(z)
        return float(val), float(jnp.max(jnp.abs(grad)))

    fpc_v, fpc_g = run("fabricpc")
    jax_v, jax_g = run("jax")
    assert fpc_v == pytest.approx(jax_v, rel=1e-5)
    assert fpc_g == pytest.approx(jax_g, rel=1e-5)


def test_cpc_project_backends_agree() -> None:
    x = _x(3, 6, D, seed=20)
    w = {"W": _x(D, 8, seed=21)}

    def run(backend: str) -> tuple[jax.Array, jax.Array]:
        val, grad = jax.value_and_grad(
            lambda p: jnp.sum(project(x, p, backend=backend) ** 2)
        )(w)
        return val, grad["W"]

    fpc_v, fpc_g = run("fabricpc")
    jax_v, jax_g = run("jax")
    np.testing.assert_allclose(fpc_v, jax_v, rtol=1e-5)
    np.testing.assert_allclose(fpc_g, jax_g, rtol=1e-5)


# --- Through-the-Model Agreement --------------------------------------------------------------


@pytest.mark.parametrize("term", ["cpc", "stab", "smoothness"])
def test_through_model_backends_agree(gpt2: FrozenSubstrate, term: str) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    ids = jnp.asarray(
        np.random.default_rng(0).integers(0, NUM_TOKENS, (BATCH, SEQ)),
        dtype=jnp.int32,
    )

    out = {}
    for backend in ("fabricpc", "jax"):
        if term == "cpc":
            cfg = make_cfg(cpc=1.0, cpc_dim=8, cpc_heads=backend)
        elif term == "stab":
            cfg = make_cfg(
                stab=1.0,
                w_anch=1.0,
                hep="lowrank",
                w_hep=1.0,
                stab_backend=backend,
            )
        else:
            cfg = make_cfg(stab=1.0, w_anch=0.0, w_smooth=1.0, stab_backend=backend)

        params = with_random_b(init_a2_params(cfg, adapter))

        def loss_fn(p: dict, c: A2Config = cfg) -> jax.Array:
            return run_objective(gpt2, adapter, c, p, ids)[0]

        val, grad = jax.value_and_grad(loss_fn)(params)
        out[backend] = (float(val), grad)

    fpc_val, fpc_grad = out["fabricpc"]
    jax_val, jax_grad = out["jax"]
    assert fpc_val == pytest.approx(jax_val, rel=1e-5)
    for k in fpc_grad:
        for leaf in fpc_grad[k]:
            np.testing.assert_allclose(
                np.asarray(fpc_grad[k][leaf]),
                np.asarray(jax_grad[k][leaf]),
                rtol=1e-4,
                atol=1e-6,
            )


def test_the_backend_choice_reaches_fabricpc(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real = fpc_energy._per_token_energy

    def spy(*a: Any, **k: Any) -> Any:
        calls.append(a[-1])
        return real(*a, **k)

    monkeypatch.setattr(fpc_energy, "_per_token_energy", spy)
    z, anchor = _x(3, 6, D, seed=25), _x(3, 6, D, seed=26)

    anchor_loss(z, anchor, backend="fabricpc")
    assert len(calls) == 1

    anchor_loss(z, anchor, backend="jax")
    assert len(calls) == 1
