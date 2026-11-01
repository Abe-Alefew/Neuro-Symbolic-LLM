"""Shared fixtures, constants and helpers for Stage A2 tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from frozenllm.substrate import FrozenSubstrate
from residual import ResidualAdapter, ResidualConfig
from residual.local_predictor import PredictorConfig
from stages.stage_A.a2 import (
    A2Config,
    Check,
    CPCConfig,
    HEPConfig,
    HiddenConfig,
    MonitorConfig,
    StabLossConfig,
    a2_objective,
    adapter_params,
    predictor_params,
)
from stages.stage_base import Domain, make_eval_set
from tests.frozenllm.conftest import BATCH, NUM_TOKENS, make_substrate

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIPPED = REPO_ROOT / "configs" / "stage_A" / "a2_pc_inspired.yaml"

SEQ = 12
LAYERS = [5, 6]
FF = 16
FAMILIES = ("gpt2", "neox")


@pytest.fixture(scope="module")
def gpt2() -> FrozenSubstrate:
    sub: FrozenSubstrate = make_substrate("gpt2")[1]
    return sub


@pytest.fixture(scope="module")
def pythia() -> FrozenSubstrate:
    sub: FrozenSubstrate = make_substrate("neox")[1]
    return sub


@pytest.fixture(scope="module")
def substrate(gpt2: FrozenSubstrate) -> FrozenSubstrate:
    return gpt2


@pytest.fixture(scope="module")
def adapter(substrate: FrozenSubstrate) -> ResidualAdapter:
    return ResidualAdapter(
        ResidualConfig(rank=4, layers=LAYERS), substrate.architecture
    )


@pytest.fixture(scope="module")
def input_ids() -> jax.Array:
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.integers(0, NUM_TOKENS, (BATCH, SEQ)), dtype=jnp.int32)


def cpc_head_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in params.items()
        if k.startswith("cpcq_") or k.startswith("cpca_")
    }


def hep_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if k.startswith("hep_")}


def make_cfg(
    *,
    cpc: float = 0.0,
    stab: float = 0.0,
    hid: float = 0.0,
    kl: float = 0.0,
    wd: float = 0.0,
    cpc_dim: int = 8,
    cpc_offsets: tuple[int, ...] = (1,),
    positive: str = "base",
    cpc_heads: str = "fabricpc",
    hep: str | None = None,
    w_anch: float = 1.0,
    w_hep: float = 0.0,
    w_smooth: float = 0.0,
    w_step: float = 0.0,
    stab_backend: str = "fabricpc",
    detach: bool = True,
    predict_delta: bool = False,
    hidden_target: str = "corrected_teacher",
    hidden_norm: str = "target_var",
    predictor_lr_mult: float = 1.0,
    monitor: str = "off",
    seed: int = 0,
    **extra: Any,
) -> A2Config:
    """Helper to build an ``A2Config`` with concise keyword overrides."""
    cpc_cfg = CPCConfig(
        weight=cpc,
        dim=cpc_dim,
        offsets=cpc_offsets,
        positive=positive,
        heads=cpc_heads,
    )
    stab_cfg = StabLossConfig(
        weight=stab,
        w_anch=w_anch,
        w_hep=w_hep,
        w_smooth=w_smooth,
        w_step=w_step,
        backend=stab_backend,
        hep=HEPConfig(kind=hep, rank=4, alpha=1.0) if hep else HEPConfig(),
    )
    hid_cfg = HiddenConfig(
        weight=hid,
        hidden_target=hidden_target,
        hidden_norm=hidden_norm,
        predictor_lr_mult=predictor_lr_mult,
        predictor=PredictorConfig(
            ff_dim=FF, detach_input=detach, predict_delta=predict_delta
        ),
    )
    monitor_cfg = MonitorConfig(mode=monitor)
    return A2Config(
        lambda_kl=kl,
        lambda_wd=wd,
        seed=seed,
        cpc=cpc_cfg,
        stab=stab_cfg,
        hid=hid_cfg,
        monitor=monitor_cfg,
        **extra,
    )


def full_cfg(**kw: Any) -> A2Config:
    """A2Config with all loss terms enabled for audit testing."""
    return make_cfg(
        kl=0.1,
        wd=0.01,
        cpc=0.05,
        stab=0.1,
        hep="lowrank",
        w_hep=0.1,
        w_smooth=0.1,
        hid=0.1,
        detach=False,
        monitor="diagnostic",
        **kw,
    )


def with_random_b(params: dict, scale: float = 0.05) -> dict:
    """Give the adapter non-zero residual parameters (B = 0 at init)."""
    rng = np.random.default_rng(1)
    out = {k: dict(v) for k, v in params.items()}
    for k in adapter_params(params):
        out[k]["B"] = jnp.asarray(
            scale * rng.normal(size=out[k]["B"].shape), jnp.float32
        )
    return out


def with_random_predictors(params: dict, scale: float = 0.1) -> dict:
    """Give predictor parameters non-zero weights and biases."""
    rng = np.random.default_rng(2)
    out = {k: dict(v) for k, v in params.items()}
    for k in predictor_params(params):
        out[k] = {
            n: v + jnp.asarray(scale * rng.normal(size=v.shape), jnp.float32)
            for n, v in params[k].items()
        }
    return out


def _domain(name: str, token: int, epochs: int = 1, num_blocks: int = 8) -> Domain:
    """Generate a test Domain with predictable token blocks."""
    rng = np.random.default_rng(token)
    blocks = np.full((num_blocks, SEQ), token, dtype=np.int32)
    noise = rng.random(blocks.shape) < 0.1
    blocks[noise] = rng.integers(0, NUM_TOKENS, int(noise.sum()))
    return Domain(
        name=name,
        train_blocks=blocks,
        eval_set=make_eval_set(blocks, BATCH, 1),
        epochs=epochs,
    )


def _events(run: Path) -> list[dict]:
    """Parse JSONL lines from RunWriter output."""
    return [
        json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
    ]


def by_name(checks: list[Check]) -> dict[str, Check]:
    """Map audit checks by name."""
    return {c.name: c for c in checks}


def rand(*shape: int, seed: int = 0) -> jax.Array:
    """Create reproducible standard normal jnp arrays."""
    return jnp.asarray(np.random.default_rng(seed).normal(size=shape), jnp.float32)


def run_objective(substrate, adapter, cfg, params, ids, **kw):
    """Convenience runner for a2_objective."""
    return a2_objective(
        params, substrate=substrate, adapter=adapter, config=cfg, input_ids=ids, **kw
    )


def grads_of(substrate, adapter, cfg, params, ids, term="total", **kw):
    """Compute gradients of a specific term of a2_objective."""
    if term == "total":
        return jax.grad(
            lambda p: run_objective(substrate, adapter, cfg, p, ids, **kw)[0]
        )(params)
    return jax.grad(
        lambda p: run_objective(substrate, adapter, cfg, p, ids, **kw)[1][term]
    )(params)


@pytest.fixture(scope="module")
def runner() -> Any:
    """Dynamically load the stage A2 CLI runner script."""
    path = REPO_ROOT / "experiments" / "run_stage_a2.py"
    spec = importlib.util.spec_from_file_location("run_stage_a2_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        sys.modules.pop(spec.name, None)
        pytest.skip("experiments/run_stage_a2.py is not available")
    yield module
    sys.modules.pop(spec.name, None)


@pytest.fixture
def setup(runner, substrate, tmp_path, monkeypatch) -> Path:
    """Minimal run environment stubbed to prevent dataset downloads."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "adapter: {rank: 4, layers: [5, 6]}\n"
        "objective:\n  lambda_kl: 0.1\n"
        "  cpc: {weight: 0.2, dim: 8}\n"
        "  stab: {weight: 0.2, hep: {kind: fixed_random, rank: 4}}\n"
        f"training: {{model: gpt2, seq_len: {SEQ}, batch_size: 2, epochs: 1, "
        "learning_rate: 1.0e-2, eval_batches: 1, log_every: 1}\n"
        "domains:\n  - {name: a, dataset: x}\n"
    )
    blocks = np.random.default_rng(0).integers(0, NUM_TOKENS, (4, SEQ)).astype(np.int32)
    monkeypatch.setattr(runner, "FrozenSubstrate", lambda _name: substrate)
    monkeypatch.setattr(runner, "load_token_blocks", lambda *a, **k: blocks)
    return Path(cfg)
