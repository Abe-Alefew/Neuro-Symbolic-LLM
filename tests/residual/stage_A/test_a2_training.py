"""Stage A2 training loop, sequential evaluation, and CLI runner tests.

Covers:
- Shared ``run_sequential`` training loop across domains with A2 objective
- Verification that evaluation ignores all auxiliary heads and predictors
- Exact parity between A2 run (aux off) and standard A1 run step-for-step
- Predictor learning rate multiplier connection to the optimizer and shared trainer
- CLI argument parsing, flags, arm defaults, and overrides
- Audit flags (--audit, --audit-only) and error handling
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from frozenllm.substrate import FrozenSubstrate
from residual import ResidualAdapter, ResidualConfig
from residual.cpc_heads import a_key, q_key
from residual.hep import hep_key
from residual.local_predictor import predictor_key
from stages.results import RunWriter, load_params
from stages.stage_A.a1 import A1Config, a1_objective
from stages.stage_A.a2 import (
    A2Config,
    Check,
    a2_objective,
    adapter_params,
    init_a2_params,
    make_a2_optimizer,
    predictor_params,
)
from stages.stage_base import (
    Objective,
    TrainingConfig,
    evaluate,
    load_configs,
    run_sequential,
)
from tests.residual.stage_A.conftest import (
    BATCH,
    LAYERS,
    NUM_TOKENS,
    SEQ,
    SHIPPED,
    _domain,
    _events,
    cpc_head_params,
    hep_params,
    make_cfg,
    with_random_b,
    with_random_predictors,
)

# --- Multi-Domain Sequential Training Loop ------------------------------------


def test_run_sequential_trains_logs_and_checkpoints_the_whole_tree(
    gpt2: FrozenSubstrate, tmp_path: Path
) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    cfg = make_cfg(
        kl=0.1,
        wd=0.01,
        cpc=0.2,
        stab=0.2,
        hep="lowrank",
        w_hep=0.2,
        hid=0.2,
        detach=False,
        monitor="diagnostic",
    )
    params = with_random_predictors(with_random_b(init_a2_params(cfg, adapter)))
    domains = [_domain("a", 5), _domain("b", 9, num_blocks=6)]
    training = TrainingConfig(
        batch_size=BATCH, learning_rate=1e-2, log_every=1, eval_batches=1
    )

    with RunWriter(tmp_path, "run") as writer:
        final_params, summaries = run_sequential(
            gpt2,
            adapter,
            params,
            domains,
            a2_objective,
            cfg,
            training,
            writer,
            optimizer=make_a2_optimizer(cfg, training.learning_rate),
        )

    # 1. Parameter tree contains all adapter, cpc, hep, and predictor components
    assert set(adapter_params(final_params)) == {f"layer_{L}" for L in LAYERS}
    assert set(cpc_head_params(final_params)) == {
        q_key(5),
        q_key(6),
        a_key(5, 1),
        a_key(6, 1),
    }
    assert set(hep_params(final_params)) == {hep_key(5), hep_key(6)}
    assert set(predictor_params(final_params)) == {
        predictor_key(5),
        predictor_key(6),
    }

    # 2. Logged metrics contain expected auxiliary and diagnostic entries
    train_events = [e for e in _events(tmp_path / "run") if e.get("event") == "train"]
    assert train_events, "expected train events in metrics.jsonl"
    for e in train_events:
        assert "task" in e
        assert "cpc" in e
        assert "stab" in e
        assert "hid" in e
        assert "res_smooth" in e

    # 3. Checkpoints on disk can be restored and contain full parameter tree
    ckpt = load_params(tmp_path / "run" / "b" / "final.npz")
    assert set(ckpt) == set(final_params)
    assert gpt2.params_unchanged()


def test_evaluation_ignores_auxiliary_heads_and_predictors(
    gpt2: FrozenSubstrate,
) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    cfg = make_cfg(
        cpc=0.2,
        stab=0.2,
        hep="lowrank",
        w_hep=0.2,
        hid=0.2,
        detach=False,
    )
    params = with_random_predictors(with_random_b(init_a2_params(cfg, adapter)))
    eval_set = _domain("a", 5).eval_set
    full_eval = evaluate(gpt2, adapter, params, eval_set)
    adapter_only_eval = evaluate(gpt2, adapter, adapter_params(params), eval_set)
    np.testing.assert_allclose(full_eval["loss"], adapter_only_eval["loss"])
    np.testing.assert_allclose(full_eval["accuracy"], adapter_only_eval["accuracy"])


def test_with_aux_terms_off_a_run_is_the_a1_run(
    gpt2: FrozenSubstrate, tmp_path: Path
) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    training = TrainingConfig(batch_size=BATCH, learning_rate=1e-2, log_every=1)
    summaries: list[Any] = []
    trains: list[Any] = []
    runs: list[tuple[str, Objective, Any]] = [
        ("a1", a1_objective, A1Config(lambda_kl=0.1, lambda_wd=0.01)),
        ("a2", a2_objective, make_cfg(kl=0.1, wd=0.01)),
    ]
    for tag, obj, cfg in runs:
        params = init_a2_params(make_cfg(), adapter)
        domains = [_domain("a", 5), _domain("b", 9)]
        with RunWriter(tmp_path, tag) as writer:
            _, s = run_sequential(
                gpt2, adapter, params, domains, obj, cfg, training, writer
            )
        summaries.append(s)
        trains.append([e for e in _events(tmp_path / tag) if e.get("event") == "train"])

    s1, s2 = summaries
    assert len(s1) == len(s2)
    for sa, sb in zip(s1, s2, strict=True):
        np.testing.assert_allclose(sa.mean_eval_loss, sb.mean_eval_loss, rtol=1e-5)
    for e1, e2 in zip(trains[0], trains[1], strict=True):
        np.testing.assert_allclose(e1["loss"], e2["loss"], rtol=1e-5)
        np.testing.assert_allclose(e1["task"], e2["task"], rtol=1e-5)


def test_predictor_lr_mult_produces_different_updates(
    gpt2: FrozenSubstrate,
) -> None:
    adapter = ResidualAdapter(ResidualConfig(rank=4, layers=LAYERS), gpt2.architecture)
    cfg_m1 = make_cfg(hid=0.2, detach=False, predictor_lr_mult=1.0)
    cfg_m2 = make_cfg(hid=0.2, detach=False, predictor_lr_mult=2.0)
    params0 = with_random_predictors(with_random_b(init_a2_params(cfg_m1, adapter)))
    params0_m2 = with_random_predictors(with_random_b(init_a2_params(cfg_m2, adapter)))
    for k in params0:
        for n in params0[k]:
            np.testing.assert_array_equal(
                np.asarray(params0[k][n]), np.asarray(params0_m2[k][n])
            )

    lr = 1e-2
    opt_m1 = make_a2_optimizer(cfg_m1, lr)
    opt_m2 = make_a2_optimizer(cfg_m2, lr)

    rng = np.random.default_rng(42)
    ids = jnp.asarray(rng.integers(0, NUM_TOKENS, (BATCH, SEQ)), dtype=jnp.int32)

    def loss_fn(p: dict, cfg: A2Config) -> jax.Array:
        return a2_objective(
            p, substrate=gpt2, adapter=adapter, config=cfg, input_ids=ids
        )[0]

    grads = jax.grad(lambda p: loss_fn(p, cfg_m1))(params0)

    state_m1 = opt_m1.init(params0)
    updates_m1, _ = opt_m1.update(grads, state_m1, params0)
    out_m1 = optax.apply_updates(params0, updates_m1)

    state_m2 = opt_m2.init(params0)
    updates_m2, _ = opt_m2.update(grads, state_m2, params0)
    out_m2 = optax.apply_updates(params0, updates_m2)

    # 1. Adapter parameters must get identical updates
    for k in adapter_params(params0):
        np.testing.assert_allclose(
            np.asarray(out_m1[k]["B"]),
            np.asarray(out_m2[k]["B"]),
            atol=1e-7,
            err_msg=f"adapter key {k} should have identical updates",
        )

    # 2. Predictor parameters must differ by ~2.0x
    pred_keys = list(predictor_params(params0))
    assert pred_keys, "expected predictor parameters in tree"
    for k in pred_keys:
        for n in params0[k]:
            delta_m1 = np.asarray(out_m1[k][n]) - np.asarray(params0[k][n])
            delta_m2 = np.asarray(out_m2[k][n]) - np.asarray(params0[k][n])
            mask = np.abs(delta_m1) > 1e-12
            if not mask.any():
                continue
            ratio = np.abs(delta_m2[mask]) / np.abs(delta_m1[mask])
            np.testing.assert_allclose(
                ratio,
                2.0,
                atol=1e-4,
                err_msg=f"predictor key {k}/{n}: expected update ratio ~2.0",
            )


# --- CLI Runner & Execution Flags (experiments/run_stage_a2.py) ---------------


def test_runner_overrides_reach_the_objective(runner: Any) -> None:
    cfg = load_configs(SHIPPED, A2Config)
    args = runner.parse_args(
        ["--epochs", "2", "--rank", "4", "--lambda-cp", "0.5", "--lambda-stab", "0"]
    )
    out = runner.apply_overrides(cfg, args)
    assert out.training.epochs == 2 and out.adapter.rank == 4
    assert out.objective.cpc.weight == 0.5 and out.objective.stab.weight == 0.0


def test_runner_arm_and_seed_overrides(runner: Any) -> None:
    cfg = load_configs(SHIPPED, A2Config)
    args = runner.parse_args(["--arm", "cpc_multi", "--seed", "7"])
    out = runner.apply_overrides(cfg, args)
    assert out.objective.cpc.offsets == (1, 2, 4) and not out.objective.stab.enabled
    assert out.adapter.seed == out.training.seed == out.objective.seed == 7
    assert (
        out.adapter.residual_seed
        == out.training.seed + out.adapter.residual_seed - out.training.seed
    )


def test_runner_predictor_and_stability_overrides(runner: Any) -> None:
    cfg = load_configs(SHIPPED, A2Config)
    args = runner.parse_args(
        [
            "--lambda-hid",
            "0.4",
            "--hidden-target",
            "base_anchor",
            "--ff-dim",
            "8",
            "--no-detach-input",
            "--predictor-lr-mult",
            "2",
            "--stability-mode",
            "off",
        ]
    )
    out = runner.apply_overrides(cfg, args)
    assert out.objective.hid.weight == 0.4
    assert out.objective.hid.hidden_target == "base_anchor"
    assert out.objective.hid.predictor.ff_dim == 8
    assert out.objective.hid.predictor.detach_input is False
    assert out.objective.hid.predictor_lr_mult == 2.0
    assert out.objective.stab.weight == 0.0


def test_runner_main_end_to_end(
    runner: Any,
    gpt2: FrozenSubstrate,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "adapter: {rank: 4, layers: [5, 6]}\n"
        "objective:\n  lambda_kl: 0.1\n"
        "  cpc: {weight: 0.2, dim: 8}\n"
        "  stab: {weight: 0.2, hep: {kind: fixed_random, rank: 4}}\n"
        "  hid: {weight: 0.2, predictor: {ff_dim: 16}}\n"
        f"training: {{model: gpt2, seq_len: {SEQ}, batch_size: 2, epochs: 1, "
        "learning_rate: 1.0e-2, eval_batches: 1, log_every: 1}\n"
        "domains:\n  - {name: a, dataset: x}\n"
    )
    blocks = np.random.default_rng(0).integers(0, NUM_TOKENS, (4, SEQ)).astype(np.int32)
    monkeypatch.setattr(runner, "FrozenSubstrate", lambda _name: gpt2)
    monkeypatch.setattr(runner, "load_token_blocks", lambda *a, **k: blocks)

    code = runner.main(
        [
            "--config",
            str(cfg_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "a2_run",
            "--eval-only",
        ]
    )
    assert code == 0
    run_dir = tmp_path / "runs" / "a2_run"
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()


def test_audit_only_prints_json_and_does_not_train(
    runner: Any, setup: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    code = runner.main(
        [
            "--config",
            str(setup),
            "--audit-only",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is True and any(c["name"] == "a1_parity" for c in data["checks"])
    assert not (tmp_path / "runs").exists()


def test_audit_before_training_writes_audit_json(
    runner: Any, setup: Path, tmp_path: Path
) -> None:
    code = runner.main(
        [
            "--config",
            str(setup),
            "--audit",
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "r",
        ]
    )
    assert code == 0
    run = tmp_path / "runs" / "r"
    assert json.loads((run / "audit.json").read_text())["ok"] is True
    assert (run / "summary.json").is_file()


def test_a_failed_audit_stops_before_training(
    runner: Any, setup: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner, "a2_audit", lambda *a, **k: [Check("boom", False, 0.0, "bad")]
    )
    code = runner.main(
        [
            "--config",
            str(setup),
            "--audit",
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "r",
        ]
    )
    assert code == 1
    assert not (tmp_path / "runs" / "r" / "summary.json").exists()


def test_no_audit_flags_means_no_audit(
    runner: Any, setup: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def must_not_run(*a: Any, **k: Any) -> None:
        raise AssertionError("audit ran without a flag")

    monkeypatch.setattr(runner, "a2_audit", must_not_run)
    code = runner.main(
        [
            "--config",
            str(setup),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "r",
        ]
    )
    assert code == 0
