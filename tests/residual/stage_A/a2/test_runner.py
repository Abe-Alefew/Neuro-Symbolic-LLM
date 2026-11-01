"""Stage A2 command line runner and CLI audit flag integration tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frozenllm.substrate import FrozenSubstrate
from stages.config import load_configs
from stages.stage_A.a2 import A2Config, Check
from tests.residual.stage_A.a2.conftest import NUM_TOKENS, SEQ, SHIPPED


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
