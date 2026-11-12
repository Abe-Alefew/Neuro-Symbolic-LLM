from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.provenance import (
    effective_condition,
    finalize_attempt,
    next_attempt_name,
    sha256_file,
    validate_complete_attempt,
)


def _condition(**overrides):
    values = {
        "method": "a3",
        "seed": 0,
        "rank": 16,
        "k": 4,
        "protocol": "sequential",
        "task_order": ("eurlex", "ledgar", "scotus"),
        "heldout_tasks": (),
        "a2_arm": None,
        "stage_config_sha256": "a" * 64,
        "runtime_config_sha256": "b" * 64,
        "runtime_source": "separate",
        "epochs": 3,
        "source_sha256": "c" * 64,
    }
    values.update(overrides)
    return effective_condition(**values)


@pytest.mark.parametrize(
    "override",
    [
        {"method": "a2", "a2_arm": "full"},
        {"seed": 1},
        {"rank": 8},
        {"k": 8},
        {"protocol": "joint"},
        {"task_order": ("scotus", "ledgar", "eurlex")},
        {"heldout_tasks": ("scotus",)},
        {"stage_config_sha256": "d" * 64},
        {"runtime_config_sha256": "e" * 64},
        {"runtime_source": "stories"},
        {"epochs": 4},
        {"source_sha256": "f" * 64},
    ],
)
def test_effective_condition_hashes_every_scientific_axis(override) -> None:
    base = _condition()
    assert base["condition_sha256"] != _condition(**override)["condition_sha256"]


def test_attempts_are_append_only(tmp_path: Path) -> None:
    assert next_attempt_name(tmp_path) == "attempt-0001"
    (tmp_path / "attempt-0001").mkdir()
    assert next_attempt_name(tmp_path) == "attempt-0002"


def test_completion_validation_rejects_mutation(tmp_path: Path) -> None:
    run = tmp_path / "attempt-0001"
    run.mkdir()
    (run / "results.json").write_text("{}\n", encoding="utf-8")
    (run / "config.yaml").write_text("adapter: {}\n", encoding="utf-8")
    provenance = run / "provenance"
    provenance.mkdir()
    (provenance / "materialized_stage.yaml").write_text(
        "adapter: {}\n", encoding="utf-8"
    )
    (provenance / "materialized_runtime.yaml").write_text(
        "runtime: {}\n", encoding="utf-8"
    )
    condition = _condition()
    spec = {
        **condition,
        "config_sha256": sha256_file(provenance / "materialized_stage.yaml"),
        "runtime_config_sha256": sha256_file(provenance / "materialized_runtime.yaml"),
    }
    finalize_attempt(run, spec)
    valid, reason = validate_complete_attempt(
        run,
        condition_sha256=condition["condition_sha256"],
        config_sha256=spec["config_sha256"],
    )
    assert valid and reason is None
    (run / "results.json").write_text(json.dumps({"changed": True}), encoding="utf-8")
    valid, reason = validate_complete_attempt(
        run,
        condition_sha256=condition["condition_sha256"],
        config_sha256=spec["config_sha256"],
    )
    assert not valid
    assert "hash mismatch" in str(reason)
