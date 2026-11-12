from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from benchmark.external_eval import (
    aggregate_lm_rows,
    evaluate_long_range,
    evaluate_symbolic_rankings,
)


def test_lm_aggregation_is_token_weighted() -> None:
    result = aggregate_lm_rows(
        [
            {
                "tokens": 1,
                "loss_sum": 1.0,
                "base_loss_sum": 2.0,
                "kl_sum": 0.1,
                "correct": 1,
            },
            {
                "tokens": 3,
                "loss_sum": 9.0,
                "base_loss_sum": 12.0,
                "kl_sum": 0.9,
                "correct": 0,
            },
        ]
    )
    assert result["loss"] == pytest.approx(2.5)
    assert result["token_accuracy"] == pytest.approx(0.25)


def test_long_range_reports_families_and_distance(tmp_path: Path) -> None:
    path = tmp_path / "probes.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "long-range/v1",
                "items": [
                    {
                        "id": "x",
                        "family": "relation",
                        "distance": "near",
                        "context": "ctx",
                        "question": " q",
                        "choices": [" correct", " wrong"],
                        "answer": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = evaluate_long_range(
        path, lambda prompt, choice: 1.0 if "correct" in choice else 0.0
    )
    assert result["accuracy"] == 1.0
    assert result["by_family"]["relation"]["accuracy"] == 1.0


def test_symbolic_rankings_are_operational(tmp_path: Path) -> None:
    path = tmp_path / "rankings.jsonl"
    path.write_text(
        json.dumps({"retrieved": ["a", "b"], "relevant": ["a"]}) + "\n",
        encoding="utf-8",
    )
    result = evaluate_symbolic_rankings(path, cutoffs=(1, 2))
    assert result["status"] == "measured"
    assert result["precision"]["@1"] == 1.0
