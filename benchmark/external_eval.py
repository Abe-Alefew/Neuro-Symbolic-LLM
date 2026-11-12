"""Model-agnostic aggregation for external benchmark suites."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from metrics.symbolic import retrieval_report


def aggregate_lm_rows(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        return {"status": "insufficient_data", "tokens": 0}
    tokens = sum(int(row["tokens"]) for row in rows)
    if tokens <= 0:
        return {"status": "insufficient_data", "tokens": 0}
    loss = sum(float(row["loss_sum"]) for row in rows) / tokens
    base_loss = sum(float(row["base_loss_sum"]) for row in rows) / tokens
    return {
        "status": "measured",
        "tokens": tokens,
        "loss": loss,
        "base_loss": base_loss,
        "perplexity": float(np.exp(loss)),
        "base_perplexity": float(np.exp(base_loss)),
        "kl": sum(float(row["kl_sum"]) for row in rows) / tokens,
        "token_accuracy": sum(float(row["correct"]) for row in rows) / tokens,
    }


def evaluate_long_range(
    path: Path,
    score_choice: Callable[[str, str], float],
) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if config.get("schema") != "long-range/v1":
        raise ValueError("long-range probe file must use schema 'long-range/v1'")
    items = config.get("items", [])
    if not items:
        return {"status": "not_configured", "reason": "no long-range probes"}
    rows = []
    for item in items:
        required = {
            "id",
            "family",
            "distance",
            "context",
            "question",
            "choices",
            "answer",
        }
        missing = required - set(item)
        if missing:
            raise ValueError(f"long-range item is missing {sorted(missing)}")
        if not isinstance(item["choices"], list) or len(item["choices"]) < 2:
            raise ValueError("long-range item requires at least two choices")
        filler = str(item.get("filler", ""))
        repeat = int(item.get("filler_repeat", 0))
        if repeat < 0:
            raise ValueError("filler_repeat must be >= 0")
        prompt = str(item["context"]) + (filler * repeat) + str(item["question"])
        scores = [float(score_choice(prompt, choice)) for choice in item["choices"]]
        answer = int(item["answer"])
        predicted = int(np.argmax(scores))
        distractors = [score for index, score in enumerate(scores) if index != answer]
        rows.append(
            {
                "id": str(item["id"]),
                "family": str(item["family"]),
                "distance": str(item["distance"]),
                "correct": predicted == answer,
                "margin": scores[answer] - max(distractors),
                "scores": scores,
            }
        )
    by_family = {}
    for family in sorted({row["family"] for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        by_family[family] = {
            "accuracy": float(np.mean([row["correct"] for row in selected])),
            "mean_margin": float(np.mean([row["margin"] for row in selected])),
        }
    by_distance = {}
    for distance in sorted({row["distance"] for row in rows}):
        selected = [row for row in rows if row["distance"] == distance]
        by_distance[distance] = float(np.mean([row["correct"] for row in selected]))
    return {
        "status": "measured",
        "accuracy": float(np.mean([row["correct"] for row in rows])),
        "mean_margin": float(np.mean([row["margin"] for row in rows])),
        "by_family": by_family,
        "by_distance": by_distance,
        "near_far_gap": (
            by_distance.get("near", np.nan) - by_distance.get("far", np.nan)
        ),
        "items": rows,
    }


def evaluate_symbolic_rankings(
    path: Path | None, cutoffs=(1, 4, 8, 16)
) -> dict[str, Any]:
    if path is None:
        return {
            "status": "not_applicable",
            "reason": "A1/A2/A3 have no symbolic retrieval head",
        }
    retrieved = []
    relevant = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            values = list(row["retrieved"])
            if len(values) != len(set(values)):
                raise ValueError("symbolic rankings must contain unique IDs")
            retrieved.append(values)
            relevant.append(set(row["relevant"]))
    return {"status": "measured", **retrieval_report(retrieved, relevant, cutoffs)}


__all__ = ["aggregate_lm_rows", "evaluate_long_range", "evaluate_symbolic_rankings"]
