"""Evaluate A4 symbolic retrieval rankings against labeled relevant templates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.external_eval import evaluate_symbolic_rankings  # noqa: E402
from stages.results import RunWriter, default_run_name, run_metadata  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_A" / "a4_symbolic_head.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    evaluation = config.get("symbolic_evaluation", {})
    cutoffs = tuple(int(value) for value in evaluation.get("cutoffs", (1, 4, 8, 16)))
    results = evaluate_symbolic_rankings(args.rankings, cutoffs=cutoffs)
    run_name = args.run_name or default_run_name("a4-symbolic")
    with RunWriter(args.output_dir, run_name) as writer:
        writer.write_config(config)
        writer.write_json(
            "meta.json",
            {**run_metadata(), "stage": "A4", "rankings": str(args.rankings)},
        )
        writer.write_json("results.json", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
