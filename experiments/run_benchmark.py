"""Thin CLI for the compact Stage A A1/A2/A3 benchmark harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.core import (  # noqa: E402
    evaluate_manifest,
    make_manifest,
    report_manifest,
    train_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "train", "eval", "report"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("a1", "a2", "a3"),
        default=["a1", "a2", "a3"],
    )
    parser.add_argument("--k", nargs="+", type=int, default=[4])
    parser.add_argument("--ranks", nargs="+", type=int, default=[16])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.command == "plan":
        manifest = make_manifest(
            output_dir,
            seeds=args.seeds,
            methods=args.methods,
            k_values=args.k,
            rank_values=args.ranks,
            epochs=args.epochs,
        )
        print(f"planned {len(manifest['runs'])} runs: {output_dir / 'manifest.json'}")
    elif args.command == "train":
        train_manifest(output_dir, resume=not args.no_resume, epochs=args.epochs)
        print(f"training complete: {output_dir}")
    elif args.command == "eval":
        evaluation = evaluate_manifest(output_dir)
        print(f"evaluated {evaluation['completed_runs']}/{evaluation['planned_runs']} runs")
    else:
        print(report_manifest(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
