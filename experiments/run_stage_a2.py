"""Run Stage A2 (local-predictor part): residual adapter + FabricPC predictor + stability diagnostics.

Wiring only, mirroring ``run_stage_a1.py``: builds the configs, substrate, adapter, local
predictor bank and stability monitor, then runs the shared loop in ``stages/stage_base.py`` with the
A2 objective. Ordinary backprop; there is no settling anywhere in this stage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run as a script: put the repo root on sys.path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frozenllm.substrate import FrozenSubstrate  # noqa: E402
from residual import ResidualAdapter  # noqa: E402
from residual.stability import (  # noqa: E402
    StabilityMonitor,
)
from residual.local_predictor import (  # noqa: E402
    LocalPredictorBank,
    merge_params,
    split_params,
)
from stages.results import RunWriter, default_run_name, run_metadata  # noqa: E402
from stages.stage_A.a2 import (  # noqa: E402
    A2Config,
    a2_diagnostics,
    make_a2_objective,
    make_a2_optimizer,
)
from stages.stage_base import (  # noqa: E402
    Domain,
    RunConfig,
    evaluate,
    evaluate_checkpoint,
    load_configs,
    load_token_blocks,
    make_eval_set,
    run_sequential,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_A" / "a2_pc_inspired.yaml"

log = logging.getLogger("stage_a2")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None, help="Adapter rank override.")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Epochs for every domain (smoke runs)."
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        default=None,
        help="Evaluate a saved checkpoint instead of training.",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--no-jit", action="store_true", default=False)
    # Local predictor
    parser.add_argument("--lambda-hid", type=float, default=None)
    parser.add_argument(
        "--hidden-target", choices=("corrected_teacher", "base_anchor"), default=None
    )
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument(
        "--detach-input",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stop L_hidden-pred gradients into the adapter.",
    )
    parser.add_argument(
        "--predict-delta", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--predictor-lr-mult", type=float, default=None)
    # Stability
    parser.add_argument("--stability-mode", choices=("off", "diagnostic"), default=None)
    return parser.parse_args(argv)


def apply_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    adapter, training, domains = cfg.adapter, cfg.training, cfg.domains
    obj: A2Config = cfg.objective
    if args.rank is not None:
        adapter = replace(adapter, rank=args.rank)
    if args.model is not None:
        training = replace(training, model=args.model)
    if args.output_dir is not None:
        training = replace(training, output_dir=args.output_dir)
    if args.no_jit:
        training = replace(training, jit=False)
    if args.eval_batches is not None:
        training = replace(training, eval_batches=args.eval_batches)
    if args.epochs is not None:
        if args.epochs < 0:
            raise ValueError(f"epochs must be >= 0, got {args.epochs}")
        training = replace(training, epochs=args.epochs)
        domains = tuple(replace(d, epochs=args.epochs) for d in domains)

    hid = {
        k: v
        for k, v in {
            "lambda_hid": args.lambda_hid,
            "hidden_target": args.hidden_target,
            "predictor_lr_mult": args.predictor_lr_mult,
        }.items()
        if v is not None
    }
    pred = {
        k: v
        for k, v in {
            "ff_dim": args.ff_dim,
            "detach_input": args.detach_input,
            "predict_delta": args.predict_delta,
        }.items()
        if v is not None
    }
    stab = {
        k: v
        for k, v in {
            "mode": args.stability_mode,
        }.items()
        if v is not None
    }
    obj = replace(
        obj,
        hid=replace(
            obj.hid,
            **hid,
            predictor=replace(obj.hid.predictor, **pred),
        ),
        monitor=replace(obj.monitor, **stab),
    )
    return replace(cfg, adapter=adapter, objective=obj, training=training, domains=domains)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args(argv)
    cfg = apply_overrides(load_configs(args.config, A2Config), args)
    train_cfg = cfg.training
    obj: A2Config = cfg.objective
    log.info("adapter=%s", cfg.adapter)
    log.info("objective=%s", obj)
    log.info("training=%s", train_cfg)
    log.info("domains=%s", [d.name for d in cfg.domains])

    substrate = FrozenSubstrate(train_cfg.model)
    arch = substrate.architecture
    max_pos = arch.max_position_embeddings
    if max_pos is not None and train_cfg.seq_len > max_pos:
        raise ValueError(f"seq_len={train_cfg.seq_len} exceeds {max_pos} positions.")

    adapter = ResidualAdapter(cfg.adapter, arch)
    bank = (
        LocalPredictorBank(obj.predictor, arch, train_cfg.seq_len, adapter.layers)
        if obj.uses_hidden_pred
        else None
    )
    monitor = (
        StabilityMonitor(obj.stability, adapter.layers)
        if obj.stability.enabled
        else None
    )
    adapter_params = adapter.init_params()
    predictor_params = bank.init_params() if bank is not None else {}
    params = merge_params(adapter_params, predictor_params)
    n_adapter = ResidualAdapter.num_params(adapter_params)
    n_pred = LocalPredictorBank.num_params(predictor_params)
    n_frozen = sum(p.numel() for p in substrate.params.values())
    log.info(
        "model=%s L=%d d=%d adapter_layers=%s predictor_layers=%s rank=%d "
        "adapter_params=%d predictor_params=%d frozen=%d",
        train_cfg.model,
        arch.num_layers,
        arch.hidden_size,
        list(adapter.layers),
        list(bank.layers) if bank else [],
        cfg.adapter.rank,
        n_adapter,
        n_pred,
        n_frozen,
    )

    frozen = substrate.verify_frozen()
    if not frozen["params_unchanged"]:
        log.error("Substrate not frozen before training: %s", frozen)
        return 1

    if args.eval_checkpoint is not None:
        log.info("Evaluating checkpoint %s on split=%s", args.eval_checkpoint, args.split)
        results = evaluate_checkpoint(
            substrate,
            adapter,
            args.eval_checkpoint,
            cfg.domains,
            split=args.split,
            seq_len=train_cfg.seq_len,
            batch_size=train_cfg.batch_size,
            eval_batches=train_cfg.eval_batches,
            jit=train_cfg.jit,
        )
        print(json.dumps(results, indent=2))
        return 0

    # Load data before creating the run directory, so bad datasets fail fast.
    domains = []
    for d in cfg.domains:
        train_blocks = load_token_blocks(substrate, d, d.train_split, train_cfg.seq_len)
        eval_blocks = load_token_blocks(substrate, d, d.eval_split, train_cfg.seq_len)
        domains.append(
            Domain(
                name=d.name,
                train_blocks=train_blocks,
                eval_set=make_eval_set(
                    eval_blocks, train_cfg.batch_size, train_cfg.eval_batches
                ),
                epochs=train_cfg.epochs if d.epochs is None else d.epochs,
            )
        )
        log.info(
            "domain=%s train_blocks=%d eval_batches=%d",
            d.name,
            len(train_blocks),
            len(domains[-1].eval_set),
        )

    def diagnostics(p: dict, domain: Domain) -> dict[str, float]:
        if bank is None:
            return {}
        return a2_diagnostics(
            substrate, adapter, bank, obj, p,
            domain.eval_set, jit=train_cfg.jit,
        )

    run_name = args.run_name or default_run_name("a2")
    with RunWriter(train_cfg.output_dir, run_name) as writer:
        log.info("run_dir=%s", writer.dir)
        writer.write_config(cfg.to_dict())
        writer.write_json(
            "meta.json",
            {
                **run_metadata(),
                "stage": "A2",
                "model": train_cfg.model,
                "num_layers": arch.num_layers,
                "hidden_size": arch.hidden_size,
                "adapted_layers": list(adapter.layers),
                "predictor_layers": list(bank.layers) if bank else [],
                "adapter_params": n_adapter,
                "predictor_params": n_pred,
                "frozen_params": n_frozen,
                "epochs": {d.name: d.epochs for d in domains},
                "jit": train_cfg.jit,
            },
        )
        init_diag = diagnostics(params, domains[0])
        writer.log("a2_diagnostics", when="init", domain=domains[0].name, **init_diag)

        params, summary = run_sequential(
            substrate,
            adapter,
            params,
            domains,
            make_a2_objective(bank, monitor),
            obj,
            train_cfg,
            writer,
            optimizer=make_a2_optimizer(obj, train_cfg.learning_rate),
        )
        frozen = substrate.verify_frozen()
        summary["params_unchanged"] = frozen["params_unchanged"]

        final_diag = {}
        for d in domains:
            final_diag[d.name] = diagnostics(params, d)
            writer.log("a2_diagnostics", when="final", domain=d.name, **final_diag[d.name])
        summary["a2_diagnostics"] = {
            "init": {domains[0].name: init_diag},
            "final": final_diag,
        }

        # Final test-set evaluation across domains for paper-table results
        test_results = {}
        adapter_only, _ = split_params(params)
        for d in cfg.domains:
            test_blocks = load_token_blocks(substrate, d, d.test_split, train_cfg.seq_len)
            test_set = make_eval_set(test_blocks, train_cfg.batch_size, train_cfg.eval_batches)
            test_results[d.name] = evaluate(
                substrate, adapter, adapter_only, test_set, jit=train_cfg.jit
            )
            writer.log("test", domain=d.name, **test_results[d.name])
        summary["test"] = test_results
        writer.write_json("results.json", summary)

    log.info("mean_seen_loss=%s", summary["mean_seen_loss"])
    log.info("mean_forgetting=%s", summary["mean_forgetting"])
    log.info("test_results=%s", summary["test"])
    log.info("verify_frozen=%s", frozen)
    if not frozen["params_unchanged"]:
        log.error("Base parameters changed during training.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())