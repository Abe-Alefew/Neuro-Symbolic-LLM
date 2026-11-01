"""Run Stage A2 (local-predictor part): residual adapter + FabricPC predictor + stability diagnostics.

Wiring only, mirroring ``run_stage_a1.py``: builds the configs, substrate, adapter, local
predictor bank and stability monitor, then runs the shared loop in ``stages/stage_base.py`` with the
A2 objective. Ordinary backprop; there is no settling anywhere in this stage.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

# Silence verbose C++ runtime warnings before JAX initializes
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run as a script: put the repo root on sys.path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frozenllm.substrate import FrozenSubstrate  # noqa: E402
from residual import ResidualAdapter  # noqa: E402
from residual.stability import (  # noqa: E402
    StabilityMonitor,
)
from stages.results import RunWriter, default_run_name, run_metadata  # noqa: E402
from stages.stage_A.a2 import (  # noqa: E402
    ARMS,
    A2Config,
    a2_audit,
    a2_diagnostics,
    apply_arm,
    audit_to_dict,
    init_a2_params,
    make_a2_objective,
    make_a2_optimizer,
    param_counts,
    predictor_bank,
)
from stages.stage_base import (  # noqa: E402
    Domain,
    RunConfig,
    configure_logging,
    evaluate_checkpoint,
    evaluate_test_domains,
    load_configs,
    load_token_blocks,
    log_experiment_header,
    log_final_summary,
    prepare_domains,
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
    parser.add_argument("--arm", choices=ARMS, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lambda-kl", type=float, default=None)
    parser.add_argument("--lambda-wd", type=float, default=None)
    parser.add_argument("--lambda-cp", type=float, default=None)
    parser.add_argument("--lambda-stab", type=float, default=None)
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
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Prepare the run and evaluate without training.",
    )
    parser.add_argument("--audit", action="store_true", default=False)
    parser.add_argument("--audit-only", action="store_true", default=False)
    return parser.parse_args(argv)


def apply_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    adapter, training, domains = cfg.adapter, cfg.training, cfg.domains
    obj: A2Config = cfg.objective
    if args.arm is not None:
        obj = apply_arm(obj, args.arm)
    if args.seed is not None:
        adapter = replace(adapter, seed=args.seed)
        training = replace(training, seed=args.seed)
        obj = replace(
            obj,
            seed=args.seed,
            hid=replace(obj.hid, predictor=replace(obj.predictor, seed=args.seed)),
        )
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
    obj = replace(
        obj,
        lambda_kl=args.lambda_kl if args.lambda_kl is not None else obj.lambda_kl,
        lambda_wd=args.lambda_wd if args.lambda_wd is not None else obj.lambda_wd,
        cpc=replace(
            obj.cpc,
            weight=args.lambda_cp if args.lambda_cp is not None else obj.cpc.weight,
        ),
        stab=replace(
            obj.stab,
            weight=(
                args.lambda_stab if args.lambda_stab is not None else obj.stab.weight
            ),
        ),
    )
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
    if args.stability_mode == "off":
        obj = replace(obj, stab=replace(obj.stab, weight=0.0))
    if args.eval_only:
        training = replace(
            training,
            epochs=0,
        )
        domains = tuple(replace(d, epochs=0) for d in domains)
    return replace(
        cfg, adapter=adapter, objective=obj, training=training, domains=domains
    )


def main(argv: Sequence[str] | None = None) -> int:
    start_time = time.time()
    configure_logging(level=logging.INFO)
    args = parse_args(argv)
    cfg = apply_overrides(load_configs(args.config, A2Config), args)
    train_cfg = cfg.training
    obj: A2Config = cfg.objective

    substrate = FrozenSubstrate(train_cfg.model)
    arch = substrate.architecture
    max_pos = arch.max_position_embeddings
    if max_pos is not None and train_cfg.seq_len > max_pos:
        raise ValueError(f"seq_len={train_cfg.seq_len} exceeds {max_pos} positions.")

    adapter = ResidualAdapter(cfg.adapter, arch)
    bank = (
        predictor_bank(obj, adapter, train_cfg.seq_len)
        if obj.uses_hidden_pred
        else None
    )
    monitor = (
        StabilityMonitor(obj.stability, adapter.layers)
        if obj.stability.enabled
        else None
    )
    params = init_a2_params(obj, adapter, train_cfg.seq_len)
    counts = param_counts(params)
    n_adapter = counts["adapter"]
    n_pred = counts["pred"]
    n_frozen = sum(p.numel() for p in substrate.params.values())

    frozen = substrate.verify_frozen()
    if not frozen["params_unchanged"]:
        log.error("Substrate not frozen before training: %s", frozen)
        return 1

    if args.eval_checkpoint is not None:
        log.info(
            "Evaluating checkpoint %s on split=%s", args.eval_checkpoint, args.split
        )
        eval_batches = (
            args.eval_batches
            if args.eval_batches is not None
            else train_cfg.eval_batches
        )
        results = evaluate_checkpoint(
            substrate,
            adapter,
            args.eval_checkpoint,
            cfg.domains,
            split=args.split,
            seq_len=train_cfg.seq_len,
            batch_size=train_cfg.batch_size,
            eval_batches=eval_batches,
            jit=train_cfg.jit,
        )
        print(json.dumps(results, indent=2))
        return 0

    # Load data before creating the run directory, so bad datasets fail fast.
    domains = prepare_domains(
        substrate, cfg.domains, train_cfg, loader_fn=load_token_blocks
    )

    audit_data = None
    if args.audit or args.audit_only:
        audit_data = audit_to_dict(
            a2_audit(substrate, adapter, obj, domains[0].eval_set[0])
        )
        if args.audit_only:
            print(json.dumps(audit_data, indent=2))
            return 0 if audit_data["ok"] else 1
        if not audit_data["ok"]:
            print(json.dumps(audit_data, indent=2))
            return 1

    def diagnostics(p: dict, domain: Domain) -> dict[str, float]:
        if bank is None:
            return {}
        return a2_diagnostics(
            substrate,
            adapter,
            bank,
            obj,
            p,
            domain.eval_set,
            jit=train_cfg.jit,
        )

    run_name = args.run_name or default_run_name("a2")
    with RunWriter(train_cfg.output_dir, run_name) as writer:
        writer.write_config(cfg.to_dict())
        if audit_data is not None:
            writer.write_json("audit.json", audit_data)
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
                "trainable_params": counts["total"],
                "trainable_params_by_group": counts,
                "frozen_params": n_frozen,
                "arm": args.arm,
                "seed": train_cfg.seed,
                "epochs": {d.name: d.epochs for d in domains},
                "jit": train_cfg.jit,
            },
        )
        log_experiment_header(
            log,
            cfg=cfg,
            arch=arch,
            adapter=adapter,
            n_trainable=counts["total"],
            n_frozen=n_frozen,
            run_dir=writer.dir,
            domains=domains,
            stage_name="A2",
            extra_info=(
                {"Predictor Sites   ": f"{list(bank.layers)} ({n_pred:,} params)"}
                if bank
                else None
            ),
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
        for domain in domains:
            final_diag[domain.name] = diagnostics(params, domain)
            writer.log(
                "a2_diagnostics",
                when="final",
                domain=domain.name,
                **final_diag[domain.name],
            )
        summary["a2_diagnostics"] = {
            "init": {domains[0].name: init_diag},
            "final": final_diag,
        }

        # Final test-set evaluation across domains for paper-table results
        summary["test"] = evaluate_test_domains(
            substrate,
            adapter,
            params,
            cfg.domains,
            train_cfg,
            eval_batches=args.eval_batches,
            writer=writer,
            loader_fn=load_token_blocks,
        )
        writer.write_json("results.json", summary)

    log_final_summary(log, summary, frozen, start_time=start_time)
    if not frozen["params_unchanged"]:
        log.error("Base parameters changed during training.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
