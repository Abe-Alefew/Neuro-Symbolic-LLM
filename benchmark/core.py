"""Small, artifact-driven A1/A2/A3 benchmark harness.

The stage runners remain the source of truth for training. This module only
coordinates isolated runs and aggregates their existing JSON artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = {
    "a1": ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml",
    "a2": ROOT / "configs" / "stage_A" / "a2_pc_inspired.yaml",
    "a3": ROOT / "configs" / "stage_A" / "a3_true_pc.yaml",
}
RUNNERS = {
    "a1": ROOT / "experiments" / "run_stage_a1.py",
    "a2": ROOT / "experiments" / "run_stage_a2.py",
    "a3": ROOT / "experiments" / "run_stage_a3.py",
}
EXPECTED_DOMAINS = ("eurlex", "ledgar", "scotus")
MEASURED_METRICS = (
    "loss_matrix",
    "perplexity",
    "mean_forgetting",
    "optimizer_steps",
    "tokens",
    "mean_step_time",
    "estimated_tokens_per_second",
    "params_unchanged",
    "residual_compressibility",
    "effective_rank",
    "invariant_redundancy",
    "fracture_index",
    "overlap",
    "taylor_delta",
    "synergy",
    "token_accuracy",
    "expected_calibration_error",
    "paired_seed_delta",
)


@dataclass(frozen=True)
class RunSpec:
    method: str
    seed: int
    protocol: str
    run_id: str
    config: str
    output_dir: str
    run_name: str
    k: int | None = None
    rank: int = 16


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_deployed_metrics(run: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the final checkpoint with the common deployed function.

    Heavy JAX/model imports are intentionally local so ``plan`` remains usable
    in lightweight environments.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from frozenllm.substrate import FrozenSubstrate, identity_modify
    from metrics.performance import (
        expected_calibration_error,
        kl_to_base,
        token_accuracy,
    )
    from metrics.representation import (
        effective_rank,
        fracture_index,
        invariant_redundancy,
        overlap,
        synergy,
        taylor_delta,
    )
    from residual import ResidualAdapter, ResidualConfig
    from stages.results import load_params
    from stages.stage_base import (
        DomainConfig,
        TrainingConfig,
        load_token_blocks,
        make_eval_set,
    )

    config_path = Path(run.get("materialized_config", run["config"]))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    training = TrainingConfig.from_dict(config["training"])
    substrate = FrozenSubstrate(training.model)
    adapter = ResidualAdapter(
        ResidualConfig.from_dict(config["adapter"]), substrate.architecture
    )
    checkpoint_dir = Path(run["output_dir"]) / run["run_name"] / "params"
    checkpoints = sorted(checkpoint_dir.glob("task_*.npz"))
    if not checkpoints:
        return {"status": "unavailable", "reason": "no checkpoint found"}
    params = {
        layer: {name: jnp.asarray(value) for name, value in values.items()}
        for layer, values in load_params(checkpoints[-1]).items()
    }
    adapter.validate_params(params)
    domains = [DomainConfig.from_dict(item) for item in config["domains"]]
    metrics_by_domain: dict[str, dict[str, float]] = {}
    batches_by_domain: dict[str, tuple[jax.Array, ...]] = {}
    latency: list[float] = []

    def deployed(ids: jax.Array) -> tuple[jax.Array, jax.Array, Any]:
        if run["method"] != "a3":
            base_result = substrate.run_with_interception(
                ids, modify_fn=identity_modify, intercept_layers=adapter.layers
            )
            adapted_result = substrate.run_with_interception(
                ids, modify_fn=adapter.modify_fn(params), intercept_layers=adapter.layers
            )
            return base_result.logits, adapted_result.logits, adapted_result
        clean = substrate.run_with_interception(
            ids, modify_fn=identity_modify, intercept_layers=adapter.layers
        )
        latent = {
            layer: adapter.encode(params[f"layer_{layer}"], clean.hidden_state(layer))
            for layer in adapter.layers
        }

        def modify(hidden: jax.Array, layer: int) -> jax.Array:
            block = params.get(f"layer_{layer}")
            if block is None:
                return hidden
            return hidden + adapter.scale * (latent[layer] @ block["B"]).astype(hidden.dtype)

        adapted_result = substrate.run_with_interception(
            ids, modify_fn=modify, intercept_layers=adapter.layers
        )
        return clean.logits, adapted_result.logits, adapted_result

    def loss_for(p: dict[str, dict[str, jax.Array]], ids: jax.Array) -> jax.Array:
        if run["method"] != "a3":
            adapted = substrate.run_with_interception(
                ids, modify_fn=adapter.modify_fn(p), intercept_layers=adapter.layers
            ).logits
        else:
            clean = substrate.run_with_interception(
                ids, modify_fn=identity_modify, intercept_layers=adapter.layers
            )
            latent = {
                layer: adapter.encode(p[f"layer_{layer}"], clean.hidden_state(layer))
                for layer in adapter.layers
            }

            def modify(hidden: jax.Array, layer: int) -> jax.Array:
                block = p.get(f"layer_{layer}")
                if block is None:
                    return hidden
                return hidden + adapter.scale * (latent[layer] @ block["B"]).astype(hidden.dtype)

            adapted = substrate.run_with_interception(
                ids, modify_fn=modify, intercept_layers=adapter.layers
            ).logits
        return substrate.compute_loss(adapted, ids, losses=("CE",))

    for domain in domains:
        blocks = load_token_blocks(
            substrate, domain, domain.test_split, training.seq_len
        )
        batches = make_eval_set(blocks, training.batch_size, training.eval_batches)
        batches_by_domain[domain.name] = batches
        rows = []
        for ids in batches:
            start = time.perf_counter()
            base, adapted, _ = deployed(ids)
            jax.block_until_ready(adapted)
            latency.append(time.perf_counter() - start)
            rows.append(
                (
                    float(substrate.compute_loss(adapted, ids, losses=("CE",))),
                    float(token_accuracy(adapted, ids)),
                    float(expected_calibration_error(adapted, ids)),
                    float(kl_to_base(base, adapted)),
                )
            )
        values = np.asarray(rows, dtype=float)
        metrics_by_domain[domain.name] = {
            "loss": float(values[:, 0].mean()),
            "ppl": float(np.exp(values[:, 0].mean())),
            "token_accuracy": float(values[:, 1].mean()),
            "ece": float(values[:, 2].mean()),
            "kl": float(values[:, 3].mean()),
        }

    gradients = {}
    for name, batches in batches_by_domain.items():
        if not batches:
            continue
        gradients[name] = jax.grad(
            lambda p, batch=batches[0]: loss_for(p, batch)
        )(params)
    conflict = {}
    names = list(gradients)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            from metrics.representation import gradient_conflict

            conflict[f"{left}:{right}"] = float(
                gradient_conflict(gradients[left], gradients[right])
            )

    representation: dict[str, Any] = {
        "effective_rank": {},
        "invariant_redundancy": {},
        "fracture_index": {},
        "overlap": {},
        "taylor_delta": {},
        "synergy": {},
        "cross_domain_conflict": conflict,
    }
    ablation_by_domain: dict[str, dict[str, float]] = {}
    for layer_name, block in params.items():
        if not layer_name.startswith("layer_"):
            continue
        a, b = block["A"], block["B"]
        representation["effective_rank"][layer_name] = float(effective_rank(a @ b))
        representation["invariant_redundancy"][layer_name] = float(
            invariant_redundancy(a, b)
        )
        atom_effects: dict[str, float] = {}
        for atom in range(int(b.shape[0])):
            ablated = {key: dict(value_map) for key, value_map in params.items()}
            ablated[layer_name]["B"] = b.at[atom].set(jnp.zeros_like(b[atom]))
            effects = []
            for domain_name, batches in batches_by_domain.items():
                if batches:
                    full = float(loss_for(params, batches[0]))
                    ablated_loss = float(loss_for(ablated, batches[0]))
                    effects.append((domain_name, ablated_loss - full))
            atom_effects[str(atom)] = float(np.mean([effect for _, effect in effects])) if effects else 0.0
        contributions = jnp.asarray([abs(value) for value in atom_effects.values()])
        representation["fracture_index"][layer_name] = float(fracture_index(contributions))
        layer_gradients = [
            gradients[name][layer_name]["B"]
            for name in names
            if layer_name in gradients[name]
        ]
        if layer_gradients:
            mean_gradient = jnp.mean(jnp.stack(layer_gradients), axis=0)
            fisher = mean_gradient**2
            atom_taylor = [
                taylor_delta(mean_gradient[i], fisher[i], -b[i])
                for i in range(int(b.shape[0]))
            ]
            atom_synergy = [
                synergy(
                    -b[i],
                    -b[(i + 1) % int(b.shape[0])],
                    fisher[i],
                )
                for i in range(int(b.shape[0]))
            ]
        else:
            atom_taylor = [jnp.asarray(0.0)]
            atom_synergy = [jnp.asarray(0.0)]
        representation["taylor_delta"][layer_name] = float(
            jnp.sum(jnp.asarray(atom_taylor))
        )
        representation["synergy"][layer_name] = float(
            jnp.mean(jnp.asarray(atom_synergy))
        )
        ablation_by_domain[layer_name] = atom_effects
    layers = list(ablation_by_domain)
    for i, left in enumerate(layers):
        for right in layers[i + 1 :]:
            representation["overlap"][f"{left}:{right}"] = float(
                overlap(
                    jnp.asarray(list(ablation_by_domain[left].values())),
                    jnp.asarray(list(ablation_by_domain[right].values())),
                )
            )
    peak_bytes = None
    for device in jax.devices():
        stats = device.memory_stats()
        if stats:
            peak_bytes = max(peak_bytes or 0, stats.get("peak_bytes_in_use", 0))
    return {
        "status": "measured",
        "domains": metrics_by_domain,
        "representation": representation,
        "ops": {
            "inference_latency_mean_s": float(np.mean(latency)) if latency else None,
            "inference_latency_p95_s": float(np.percentile(latency, 95)) if latency else None,
            "peak_memory_bytes": peak_bytes,
        },
        "anchor_kl": {"status": "unavailable", "reason": "anchor dataset not configured"},
        "heldout_transfer": {
            "status": "unavailable",
            "reason": "held-out datasets not configured",
        },
        "joint_gap": {"status": "unavailable", "reason": "joint protocol not implemented"},
        "symbolic_retrieval": {"status": "excluded", "reason": "A4 not implemented"},
    }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _config_fingerprint(config: dict[str, Any]) -> str:
    return _sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _compressibility_curve(deltas: dict[int, float]) -> dict[int, float]:
    reference = deltas[max(deltas)]
    return {rank: value / (reference + 1e-8) for rank, value in deltas.items()}


def _set_seed(config: dict[str, Any], method: str, seed: int) -> None:
    config.setdefault("adapter", {})["seed"] = seed
    config.setdefault("training", {})["seed"] = seed
    if method == "a2":
        config.setdefault("objective", {})["seed"] = seed
        hid = config["objective"].get("hid")
        if isinstance(hid, dict):
            predictor = hid.get("predictor")
            if isinstance(predictor, dict):
                predictor["seed"] = seed


def _set_epochs(config: dict[str, Any], epochs: int | None) -> None:
    if epochs is None:
        return
    config.setdefault("training", {})["epochs"] = epochs
    for domain in config.get("domains", []):
        if isinstance(domain, dict):
            domain["epochs"] = epochs


def _set_protocol(config: dict[str, Any], protocol: str) -> None:
    if protocol != "sequential":
        raise ValueError(
            "The current stage runners implement sequential training only; "
            "joint protocol support is intentionally not fabricated."
        )


def _set_rank(config: dict[str, Any], rank: int) -> None:
    if rank < 1:
        raise ValueError("rank must be >= 1")
    config.setdefault("adapter", {})["rank"] = rank


def _validate_domains(config: dict[str, Any], source: Path) -> None:
    domains = config.get("domains")
    names = tuple(
        domain.get("name")
        for domain in domains
        if isinstance(domain, dict)
    ) if isinstance(domains, list) else ()
    if names != EXPECTED_DOMAINS:
        raise ValueError(
            f"{source} must define domains in order {list(EXPECTED_DOMAINS)}, "
            f"got {list(names)}"
        )


def expand_specs(
    output_dir: Path,
    *,
    seeds: list[int],
    methods: list[str],
    k_values: list[int],
    rank_values: list[int] | None = None,
    epochs: int | None = None,
    protocol: str = "sequential",
) -> list[RunSpec]:
    if not seeds:
        raise ValueError("At least one seed is required.")
    unknown = sorted(set(methods) - set(DEFAULT_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; choose from {sorted(DEFAULT_CONFIGS)}")
    if not k_values or any(k < 0 for k in k_values):
        raise ValueError("k_values must contain non-negative integers.")
    ranks = rank_values or [16]
    if not ranks or any(rank < 1 for rank in ranks):
        raise ValueError("rank_values must contain positive integers.")
    specs: list[RunSpec] = []
    for method in methods:
        for seed in seeds:
            ks = k_values if method == "a3" else [None]
            for rank in ranks:
                for k in ks:
                    suffix = f"-r{rank}" + (f"-k{k}" if k is not None else "")
                    run_id = f"{method}-seed{seed}{suffix}-{protocol}"
                    specs.append(
                        RunSpec(
                            method=method,
                            seed=seed,
                            protocol=protocol,
                            run_id=run_id,
                            config=str(DEFAULT_CONFIGS[method]),
                            output_dir=str(output_dir / "runs" / run_id),
                            run_name=run_id,
                            k=k,
                            rank=rank,
                        )
                    )
    return specs


def _materialize(spec: RunSpec, root: Path, epochs: int | None) -> Path:
    source = Path(spec.config)
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    _validate_domains(config, source)
    _set_seed(config, spec.method, spec.seed)
    _set_rank(config, spec.rank)
    _set_epochs(config, epochs)
    _set_protocol(config, spec.protocol)
    target = root / "configs" / f"{spec.run_id}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def make_manifest(
    output_dir: Path,
    *,
    seeds: list[int],
    methods: list[str],
    k_values: list[int],
    rank_values: list[int] | None = None,
    epochs: int | None = None,
    protocol: str = "sequential",
) -> dict[str, Any]:
    specs = expand_specs(
        output_dir,
        seeds=seeds,
        methods=methods,
        k_values=k_values,
        rank_values=rank_values,
        epochs=epochs,
        protocol=protocol,
    )
    manifest = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repository": str(ROOT),
        "protocol": protocol,
        "methods": methods,
        "seeds": seeds,
        "k_values": k_values,
        "rank_values": rank_values or [16],
        "epochs": epochs,
        "domains": list(EXPECTED_DOMAINS),
        "measured_metrics": list(MEASURED_METRICS),
        "runs": [{**asdict(spec), "status": "pending"} for spec in specs],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _update_manifest(path: Path, run_id: str, **fields: Any) -> None:
    manifest = _read_json(path)
    for run in manifest["runs"]:
        if run["run_id"] == run_id:
            run.update(fields)
            break
    else:
        raise KeyError(run_id)
    _write_json(path, manifest)


def _command(spec: RunSpec, config: Path) -> list[str]:
    command = [
        sys.executable,
        str(RUNNERS[spec.method]),
        "--config",
        str(config),
        "--output-dir",
        spec.output_dir,
        "--run-name",
        spec.run_name,
    ]
    if spec.method == "a2":
        command.extend(["--arm", "full"])
    if spec.method == "a3" and spec.k is not None:
        command.extend(["--K", str(spec.k)])
    return command


def train_manifest(
    output_dir: Path, *, resume: bool = True, epochs: int | None = None
) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    config_root = output_dir
    for raw in manifest["runs"]:
        spec = RunSpec(
            method=raw["method"],
            seed=raw["seed"],
            protocol=raw["protocol"],
            run_id=raw["run_id"],
            config=raw["config"],
            output_dir=raw["output_dir"],
            run_name=raw["run_name"],
            k=raw.get("k"),
            rank=raw.get("rank", 16),
        )
        run_dir = Path(spec.output_dir) / spec.run_name
        results = run_dir / "results.json"
        if resume and results.is_file():
            _update_manifest(manifest_path, spec.run_id, status="complete", results=str(results))
            continue
        config = _materialize(spec, config_root, epochs if epochs is not None else manifest["epochs"])
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        _update_manifest(
            manifest_path,
            spec.run_id,
            status="running",
            materialized_config=str(config),
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        log_path = output_dir / "logs" / f"{spec.run_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                _command(spec, config),
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        if completed.returncode != 0:
            _update_manifest(
                manifest_path,
                spec.run_id,
                status="failed",
                returncode=completed.returncode,
                log=str(log_path),
            )
            raise RuntimeError(f"{spec.run_id} failed; see {log_path}")
        if not results.is_file():
            raise RuntimeError(f"{spec.run_id} exited successfully but wrote no results.json")
        _update_manifest(
            manifest_path,
            spec.run_id,
            status="complete",
            returncode=0,
            results=str(results),
            log=str(log_path),
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )


def _result_record(raw: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    test = raw.get("test", {})
    domains = raw.get("domains", [])
    losses = raw.get("loss", [])
    final_losses = losses[-1] if losses else []
    base_losses = raw.get("base_loss", [])
    final_mean_loss = sum(final_losses) / len(final_losses) if final_losses else None
    base_mean_loss = sum(base_losses) / len(base_losses) if base_losses else None
    return {
        "run_id": run["run_id"],
        "method": run["method"],
        "seed": run["seed"],
        "k": run.get("k"),
        "rank": run.get("rank", 16),
        "domains": domains,
        "final_mean_loss": final_mean_loss,
        "loss_improvement": (
            base_mean_loss - final_mean_loss
            if base_mean_loss is not None and final_mean_loss is not None
            else None
        ),
        "final_mean_ppl": (
            sum(row.get("ppl", 0.0) for row in test.values()) / len(test)
            if test
            else None
        ),
        "mean_forgetting": (
            raw.get("mean_forgetting", [])[-1]
            if raw.get("mean_forgetting")
            else None
        ),
        "steps": raw.get("steps"),
        "tokens": raw.get("tokens"),
        "mean_step_time": raw.get("mean_step_time"),
        "params_unchanged": raw.get("params_unchanged"),
        "results": run.get("results"),
    }


def evaluate_manifest(output_dir: Path) -> dict[str, Any]:
    manifest = _read_json(output_dir / "manifest.json")
    records = []
    fingerprints = {}
    for run in manifest["runs"]:
        result_path = Path(run.get("results", ""))
        if not result_path.is_file():
            continue
        config_path = Path(run.get("materialized_config", run["config"]))
        config_bytes = config_path.read_bytes()
        fingerprints[run["run_id"]] = _sha256(config_bytes)
        raw_result = _read_json(result_path)
        record = _result_record(raw_result, run)
        try:
            record["deployed_evaluation"] = _load_deployed_metrics(run, raw_result)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            record["deployed_evaluation"] = {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
    by_method_rank_seed: dict[str, dict[int, dict[int, float]]] = {}
    for record in records:
        loss = record.get("loss_improvement")
        if loss is not None:
            by_method_rank_seed.setdefault(record["method"], {}).setdefault(
                record["seed"], {}
            )[record["rank"]] = loss
    for record in records:
        curve_data = by_method_rank_seed.get(record["method"], {}).get(record["seed"], {})
        if len(curve_data) >= 2:
            record["residual_compressibility"] = {
                str(rank): float(value)
                for rank, value in _compressibility_curve(curve_data).items()
            }
        else:
            record["residual_compressibility"] = None
    reference = {
        (record["seed"], record["rank"]): record["final_mean_loss"]
        for record in records
        if record["method"] == "a1"
    }
    for record in records:
        baseline = reference.get((record["seed"], record["rank"]))
        record["paired_delta_vs_a1"] = (
            record["final_mean_loss"] - baseline
            if baseline is not None and record["method"] != "a1"
            else None
        )
    evaluation = {
        "schema": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "domains": manifest.get("domains", list(EXPECTED_DOMAINS)),
        "measured_metrics": manifest.get("measured_metrics", list(MEASURED_METRICS)),
        "config_fingerprints": fingerprints,
        "records": records,
        "completed_runs": len(records),
        "planned_runs": len(manifest["runs"]),
    }
    _write_json(output_dir / "evaluation.json", evaluation)
    return evaluation


def report_manifest(output_dir: Path) -> Path:
    evaluation = _read_json(output_dir / "evaluation.json")
    by_method: dict[str, list[dict[str, Any]]] = {}
    for record in evaluation["records"]:
        by_method.setdefault(record["method"], []).append(record)
    lines = [
        "# Stage A A1/A2/A3 Benchmark Report",
        "",
        f"- Completed runs: {evaluation['completed_runs']}/{evaluation['planned_runs']}",
        f"- Domains: {', '.join(evaluation.get('domains', []))}",
        "",
        "| Method | Runs | Final loss mean | Final loss std | Forgetting mean | C(r) | Tokens/s | Eval status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for method, records in sorted(by_method.items()):
        losses = [r["final_mean_loss"] for r in records if r["final_mean_loss"] is not None]
        forgetting = [r["mean_forgetting"] for r in records if r["mean_forgetting"] is not None]
        rates = [
            r["tokens"] / (r["steps"] * r["mean_step_time"])
            for r in records
            if r.get("tokens") and r.get("steps") and r.get("mean_step_time")
        ]
        curves = [
            r["residual_compressibility"]
            for r in records
            if r.get("residual_compressibility")
        ]
        c_rank = (
            sum(float(curve.get("4", 0.0)) for curve in curves) / len(curves)
            if curves
            else None
        )
        deployed = [
            r.get("deployed_evaluation", {}) for r in records
            if r.get("deployed_evaluation", {}).get("status") == "measured"
        ]
        mean = sum(losses) / len(losses) if losses else None
        std = (
            (sum((x - mean) ** 2 for x in losses) / (len(losses) - 1)) ** 0.5
            if len(losses) > 1
            else None
        )
        lines.append(
            f"| {method} | {len(records)} | {mean if mean is not None else 'N/A'} | "
            f"{std if std is not None else 'N/A'} | "
            f"{sum(forgetting) / len(forgetting) if forgetting else 'N/A'} | "
            f"{c_rank if c_rank is not None else 'N/A'} | "
            f"{sum(rates) / len(rates) if rates else 'N/A'} | "
            f"{'measured' if deployed else 'unavailable'} |"
        )
    lines.extend(
        [
            "",
            "## Scope and limitations",
            "",
            "Measured metrics: loss matrix, perplexity, mean forgetting, optimizer "
            "steps, tokens, mean step time, estimated tokens/second, and frozen-base "
            "parameter integrity. Completed checkpoints additionally receive logits-"
            "based accuracy, ECE, domain KL, cross-domain conflict, ablation-based "
            "fracture/overlap, Taylor/synergy, inference latency, and memory metrics.",
            "",
            "This compact implementation reuses the stage runners and reports their "
            "existing artifacts. Anchor/held-out evaluation, joint training, and A4 "
            "symbolic retrieval remain explicitly unavailable.",
            "A3 `K` is a training-time settling parameter; deployment uses the "
            "label-free predictor path.",
        ]
    )
    path = output_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
