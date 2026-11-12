"""Small, artifact-driven A1/A2/A3 benchmark harness."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
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
    "task_accuracy",
    "task_f1",
    "accuracy_forgetting",
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


def _sanitize_value(value: Any) -> Any:
    """Recursively sanitize data structure by converting non-finite floats to None."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitize_value(v) for v in value]
    return value


def _jsonable(value: Any) -> Any:
    return _sanitize_value(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _jsonable(value)
    path.write_text(
        json.dumps(clean, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _spearman_corr(x: list[float] | Any, y: list[float] | Any) -> float:
    """Compute Spearman rank correlation between two vectors without external dependencies."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) < 2 or np.all(xa == xa[0]) or np.all(ya == ya[0]):
        return 0.0
    rx = np.argsort(np.argsort(xa)).astype(float)
    ry = np.argsort(np.argsort(ya)).astype(float)
    vx = rx - rx.mean()
    vy = ry - ry.mean()
    denom = np.sqrt(np.sum(vx**2) * np.sum(vy**2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(vx * vy) / denom)


def compute_domain_overlap(
    domain_atom_effects: dict[str, list[float] | Any],
    eps: float = 1e-8,
) -> dict[str, float]:
    import jax
    import jax.numpy as jnp

    from metrics.representation import overlap

    domain_names = sorted(domain_atom_effects)
    normalized: dict[str, jax.Array] = {}
    for name in domain_names:
        raw = jnp.maximum(
            jnp.asarray(domain_atom_effects[name], dtype=jnp.float32), 0.0
        )
        total = jnp.maximum(jnp.sum(raw), eps)
        normalized[name] = raw / total

    matrix: dict[str, float] = {}
    for i, d1 in enumerate(domain_names):
        for d2 in domain_names[i + 1 :]:
            matrix[f"{d1}:{d2}"] = float(
                overlap(normalized[d1], normalized[d2], eps=eps)
            )
    return matrix


def compute_semantic_overlap(
    category_atom_effects: dict[str, list[float] | Any],
    eps: float = 1e-8,
) -> dict[str, float]:
    """Calculate overlap between semantic/category factors within a layer."""
    return compute_domain_overlap(category_atom_effects, eps=eps)


def _load_deployed_metrics(
    run: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the final checkpoint with the common deployed function.

    Heavy JAX/model imports are intentionally local so ``plan`` remains usable
    in lightweight environments.
    """
    import jax
    import jax.numpy as jnp

    from frozenllm.substrate import FrozenSubstrate, identity_modify
    from metrics.performance import (
        calibration_stats,
        ece_from_stats,
        kl_to_base,
        token_accuracy,
    )
    from metrics.representation import (
        effective_rank,
        fracture_index,
        gradient_conflict,
        invariant_redundancy,
        pairwise_ablation_interaction,
        synergy_proxy,
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
                ids,
                modify_fn=adapter.modify_fn(params),
                intercept_layers=adapter.layers,
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
            return hidden + adapter.scale * (latent[layer] @ block["B"]).astype(
                hidden.dtype
            )

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
                return hidden + adapter.scale * (latent[layer] @ block["B"]).astype(
                    hidden.dtype
                )

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
        conf_sums = []
        acc_sums = []
        count_sums = []
        for ids in batches:
            start = time.perf_counter()
            base, adapted, _ = deployed(ids)
            jax.block_until_ready(adapted)
            latency.append(time.perf_counter() - start)
            c_sum, a_sum, cnt = calibration_stats(adapted, ids)
            conf_sums.append(c_sum)
            acc_sums.append(a_sum)
            count_sums.append(cnt)
            rows.append(
                (
                    float(substrate.compute_loss(adapted, ids, losses=("CE",))),
                    float(token_accuracy(adapted, ids)),
                    float(kl_to_base(base, adapted)),
                )
            )
        values = np.asarray(rows, dtype=float)
        if conf_sums:
            pooled_conf = jnp.sum(jnp.stack(conf_sums), axis=0)
            pooled_acc = jnp.sum(jnp.stack(acc_sums), axis=0)
            pooled_cnt = jnp.sum(jnp.stack(count_sums), axis=0)
            pooled_ece = float(ece_from_stats(pooled_conf, pooled_acc, pooled_cnt))
        else:
            pooled_ece = 0.0
        metrics_by_domain[domain.name] = {
            "loss": float(values[:, 0].mean()) if len(values) else 0.0,
            "ppl": float(np.exp(values[:, 0].mean())) if len(values) else 0.0,
            "token_accuracy": float(values[:, 1].mean()) if len(values) else 0.0,
            "ece": pooled_ece,
            "kl": float(values[:, 2].mean()) if len(values) else 0.0,
        }

    gradients = {}
    for name, batches in batches_by_domain.items():
        if not batches:
            continue
        gradients[name] = jax.grad(lambda p, batch=batches[0]: loss_for(p, batch))(
            params
        )
    conflict = {}
    names = list(gradients)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            conflict[f"{left}:{right}"] = float(
                gradient_conflict(gradients[left], gradients[right])
            )

    representation: dict[str, Any] = {
        "effective_rank": {},
        "invariant_redundancy": {},
        "fracture_index": {},
        "overlap": {},
        "overlap_matrices": {},
        "semantic_overlap": {},
        "taylor_delta": {},
        "synergy": {},
        "synergy_proxy": {},
        "measured_synergy": {},
        "synergy_validation": {},
        "cross_domain_conflict": conflict,
    }
    ablation_by_domain: dict[str, dict[str, list[float]]] = {}
    for layer_name, block in params.items():
        if not layer_name.startswith("layer_"):
            continue
        a, b = block["A"], block["B"]
        num_atoms = int(b.shape[0])
        representation["effective_rank"][layer_name] = float(effective_rank(a @ b))
        representation["invariant_redundancy"][layer_name] = float(
            invariant_redundancy(a, b)
        )
        layer_domain_effects: dict[str, list[float]] = {d.name: [] for d in domains}
        for atom in range(num_atoms):
            ablated = {key: dict(value_map) for key, value_map in params.items()}
            ablated[layer_name]["B"] = b.at[atom].set(jnp.zeros_like(b[atom]))
            for domain in domains:
                batches = batches_by_domain.get(domain.name, ())
                if batches:
                    full = float(loss_for(params, batches[0]))
                    ablated_loss = float(loss_for(ablated, batches[0]))
                    layer_domain_effects[domain.name].append(ablated_loss - full)
                else:
                    layer_domain_effects[domain.name].append(0.0)

        ablation_by_domain[layer_name] = layer_domain_effects
        domain_fractures = [
            float(fracture_index(jnp.asarray([abs(e) for e in effects])))
            for effects in layer_domain_effects.values()
            if effects
        ]
        representation["fracture_index"][layer_name] = (
            float(np.mean(domain_fractures)) if domain_fractures else 0.0
        )
        layer_overlap = compute_domain_overlap(layer_domain_effects)
        representation["overlap"][layer_name] = layer_overlap
        representation["overlap_matrices"][layer_name] = layer_overlap

        semantic_cats = config.get("semantic_categories")
        if semantic_cats and isinstance(semantic_cats, dict):
            cat_effects: dict[str, list[float]] = {}
            for cat_name, dom_list in semantic_cats.items():
                matching = [
                    layer_domain_effects[d]
                    for d in dom_list
                    if d in layer_domain_effects
                ]
                if matching:
                    cat_effects[cat_name] = list(np.mean(matching, axis=0))
            if len(cat_effects) >= 2:
                representation["semantic_overlap"][layer_name] = (
                    compute_semantic_overlap(cat_effects)
                )
            else:
                representation["semantic_overlap"][layer_name] = {
                    "status": "unavailable",
                    "reason": "insufficient categories with matching domains",
                }
        else:
            representation["semantic_overlap"][layer_name] = {
                "status": "unavailable",
                "reason": "no category labels provided",
            }

        layer_gradients = [
            gradients[name][layer_name]["B"]
            for name in names
            if layer_name in gradients[name]
        ]
        if layer_gradients:
            stacked_grads = jnp.stack(layer_gradients)
            mean_gradient = jnp.mean(stacked_grads, axis=0)
            fisher = jnp.mean(stacked_grads**2, axis=0)
            atom_taylor = [
                taylor_delta(mean_gradient[i], fisher[i], -b[i])
                for i in range(num_atoms)
            ]
            metric_g = jnp.mean(fisher, axis=0)
        else:
            atom_taylor = [jnp.asarray(0.0)]
            metric_g = jnp.ones(int(b.shape[1]), dtype=jnp.float32)

        representation["taylor_delta"][layer_name] = float(
            jnp.sum(jnp.asarray(atom_taylor))
        )

        proxy_pairs: dict[str, float] = {}
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                delta_i = -b[i]
                delta_j = -b[j]
                proxy_pairs[f"{i}:{j}"] = float(
                    synergy_proxy(delta_i, delta_j, metric_g)
                )

        top_atoms = sorted(
            range(num_atoms),
            key=lambda a: float(
                np.mean([abs(layer_domain_effects[dn][a]) for dn in names])
            ),
            reverse=True,
        )[: min(num_atoms, 8)]

        measured_pairs: dict[str, float] = {}
        if len(top_atoms) >= 2 and names:
            eval_batch = batches_by_domain[names[0]][0]
            loss_full = float(loss_for(params, eval_batch))
            atom_single_losses: dict[int, float] = {}
            for atom_i in top_atoms:
                ab_i = {k: dict(v) for k, v in params.items()}
                ab_i[layer_name]["B"] = b.at[atom_i].set(jnp.zeros_like(b[atom_i]))
                atom_single_losses[atom_i] = float(loss_for(ab_i, eval_batch))

            for idx_i, atom_i in enumerate(top_atoms):
                loss_i = atom_single_losses[atom_i]
                for atom_j in top_atoms[idx_i + 1 :]:
                    loss_j = atom_single_losses[atom_j]
                    ab_ij = {k: dict(v) for k, v in params.items()}
                    ab_ij[layer_name]["B"] = b.at[atom_i].set(0.0).at[atom_j].set(0.0)
                    loss_ij = float(loss_for(ab_ij, eval_batch))
                    s_meas = float(
                        pairwise_ablation_interaction(
                            loss_full, loss_i, loss_j, loss_ij
                        )
                    )
                    measured_pairs[f"{atom_i}:{atom_j}"] = s_meas

        common_pairs = [k for k in measured_pairs if k in proxy_pairs]
        if len(common_pairs) >= 3:
            p_vals = [proxy_pairs[k] for k in common_pairs]
            m_vals = [measured_pairs[k] for k in common_pairs]
            val_corr = _spearman_corr(p_vals, m_vals)
        else:
            val_corr = None

        representation["synergy_proxy"][layer_name] = {
            "mean": float(np.mean(list(proxy_pairs.values()))) if proxy_pairs else 0.0,
            "pairs": proxy_pairs,
        }
        representation["measured_synergy"][layer_name] = {
            "mean": (
                float(np.mean(list(measured_pairs.values()))) if measured_pairs else 0.0
            ),
            "pairs": measured_pairs,
        }
        representation["synergy_validation"][layer_name] = {
            "spearman_correlation": val_corr,
            "num_pairs": len(common_pairs),
        }
        representation["synergy"][layer_name] = (
            float(np.mean(list(proxy_pairs.values()))) if proxy_pairs else 0.0
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
            "inference_latency_p95_s": (
                float(np.percentile(latency, 95)) if latency else None
            ),
            "peak_memory_bytes": peak_bytes,
        },
        "anchor_kl": {
            "status": "unavailable",
            "reason": "anchor dataset not configured",
        },
        "heldout_transfer": {
            "status": "unavailable",
            "reason": "held-out datasets not configured",
        },
        "joint_gap": {
            "status": "unavailable",
            "reason": "joint protocol not implemented",
        },
        "symbolic_retrieval": {"status": "excluded", "reason": "A4 not implemented"},
    }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _config_fingerprint(config: dict[str, Any]) -> str:
    return _sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _compressibility_curve(deltas: dict[int, float]) -> dict[int, float]:

    return _compressibility_detail(deltas)["curve"] or {}


def _compressibility_detail(deltas: dict[int, float]) -> dict[str, Any]:
    from metrics.representation import guarded_compressibility

    return guarded_compressibility(deltas)


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
    names = (
        tuple(domain.get("name") for domain in domains if isinstance(domain, dict))
        if isinstance(domains, list)
        else ()
    )
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
        raise ValueError(
            f"Unknown methods: {unknown}; choose from {sorted(DEFAULT_CONFIGS)}"
        )
    if not k_values or any(k < 0 for k in k_values):
        raise ValueError("k_values must contain non-negative integers.")
    ranks = rank_values or [16]
    if not ranks or any(rank < 1 for rank in ranks):
        raise ValueError("rank_values must contain positive integers.")
    specs: list[RunSpec] = []
    for method in methods:
        for seed in seeds:
            ks: list[int | None] = (
                [int(k) for k in k_values] if method == "a3" else [None]
            )
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
        config = _materialize(
            spec, config_root, epochs if epochs is not None else manifest["epochs"]
        )
        config_bytes = config.read_bytes()
        config_sha = _sha256(config_bytes)
        if resume and results.is_file():
            _update_manifest(
                manifest_path,
                spec.run_id,
                status="complete",
                results=str(results),
                materialized_config=str(config),
                config_sha=config_sha,
            )
            continue
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        _update_manifest(
            manifest_path,
            spec.run_id,
            status="running",
            materialized_config=str(config),
            config_sha=config_sha,
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
            try:
                log_tail = "\n".join(
                    log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
                        -40:
                    ]
                )
            except OSError as exc:
                log_tail = f"Unable to read failure log: {exc}"
            raise RuntimeError(
                f"{spec.run_id} failed with exit code {completed.returncode}; "
                f"see {log_path}\n\nLast log lines:\n{log_tail}"
            )
        if not results.is_file():
            raise RuntimeError(
                f"{spec.run_id} exited successfully but wrote no results.json"
            )
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
        "domain_loss_improvement": (
            {
                str(name): float(b) - float(f)
                for name, b, f in zip(
                    domains, base_losses, final_losses, strict=False
                )
            }
            if domains and len(base_losses) == len(final_losses) == len(domains)
            else None
        ),
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
            raw.get("mean_forgetting", [])[-1] if raw.get("mean_forgetting") else None
        ),
        "accuracy_forgetting": (
            raw.get("accuracy_forgetting", [])[-1]
            if raw.get("accuracy_forgetting")
            else None
        ),
        "task_accuracy": (
            float(np.nanmean(np.asarray(raw["task_accuracy"], dtype=float)[-1]))
            if raw.get("task_accuracy")
            and np.isfinite(np.asarray(raw["task_accuracy"], dtype=float)[-1]).any()
            else None
        ),
        "task_f1": (
            float(np.nanmean(np.asarray(raw["task_f1"], dtype=float)[-1]))
            if raw.get("task_f1")
            and np.isfinite(np.asarray(raw["task_f1"], dtype=float)[-1]).any()
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
        except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
            record["deployed_evaluation"] = {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
    by_group: dict[tuple[str, int, int | None], dict[int, float]] = {}
    by_group_domain: dict[
        tuple[str, int, int | None], dict[str, dict[int, float]]
    ] = {}
    for record in records:
        group_key = (record["method"], record["seed"], record.get("k"))
        loss = record.get("loss_improvement")
        if loss is not None:
            by_group.setdefault(group_key, {})[record["rank"]] = loss
        for name, value in (record.get("domain_loss_improvement") or {}).items():
            by_group_domain.setdefault(group_key, {}).setdefault(name, {})[
                record["rank"]
            ] = value
    for record in records:
        group_key = (record["method"], record["seed"], record.get("k"))
        detail = _compressibility_detail(by_group.get(group_key, {}))
        record["compressibility_status"] = detail["status"]
        record["compressibility_detail"] = {
            k: v for k, v in detail.items() if k != "curve"
        }
        record["residual_compressibility"] = (
            {str(rank): float(v) for rank, v in detail["curve"].items()}
            if detail["curve"]
            else None
        )
        per_domain = {}
        for name, deltas in by_group_domain.get(group_key, {}).items():
            d = _compressibility_detail(deltas)
            per_domain[name] = {
                "status": d["status"],
                "reference_improvement": d["reference_improvement"],
                "curve": (
                    {str(r): float(v) for r, v in d["curve"].items()}
                    if d["curve"]
                    else None
                ),
            }
        record["residual_compressibility_by_domain"] = per_domain or None
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


def _numeric_values(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return [float(value)]
    if isinstance(value, dict):
        values: list[float] = []
        for item in value.values():
            values.extend(_numeric_values(item))
        return values
    if isinstance(value, list | tuple):
        values = []
        for item in value:
            values.extend(_numeric_values(item))
        return values
    return []


def _report_metric_value(record: dict[str, Any], metric: str, rank: int) -> list[float]:
    deployed = record.get("deployed_evaluation", {})
    representation = deployed.get("representation", {})
    if metric == "residual_compressibility":
        curve = record.get("residual_compressibility") or {}
        value = curve.get(str(rank))
    elif metric == "validation_loss":
        value = record.get("final_mean_loss")
    elif metric == "perplexity":
        value = record.get("final_mean_ppl")
    elif metric == "forgetting":
        value = record.get("mean_forgetting")
    elif metric == "accuracy_forgetting":
        value = record.get("accuracy_forgetting")
    elif metric == "task_accuracy":
        value = record.get("task_accuracy")
    elif metric == "task_f1":
        value = record.get("task_f1")
    elif metric == "calibration":
        value = {
            domain: values.get("ece")
            for domain, values in deployed.get("domains", {}).items()
            if isinstance(values, dict)
        }
    elif metric == "gradient_conflict":
        value = representation.get("cross_domain_conflict")
    elif metric == "factor_overlap":
        value = representation.get("overlap")
    elif metric == "fracture":
        value = representation.get("fracture_index")
    elif metric == "mixed_curvature":
        value = None
    elif metric == "category_alignment":
        value = representation.get("semantic_overlap")
    elif metric == "ablation_fidelity":
        value = None
    elif metric == "taylor_delta":
        value = representation.get("taylor_delta")
    elif metric == "synergy_validation":
        value = representation.get("synergy_validation")
    elif metric == "synergy":
        value = representation.get("synergy")
    else:
        value = None
    return _numeric_values(value)


def _format_config_label(method: str, rank: int | None, k: int | None) -> str:
    m = method.upper()
    parts: list[str] = []
    if rank is not None:
        parts.append(f"r={rank}")
    if k is not None:
        parts.append(f"K={k}")
    if parts:
        return f"{m} ({', '.join(parts)})"
    return m


def _append_metric_table(
    lines: list[str],
    title: str,
    metric_specs: list[tuple[str, str]],
    configs: list[tuple[str, int | None, int | None]],
    by_config: dict[tuple[str, int | None, int | None], list[dict[str, Any]]],
    report_rank: int,
) -> None:
    labels = [_format_config_label(m, r, k) for m, r, k in configs]
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            f"| Metric | {' | '.join(labels)} | Coverage |",
            f"|---|{'---:|' * len(labels)}---|",
        ]
    )
    for label, key in metric_specs:
        cells: list[str] = []
        measured = False
        for cfg in configs:
            values = [
                value
                for record in by_config.get(cfg, [])
                for value in _report_metric_value(record, key, report_rank)
            ]
            if values:
                measured = True
                cells.append(f"{sum(values) / len(values):.6g}")
            else:
                cells.append("N/A")
        coverage = "measured" if measured else "not measured"
        lines.append(f"| {label} | {' | '.join(cells)} | {coverage} |")


def report_manifest(output_dir: Path, report_rank: int = 4) -> Path:
    evaluation = _read_json(output_dir / "evaluation.json")
    by_method: dict[str, list[dict[str, Any]]] = {}
    by_config: dict[tuple[str, int | None, int | None], list[dict[str, Any]]] = {}
    for record in evaluation["records"]:
        by_method.setdefault(record["method"], []).append(record)
        cfg_key = (
            str(record.get("method", "unknown")).lower(),
            record.get("rank"),
            record.get("k"),
        )
        by_config.setdefault(cfg_key, []).append(record)

    def _config_sort_key(
        cfg: tuple[str, int | None, int | None]
    ) -> tuple[int, int, int]:
        m, r, k = cfg
        m_order = {"a1": 1, "a2": 2, "a3": 3}.get(m, 99)
        return (m_order, r if r is not None else 0, k if k is not None else 0)

    configs = sorted(by_config.keys(), key=_config_sort_key)
    if not configs:
        configs = [("a1", None, None), ("a2", None, None), ("a3", None, None)]
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
        losses = [
            r["final_mean_loss"]
            for r in records
            if r.get("final_mean_loss") is not None
        ]
        forgetting = [
            r["mean_forgetting"]
            for r in records
            if r.get("mean_forgetting") is not None
        ]
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
        target_rank = str(report_rank)
        valid_c = [
            float(curve[target_rank]) for curve in curves if target_rank in curve
        ]
        c_rank = sum(valid_c) / len(valid_c) if valid_c else None
        deployed = [
            r.get("deployed_evaluation", {})
            for r in records
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
    _append_metric_table(
        lines,
        "Internal metrics",
        [
            ("Residual compressibility", "residual_compressibility"),
            ("Representation fracture index", "fracture"),
            ("Factor overlap and entanglement", "factor_overlap"),
            ("Gradient conflict", "gradient_conflict"),
            ("Mixed curvature coupling", "mixed_curvature"),
            ("Causal modularity and off-support leakage", "unavailable"),
            ("Update commutator proxy", "unavailable"),
            ("Causal fingerprint separability", "unavailable"),
            ("Category alignment and naturality", "category_alignment"),
            ("Interpretability and ablation fidelity", "ablation_fidelity"),
        ],
        configs,
        by_config,
        report_rank,
    )
    _append_metric_table(
        lines,
        "External metrics",
        [
            ("Validation loss", "validation_loss"),
            ("Perplexity", "perplexity"),
            ("Task accuracy", "task_accuracy"),
            ("Task macro-F1", "task_f1"),
            ("Calibration error", "calibration"),
            ("Forgetting", "forgetting"),
            ("Accuracy-based forgetting", "accuracy_forgetting"),
            ("Held-out transfer", "unavailable"),
            ("Sequential versus joint", "unavailable"),
            ("Anchor KL drift", "unavailable"),
            ("Symbolic retrieval precision / recall", "unavailable"),
            ("Long-range coherence", "unavailable"),
        ],
        configs,
        by_config,
        report_rank,
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
            "The internal and external tables include every metric required by "
            "metrics.md. N/A means the protocol or implementation does not measure "
            "that metric; it is not a zero result. Token accuracy and domain KL are "
            "available as diagnostics, but are not substituted for task accuracy or "
            "dedicated anchor KL.",
            "",
            "This compact implementation reuses the stage runners and reports their "
            "existing artifacts. Accuracy-based task metrics require labeled domain "
            "examples and are measured through the configured label-token protocol. "
            "Anchor/held-out evaluation, joint training, A4 symbolic retrieval, "
            "and long-range coherence remain explicitly unavailable.",
            "A3 `K` is a training-time settling parameter; deployment uses the "
            "label-free predictor path.",
        ]
    )
    path = output_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
