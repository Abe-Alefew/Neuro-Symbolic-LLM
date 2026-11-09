"""Consolidated test suite for the Stage A A1/A2/A3 benchmark harness and metrics.

Covers Phase 2 bug fixes and enhanced representation metrics:
- 2.1: ECE handles empty bins without producing NaN; jits cleanly
- 2.2: Pooled ECE across batches matches evaluation on concatenated batch
- 2.3: A3 K-sweep compressibility curves are grouped by (method, seed, k)
- 2.4: C(r) reports None / N/A when target rank is absent (not 0.0)
- 2.5: NaN / Inf sanitized to null; written with allow_nan=False
- 2.6: Resume rematerializes config and records config_sha
- Requirements 1-4: Per-domain overlap matrices within layer, normalized separately
- Requirement 5: Semantic / category factor overlap
- Requirement 6: Metric-weighted synergy proxy Syn_proxy(i,j) = delta_h_i^T G delta_h_j
- Requirement 7: Measured pairwise ablation interaction S_ij and proxy validation
- Orchestration: planning, evaluation, and report rendering
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from benchmark.core import (
    _read_json,
    _spearman_corr,
    _write_json,
    compute_domain_overlap,
    compute_semantic_overlap,
    evaluate_manifest,
    make_manifest,
    report_manifest,
    train_manifest,
)
from metrics.performance import (
    calibration_stats,
    ece_from_stats,
    expected_calibration_error,
)
from metrics.representation import (
    pairwise_ablation_interaction,
    synergy,
    synergy_proxy,
)

# ==============================================================================
# Phase 2 Bug 2.1: ECE finite with empty bins and JIT-compatible
# ==============================================================================


def test_ece_finite_when_bins_are_empty() -> None:
    """Peaked logits with high confidence leave several bins completely empty.

    Must return a finite float, never NaN.
    """
    # 2 batches, 5 tokens, vocab 10
    logits = jnp.zeros((2, 5, 10), dtype=jnp.float32)
    # Peak class 0 to have confidence > 0.95
    logits = logits.at[:, :, 0].set(50.0)
    labels = jnp.zeros((2, 5), dtype=jnp.int32)

    ece = float(expected_calibration_error(logits, labels, bins=15))
    assert math.isfinite(ece)
    assert not math.isnan(ece)
    # Since confidence is ~1.0 and predictions are 100% correct, ECE should be near 0
    assert ece == pytest.approx(0.0, abs=1e-3)


def test_ece_confident_wrong_predictions_near_one() -> None:
    """When the model is 100% confident but always wrong, ECE ≈ 1.0."""
    logits = jnp.zeros((2, 5, 10), dtype=jnp.float32)
    logits = logits.at[:, :, 0].set(50.0)  # Predicts 0 with ~100% conf
    labels = jnp.ones((2, 5), dtype=jnp.int32)  # True label is 1

    ece = float(expected_calibration_error(logits, labels, bins=15))
    assert math.isfinite(ece)
    assert ece == pytest.approx(1.0, abs=1e-3)


def test_ece_jits_cleanly() -> None:
    """calibration_stats and expected_calibration_error must be JIT-compilable."""
    logits = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 16))
    labels = jnp.ones((2, 8), dtype=jnp.int32)

    jitted_ece = jax.jit(expected_calibration_error)
    val1 = float(expected_calibration_error(logits, labels, bins=10))
    val2 = float(jitted_ece(logits, labels, bins=10))
    assert math.isfinite(val2)
    assert val1 == pytest.approx(val2, rel=1e-5)


def test_ece_ignores_masked_labels() -> None:
    """Labels with -100 must not contribute to calibration statistics."""
    logits = jnp.zeros((1, 4, 5), dtype=jnp.float32)
    logits = logits.at[:, :, 0].set(20.0)
    labels = jnp.array([[-100, -100, -100, -100]], dtype=jnp.int32)

    conf_sum, acc_sum, counts = calibration_stats(logits, labels, bins=5)
    assert float(counts.sum()) == 0.0
    ece = float(ece_from_stats(conf_sum, acc_sum, counts))
    assert ece == pytest.approx(0.0)


# ==============================================================================
# Phase 2 Bug 2.2: Pooled ECE across batches
# ==============================================================================


def test_pooled_ece_across_batches_equals_concatenated() -> None:
    """Accumulating calibration_stats across batches equals evaluating on concatenated batch."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    logits1 = jax.random.normal(k1, (2, 6, 8))
    labels1 = jax.random.randint(k2, (2, 6), 0, 8)
    logits2 = jax.random.normal(k3, (3, 6, 8))
    labels2 = jax.random.randint(k4, (3, 6), 0, 8)

    # 1. Pooled stats
    c1, a1, cnt1 = calibration_stats(logits1, labels1, bins=10)
    c2, a2, cnt2 = calibration_stats(logits2, labels2, bins=10)
    pooled_ece = float(ece_from_stats(c1 + c2, a1 + a2, cnt1 + cnt2))

    # 2. Concatenated batch
    logits_cat = jnp.concatenate([logits1, logits2], axis=0)
    labels_cat = jnp.concatenate([labels1, labels2], axis=0)
    cat_ece = float(expected_calibration_error(logits_cat, labels_cat, bins=10))

    assert math.isfinite(pooled_ece)
    assert pooled_ece == pytest.approx(cat_ece, rel=1e-5)


# ==============================================================================
# Phase 2 Bug 2.3: A3 K-sweep compressibility grouping
# ==============================================================================


def test_a3_k_sweep_curves_do_not_overwrite(tmp_path: Path) -> None:
    """Compressibility curves must be grouped by (method, seed, k) so k=1 and k=4 stay distinct."""
    manifest = {
        "schema": 1,
        "runs": [
            {
                "run_id": "a3-seed0-r4-k1-sequential",
                "method": "a3",
                "seed": 0,
                "k": 1,
                "rank": 4,
                "config": str(tmp_path / "cfg.yaml"),
                "results": str(tmp_path / "r1.json"),
            },
            {
                "run_id": "a3-seed0-r16-k1-sequential",
                "method": "a3",
                "seed": 0,
                "k": 1,
                "rank": 16,
                "config": str(tmp_path / "cfg.yaml"),
                "results": str(tmp_path / "r2.json"),
            },
            {
                "run_id": "a3-seed0-r4-k4-sequential",
                "method": "a3",
                "seed": 0,
                "k": 4,
                "rank": 4,
                "config": str(tmp_path / "cfg.yaml"),
                "results": str(tmp_path / "r3.json"),
            },
            {
                "run_id": "a3-seed0-r16-k4-sequential",
                "method": "a3",
                "seed": 0,
                "k": 4,
                "rank": 16,
                "config": str(tmp_path / "cfg.yaml"),
                "results": str(tmp_path / "r4.json"),
            },
        ],
    }
    _write_json(tmp_path / "manifest.json", manifest)
    (tmp_path / "cfg.yaml").write_text("adapter:\n  rank: 16\n", encoding="utf-8")

    # Mock results: k=1 has improvement (r4=0.2, r16=0.4); k=4 has improvement (r4=0.3, r16=0.6)
    _write_json(tmp_path / "r1.json", {"base_loss": [3.0], "loss": [[2.8]]})
    _write_json(tmp_path / "r2.json", {"base_loss": [3.0], "loss": [[2.6]]})
    _write_json(tmp_path / "r3.json", {"base_loss": [3.0], "loss": [[2.7]]})
    _write_json(tmp_path / "r4.json", {"base_loss": [3.0], "loss": [[2.4]]})

    evaluation = evaluate_manifest(tmp_path)
    records_by_id = {r["run_id"]: r for r in evaluation["records"]}

    # k=1 curve: r4 relative to r16 (0.2 / 0.4 = 0.5)
    rec_k1 = records_by_id["a3-seed0-r4-k1-sequential"]
    assert rec_k1["residual_compressibility"] is not None
    assert float(rec_k1["residual_compressibility"]["4"]) == pytest.approx(
        0.5, rel=1e-3
    )

    # k=4 curve: r4 relative to r16 (0.3 / 0.6 = 0.5)
    rec_k4 = records_by_id["a3-seed0-r4-k4-sequential"]
    assert rec_k4["residual_compressibility"] is not None
    assert float(rec_k4["residual_compressibility"]["4"]) == pytest.approx(
        0.5, rel=1e-3
    )


# ==============================================================================
# Phase 2 Bug 2.4: C(r) reports N/A when target rank is absent
# ==============================================================================


def test_cr_reports_na_when_target_rank_absent(tmp_path: Path) -> None:
    """When rank 4 is not evaluated (e.g. only ranks 8 and 16), report_manifest reports N/A, not 0.0."""
    evaluation = {
        "completed_runs": 2,
        "planned_runs": 2,
        "domains": ["d1"],
        "records": [
            {
                "method": "a1",
                "final_mean_loss": 2.0,
                "residual_compressibility": {"8": 0.8, "16": 1.0},
            },
            {
                "method": "a1",
                "final_mean_loss": 2.0,
                "residual_compressibility": {"8": 0.8, "16": 1.0},
            },
        ],
    }
    _write_json(tmp_path / "evaluation.json", evaluation)
    report_path = report_manifest(tmp_path, report_rank=4)
    text = report_path.read_text(encoding="utf-8")
    # Must NOT report "0.0", must report "N/A"
    assert "| 0.0 |" not in text
    assert "| N/A |" in text

    # When report_rank is 8, it should report 0.8
    report_path_8 = report_manifest(tmp_path, report_rank=8)
    text_8 = report_path_8.read_text(encoding="utf-8")
    assert "| 0.8 |" in text_8


# ==============================================================================
# Phase 2 Bug 2.5: NaN / Inf written as null with allow_nan=False
# ==============================================================================


def test_nan_and_inf_sanitized_to_valid_json(tmp_path: Path) -> None:
    """Writing records with NaN/Inf must produce strict valid JSON (nulls), never fail."""
    bad_data = {
        "metric_nan": float("nan"),
        "metric_inf": float("inf"),
        "metric_neg_inf": float("-inf"),
        "nested": {"val": float("nan")},
        "list": [1.0, float("nan"), 3.0],
    }
    out_file = tmp_path / "sanitized.json"
    _write_json(out_file, bad_data)

    # Must be readable with strict json parser without error
    raw_text = out_file.read_text(encoding="utf-8")
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text

    loaded = json.loads(raw_text)
    assert loaded["metric_nan"] is None
    assert loaded["metric_inf"] is None
    assert loaded["metric_neg_inf"] is None
    assert loaded["nested"]["val"] is None
    assert loaded["list"] == [1.0, None, 3.0]


# ==============================================================================
# Phase 2 Bug 2.6: Resume materializes config and sets SHA
# ==============================================================================


def test_resume_preserves_materialized_config_and_sha(tmp_path: Path) -> None:
    """train_manifest with resume=True when results exist still records config and SHA."""
    run_dir = tmp_path / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "results.json", {"test": {}})

    dummy_cfg = tmp_path / "base_cfg.yaml"
    dummy_cfg.write_text(
        "adapter:\n  rank: 16\ntraining:\n  model: gpt2\ndomains:\n  - name: eurlex\n  - name: ledgar\n  - name: scotus\n",
        encoding="utf-8",
    )

    manifest = {
        "schema": 1,
        "epochs": 1,
        "runs": [
            {
                "run_id": "test-run",
                "method": "a1",
                "seed": 0,
                "protocol": "sequential",
                "config": str(dummy_cfg),
                "output_dir": str(tmp_path / "runs"),
                "run_name": "test-run",
                "status": "pending",
            }
        ],
    }
    _write_json(tmp_path / "manifest.json", manifest)

    train_manifest(tmp_path, resume=True)

    updated = _read_json(tmp_path / "manifest.json")
    run_entry = updated["runs"][0]
    assert run_entry["status"] == "complete"
    assert "materialized_config" in run_entry
    assert Path(run_entry["materialized_config"]).is_file()
    assert "config_sha" in run_entry
    assert isinstance(run_entry["config_sha"], str)
    assert len(run_entry["config_sha"]) == 64


# ==============================================================================
# Requirements 1-4: Domain overlap within layer
# ==============================================================================


def test_domain_overlap_within_layer() -> None:
    """Domain overlap compares domains within a layer using normalized contributions.

    1. Same residual component definitions across domains (atoms of the layer).
    2. Normalize contributions separately for each domain.
    3. Retain per-domain overlap matrix entries.
    """
    # 3 domains, 4 atoms
    domain_effects = {
        "domain_A": [1.0, 0.0, 0.0, 0.0],
        "domain_B": [0.5, 0.5, 0.0, 0.0],
        "domain_C": [0.0, 0.0, 1.0, 1.0],
    }
    matrix = compute_domain_overlap(domain_effects)

    # Domain A normalized: [1, 0, 0, 0]
    # Domain B normalized: [0.5, 0.5, 0, 0]
    # Domain C normalized: [0, 0, 0.5, 0.5]
    # Overlap(A, B) = min(1, 0.5) + min(0, 0.5) = 0.5
    # Overlap(A, C) = 0.0
    # Overlap(B, C) = 0.0
    assert matrix["domain_A:domain_B"] == pytest.approx(0.5)
    assert matrix["domain_A:domain_C"] == pytest.approx(0.0)
    assert matrix["domain_B:domain_C"] == pytest.approx(0.0)


def test_domain_overlap_identical_domains_give_one() -> None:
    """Identical contribution profiles yield an overlap of 1.0."""
    domain_effects = {
        "d1": [0.2, 0.4, 0.6, 0.8],
        "d2": [0.2, 0.4, 0.6, 0.8],
    }
    matrix = compute_domain_overlap(domain_effects)
    assert matrix["d1:d2"] == pytest.approx(1.0)


def test_domain_overlap_ignores_negative_contributions() -> None:
    """Harmful atoms (negative ablation effect) are clamped to 0."""
    domain_effects = {
        "d1": [1.0, -2.0, 0.0],
        "d2": [1.0, -1.0, 0.0],
    }
    # Normalized: both become [1.0, 0.0, 0.0]
    matrix = compute_domain_overlap(domain_effects)
    assert matrix["d1:d2"] == pytest.approx(1.0)


# ==============================================================================
# Requirement 5: Semantic / category label overlap
# ==============================================================================


def test_semantic_category_overlap() -> None:
    """Evaluate overlap between semantic / category factors within a layer."""
    cat_effects = {
        "concept_syntax": [0.8, 0.2, 0.0, 0.0],
        "concept_semantics": [0.0, 0.0, 0.5, 0.5],
        "concept_discourse": [0.4, 0.1, 0.25, 0.25],
    }
    cat_matrix = compute_semantic_overlap(cat_effects)
    assert "concept_discourse:concept_semantics" in cat_matrix
    assert "concept_syntax:concept_semantics" in cat_matrix
    assert cat_matrix["concept_syntax:concept_semantics"] == pytest.approx(0.0)
    assert cat_matrix["concept_discourse:concept_semantics"] == pytest.approx(0.5)


# ==============================================================================
# Requirement 6: Metric-weighted synergy proxy
# ==============================================================================


def test_synergy_proxy_diagonal_metric() -> None:
    """Syn_proxy(i, j) = delta_h_i^T G delta_h_j with 1-D diagonal metric."""
    delta_i = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    delta_j = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32)
    g_diag = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float32)

    # Expected: 0.5*1*4 + 1.0*2*5 + 2.0*3*6 = 2 + 10 + 36 = 48
    val = float(synergy_proxy(delta_i, delta_j, g_diag))
    assert val == pytest.approx(48.0)


def test_synergy_proxy_matrix_metric() -> None:
    """Syn_proxy(i, j) with 2-D metric tensor G."""
    delta_i = jnp.array([1.0, 2.0], dtype=jnp.float32)
    delta_j = jnp.array([3.0, 4.0], dtype=jnp.float32)
    g_matrix = jnp.array([[2.0, 0.5], [0.5, 1.0]], dtype=jnp.float32)

    # delta_i @ G @ delta_j = [1, 2] @ [[2, 0.5], [0.5, 1]] @ [3, 4]
    # [1, 2] @ G = [1*2 + 2*0.5, 1*0.5 + 2*1] = [3.0, 2.5]
    # [3.0, 2.5] @ [3, 4] = 9.0 + 10.0 = 19.0
    val = float(synergy_proxy(delta_i, delta_j, g_matrix))
    assert val == pytest.approx(19.0)


def test_synergy_backward_compat() -> None:
    """The legacy synergy function delegates to synergy_proxy."""
    d1 = jnp.array([1.0, 2.0])
    d2 = jnp.array([3.0, 4.0])
    fisher = jnp.array([0.5, 1.5])
    assert float(synergy(d1, d2, fisher)) == float(synergy_proxy(d1, d2, fisher))


# ==============================================================================
# Requirement 7: Measured pairwise ablation interaction & validation
# ==============================================================================


def test_measured_pairwise_ablation_interaction_additive() -> None:
    """For an additive model L = sum_k a_k, S_ij must be zero."""
    # Full loss: 10.0
    # Ablating i increases loss by 2.0 -> loss_i = 12.0
    # Ablating j increases loss by 3.0 -> loss_j = 13.0
    # Ablating both increases loss by 5.0 -> loss_ij = 15.0
    # S_ij = 15 - 12 - 13 + 10 = 0
    s_ij = float(
        pairwise_ablation_interaction(
            loss_full=10.0,
            loss_ablate_i=12.0,
            loss_ablate_j=13.0,
            loss_ablate_ij=15.0,
        )
    )
    assert s_ij == pytest.approx(0.0)


def test_measured_pairwise_ablation_interaction_synergistic() -> None:
    """Non-linear synergy produces non-zero S_ij."""
    # Synergistic interaction: removing both causes extra harm
    s_ij = float(
        pairwise_ablation_interaction(
            loss_full=10.0,
            loss_ablate_i=12.0,
            loss_ablate_j=13.0,
            loss_ablate_ij=17.0,  # +2 extra synergy
        )
    )
    assert s_ij == pytest.approx(2.0)


def test_spearman_correlation_validation() -> None:
    """Spearman correlation correctly validates monotonic alignment."""
    proxy_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    measured_vals = [2.0, 4.0, 6.0, 8.0, 10.0]
    corr = _spearman_corr(proxy_vals, measured_vals)
    assert corr == pytest.approx(1.0)

    # Reverse ranking
    rev_vals = [5.0, 4.0, 3.0, 2.0, 1.0]
    corr_rev = _spearman_corr(proxy_vals, rev_vals)
    assert corr_rev == pytest.approx(-1.0)


# ==============================================================================
# Orchestration: Specs, Manifest, and Report
# ==============================================================================


def test_expand_specs_and_make_manifest(tmp_path: Path) -> None:
    """Planning expands combinations of methods, seeds, ranks, and k-values."""
    manifest = make_manifest(
        tmp_path,
        seeds=[0, 1],
        methods=["a1", "a3"],
        k_values=[1, 4],
        rank_values=[16],
    )
    # a1 has 2 seeds x 1 rank x 1 k(None) = 2 runs
    # a3 has 2 seeds x 1 rank x 2 k([1, 4]) = 4 runs
    # Total = 6 runs
    assert len(manifest["runs"]) == 6
    assert (tmp_path / "manifest.json").is_file()
