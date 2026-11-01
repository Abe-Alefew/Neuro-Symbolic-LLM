"""Stage A2 audit test suite: checkpoints, break-it detection, and reporting."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import pytest

import stages.stage_A.a2 as a2
from frozenllm.substrate import FrozenSubstrate
from residual import ResidualAdapter
from stages.stage_A.a2 import (
    ARMS,
    A2Config,
    Check,
    a2_audit,
    apply_arm,
    audit_ok,
    audit_to_dict,
    format_audit,
)
from tests.residual.stage_A.a2.conftest import by_name, full_cfg, make_cfg


@pytest.mark.parametrize("arm", ARMS)
def test_audit_passes_across_all_arms(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
    arm: str,
) -> None:
    cfg = apply_arm(full_cfg(), arm)
    checks = a2_audit(substrate, adapter, cfg, input_ids)
    failed = [c for c in checks if c.ok is False]
    assert not failed, format_audit(checks)
    assert audit_ok(checks)


def test_gradient_ratios_are_information_not_failures(
    substrate: FrozenSubstrate, adapter: ResidualAdapter, input_ids: jax.Array
) -> None:
    c = by_name(a2_audit(substrate, adapter, full_cfg(), input_ids))
    for name in ("grad_ratio_cpc", "grad_ratio_stab_anch"):
        assert c[name].ok is None and c[name].value > 0


def test_only_the_enabled_terms_are_audited(
    substrate: FrozenSubstrate, adapter: ResidualAdapter, input_ids: jax.Array
) -> None:
    off = by_name(a2_audit(substrate, adapter, A2Config(lambda_kl=0.1), input_ids))
    assert not any(k.startswith("grad_ratio") for k in off)
    assert "grad_routing_cpc" not in off and "finite_with_ignored_labels" not in off
    assert audit_ok(list(off.values()))
    fixed = by_name(
        a2_audit(
            substrate,
            adapter,
            make_cfg(stab=0.1, hep="fixed_random"),
            input_ids,
        )
    )
    assert "grad_routing_stab_hep" not in fixed


def test_a_wrong_total_is_caught(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = a2.a2_objective

    def off_by_one(params, **kw):
        total, terms = real(params, **kw)
        return total + 1.0, {**terms, "total": total + 1.0}

    monkeypatch.setattr(a2, "a2_objective", off_by_one)
    checks = by_name(a2_audit(substrate, adapter, full_cfg(), input_ids))
    assert checks["total_decomposition"].ok is False


def test_a_leak_into_the_adapter_is_caught(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = a2.a2_objective

    def leaky(params, **kw):
        total, terms = real(params, **kw)
        leak = 1e-3 * sum(jnp.sum(v["B"]) for v in a2.adapter_params(params).values())
        if "stab_hep" in terms:
            terms = {**terms, "stab_hep": terms["stab_hep"] + leak}
        return total, terms

    monkeypatch.setattr(a2, "a2_objective", leaky)
    checks = by_name(a2_audit(substrate, adapter, full_cfg(), input_ids))
    assert checks["grad_routing_stab_hep"].ok is False


def test_non_finite_terms_are_caught(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = a2.a2_objective

    def nan_cpc(params, **kw):
        total, terms = real(params, **kw)
        if "cpc" in terms:
            terms = {**terms, "cpc": terms["cpc"] * jnp.nan}
        return total, terms

    monkeypatch.setattr(a2, "a2_objective", nan_cpc)
    checks = by_name(a2_audit(substrate, adapter, full_cfg(), input_ids))
    assert checks["probe_terms_finite"].ok is False


def test_report_formats(
    substrate: FrozenSubstrate, adapter: ResidualAdapter, input_ids: jax.Array
) -> None:
    checks = a2_audit(substrate, adapter, full_cfg(), input_ids)
    text = format_audit(checks)
    assert "[PASS] a1_parity" in text and "[info] grad_ratio_cpc" in text
    data = audit_to_dict(checks)
    assert data["ok"] is True and len(data["checks"]) == len(checks)
    json.dumps(data)
    broken = [Check("x", False, 1.0, "bad"), Check("y", True, 0.0)]
    assert audit_ok(broken) is False
    assert "[FAIL] x: bad" in format_audit(broken)
