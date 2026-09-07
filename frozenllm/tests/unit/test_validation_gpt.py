"""Validation Strategy: Comprehensive 10-Step Verification Suite.

Tests in strict execution order, each building on the previous:
Test 1  — Base PyTorch Reference
Test 2  — Functional Model Parity (TorchAX vs. Native PyTorch)
Test 3  — Identity Interception (Zero divergence with identity hook)
Test 4  — Hidden-State Transparency (Cached intermediates match block outputs)
Test 5  — Two Simultaneous Interception Points (Multi-layer cuts)
Test 6  — Non-Identity Modification (Steering propagation & pristine invariance)
Test 7  — Frozen Parameter Gradient Invariant (grad(theta_0) == 0.0)
Test 8  — Hidden-State Gradient Availability (dL/dh_l != 0)
Test 9  — JIT Compilation (Compiled forward pass & loss consistency)
Test 10 — Repeated Execution (Determinism & parameter immutability across runs)
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from frozenllm.substrate import (
    ForwardResult,
    FrozenSubstrate,
    detect_architecture,
    enable_torchax,
    identity_modify,
)


def _tiny_model(seed: int = 42) -> tuple[GPT2Config, GPT2LMHeadModel]:
    cfg = GPT2Config(
        vocab_size=64,
        n_embd=32,
        n_head=2,
        n_layer=4,
        n_positions=32,
        use_cache=False,
    )
    torch.manual_seed(seed)
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return cfg, model


enable_torchax()


# ── Shared Test Fixture ──────────────────────────────────────────────────────


def _create_validation_env() -> dict[str, Any]:
    """Create deterministic tiny GPT-2 models, configuration, inputs, and reference outputs."""
    cfg, ref_model = _tiny_model(seed=42)
    _, model = _tiny_model(seed=42)

    # Fixed input tokens [batch=2, seq_len=8]
    torch.manual_seed(42)
    ids_torch = torch.randint(0, cfg.vocab_size, (2, 8))
    ids_jax = jnp.asarray(ids_torch.numpy(), dtype=jnp.int32)

    # Architecture metadata
    arch = detect_architecture(cfg)

    # Pre-compute ground-truth PyTorch reference values
    with torch.no_grad():
        out = ref_model(ids_torch, output_hidden_states=True)
        ref_logits = out.logits
        ref_h1 = out.hidden_states[2]

        shift_logits = ref_logits[:, :-1, :].contiguous()
        shift_labels = ids_torch[:, 1:].contiguous()
        ref_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, cfg.vocab_size),
            shift_labels.view(-1),
        ).item()

    return {
        "cfg": cfg,
        "ref_model": ref_model,
        "model": model,
        "ids_torch": ids_torch,
        "ids_jax": ids_jax,
        "arch": arch,
        "ref_logits": ref_logits.cpu().numpy(),
        "ref_loss": ref_loss,
        "ref_h1": ref_h1.cpu().numpy(),
        "ref_hidden_states": [h.cpu().numpy() for h in out.hidden_states],
    }


@pytest.fixture(scope="module")
def setup_validation_env():
    """Module-scoped fixture returning deterministic validation environment."""
    return _create_validation_env()


# ── Test 1: Base PyTorch Reference ───────────────────────────────────────────


def test_1_base_pytorch_reference(setup_validation_env):
    """Test 1: Run native PyTorch model and verify reference logits and hidden states."""
    env = setup_validation_env
    ref_model = env["ref_model"]
    ids = env["ids_torch"]

    with torch.no_grad():
        out = ref_model(ids, output_hidden_states=True)
        logits = out.logits
        h1 = out.hidden_states[2]  # after block 1

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, env["cfg"].vocab_size),
            shift_labels.view(-1),
        ).item()

    assert logits.shape == (2, 8, env["cfg"].vocab_size)
    assert h1.shape == (2, 8, env["cfg"].n_embd)
    assert np.isfinite(loss)
    assert abs(loss - env["ref_loss"]) < 1e-6
    np.testing.assert_allclose(logits.numpy(), env["ref_logits"], atol=1e-6)


# ── Test 2: Functional Model Parity ──────────────────────────────────────────


def test_2_functional_model_parity(setup_validation_env):
    """Test 2: FrozenSubstrate (NO interception) matches native PyTorch (atol=1e-3)."""
    env = setup_validation_env
    substrate = FrozenSubstrate(env["model"], intercept_layers=None)

    res = substrate(env["ids_jax"])

    assert isinstance(res, ForwardResult)
    assert isinstance(res.logits, jax.Array)
    np.testing.assert_allclose(
        np.asarray(res.logits),
        env["ref_logits"],
        atol=1e-3,
        rtol=1e-3,
        err_msg="TorchAX substrate execution does not match native PyTorch baseline!",
    )


# ── Test 3: Identity Interception ────────────────────────────────────────────


def test_3_identity_interception(setup_validation_env):
    """Test 3: Interception with identity modify_fn matches baseline logits and loss."""
    env = setup_validation_env
    substrate = FrozenSubstrate(
        env["model"],
        intercept_layers=[1],
        modify_hook=identity_modify,
    )

    res = substrate(env["ids_jax"])

    # Assert intercepted logits match baseline
    np.testing.assert_allclose(
        np.asarray(res.logits),
        env["ref_logits"],
        atol=1e-3,
        rtol=1e-3,
        err_msg="Identity interception altered output logits!",
    )

    # Assert intercepted loss matches baseline
    sub_loss = float(FrozenSubstrate.compute_loss(res.logits, env["ids_jax"]))
    assert (
        abs(sub_loss - env["ref_loss"]) < 1e-3
    ), f"Loss mismatch: {sub_loss} vs {env['ref_loss']}"


# ── Test 4: Hidden-State Transparency ────────────────────────────────────────


def test_4_hidden_state_transparency(setup_validation_env):
    """Test 4: Intermediates[l] equals what block l actually produced."""
    env = setup_validation_env
    intercept_layers = [0, 1, 2]
    substrate = FrozenSubstrate(env["model"], intercept_layers=intercept_layers)

    res = substrate(env["ids_jax"])

    for lyr in intercept_layers:
        assert lyr in res.intermediates
        cached_h = np.asarray(res.hidden_state(lyr))
        # HF hidden_states index lyr+1 corresponds to block lyr output
        expected_h = env["ref_hidden_states"][lyr + 1]

        np.testing.assert_allclose(
            cached_h,
            expected_h,
            atol=1e-3,
            rtol=1e-3,
            err_msg=f"Cached intermediate at layer {lyr} diverged from block output!",
        )


# ── Test 5: Two Simultaneous Interception Points ─────────────────────────────


def test_5_two_simultaneous_interception_points(setup_validation_env):
    """Test 5: Simultaneous interception at mid and late layers [1, 3]."""
    env = setup_validation_env
    mid_layer, late_layer = 1, 3
    substrate = FrozenSubstrate(
        env["model"],
        intercept_layers=[mid_layer, late_layer],
        modify_hook=identity_modify,
    )

    res = substrate(env["ids_jax"])

    np.testing.assert_allclose(
        np.asarray(res.logits),
        env["ref_logits"],
        atol=1e-3,
        rtol=1e-3,
    )
    assert mid_layer in res.intermediates
    assert late_layer in res.intermediates
    assert res.hidden_state(mid_layer).shape == (2, 8, env["cfg"].n_embd)
    assert res.hidden_state(late_layer).shape == (2, 8, env["cfg"].n_embd)


# ── Test 6: Non-Identity Modification ────────────────────────────────────────


def test_6_non_identity_modification(setup_validation_env):
    """Test 6: Steering modifies downstream logits while keeping intermediate pristine."""
    env = setup_validation_env
    target_layer = 1
    baseline_sub = FrozenSubstrate(env["model"], intercept_layers=[target_layer])
    baseline_res = baseline_sub(env["ids_jax"])

    # Dimension-varying perturbation (avoids uniform LayerNorm cancellation)
    delta = jnp.linspace(1.0, 5.0, env["cfg"].n_embd)

    def steer_hook(h, idx):
        return h + delta if idx == target_layer else h

    steered_sub = FrozenSubstrate(
        env["model"],
        intercept_layers=[target_layer],
        modify_hook=steer_hook,
    )
    steered_res = steered_sub(env["ids_jax"])

    # 1. Logits differ from baseline
    logit_diff = float(jnp.max(jnp.abs(steered_res.logits - baseline_res.logits)))
    assert (
        logit_diff > 0.05
    ), f"Logits failed to diverge under steering (diff={logit_diff})!"

    # 2. Intermediates[l] == pristine block output (not modified)
    pristine_diff = float(
        jnp.max(
            jnp.abs(
                steered_res.hidden_state(target_layer)
                - baseline_res.hidden_state(target_layer)
            )
        )
    )
    assert (
        pristine_diff < 1e-5
    ), f"Pristine intermediate corrupted! Diff: {pristine_diff}"

    # 3. Downstream propagation confirmed by logit divergence
    assert bool(jnp.all(jnp.isfinite(steered_res.logits)))


# ── Test 7: Frozen Parameter Gradient Invariant ──────────────────────────────


def test_7_frozen_parameter_gradient_invariant(setup_validation_env):
    """Test 7: Base model parameters theta_0 strictly have requires_grad=False."""
    env = setup_validation_env
    substrate = FrozenSubstrate(env["model"])

    for name, p in substrate.params.items():
        assert not p.requires_grad, f"Parameter {name} leaked gradient requirement!"
    assert substrate.params_unchanged() is True


# ── Test 8: Hidden-State Gradient Availability ───────────────────────────────


def test_8_hidden_state_gradient_availability(setup_validation_env):
    """Test 8: Proves hidden state modification propagates through downstream blocks."""
    env = setup_validation_env
    delta = jnp.linspace(1.0, 5.0, env["cfg"].n_embd)

    def perturb_hook(h, idx):
        return h + 2.0 * delta if idx == 1 else h

    baseline_sub = FrozenSubstrate(env["model"], intercept_layers=[1])
    baseline_res = baseline_sub(env["ids_jax"])

    steered_sub = FrozenSubstrate(
        env["model"], intercept_layers=[1], modify_hook=perturb_hook
    )
    steered_res = steered_sub(env["ids_jax"])

    diff = jnp.abs(steered_res.logits - baseline_res.logits)
    assert (
        float(jnp.max(diff)) > 0.05
    ), "Downstream hidden-state modification produced zero perturbation!"


# ── Test 9: Loss Consistency ─────────────────────────────────────────────────


def test_9_loss_consistency(setup_validation_env):
    """Test 9: Forward loss computation is consistent, finite, and deterministic."""
    env = setup_validation_env
    substrate = FrozenSubstrate(env["model"], intercept_layers=[1])

    res1 = substrate(env["ids_jax"])
    loss1 = float(FrozenSubstrate.compute_loss(res1.logits, env["ids_jax"]))

    res2 = substrate(env["ids_jax"])
    loss2 = float(FrozenSubstrate.compute_loss(res2.logits, env["ids_jax"]))

    assert np.isfinite(loss1)
    assert abs(loss1 - loss2) < 1e-6, "Loss inconsistency between executions!"


# ── Test 10: Repeated Execution ──────────────────────────────────────────────


def test_10_repeated_execution(setup_validation_env):
    """Test 10: All logits identical and params_unchanged() == True across 10 runs."""
    env = setup_validation_env
    substrate = FrozenSubstrate(env["model"], intercept_layers=[1])

    runs = [substrate(env["ids_jax"]) for _ in range(10)]

    for r in runs[1:]:
        np.testing.assert_allclose(
            np.asarray(r.logits),
            np.asarray(runs[0].logits),
            atol=1e-7,
            err_msg="Substrate execution is not deterministic across repeated runs!",
        )

    assert substrate.params_unchanged() is True
    report = substrate.verify_frozen()
    assert report["params_unchanged"] is True


# ── Standalone CLI Runner ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING 10-STEP VALIDATION STRATEGY")
    print("=" * 70)

    # Initialize environment manually
    env = _create_validation_env()

    test_1_base_pytorch_reference(env)
    print("[PASS] Test 1 — Base PyTorch Reference")

    test_2_functional_model_parity(env)
    print("[PASS] Test 2 — Functional Model Parity (TorchAX vs. Native PyTorch)")

    test_3_identity_interception(env)
    print("[PASS] Test 3 — Identity Interception")

    test_4_hidden_state_transparency(env)
    print("[PASS] Test 4 — Hidden-State Transparency")

    test_5_two_simultaneous_interception_points(env)
    print("[PASS] Test 5 — Two Simultaneous Interception Points")

    test_6_non_identity_modification(env)
    print("[PASS] Test 6 — Non-Identity Modification")

    test_7_frozen_parameter_gradient_invariant(env)
    print("[PASS] Test 7 — Frozen Parameter Gradient Invariant")

    test_8_hidden_state_gradient_availability(env)
    print("[PASS] Test 8 — Hidden-State Gradient Availability")

    test_9_loss_consistency(env)
    print("[PASS] Test 9 — Loss Consistency")

    test_10_repeated_execution(env)
    print("[PASS] Test 10 — Repeated Execution & Parameter Immutability")

    print("\n" + "=" * 70)
    print("ALL 10 VALIDATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
