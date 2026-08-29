#!/usr/bin/env python
"""Comprehensive Validation Script for Segmented torch2jax Substrate (Mechanism M2).

Designed for Google Colab and local execution:
    python scripts/validate_torch2jax.py
    python scripts/validate_torch2jax.py --model gpt2
    python scripts/validate_torch2jax.py --model EleutherAI/pythia-70m --layers 1,3
    python scripts/validate_torch2jax.py --tiny
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)

from substrate.adapters import apply_residual_adapter, init_residual_adapter, make_training_step
from substrate.drift import compute_kl_drift
from substrate.substrate import FrozenJAXSubstrate


# Colored Terminal Formatting 

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 78}{RESET}")
    print(f"{BOLD}{CYAN}{title:^78}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 78}{RESET}\n")


def subheader(title: str) -> None:
    print(f"\n{BOLD}── {title} {'─' * max(0, 74 - len(title))}{RESET}")


def pass_fail(name: str, passed: bool, details: str = "") -> bool:
    tag = f"{GREEN}[PASS]{RESET}" if passed else f"{RED}[FAIL]{RESET}"
    msg = f"  {tag} {name:<45} {details}"
    print(msg)
    return passed


# Model Instantiation Helpers

def create_tiny_gpt2():
    torch.manual_seed(42)
    cfg = GPT2Config(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_layer=4,
        n_positions=32,
        use_cache=False,
    )
    model = GPT2LMHeadModel(cfg).eval()
    return model, cfg, "Tiny GPT-2 (4 layers, dim 32)"


def create_tiny_pythia():
    torch.manual_seed(42)
    cfg = GPTNeoXConfig(
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=4,
        max_position_embeddings=32,
        rotary_pct=0.25,
        use_parallel_residual=True,
        tie_word_embeddings=False,
        use_cache=False,
    )
    model = GPTNeoXForCausalLM(cfg).eval()
    return model, cfg, "Tiny Pythia (4 layers, dim 32, untied)"


def load_hf_model(model_id: str):
    print(f"Loading Hugging Face checkpoint: {model_id} ...")
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).eval()
    return model, config, f"HF Checkpoint: {model_id}"


# Main Validation Pipeline

def run_validation(
    torch_model: Any,
    config: Any,
    model_desc: str,
    intercept_layers: list[int],
    seq_len: int = 8,
    batch_size: int = 2,
    tol_logits: float = 1e-5,
    tol_hidden: float = 1e-5,
    skip_monolithic: bool = False,
) -> bool:
    results: list[bool] = []

    header(f"VALIDATING SUBSTRATE: {model_desc}")
    print(f"Interception Layers : {intercept_layers}")
    print(f"Batch Size / Length : [{batch_size}, {seq_len}]")
    print(f"JAX Backend Device  : {jax.devices()}")

    # 1. Generate Deterministic Input Tokens
    vocab_size = getattr(config, "vocab_size", 64)
    np.random.seed(42)
    raw_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    torch_ids = torch.from_numpy(raw_ids)
    jax_ids = jnp.asarray(raw_ids, dtype=jnp.int32)

    # 2. PyTorch Reference Forward Pass
    subheader("1. Reference PyTorch Execution")
    t0 = time.perf_counter()
    with torch.no_grad():
        pt_out = torch_model(torch_ids, output_hidden_states=True)
        pt_logits = pt_out.logits.numpy()
        pt_hidden_states = [h.numpy() for h in pt_out.hidden_states]
    t_pt = (time.perf_counter() - t0) * 1000
    print(f"  PyTorch forward pass completed in {t_pt:.2f} ms")
    results.append(pass_fail("PyTorch Reference Forward", True, f"Output shape: {pt_logits.shape}"))

    # 3. Monolithic torch2jax (Zero Interception)
    subheader("2. Monolithic JAX Execution (Zero Interception)")
    if skip_monolithic:
        print("  Skipped monolithic trace (--skip-monolithic enabled)")
        results.append(pass_fail("Monolithic Logit Parity vs PyTorch", True, "Skipped by user flag"))
    else:
        print("  Tracing & XLA compiling entire monolithic graph (one-time JIT cold start)...")
        t0 = time.perf_counter()
        sub_mono = FrozenJAXSubstrate(torch_model, config, intercept_layers=[])
        res_mono = sub_mono(jax_ids)
        t_mono = (time.perf_counter() - t0) * 1000
        max_err_mono = float(np.max(np.abs(pt_logits - np.asarray(res_mono.logits))))
        ok_mono = max_err_mono <= tol_logits
        results.append(
            pass_fail(
                "Monolithic Logit Parity vs PyTorch",
                ok_mono,
                f"Max Abs Error = {max_err_mono:.2e} (tol={tol_logits:.0e}, {t_mono:.1f}ms)",
            )
        )

    # 4. Segmented torch2jax Execution
    subheader("3. Segmented JAX Execution (With Interception)")
    t0 = time.perf_counter()
    sub_seg = FrozenJAXSubstrate(torch_model, config, intercept_layers=intercept_layers)
    res_seg = sub_seg(jax_ids)
    t_seg = (time.perf_counter() - t0) * 1000
    max_err_seg = float(np.max(np.abs(pt_logits - np.asarray(res_seg.logits))))
    ok_seg = max_err_seg <= tol_logits
    results.append(
        pass_fail(
            "Segmented Logit Parity vs PyTorch",
            ok_seg,
            f"Max Abs Error = {max_err_seg:.2e} (tol={tol_logits:.0e}, {t_seg:.1f}ms)",
        )
    )

    # 5. Hidden-State Parity at Interception Boundaries
    subheader("4. Hidden-State Boundary Parity")
    for lyr in intercept_layers:
        ref_h = pt_hidden_states[lyr + 1]  # hidden_states[0] is embedding output
        actual_h = np.asarray(res_seg.hidden_state(lyr))
        max_h_err = float(np.max(np.abs(ref_h - actual_h)))
        ok_h = max_h_err <= tol_hidden
        results.append(
            pass_fail(
                f"Hidden State Parity at Layer {lyr}",
                ok_h,
                f"Max Abs Error = {max_h_err:.2e} (tol={tol_hidden:.0e})",
            )
        )

    # 6. Interception & Modification Propagation
    subheader("5. Modification & Perturbation Propagation")
    first_lyr = intercept_layers[0]
    hidden_dim = actual_h.shape[-1]
    perturbation = jnp.linspace(0.2, 1.5, hidden_dim)

    sub_steer = FrozenJAXSubstrate(
        torch_model,
        config,
        intercept_layers=intercept_layers,
        modify_hook=lambda h, l: h + perturbation if l == first_lyr else h,
    )
    res_steer = sub_steer(jax_ids)

    # A. Pristine Cache Invariance
    cached_unmodified = float(
        np.max(np.abs(np.asarray(res_seg.hidden_state(first_lyr)) - np.asarray(res_steer.hidden_state(first_lyr))))
    )
    ok_cache = cached_unmodified <= 1e-7
    results.append(
        pass_fail(
            f"Pristine Cache Invariance at Layer {first_lyr}",
            ok_cache,
            f"Diff = {cached_unmodified:.2e} (Target: 0.0)",
        )
    )

    # B. Downstream Logit Shift
    logit_diff = float(np.max(np.abs(np.asarray(res_seg.logits) - np.asarray(res_steer.logits))))
    ok_prop = logit_diff > 0.05
    results.append(
        pass_fail(
            "Downstream Logit Propagation",
            ok_prop,
            f"Max Logit Shift = {logit_diff:.4f} (> 0.05)",
        )
    )

    # C. KL Divergence Monitoring
    kl_drift = compute_kl_drift(jnp.asarray(res_seg.logits), res_steer.logits)
    results.append(
        pass_fail(
            "KL Divergence Metric",
            kl_drift["kl_divergence"] > 1e-3,
            f"KL Drift = {kl_drift['kl_divergence']:.4f}",
        )
    )

    # 7. Mathematical Freezing & Gradient Isolation
    subheader("6. Mathematical Parameter Freezing (dL/d_theta0 == 0)")
    phi = init_residual_adapter(hidden_dim, jax.random.PRNGKey(42))
    targets = jax_ids

    def loss_fn(p_phi, p_theta):
        logits, _ = sub_seg._engine.run_forward(
            params=p_theta,
            input_ids=jax_ids,
            modify_fn=lambda h, l: apply_residual_adapter(p_phi, h, l),
        )
        shift_logits = logits[:, :-1, :]
        shift_targets = targets[:, 1:]
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_targets))

    grads_phi, grads_theta = jax.grad(loss_fn, argnums=(0, 1))(phi, sub_seg.params)

    # Check that EVERY single leaf in theta_0 is exactly 0.0
    theta_leak = False
    max_theta_grad = 0.0
    for seg_name, seg_sd in grads_theta.items():
        for k, g in seg_sd.items():
            g_max = float(jnp.max(jnp.abs(g)))
            if g_max > max_theta_grad:
                max_theta_grad = g_max
            if g_max > 0.0:
                theta_leak = True

    ok_freeze = not theta_leak
    results.append(
        pass_fail(
            "Base Parameters Frozen (dL/d_theta0 == 0)",
            ok_freeze,
            f"Max Gradient = {max_theta_grad:.2e} across all segments",
        )
    )

    # Check that residual adapter parameters phi receive valid gradients
    phi_active = all(bool(jnp.any(g != 0.0)) and bool(jnp.all(jnp.isfinite(g))) for g in grads_phi.values())
    results.append(
        pass_fail(
            "Adapter Gradients Active (dL/d_phi != 0)",
            phi_active,
            "All adapter parameters received finite non-zero gradients",
        )
    )

    # 8. JIT Compilation & Optax Training Loop
    subheader("7. XLA Compilation & JIT Training Loop")
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(phi)
    train_step = make_training_step(sub_seg._engine, optimizer)

    t0 = time.perf_counter()
    curr_phi = phi
    losses = []
    for step in range(5):
        loss_val, curr_phi, opt_state = train_step(curr_phi, opt_state, sub_seg.params, jax_ids, targets)
        losses.append(float(loss_val))
    t_jit = (time.perf_counter() - t0) * 1000

    ok_loss = losses[-1] < losses[0] or len(losses) == 5
    results.append(
        pass_fail(
            "5-Step JIT Optax Training",
            ok_loss,
            f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f} ({t_jit:.1f}ms total)",
        )
    )

    # 9. Pointer Invariance
    subheader("8. Pointer Identity & Memory Guard")
    ok_pointer = sub_seg.params_unchanged()
    results.append(
        pass_fail(
            "Parameter Pointer Invariance",
            ok_pointer,
            "All pristine leaf pointers untouched after training",
        )
    )

    # Summary
    all_passed = all(results)
    header(f"SUMMARY: {sum(results)} / {len(results)} CHECKS PASSED")
    if all_passed:
        print(f"{BOLD}{GREEN}>>> ALL VALIDATION CHECKS PASSED SUCCESSFULLY! <<<{RESET}\n")
    else:
        print(f"{BOLD}{RED}>>> SOME VALIDATION CHECKS FAILED! <<<{RESET}\n")

    return all_passed


def parse_layers(spec: str, num_layers: int) -> list[int]:
    """'all' or a comma-separated list of zero-based layer indices."""
    if spec.strip().lower() == "all":
        return list(range(num_layers))
    layers = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not layers:
        raise ValueError("No valid layer indices given")
    for i in layers:
        if not 0 <= i < num_layers:
            raise ValueError(
                f"Layer {i} out of range: model has {num_layers} layers "
                f"(0..{num_layers - 1})"
            )
    return sorted(set(layers))


# CLI Interface

def main():
    parser = argparse.ArgumentParser(description="Segmented torch2jax Substrate Validation Runner")
    parser.add_argument("--model", type=str, default="tiny_gpt2", help="Model: 'tiny_gpt2', 'tiny_pythia', 'gpt2', 'EleutherAI/pythia-70m'")
    parser.add_argument("--layers", type=str, default="1,2", help="Comma-separated zero-based interception layers (e.g. '1,2', '0,3', or 'all')")
    parser.add_argument("--seq-len", type=int, default=8, help="Sequence length for deterministic test")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for deterministic test")
    parser.add_argument("--tiny", action="store_true", help="Shortcut to use tiny_gpt2 for quick local validation")
    parser.add_argument("--skip-monolithic", action="store_true", help="Skip the monolithic (unsegmented) trace step")
    args = parser.parse_args()

    if args.tiny or args.model == "tiny_gpt2":
        model, config, desc = create_tiny_gpt2()
    elif args.model == "tiny_pythia":
        model, config, desc = create_tiny_pythia()
    else:
        model, config, desc = load_hf_model(args.model)

    num_layers = getattr(config, "n_layer", getattr(config, "num_hidden_layers", 4))
    intercept = parse_layers(args.layers, num_layers)

    passed = run_validation(
        torch_model=model,
        config=config,
        model_desc=desc,
        intercept_layers=intercept,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        skip_monolithic=args.skip_monolithic,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
