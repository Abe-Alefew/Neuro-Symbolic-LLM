"""Comprehensive Test Suite for Segmented torch2jax Substrate (Mechanism M2)."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import torch
import optax
from transformers import GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM

from substrate.substrate import FrozenJAXSubstrate
from substrate.models import SegmentedTorch2JaxEngine, validate_segment_state_dict
from substrate.adapters import init_residual_adapter, apply_residual_adapter, make_training_step


# Test Fixtures
@pytest.fixture
def gpt2_tiny():
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
    ids = np.random.randint(0, cfg.vocab_size, size=(2, 8))
    return model, cfg, jnp.asarray(ids)


@pytest.fixture
def pythia_tiny():
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
    ids = np.random.randint(0, cfg.vocab_size, size=(2, 8))
    return model, cfg, jnp.asarray(ids)


# Test 1: Logit Parity
def test_1_gpt2_logit_parity(gpt2_tiny):
    """Test 1: Output logit parity against PyTorch reference (< 1e-5)."""
    torch_model, cfg, ids = gpt2_tiny
    substrate = FrozenJAXSubstrate(torch_model, cfg, intercept_layers=[1, 2])

    with torch.no_grad():
        ref_logits = torch_model(torch.from_numpy(np.asarray(ids))).logits.numpy()

    # Eager forward
    res_eager = substrate(ids)
    max_err_eager = np.max(np.abs(ref_logits - np.asarray(res_eager.logits)))

    # JIT forward
    jitted_sub = jax.jit(substrate)
    res_jit = jitted_sub(ids)
    max_err_jit = np.max(np.abs(ref_logits - np.asarray(res_jit.logits)))

    assert max_err_eager <= 1e-5, f"Eager logit error {max_err_eager} > 1e-5"
    assert max_err_jit <= 1e-5, f"JIT logit error {max_err_jit} > 1e-5"


def test_1_pythia_logit_parity(pythia_tiny):
    """Test 1 (Pythia): Untied Pythia logit parity (< 1e-5)."""
    torch_model, cfg, ids = pythia_tiny
    substrate = FrozenJAXSubstrate(torch_model, cfg, intercept_layers=[1, 2])

    with torch.no_grad():
        ref_logits = torch_model(torch.from_numpy(np.asarray(ids))).logits.numpy()

    res = substrate(ids)
    max_err = np.max(np.abs(ref_logits - np.asarray(res.logits)))
    assert max_err <= 1e-5, f"Pythia logit error {max_err} > 1e-5"


# Test 2: Hidden-State Parity
def test_2_hidden_state_parity(gpt2_tiny):
    """Test 2: Intermediate hidden-state parity at interception points (< 1e-6)."""
    torch_model, cfg, ids = gpt2_tiny
    substrate = FrozenJAXSubstrate(torch_model, cfg, intercept_layers=[0, 2])

    with torch.no_grad():
        out = torch_model(torch.from_numpy(np.asarray(ids)), output_hidden_states=True)
        ref_h0 = out.hidden_states[1].numpy()  # Layer 0 output
        ref_h2 = out.hidden_states[3].numpy()  # Layer 2 output

    res = substrate(ids)
    err_h0 = np.max(np.abs(ref_h0 - np.asarray(res.hidden_state(0))))
    err_h2 = np.max(np.abs(ref_h2 - np.asarray(res.hidden_state(2))))

    assert err_h0 <= 1e-6, f"Layer 0 hidden state error {err_h0} > 1e-6"
    assert err_h2 <= 1e-6, f"Layer 2 hidden state error {err_h2} > 1e-6"


# Test 3: Modification Propagation
def test_3_modification_propagation(gpt2_tiny):
    """Test 3: Verify h' = h + delta alters downstream states and logits."""
    torch_model, cfg, ids = gpt2_tiny
    delta = jnp.linspace(0.5, 2.0, cfg.n_embd)

    base_sub = FrozenJAXSubstrate(torch_model, cfg, intercept_layers=[1])
    base_res = base_sub(ids)

    mod_sub = FrozenJAXSubstrate(
        torch_model,
        cfg,
        intercept_layers=[1],
        modify_hook=lambda h, l: h + delta,
    )
    mod_res = mod_sub(ids)

    # 1. Pristine cache at layer 1 remains untouched
    np.testing.assert_allclose(
        np.asarray(base_res.hidden_state(1)),
        np.asarray(mod_res.hidden_state(1)),
        atol=1e-7,
    )
    # 2. Output logits changed significantly
    logit_diff = np.max(np.abs(np.asarray(base_res.logits) - np.asarray(mod_res.logits)))
    assert logit_diff > 0.1, "Logits failed to respond to perturbation."


#  Test 4: JIT Compilation
def test_4_jit_compilation(gpt2_tiny):
    """Test 4: Full segmented forward compiles cleanly under JIT."""
    torch_model, cfg, ids = gpt2_tiny
    sub = FrozenJAXSubstrate(torch_model, cfg, intercept_layers=[0, 1, 2])

    @jax.jit
    def jitted_run(x):
        return sub(x)

    res = jitted_run(ids)
    assert bool(jnp.all(jnp.isfinite(res.logits)))


# Tests 5 & 6: Gradient Isolation
def test_5_and_6_gradient_isolation(gpt2_tiny):
    """Tests 5 & 6: grad(theta_0) == 0.0 leaf-by-leaf AND grad(phi) != 0.0."""
    torch_model, cfg, ids = gpt2_tiny
    sub = FrozenJAXSubstrate(torch_model, cfg, intercept_layers=[1])
    targets = ids
    phi = init_residual_adapter(cfg.n_embd, jax.random.PRNGKey(42))

    def loss_fn(p_phi, p_theta):
        logits, _ = sub._engine.run_forward(
            p_theta,
            ids,
            modify_fn=lambda h, l: apply_residual_adapter(p_phi, h, l),
        )
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))

    # Compute joint gradients
    grads_phi, grads_theta = jax.grad(loss_fn, argnums=(0, 1))(phi, sub.params)

    # Test 6: Every single leaf of theta0 must be EXACTLY zero
    for seg_name, seg_dict in grads_theta.items():
        for k, grad_tensor in seg_dict.items():
            assert jnp.all(grad_tensor == 0.0), f"Gradient leaked into theta0[{seg_name}][{k}]!"

    # Test 5: Residual parameters phi must receive non-zero finite gradients
    for k, g in grads_phi.items():
        assert jnp.any(g != 0.0), f"Gradient for residual phi[{k}] is zero!"
        assert jnp.all(jnp.isfinite(g)), f"Gradient for phi[{k}] contains NaN/Inf!"


# Test 7: Pristine Cache Invariance
def test_7_cache_is_pristine(gpt2_tiny):
    """Test 7: intermediates[l] contains untouched h0 even under strong modification."""
    torch_model, cfg, ids = gpt2_tiny
    base_sub = FrozenJAXSubstrate(torch_model, cfg, intercept_layers=[1])
    base_res = base_sub(ids)

    mod_sub = FrozenJAXSubstrate(
        torch_model,
        cfg,
        intercept_layers=[1],
        modify_hook=lambda h, l: h * 50.0 + 10.0,
    )
    mod_res = mod_sub(ids)

    np.testing.assert_allclose(
        np.asarray(mod_res.hidden_state(1)),
        np.asarray(base_res.hidden_state(1)),
        atol=1e-7,
    )


# Test 8: Multiple Interceptions
def test_8_multiple_interceptions(gpt2_tiny):
    """Test 8: Verify multi-cut configurations [1, 3] and [0, 2, 3]."""
    torch_model, cfg, ids = gpt2_tiny

    for layers in ([1, 3], [0, 2, 3]):
        sub = FrozenJAXSubstrate(torch_model, cfg, intercept_layers=layers)
        assert len(sub._engine.segments) == len(layers) + 1
        res = sub(ids)
        for lyr in layers:
            assert lyr in res.intermediates
            assert res.hidden_state(lyr).shape == (2, 8, cfg.n_embd)


# Test 9: State-Dict Validation
def test_9_state_dict_validation_error(gpt2_tiny):
    """Test 9: Verify state_dict validator detects missing and unexpected keys."""
    torch_model, cfg, _ = gpt2_tiny
    engine = SegmentedTorch2JaxEngine(torch_model, intercept_layers=[1])
    seg0 = engine.segments[0]

    # Missing key
    invalid_sd = {k: jnp.zeros(v.shape) for k, v in seg0.state_dict().items()}
    del invalid_sd["wte.weight"]
    with pytest.raises(KeyError, match="Missing keys"):
        validate_segment_state_dict("test_seg", seg0, invalid_sd)

    # Unexpected key
    invalid_sd2 = {k: jnp.zeros(v.shape) for k, v in seg0.state_dict().items()}
    invalid_sd2["bogus.bias"] = jnp.zeros((32,))
    with pytest.raises(KeyError, match="Unexpected keys"):
        validate_segment_state_dict("test_seg", seg0, invalid_sd2)
