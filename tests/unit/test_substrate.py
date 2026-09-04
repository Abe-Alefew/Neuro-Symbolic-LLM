"""Unit tests for substrate.substrate (FrozenSubstrate with TorchAX execution).

Verifies:
1. FrozenSubstrate initialization with a live PyTorch model.
2. Architecture auto-detection and layer discovery.
3. Strict parameter immutability and theta_0 freezing invariant.
4. Forward pass execution with pure JAX logits and pristine intermediates.
5. In-flight hidden state interception and custom steering hooks.
6. Clean hook lifecycle (all hooks detached after execution).
7. Causal LM loss computation with label shifting and padding masking.
8. Memory status and input validation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from substrate.substrate import ForwardResult, FrozenSubstrate
from substrate.torchax_backend import enable_torchax, from_jax_array, to_jax_array

enable_torchax()


def _tiny_gpt2_model() -> GPT2LMHeadModel:
    """Create a small local GPT-2 model without needing network access."""
    cfg = GPT2Config(
        n_layer=4,
        n_head=2,
        n_embd=32,
        vocab_size=100,
        n_positions=16,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model


class TestFrozenSubstrateInit:
    def test_initialization_with_torch_model(self):
        model = _tiny_gpt2_model()
        sub = FrozenSubstrate(model, intercept_layers=[1, 3])

        assert sub.architecture.model_family == "gpt2"
        assert sub.architecture.num_layers == 4
        assert sub.architecture.hidden_size == 32
        assert sub.intercept_layers == (1, 3)

    def test_parameters_are_strictly_frozen(self):
        model = _tiny_gpt2_model()
        sub = FrozenSubstrate(model)

        # Invariant: every parameter leaf has requires_grad=False
        for name, param in sub.params.items():
            assert not param.requires_grad, f"Param {name} was not frozen!"

        assert sub.params_unchanged()
        report = sub.verify_frozen()
        assert report["params_unchanged"] is True
        assert report["architecture"]["num_layers"] == 4

    def test_invalid_interception_layers_rejected(self):
        model = _tiny_gpt2_model()
        with pytest.raises(ValueError, match="non-negative"):
            FrozenSubstrate(model, intercept_layers=[-1])

        with pytest.raises(ValueError, match="only 4 transformer layers"):
            FrozenSubstrate(model, intercept_layers=[4])


class TestFrozenSubstrateForward:
    def test_forward_produces_jax_logits(self):
        model = _tiny_gpt2_model()
        sub = FrozenSubstrate(model, intercept_layers=[1])

        batch, seq_len = 2, 8
        input_ids = jnp.zeros((batch, seq_len), dtype=jnp.int32)

        result = sub(input_ids)

        assert isinstance(result, ForwardResult)
        assert isinstance(result.logits, jax.Array)
        assert result.logits.shape == (batch, seq_len, 100)

        # Intermediates check
        assert result.layer_indices() == (1,)
        h1 = result.hidden_state(1)
        assert isinstance(h1, jax.Array)
        assert h1.shape == (batch, seq_len, 32)
        assert result.hidden_shapes() == {1: (batch, seq_len, 32)}

    def test_input_shape_validation(self):
        model = _tiny_gpt2_model()
        sub = FrozenSubstrate(model)

        with pytest.raises(ValueError, match="2D"):
            sub(jnp.zeros((4,), dtype=jnp.int32))

        with pytest.raises(ValueError, match="at least one"):
            sub(jnp.zeros((2, 0), dtype=jnp.int32))

    def test_tokenize_helper(self):
        # Mock tokenizer returning PyTorch tensor
        class MockTokenizer:
            def __call__(self, text, return_tensors="pt", **kw):
                return {"input_ids": torch.tensor([[10, 20, 30]])}

        mock_tok = MockTokenizer()
        model = _tiny_gpt2_model()
        sub = FrozenSubstrate(model, tokenizer=mock_tok)

        assert sub.tokenizer is mock_tok
        ids_jax = sub.tokenize("test prompt", return_tensors="jax")
        assert isinstance(ids_jax, jax.Array)
        assert ids_jax.shape == (1, 3)

        ids_pt = sub.tokenize("test prompt", return_tensors="pt")
        assert isinstance(ids_pt, torch.Tensor)
        assert ids_pt.shape == (1, 3)


class TestInterceptionAndSteering:
    def test_steering_hook_modifies_output(self):
        model = _tiny_gpt2_model()
        sub = FrozenSubstrate(model, intercept_layers=[2])

        input_ids = jnp.ones((1, 4), dtype=jnp.int32)

        # 1. Baseline unsteered run
        baseline = sub(input_ids)

        # 2. Steered run: add dimension-varying perturbation at layer 2
        def steer(h: jax.Array, layer_idx: int) -> jax.Array:
            delta = jnp.linspace(1.0, 5.0, h.shape[-1])
            return h + delta

        steered = sub.run_with_interception(
            input_ids=input_ids,
            modify_fn=steer,
            intercept_layers=[2],
        )

        # Both results return valid JAX arrays
        assert isinstance(steered.logits, jax.Array)

        # Logits must diverge due to downstream propagation
        diff = float(jnp.max(jnp.abs(steered.logits - baseline.logits)))
        assert diff > 0.0, "Steering did not affect downstream logits!"

        # Pristine intermediate at layer 2 should match between both runs
        pristine_diff = float(
            jnp.max(jnp.abs(steered.hidden_state(2) - baseline.hidden_state(2)))
        )
        assert pristine_diff < 1e-5, "Intermediates did not capture the pristine state!"

    def test_clean_hook_lifecycle(self):
        model = _tiny_gpt2_model()
        sub = FrozenSubstrate(model, intercept_layers=[1, 2])

        input_ids = jnp.zeros((1, 4), dtype=jnp.int32)
        _ = sub(input_ids)

        # Hooks must be cleaned up after forward pass
        for block in model.transformer.h:
            assert len(block._forward_hooks) == 0


class TestCausalLossComputation:
    def test_compute_loss_causal_shift(self):
        batch, seq_len, vocab_size = 2, 6, 100
        # Deterministic logits
        logits = jnp.zeros((batch, seq_len, vocab_size))
        labels = jnp.zeros((batch, seq_len), dtype=jnp.int32)

        loss = FrozenSubstrate.compute_loss(logits, labels)

        assert isinstance(loss, jax.Array)
        assert loss.ndim == 0
        # Uniform distribution over 100 tokens: -ln(1/100) = ln(100) ≈ 4.605
        expected = float(jnp.log(vocab_size))
        assert abs(float(loss) - expected) < 1e-4

    def test_compute_loss_ignores_padding(self):
        batch, seq_len, vocab_size = 1, 4, 10
        logits = jnp.zeros((batch, seq_len, vocab_size))
        # Mask out all target positions except one
        labels = jnp.array([[-100, -100, 0, -100]], dtype=jnp.int32)

        loss = FrozenSubstrate.compute_loss(logits, labels)
        expected = float(jnp.log(vocab_size))
        assert abs(float(loss) - expected) < 1e-4


class TestMemoryStatus:
    def test_memory_status_returns_valid_object(self):
        model = _tiny_gpt2_model()
        sub = FrozenSubstrate(model)

        status = sub.memory_status()
        assert status is not None
        assert hasattr(status, "allocated_bytes")
        assert hasattr(status, "bytes_in_use")
        assert hasattr(status, "available")
