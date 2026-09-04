"""Unit tests for substrate.architecture module.

Tests architecture detection from HuggingFace config objects and parameter trees,
block accessors, module accessors, and interception layer validation using real
HuggingFace GPT-2 and Pythia/GPT-NeoX models and configurations.
"""

from __future__ import annotations

import pytest
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    PretrainedConfig,
)

from substrate.architecture import (
    Architecture,
    detect_architecture_from_config,
    discover_layers_from_config,
    get_block_accessor,
    get_embedding_module,
    get_head_modules,
    get_position_embedding_module,
    validate_interception_layers,
)


class TestDetectArchitectureFromConfig:
    def test_gpt2_config_object(self):
        cfg = GPT2Config(
            n_layer=12,
            n_embd=768,
            n_head=12,
            vocab_size=50257,
            n_positions=1024,
            layer_norm_epsilon=1e-5,
        )
        arch = detect_architecture_from_config(cfg)
        assert arch.model_family == "gpt2"
        assert arch.num_layers == 12
        assert arch.hidden_size == 768
        assert arch.num_heads == 12
        assert arch.head_dim == 64
        assert arch.vocab_size == 50257
        assert arch.max_position_embeddings == 1024
        assert arch.layer_norm_eps == 1e-5
        assert not arch.use_parallel_residual

    def test_gpt2_config_dict(self):
        cfg_dict = {
            "model_type": "gpt2",
            "n_layer": 6,
            "n_embd": 512,
            "n_head": 8,
            "vocab_size": 32000,
            "n_positions": 512,
        }
        arch = detect_architecture_from_config(cfg_dict)
        assert arch.model_family == "gpt2"
        assert arch.num_layers == 6
        assert arch.hidden_size == 512
        assert arch.num_heads == 8
        assert arch.head_dim == 64
        assert arch.vocab_size == 32000

    def test_neox_config_object(self):
        cfg = GPTNeoXConfig(
            num_hidden_layers=16,
            hidden_size=1024,
            num_attention_heads=16,
            vocab_size=50304,
            max_position_embeddings=2048,
            layer_norm_eps=1e-5,
            rope_theta=10000.0,
            rotary_pct=0.25,
            use_parallel_residual=True,
        )
        arch = detect_architecture_from_config(cfg)
        assert arch.model_family == "neox"
        assert arch.num_layers == 16
        assert arch.hidden_size == 1024
        assert arch.num_heads == 16
        assert arch.head_dim == 64
        assert arch.vocab_size == 50304
        assert arch.max_position_embeddings == 2048
        assert arch.rotary_pct == 0.25
        assert arch.use_parallel_residual is True

    def test_neox_config_dict(self):
        cfg_dict = {
            "model_type": "gpt_neox",
            "num_hidden_layers": 8,
            "hidden_size": 512,
            "num_attention_heads": 8,
        }
        arch = detect_architecture_from_config(cfg_dict)
        assert arch.model_family == "neox"
        assert arch.num_layers == 8
        assert arch.hidden_size == 512
        assert arch.num_heads == 8
        assert arch.head_dim == 64
        assert arch.use_parallel_residual is True

    def test_pythia_detection_via_architectures(self):
        cfg = GPTNeoXConfig(
            architectures=["GPTNeoXForCausalLM"],
            num_hidden_layers=6,
            hidden_size=256,
            num_attention_heads=4,
        )
        arch = detect_architecture_from_config(cfg)
        assert arch.model_family == "neox"
        assert arch.num_layers == 6
        assert arch.hidden_size == 256
        assert arch.head_dim == 64

    def test_unsupported_model_type_raises(self):
        class LlamaLikeConfig(PretrainedConfig):
            model_type = "llama"

        cfg = LlamaLikeConfig(num_hidden_layers=32)
        with pytest.raises(ValueError, match="Unsupported or unrecognized"):
            detect_architecture_from_config(cfg)

    def test_indivisible_heads_raises(self):
        cfg = GPT2Config(
            n_layer=4,
            n_embd=100,
            n_head=3,  # 100 is not divisible by 3
        )
        with pytest.raises(ValueError, match="not divisible by num_heads"):
            detect_architecture_from_config(cfg)

    def test_invalid_num_layers_raises(self):
        cfg = GPT2Config(n_layer=0, n_embd=64, n_head=1)
        with pytest.raises(ValueError, match="Invalid num_layers"):
            detect_architecture_from_config(cfg)


class TestDiscoverLayersFromConfig:
    def test_discover_layers(self):
        cfg = GPT2Config(n_layer=24, n_embd=1024, n_head=16)
        assert discover_layers_from_config(cfg) == 24


# ── Real Model Factory Functions ─────────────────────────────────────────────

def _real_gpt2_model() -> GPT2LMHeadModel:
    """Instantiate a real HuggingFace GPT-2 model with compact dimensions."""
    cfg = GPT2Config(
        n_layer=4,
        n_head=2,
        n_embd=64,
        vocab_size=100,
        n_positions=128,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model


def _real_neox_model() -> GPTNeoXForCausalLM:
    """Instantiate a real HuggingFace Pythia/GPT-NeoX model with compact dimensions."""
    cfg = GPTNeoXConfig(
        num_hidden_layers=4,
        num_attention_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=128,
        use_parallel_residual=True,
    )
    model = GPTNeoXForCausalLM(cfg)
    model.eval()
    return model


# ── Accessor Tests with Real Models ──────────────────────────────────────────

class TestAccessors:
    def test_gpt2_accessors(self):
        model = _real_gpt2_model()
        arch = detect_architecture_from_config(model.config)

        # Block accessor
        accessor = get_block_accessor(model, arch)
        assert accessor(0) is model.transformer.h[0]
        assert accessor(3) is model.transformer.h[3]

        # Embedding accessors
        wte = get_embedding_module(model, arch)
        assert wte is model.transformer.wte
        wpe = get_position_embedding_module(model, arch)
        assert wpe is not None
        assert wpe is model.transformer.wpe

        # Head modules
        head_mods = get_head_modules(model, arch)
        assert len(head_mods) == 2
        assert head_mods[0] is model.transformer.ln_f
        assert head_mods[1] is model.lm_head

    def test_neox_accessors(self):
        model = _real_neox_model()
        arch = detect_architecture_from_config(model.config)

        # Block accessor
        accessor = get_block_accessor(model, arch)
        assert accessor(0) is model.gpt_neox.layers[0]
        assert accessor(2) is model.gpt_neox.layers[2]

        # Embedding accessors
        embed_in = get_embedding_module(model, arch)
        assert embed_in is model.gpt_neox.embed_in
        assert get_position_embedding_module(model, arch) is None

        # Head modules
        head_mods = get_head_modules(model, arch)
        assert len(head_mods) == 2
        assert head_mods[0] is model.gpt_neox.final_layer_norm
        assert head_mods[1] is model.embed_out

    def test_wrapped_model_accessor(self):
        inner = _real_gpt2_model()

        class WrappedModel(torch.nn.Module):
            def __init__(self, mod: torch.nn.Module):
                super().__init__()
                self.module = mod

        wrapped = WrappedModel(inner)
        arch = detect_architecture_from_config(inner.config)
        accessor = get_block_accessor(wrapped, arch)
        assert accessor(1) is inner.transformer.h[1]


# ── Interception Layer Validation Tests ───────────────────────────────────────

class TestValidateInterceptionLayers:
    def test_valid_layers(self):
        assert validate_interception_layers([3, 1, 0], 4) == (0, 1, 3)
        assert validate_interception_layers(None, 4) == ()
        assert validate_interception_layers([], 4) == ()

    def test_out_of_range(self):
        with pytest.raises(ValueError, match="Invalid interception layer"):
            validate_interception_layers([4], 4)

    def test_negative_layer(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_interception_layers([-1], 4)

    def test_duplicate_layer(self):
        with pytest.raises(ValueError, match="Duplicate interception layer"):
            validate_interception_layers([1, 1], 4)