"""Unit tests for substrate.architecture module.

Tests architecture detection from HuggingFace config objects and parameter trees,
block accessors, module accessors, and interception layer validation.
"""

from __future__ import annotations

from types import SimpleNamespace
import pytest

from substrate.architecture import (
    Architecture,
    detect_architecture,
    detect_architecture_from_config,
    discover_layers,
    discover_layers_from_config,
    get_block_accessor,
    get_embedding_module,
    get_head_modules,
    get_position_embedding_module,
    validate_interception_layers,
)


class TestDetectArchitectureFromConfig:
    def test_gpt2_config_object(self):
        cfg = SimpleNamespace(
            model_type="gpt2",
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
        cfg = SimpleNamespace(
            model_type="gpt_neox",
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
        cfg = SimpleNamespace(
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
        cfg = SimpleNamespace(model_type="llama", num_hidden_layers=32)
        with pytest.raises(ValueError, match="Unsupported or unrecognized"):
            detect_architecture_from_config(cfg)

    def test_indivisible_heads_raises(self):
        cfg = SimpleNamespace(
            model_type="gpt2",
            n_layer=4,
            n_embd=100,
            n_head=3,  # 100 is not divisible by 3
        )
        with pytest.raises(ValueError, match="not divisible by num_heads"):
            detect_architecture_from_config(cfg)

    def test_invalid_num_layers_raises(self):
        cfg = SimpleNamespace(model_type="gpt2", n_layer=0, n_embd=64, n_head=1)
        with pytest.raises(ValueError, match="Invalid num_layers"):
            detect_architecture_from_config(cfg)


class TestDiscoverLayersFromConfig:
    def test_discover_layers(self):
        cfg = {"model_type": "gpt2", "n_layer": 24, "n_embd": 1024, "n_head": 16}
        assert discover_layers_from_config(cfg) == 24


class MockSubmodule:
    def __init__(self, name: str):
        self.name = name


class MockGPT2Model:
    def __init__(self):
        self.transformer = SimpleNamespace(
            wte=MockSubmodule("wte"),
            wpe=MockSubmodule("wpe"),
            h=[MockSubmodule(f"h_{i}") for i in range(4)],
            ln_f=MockSubmodule("ln_f"),
        )
        self.lm_head = MockSubmodule("lm_head")


class MockNeoXModel:
    def __init__(self):
        self.gpt_neox = SimpleNamespace(
            embed_in=MockSubmodule("embed_in"),
            layers=[MockSubmodule(f"layer_{i}") for i in range(4)],
            final_layer_norm=MockSubmodule("final_ln"),
        )
        self.embed_out = MockSubmodule("embed_out")


class TestAccessors:
    def test_gpt2_accessors(self):
        model = MockGPT2Model()
        arch = Architecture(
            model_family="gpt2",
            num_layers=4,
            hidden_size=64,
            num_heads=2,
            head_dim=32,
            vocab_size=100,
        )

        # Block accessor
        accessor = get_block_accessor(model, arch)
        assert accessor(0).name == "h_0"
        assert accessor(3).name == "h_3"

        # Embedding accessors
        wte = get_embedding_module(model, arch)
        assert wte.name == "wte"
        wpe = get_position_embedding_module(model, arch)
        assert wpe is not None and wpe.name == "wpe"

        # Head modules
        head_mods = get_head_modules(model, arch)
        assert len(head_mods) == 2
        assert head_mods[0].name == "ln_f"
        assert head_mods[1].name == "lm_head"

    def test_neox_accessors(self):
        model = MockNeoXModel()
        arch = Architecture(
            model_family="neox",
            num_layers=4,
            hidden_size=64,
            num_heads=2,
            head_dim=32,
            vocab_size=100,
        )

        # Block accessor
        accessor = get_block_accessor(model, arch)
        assert accessor(0).name == "layer_0"
        assert accessor(2).name == "layer_2"

        # Embedding accessors
        embed_in = get_embedding_module(model, arch)
        assert embed_in.name == "embed_in"
        assert get_position_embedding_module(model, arch) is None

        # Head modules
        head_mods = get_head_modules(model, arch)
        assert len(head_mods) == 2
        assert head_mods[0].name == "final_ln"
        assert head_mods[1].name == "embed_out"

    def test_wrapped_model_accessor(self):
        inner = MockGPT2Model()
        wrapped = SimpleNamespace(module=inner)
        arch = Architecture(
            model_family="gpt2",
            num_layers=4,
            hidden_size=64,
            num_heads=2,
            head_dim=32,
            vocab_size=100,
        )
        accessor = get_block_accessor(wrapped, arch)
        assert accessor(1).name == "h_1"


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
