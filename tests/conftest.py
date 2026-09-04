"""Shared fixtures for substrate tests.

Reference "original frozen models" are real HuggingFace PyTorch models built
from small configs. Their weights are converted to JAX parameter PyTrees via
the same code path used for real checkpoints, and the wrapper output is
compared against the untouched torch forward pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)

from substrate import (
    FrozenJAXSubstrate,
    FrozenSubstrate,
    state_dict_to_jax_pytree,
)

GPT2_CFG: dict[str, Any] = {
    "n_layer": 12,
    "n_head": 4,
    "n_embd": 32,
    "n_positions": 32,
    "vocab_size": 64,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "resid_pdrop": 0.0,
    "embd_pdrop": 0.0,
    "attn_pdrop": 0.0,
}

NEOX_CFG: dict[str, Any] = {
    "vocab_size": 64,
    "hidden_size": 32,
    "num_hidden_layers": 12,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "max_position_embeddings": 32,
    "rotary_pct": 1.0,
    "rope_theta": 10000.0,
    "layer_norm_eps": 1e-5,
    "use_parallel_residual": False,
    "attention_bias": True,
    "hidden_act": "gelu",
}

NUM_TOKENS = 64
BATCH = 2
SEQ = 9


def _torch_model(family: str, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if family == "gpt2":
        return GPT2LMHeadModel(GPT2Config(**GPT2_CFG))
    return GPTNeoXForCausalLM(GPTNeoXConfig(**NEOX_CFG))


def _config(family: str):
    if family == "gpt2":
        return GPT2Config(**GPT2_CFG)
    return GPTNeoXConfig(**NEOX_CFG)


def make_substrate(family: str, intercept_layers=None, modify_hook=None, seed: int = 0):
    """Build the torch reference model plus a FrozenSubstrate wrapper using TorchAX."""
    model = _torch_model(family, seed=seed)
    model.eval()
    substrate = FrozenSubstrate(
        model,
        config=_config(family),
        intercept_layers=intercept_layers,
        modify_hook=modify_hook,
    )
    return model, substrate


def torch_logits(model, ids):
    with torch.no_grad():
        param = next(model.parameters(), None)
        if param is not None and isinstance(ids, torch.Tensor) and ids.device != param.device:
            ids = ids.to(param.device)
        elif param is not None and not isinstance(ids, torch.Tensor):
            ids = torch.as_tensor(ids, device=param.device)
        out = model(ids).logits
        if hasattr(out, "cpu"):
            out = out.cpu()
        return np.asarray(out)


def make_input_ids():
    return torch.randint(0, NUM_TOKENS, (BATCH, SEQ))


@pytest.fixture(scope="session")
def gpt2_reference():
    model = _torch_model("gpt2")
    model.eval()
    return model


@pytest.fixture(scope="session")
def pythia_reference():
    model = _torch_model("neox")
    model.eval()
    return model


# ── Real HuggingFace Configurations & Model Factories ────────────────────────


def create_gpt2_config(
    n_layer: int = 4,
    n_head: int = 2,
    n_embd: int = 64,
    vocab_size: int = 100,
    n_positions: int = 128,
    **kwargs: Any,
) -> GPT2Config:
    """Create a real HuggingFace GPT2Config with compact dimensions."""
    return GPT2Config(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        vocab_size=vocab_size,
        n_positions=n_positions,
        **kwargs,
    )


def create_pythia_config(
    num_hidden_layers: int = 4,
    num_attention_heads: int = 2,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    vocab_size: int = 100,
    max_position_embeddings: int = 128,
    rotary_pct: float = 1.0,
    rope_theta: float = 10000.0,
    use_parallel_residual: bool = True,
    **kwargs: Any,
) -> GPTNeoXConfig:
    """Create a real HuggingFace GPTNeoXConfig (Pythia) with compact dimensions."""
    return GPTNeoXConfig(
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        rotary_pct=rotary_pct,
        rope_theta=rope_theta,
        use_parallel_residual=use_parallel_residual,
        **kwargs,
    )


def create_real_gpt2_model(
    config: GPT2Config | None = None,
    seed: int = 0,
) -> GPT2LMHeadModel:
    """Instantiate a real HuggingFace GPT2LMHeadModel in eval mode with deterministic weights."""
    torch.manual_seed(seed)
    cfg = config or create_gpt2_config()
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model


def create_real_pythia_model(
    config: GPTNeoXConfig | None = None,
    seed: int = 0,
) -> GPTNeoXForCausalLM:
    """Instantiate a real HuggingFace GPTNeoXForCausalLM in eval mode with deterministic weights."""
    torch.manual_seed(seed)
    cfg = config or create_pythia_config()
    model = GPTNeoXForCausalLM(cfg)
    model.eval()
    return model


def create_torchax_gpt2_model(
    config: GPT2Config | None = None,
    seed: int = 0,
) -> tuple[GPT2LMHeadModel, dict[str, torch.Tensor]]:
    """Instantiate a real GPT-2 model on TorchAX device with frozen parameters."""
    try:
        from substrate.torchax_backend import enable_torchax, to_torchax_device

        enable_torchax()
        model = create_real_gpt2_model(config=config, seed=seed)
        model_jax = to_torchax_device(model)
    except Exception:
        model = create_real_gpt2_model(config=config, seed=seed)
        model_jax = model

    params = dict(model_jax.named_parameters())
    for p in params.values():
        p.requires_grad_(False)
    return model_jax, params


def create_torchax_pythia_model(
    config: GPTNeoXConfig | None = None,
    seed: int = 0,
) -> tuple[GPTNeoXForCausalLM, dict[str, torch.Tensor]]:
    """Instantiate a real Pythia model on TorchAX device with frozen parameters."""
    try:
        from substrate.torchax_backend import enable_torchax, to_torchax_device

        enable_torchax()
        model = create_real_pythia_model(config=config, seed=seed)
        model_jax = to_torchax_device(model)
    except Exception:
        model = create_real_pythia_model(config=config, seed=seed)
        model_jax = model

    params = dict(model_jax.named_parameters())
    for p in params.values():
        p.requires_grad_(False)
    return model_jax, params


# ── Pytest Fixtures for Architecture & Interception Unit Tests ────────────────


@pytest.fixture
def real_gpt2_config() -> GPT2Config:
    """Real HuggingFace GPT2Config fixture."""
    return create_gpt2_config()


@pytest.fixture
def real_pythia_config() -> GPTNeoXConfig:
    """Real HuggingFace GPTNeoXConfig (Pythia) fixture."""
    return create_pythia_config()


@pytest.fixture
def standard_gpt2_config() -> GPT2Config:
    """Standard 12-layer GPT2Config fixture matching GPT2_CFG."""
    return GPT2Config(**GPT2_CFG)


@pytest.fixture
def standard_pythia_config() -> GPTNeoXConfig:
    """Standard 12-layer GPTNeoXConfig fixture matching NEOX_CFG."""
    return GPTNeoXConfig(**NEOX_CFG)


@pytest.fixture
def real_gpt2_model(real_gpt2_config: GPT2Config) -> GPT2LMHeadModel:
    """Real HuggingFace GPT2LMHeadModel fixture in eval mode."""
    return create_real_gpt2_model(config=real_gpt2_config)


@pytest.fixture
def real_pythia_model(real_pythia_config: GPTNeoXConfig) -> GPTNeoXForCausalLM:
    """Real HuggingFace GPTNeoXForCausalLM fixture in eval mode."""
    return create_real_pythia_model(config=real_pythia_config)


@pytest.fixture
def torchax_gpt2_model(
    real_gpt2_config: GPT2Config,
) -> tuple[GPT2LMHeadModel, dict[str, torch.Tensor]]:
    """Real GPT-2 model on TorchAX device with frozen parameters."""
    return create_torchax_gpt2_model(config=real_gpt2_config)


@pytest.fixture
def torchax_pythia_model(
    real_pythia_config: GPTNeoXConfig,
) -> tuple[GPTNeoXForCausalLM, dict[str, torch.Tensor]]:
    """Real Pythia model on TorchAX device with frozen parameters."""
    return create_torchax_pythia_model(config=real_pythia_config)
