"""Frozen LLM substrate based in JAX/Flax.

A reusable wrapper around pretrained causal language models (GPT-2 and
Pythia/GPT-NeoX) that keeps the base model completely frozen while allowing
arbitrary per-layer hidden-state interception, activation caching, drift
monitoring and device memory monitoring.
"""

from .architecture import (
    Architecture,
    detect_architecture,
    detect_architecture_from_config,
    discover_layers_from_config,
    get_block_accessor,
    get_embedding_module,
    get_head_modules,
    get_position_embedding_module,
    discover_layers,
    validate_interception_layers,
)
from .drift import compute_kl_drift
from .loader import (
    build_substrate_from_state_dict,
    load_substrate_from_hf,
    state_dict_to_jax_pytree,
)
from .memory import (
    MemoryStatus,
    check_memory_headroom,
    compute_memory_headroom,
    get_memory_status,
    maybe_reduce_batch_size,
)
from .torchax_backend import (
    enable_torchax,
    from_jax_array,
    
    is_on_torchax_device,
    is_torchax_enabled,
    to_jax_array,
    to_torchax_device,
)

from .torchax_models import (
    check_numerical_fidelity,
    functional_model,
    load_tokenizer,
    load_torchax_model,
)

from .substrate import ForwardResult, FrozenJAXSubstrate


__all__ = [
    "Architecture",
    "ForwardResult",
    "FrozenJAXSubstrate",
    "MemoryStatus",
    "build_substrate_from_state_dict",
    "check_memory_headroom",
    "check_numerical_fidelity",
    "compute_kl_drift",
    "compute_memory_headroom",
    "detect_architecture",
    "detect_architecture_from_config",
    "discover_layers",
    "discover_layers_from_config",
    "from_jax_array",
    "get_block_accessor",
    "get_embedding_module",
    "get_head_modules",
    "get_position_embedding_module",
    "get_memory_status",
    "load_substrate_from_hf",
    "load_tokenizer",
    "load_torchax_model",
    "maybe_reduce_batch_size",
    "state_dict_to_jax_pytree",
    "to_jax_array",
    "to_torchax_device",
    "validate_interception_layers",
    "enable_torchax",
    
    "is_on_torchax_device",
    "is_torchax_enabled",
    
    "functional_model",
]

__version__ = "0.1.0"
