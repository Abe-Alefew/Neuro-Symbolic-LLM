"""Frozen LLM substrate in JAX/Flax.

A reusable wrapper around pretrained causal language models (GPT-2 and
Pythia/GPT-NeoX) that keeps the base model completely frozen while allowing
arbitrary per-layer hidden-state interception, activation caching, drift
monitoring and device memory monitoring.
"""

from .architecture import (
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
from .drift import compute_kl_drift
from .interception import (
    InterceptionContext,
    ModifyFn,
    identity_modify,
    run_with_hooks,
    run_with_interception,
)
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
from .substrate import ForwardResult, FrozenJAXSubstrate, FrozenSubstrate
from .torchax_backend import (
    enable_torchax,
    from_jax_array,
    initialize_torchax,
    is_on_torchax_device,
    is_torchax_enabled,
    model_to_jax,
    to_jax_array,
    to_torchax_device,
)
from .torchax_gpt2 import (
    check_numerical_fidelity,
    functional_gpt2,
    load_tokenizer,
    load_torchax_gpt2,
)

__all__ = [
    "Architecture",
    "ForwardResult",
    "FrozenJAXSubstrate",
    "FrozenSubstrate",
    "InterceptionContext",
    "MemoryStatus",
    "ModifyFn",
    "build_substrate_from_state_dict",
    "check_memory_headroom",
    "check_numerical_fidelity",
    "compute_kl_drift",
    "compute_memory_headroom",
    "detect_architecture",
    "detect_architecture_from_config",
    "discover_layers",
    "discover_layers_from_config",
    "enable_torchax",
    "from_jax_array",
    "functional_gpt2",
    "get_block_accessor",
    "get_embedding_module",
    "get_head_modules",
    "get_memory_status",
    "get_position_embedding_module",
    "identity_modify",
    "initialize_torchax",
    "is_on_torchax_device",
    "is_torchax_enabled",
    "load_substrate_from_hf",
    "load_tokenizer",
    "load_torchax_gpt2",
    "maybe_reduce_batch_size",
    "model_to_jax",
    "run_with_hooks",
    "run_with_interception",
    "state_dict_to_jax_pytree",
    "to_jax_array",
    "to_torchax_device",
    "validate_interception_layers",
]

__version__ = "0.1.0"
