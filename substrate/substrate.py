"""Frozen LLM substrate: a reusable JAX wrapper around pretrained causal LMs.

Uses Segmented torch2jax Mechanism M2 to convert PyTorch segments to JAX dynamically
and compile them under XLA, while enforcing mathematical freezing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import flax
from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn

from .architecture import (
    Architecture,
    detect_architecture,
    validate_interception_layers,
)
from .memory import (
    MemoryStatus,
    check_memory_headroom,
    get_memory_status,
    maybe_reduce_batch_size,
)
from .models import SegmentedTorch2JaxEngine


@dataclass(frozen=True)
class ForwardResult:
    """Output of a substrate forward pass.

    ``intermediates`` maps each intercepted zero-based layer index to the
    pre-modification hidden state cached at that layer.
    """

    logits: jax.Array
    intermediates: dict[int, jax.Array]

    def layer_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self.intermediates.keys()))

    def hidden_state(self, layer_idx: int) -> jax.Array:
        if layer_idx not in self.intermediates:
            raise KeyError(
                f"Layer {layer_idx} was not intercepted. Intercepted layers: "
                f"{self.layer_indices()}"
            )
        return self.intermediates[layer_idx]

    def hidden_shapes(self) -> dict[int, tuple[int, ...]]:
        return {i: tuple(h.shape) for i, h in self.intermediates.items()}


jax.tree_util.register_dataclass(
    ForwardResult, data_fields=["logits", "intermediates"], meta_fields=[]
)


class FrozenJAXSubstrate:
    """Frozen LLM Substrate backed by Segmented torch2jax XLA compilation."""

    def __init__(
        self,
        torch_model: nn.Module | Mapping[str, Any] | None = None,
        config: Any = None,
        intercept_layers: Sequence[int] | None = None,
        modify_hook: Callable[[jax.Array, int], jax.Array] | None = None,
        min_memory_headroom: float = 0.5,
        **kwargs: Any,
    ) -> None:
        if torch_model is None:
            if "params" in kwargs:
                torch_model = kwargs["params"]
            else:
                raise ValueError("Must provide either torch_model or params.")

        self._min_memory_headroom = float(min_memory_headroom)
        self._modify_hook = modify_hook
        self._call_count = 0

        # Handle backward compatibility: if torch_model is a JAX PyTree parameter mapping
        if isinstance(torch_model, Mapping):
            params_pytree = torch_model
            # 1. Detect architecture from PyTree and config
            self._architecture = detect_architecture(params_pytree, config)
            self._intercept_layers = validate_interception_layers(
                intercept_layers, self._architecture.num_layers
            )

            # 2. Reconstruct config if None
            if config is None:
                if self._architecture.model_family == "gpt2":
                    from transformers import GPT2Config
                    config = GPT2Config(
                        n_layer=self._architecture.num_layers,
                        n_embd=self._architecture.hidden_size,
                        n_head=self._architecture.num_heads,
                        vocab_size=self._architecture.vocab_size,
                        layer_norm_epsilon=self._architecture.layer_norm_eps,
                    )
                elif self._architecture.model_family == "neox":
                    from transformers import GPTNeoXConfig
                    config = GPTNeoXConfig(
                        num_hidden_layers=self._architecture.num_layers,
                        hidden_size=self._architecture.hidden_size,
                        num_attention_heads=self._architecture.num_heads,
                        vocab_size=self._architecture.vocab_size,
                        layer_norm_eps=self._architecture.layer_norm_eps,
                        rope_theta=self._architecture.rope_theta,
                        rotary_pct=self._architecture.rotary_pct,
                        use_parallel_residual=self._architecture.use_parallel_residual,
                    )

            # 3. Instantiate the PyTorch model class
            if self._architecture.model_family == "gpt2":
                from transformers import GPT2LMHeadModel
                py_model = GPT2LMHeadModel(config)
            elif self._architecture.model_family == "neox":
                from transformers import GPTNeoXForCausalLM
                py_model = GPTNeoXForCausalLM(config)
            else:
                raise ValueError(f"Unsupported model family: {self._architecture.model_family}")

            # 4. Flatten the JAX parameters PyTree back to a state dict
            state_dict = {}
            import numpy as np
            def _flatten(node: Any, prefix: str) -> None:
                if isinstance(node, Mapping):
                    for k, v in node.items():
                        _flatten(v, f"{prefix}.{k}" if prefix else k)
                elif isinstance(node, (list, tuple)):
                    for i, v in enumerate(node):
                        _flatten(v, f"{prefix}.{i}" if prefix else str(i))
                else:
                    arr = np.asarray(node)
                    state_dict[prefix] = torch.from_numpy(arr)

            _flatten(params_pytree, "")

            # 5. Load state dict into PyTorch model
            py_model.eval()
            py_model.load_state_dict(state_dict, strict=False)
            real_torch_model = py_model

        else:
            # Standard nn.Module instantiation
            real_torch_model = torch_model
            real_torch_model.eval()
            self._config = config or getattr(torch_model, "config", None)
            from .loader import state_dict_to_jax_pytree
            params_pytree = state_dict_to_jax_pytree(real_torch_model.state_dict())
            self._architecture = detect_architecture(params_pytree, self._config)
            self._intercept_layers = validate_interception_layers(
                intercept_layers, self._architecture.num_layers
            )

        # 6. Initialize Segmented Engine
        self._engine = SegmentedTorch2JaxEngine(
            torch_model=real_torch_model,
            arch=self._architecture,
            intercept_layers=self._intercept_layers,
        )

        # 7. Extract parameters and save pristine copy
        raw_params = self._engine.extract_and_validate_params()
        self._params = freeze(raw_params)
        self._pristine = freeze(raw_params)

    @property
    def architecture(self) -> Architecture:
        return self._architecture

    @property
    def intercept_layers(self) -> tuple[int, ...]:
        return self._intercept_layers

    @property
    def params(self) -> Any:
        return unfreeze(self._params)

    def get_params(self) -> Any:
        return unfreeze(self._params)

    def intercept_and_modify(self, hidden_state: jax.Array, layer_idx: int) -> jax.Array:
        """Default identity modification."""
        return hidden_state + 0.0

    def __call__(
        self,
        input_ids: jax.Array,
        position_ids: jax.Array | None = None,
    ) -> ForwardResult:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be a 2D array of shape [batch, seq_len], "
                f"got shape {tuple(input_ids.shape)}"
            )
        if input_ids.shape[1] < 1:
            raise ValueError("input_ids must contain at least one token position")

        # Enforce exact mathematical parameter freezing: grad(theta_0) == 0.0
        frozen_params = jax.tree.map(jax.lax.stop_gradient, self._params)
        hook = self._modify_hook or self.intercept_and_modify

        logits, intermediates = self._engine.run_forward(
            params=frozen_params,
            input_ids=input_ids,
            position_ids=position_ids,
            modify_fn=hook,
        )

        self._call_count += 1
        return ForwardResult(logits=logits, intermediates=intermediates)

    def params_unchanged(self) -> bool:
        """Verify leaf pointer identities against pristine copy."""
        pristine_leaves = jax.tree_util.tree_leaves(self._pristine)
        current_leaves = jax.tree_util.tree_leaves(self._params)
        return all(a is b for a, b in zip(pristine_leaves, current_leaves))

    def verify_frozen(self) -> dict[str, Any]:
        """Run original-vs-wrapper param identity check and return report."""
        unchanged = self.params_unchanged()
        return {
            "params_unchanged": unchanged,
            "param_leaves": len(jax.tree_util.tree_leaves(self._params)),
            "architecture": {
                "model_family": self._architecture.model_family,
                "num_layers": self._architecture.num_layers,
                "hidden_size": self._architecture.hidden_size,
            },
        }

    # ── memory monitoring ───────────────────────────────────────────────────

    def memory_status(self, device: jax.Device | None = None) -> MemoryStatus:
        return get_memory_status(device)

    def memory_warnings(self, status: MemoryStatus | None = None) -> list[str]:
        status = status or self.memory_status()
        return check_memory_headroom(status, self._min_memory_headroom)

    def run_with_memory_guard(
        self,
        input_ids: jax.Array,
        min_headroom: float | None = None,
        auto_reduce_batch_size: bool = False,
    ) -> tuple[ForwardResult, dict[str, Any]]:
        """Forward pass plus the memory headroom safety rule."""
        headroom = (
            min_headroom if min_headroom is not None else self._min_memory_headroom
        )
        status = self.memory_status()
        batch_size, reduced, warnings = maybe_reduce_batch_size(
            status, input_ids.shape[0], headroom, auto_reduce_batch_size
        )
        ids = input_ids[:batch_size] if reduced else input_ids
        result = self(ids)
        report = {
            "memory_status": status,
            "warnings": warnings,
            "batch_size_reduced": reduced,
            "effective_batch_size": ids.shape[0],
        }
        return result, report

    def __repr__(self) -> str:
        return (
            f"FrozenJAXSubstrate(model_family={self._architecture.model_family!r}, "
            f"num_layers={self._architecture.num_layers}, "
            f"hidden_size={self._architecture.hidden_size}, "
            f"intercept_layers={list(self._intercept_layers)})"
        )
