"""Pure-JAX trainable residual adapter R_phi(h) for Mechanism M2."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast
import jax
import jax.numpy as jnp
import optax

TrainingStepFn = Callable[
    [dict[str, jax.Array], Any, dict[str, dict[str, jax.Array]], jax.Array, jax.Array],
    tuple[jax.Array, dict[str, jax.Array], Any],
]


def init_residual_adapter(dim: int, key: jax.Array) -> dict[str, jax.Array]:
    """Initialize residual parameters phi = {w1, b1, w2, b2}."""
    k1, k2 = jax.random.split(key)
    scale = 1.0 / jnp.sqrt(dim)
    return {
        "w1": jax.random.uniform(k1, (dim, dim), minval=-scale, maxval=scale),
        "b1": jnp.zeros((dim,)),
        "w2": jax.random.uniform(k2, (dim, dim), minval=-scale, maxval=scale),
        "b2": jnp.zeros((dim,)),
    }


def apply_residual_adapter(
    phi: dict[str, jax.Array],
    hidden_state: jax.Array,
    layer_idx: int,
) -> jax.Array:
    """Compute h_modified = h + W2 * tanh(W1 * h + b1) + b2."""
    h = hidden_state @ phi["w1"] + phi["b1"]
    h = jnp.tanh(h)
    delta = h @ phi["w2"] + phi["b2"]
    return hidden_state + delta


def make_training_step(
    engine: Any,
    optimizer: optax.GradientTransformation,
) -> TrainingStepFn:
    """Constructs a JIT-compiled training step optimizing residual parameters phi."""

    @jax.jit
    def train_step(
        phi: dict[str, jax.Array],
        opt_state: Any,
        theta0: dict[str, dict[str, jax.Array]],
        input_ids: jax.Array,
        targets: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array], Any]:
        def loss_fn(p: dict[str, jax.Array]) -> jax.Array:
            logits, _ = engine.run_forward(
                params=theta0,
                input_ids=input_ids,
                modify_fn=lambda h, l: apply_residual_adapter(p, h, l),
            )
            # Causal shift
            shift_logits = logits[:, :-1, :]
            shift_targets = targets[:, 1:]
            return jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(
                    shift_logits, shift_targets
                )
            )

        loss, grads = jax.value_and_grad(loss_fn)(phi)
        updates, new_opt_state = optimizer.update(grads, opt_state, phi)
        new_phi = optax.apply_updates(phi, updates)
        return loss, new_phi, new_opt_state

    return cast(TrainingStepFn, train_step)
