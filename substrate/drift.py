"""Drift monitoring between a frozen substrate and the untouched model."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp


def _log_softmax(logits: jax.Array) -> jax.Array:
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    return shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))


def compute_kl_drift(
    original_logits: jax.Array,
    modified_logits: jax.Array,
    mask: jax.Array | None = None,
) -> Mapping[str, float]:
    """Compute the mean KL divergence ``KL(original || modified)`` over all
    batch/sequence positions using numerically stable log-softmax.

    With the identity interception hook this should be approximately zero.
    """
    log_p = _log_softmax(original_logits)
    log_q = _log_softmax(modified_logits)
    p = jnp.exp(log_p)
    kl_per_position = jnp.sum(p * (log_p - log_q), axis=-1)
    if mask is not None:
        mask = mask.astype(jnp.float32)
        # Sum only valid token positions
        total_kl_masked = jnp.sum(kl_per_position * mask)
        total_tokens = jnp.sum(mask)
        mean_kl_masked = float(total_kl_masked / jnp.maximum(total_tokens, 1e-8))
        mean_kl_masked = max(0.0, mean_kl_masked)
        return {"kl_divergence": mean_kl_masked}
    mean_kl = float(jnp.mean(kl_per_position))
    # KL divergence is non-negative in exact arithmetic; clamp float noise.
    mean_kl = max(0.0, mean_kl)
    return {"kl_divergence": mean_kl}
