"""Task-performance, drift, continual-learning and compute metrics.

Continual-learning metrics take ``A[t, j]``: the metric on task ``j`` after
training task ``t`` (0-based, training order).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

T = TypeVar("T")


def cross_entropy_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Causal cross-entropy loss with next-token shifting and padding (-100) masking."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
    mask = shift_labels != -100
    safe_labels = jnp.where(mask, shift_labels, 0)

    gathered_log_probs = jnp.take_along_axis(
        log_probs, safe_labels[..., None], axis=-1
    ).squeeze(-1)

    loss = -jnp.sum(gathered_log_probs * mask) / jnp.maximum(1.0, jnp.sum(mask))
    return loss


def kl_to_base(base_logits: jax.Array, adapted_logits: jax.Array) -> jax.Array:
    # float32 log-space for stability; sum over vocab, mean over batch and positions.
    log_p = jax.nn.log_softmax(base_logits.astype(jnp.float32), axis=-1)
    log_q = jax.nn.log_softmax(adapted_logits.astype(jnp.float32), axis=-1)
    return jnp.mean(jnp.sum(jnp.exp(log_p) * (log_p - log_q), axis=-1))


def perplexity(loss: jax.Array | float) -> jax.Array:
    return jnp.exp(jnp.asarray(loss, dtype=jnp.float32))


def token_accuracy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Causal next-token top-1 accuracy, ignoring ``-100`` labels."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    target = labels[:, 1:]
    mask = target != -100
    predicted = jnp.argmax(logits[:, :-1], axis=-1)
    correct = (predicted == jnp.where(mask, target, 0)) & mask
    return jnp.sum(correct) / jnp.maximum(jnp.sum(mask), 1)


def expected_calibration_error(
    logits: jax.Array, labels: jax.Array, bins: int = 15
) -> jax.Array:
    """Token-level ECE using confidence of the top-1 next-token prediction."""
    if bins < 1:
        raise ValueError("bins must be >= 1")
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    target = labels[:, 1:]
    mask = target != -100
    probs = jax.nn.softmax(logits[:, :-1].astype(jnp.float32), axis=-1)
    confidence = jnp.max(probs, axis=-1)[mask]
    predicted = jnp.argmax(probs, axis=-1)[mask]
    correct = (predicted == target[mask]).astype(jnp.float32)
    edges = jnp.linspace(0.0, 1.0, bins + 1)
    total = jnp.maximum(confidence.size, 1)
    return sum(
        jnp.abs(jnp.mean(confidence[which]) - jnp.mean(correct[which]))
        * (jnp.sum(which) / total)
        for i in range(bins)
        for which in [(confidence >= edges[i]) & (confidence <= edges[i + 1])]
        if confidence.size
    )


def paired_deltas(
    values: dict[str, dict[int, float]], reference: str
) -> dict[str, dict[int, float]]:
    """Return method-minus-reference values paired by seed."""
    if reference not in values:
        raise KeyError(reference)
    return {
        method: {
            seed: float(seed_values[seed] - values[reference][seed])
            for seed in seed_values
            if seed in values[reference]
        }
        for method, seed_values in values.items()
        if method != reference
    }


def sample_mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """Return sample mean and sample standard deviation."""
    if not values:
        return None, None
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return mean, std


def _task_matrix(acc: Any, t: int) -> jax.Array:
    a = jnp.asarray(acc, dtype=jnp.float32)
    if a.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix A[t, j], got shape {a.shape}.")
    if not isinstance(t, int) or t < 0:
        raise ValueError(f"t must be an integer >= 0, got {t!r}.")
    if t >= min(a.shape):
        raise ValueError(f"t={t} is out of range for A with shape {a.shape}.")
    return a


def mean_seen_accuracy(acc: Any, t: int) -> jax.Array:
    a = _task_matrix(acc, t)
    return jnp.mean(a[t, : t + 1])


def mean_forgetting(acc: Any, t: int, *, higher_is_better: bool = True) -> jax.Array:
    a = _task_matrix(acc, t)
    if t == 0:
        raise ValueError("Forgetting is undefined at t=0: there are no earlier tasks.")
    # Negate losses so "best" is always the max. The peak spans s >= j only
    # (Chaudhry et al., 2018), so negative transfer is not counted as forgetting.
    x = a if higher_is_better else -a
    s = jnp.arange(t)[:, None]
    j = jnp.arange(t)[None, :]
    peak = jnp.max(jnp.where(s >= j, x[:t, :t], -jnp.inf), axis=0)
    return jnp.mean(peak - x[t, :t])


def training_step_time(
    step_fn: Callable[..., T], *args: Any, **kwargs: Any
) -> tuple[T, float]:
    start = time.perf_counter()
    out = step_fn(*args, **kwargs)
    # JAX is async: wait for the outputs before stopping the timer.
    jax.block_until_ready(out)
    return out, time.perf_counter() - start


__all__ = [
    "cross_entropy_loss",
    "expected_calibration_error",
    "kl_to_base",
    "mean_forgetting",
    "mean_seen_accuracy",
    "paired_deltas",
    "perplexity",
    "sample_mean_std",
    "token_accuracy",
    "training_step_time",
]
