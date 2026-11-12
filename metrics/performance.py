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


def classification_metrics(
    targets: Any, predictions: Any, *, labels: Any | None = None
) -> dict[str, float]:
    target_array = np.asarray(targets)
    prediction_array = np.asarray(predictions)
    if target_array.shape != prediction_array.shape:
        raise ValueError(
            f"targets and predictions must have equal shapes, got "
            f"{target_array.shape} and {prediction_array.shape}"
        )
    if target_array.ndim != 1 or target_array.size == 0:
        raise ValueError("targets and predictions must be non-empty 1-D arrays")
    classes = list(labels) if labels is not None else sorted(
        set(target_array.tolist()) | set(prediction_array.tolist()), key=str
    )
    if not classes:
        raise ValueError("labels must contain at least one class")
    true_positive = []
    false_positive = []
    false_negative = []
    for label in classes:
        target_mask = target_array == label
        prediction_mask = prediction_array == label
        true_positive.append(int(np.count_nonzero(target_mask & prediction_mask)))
        false_positive.append(int(np.count_nonzero(~target_mask & prediction_mask)))
        false_negative.append(int(np.count_nonzero(target_mask & ~prediction_mask)))
    precision = [
        tp / (tp + fp) if tp + fp else 0.0
        for tp, fp in zip(true_positive, false_positive, strict=True)
    ]
    recall = [
        tp / (tp + fn) if tp + fn else 0.0
        for tp, fn in zip(true_positive, false_negative, strict=True)
    ]
    f1 = [
        2 * p * r / (p + r) if p + r else 0.0
        for p, r in zip(precision, recall, strict=True)
    ]
    return {
        "accuracy": float(np.mean(target_array == prediction_array)),
        "macro_f1": float(np.mean(f1)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
    }


def classification_predictions(
    logits: jax.Array, positions: jax.Array, label_token_ids: jax.Array
) -> jax.Array:
    if logits.ndim != 3 or positions.ndim != 1 or label_token_ids.ndim != 2:
        raise ValueError("invalid logits, positions, or label-token shapes")
    if logits.shape[0] != positions.shape[0]:
        raise ValueError("one position is required for each example")
    if label_token_ids.shape[1] < 1:
        raise ValueError("each class must have at least one label token")
    scores = logits[jnp.arange(logits.shape[0]), positions][:, label_token_ids]
    return jnp.argmax(jnp.max(scores, axis=-1), axis=-1)


def calibration_stats(
    logits: jax.Array, labels: jax.Array, bins: int = 15
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return per-bin (conf_sum, acc_sum, count) for top-1 predictions.

    Uses half-open bins and segment summation so it jits without dynamic shapes
    and remains finite when some bins are empty.
    """
    if bins < 1:
        raise ValueError("bins must be >= 1")
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    target = labels[:, 1:]
    mask = (target != -100).astype(jnp.float32)
    probs = jax.nn.softmax(logits[:, :-1].astype(jnp.float32), axis=-1)
    conf = jnp.max(probs, axis=-1)
    predicted = jnp.argmax(probs, axis=-1)
    safe_target = jnp.where(target != -100, target, 0)
    correct = (predicted == safe_target).astype(jnp.float32)

    idx = jnp.clip((conf * bins).astype(jnp.int32), 0, bins - 1)
    conf_flat = (conf * mask).ravel()
    acc_flat = (correct * mask).ravel()
    mask_flat = mask.ravel()
    idx_flat = idx.ravel()

    conf_sum = jax.ops.segment_sum(conf_flat, idx_flat, num_segments=bins)
    acc_sum = jax.ops.segment_sum(acc_flat, idx_flat, num_segments=bins)
    counts = jax.ops.segment_sum(mask_flat, idx_flat, num_segments=bins)
    return conf_sum, acc_sum, counts


def ece_from_stats(
    conf_sum: jax.Array, acc_sum: jax.Array, counts: jax.Array
) -> jax.Array:
    """Compute ECE from pooled per-bin calibration statistics.

    ECE = sum_b |conf_sum_b - acc_sum_b| / max(N, 1).
    Empty bins contribute 0 (since |0 - 0| = 0) and never produce NaN.
    """
    total = jnp.maximum(counts.sum(), 1.0)
    return jnp.abs(conf_sum - acc_sum).sum() / total


def expected_calibration_error(
    logits: jax.Array, labels: jax.Array, bins: int = 15
) -> jax.Array:
    """Token-level ECE using confidence of the top-1 next-token prediction."""
    conf_sum, acc_sum, counts = calibration_stats(logits, labels, bins=bins)
    return ece_from_stats(conf_sum, acc_sum, counts)


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
    x = a if higher_is_better else -a
    s = jnp.arange(t)[:, None]
    j = jnp.arange(t)[None, :]
    peak = jnp.max(jnp.where(s >= j, x[:t, :t], -jnp.inf), axis=0)
    return jnp.mean(peak - x[t, :t])


def accuracy_forgetting(accuracy_matrix: Any, t: int) -> jax.Array:
    return mean_forgetting(accuracy_matrix, t, higher_is_better=True)


def training_step_time(
    step_fn: Callable[..., T], *args: Any, **kwargs: Any
) -> tuple[T, float]:
    start = time.perf_counter()
    out = step_fn(*args, **kwargs)
    jax.block_until_ready(out)
    return out, time.perf_counter() - start


__all__ = [
    "calibration_stats",
    "classification_metrics",
    "classification_predictions",
    "cross_entropy_loss",
    "ece_from_stats",
    "expected_calibration_error",
    "accuracy_forgetting",
    "kl_to_base",
    "mean_forgetting",
    "mean_seen_accuracy",
    "paired_deltas",
    "perplexity",
    "sample_mean_std",
    "token_accuracy",
    "training_step_time",
]
