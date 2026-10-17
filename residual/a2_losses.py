"""Loss and diagnostic terms used by Stage A2.

This module owns auxiliary loss terms (such as hidden-state prediction) and stability
metric collection for Stage A2.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from residual import ResidualAdapter
from residual.stability import (
    STEP_KEYS,
    StabilityMonitor,
    residual_smoothness,
    step_metrics,
)

_VAR_FLOOR = 1e-6
A2_LOSS_KEYS = ("task", "kl", "wd", "hid")
STABILITY_METRIC_KEYS = (
    "res_smooth",
    "input_smooth",
    "res_vs_input",
    *STEP_KEYS,
)


def hidden_prediction_loss(
    pred: jax.Array, target: jax.Array, source: jax.Array, config: Any
) -> dict[str, jax.Array]:
    """Compute hidden-prediction loss and its scale-free diagnostics.

    ``pred``, ``target`` and ``source`` have shape ``[B, S, D]``. The target
    is always detached. ``loss`` is target-variance-normalized unless
    ``config.hidden_norm`` is ``"none"``.
    """
    target = jax.lax.stop_gradient(target.astype(jnp.float32))
    source = source.astype(jnp.float32)
    if config.skip_first_token:
        pred, target, source = pred[:, 1:], target[:, 1:], source[:, 1:]
    mse = jnp.mean(jnp.square(pred - target))
    var = jnp.mean(jnp.var(target, axis=(0, 1))) + _VAR_FLOOR
    copy_mse = jnp.mean(jnp.square(jax.lax.stop_gradient(source) - target))
    rel = mse / var
    return {
        "loss": rel if config.hidden_norm == "target_var" else mse,
        "mse": mse,
        "rel": rel,
        "copy_rel": copy_mse / var,
    }


def collect_stability_metrics(
    monitor: StabilityMonitor,
    adapter: ResidualAdapter,
    adapter_params: Mapping[str, Any],
    site_inputs: Mapping[int, jax.Array],
    corrected: Mapping[int, jax.Array] | None = None,
) -> dict[str, jax.Array]:
    """Collect A2 stability metrics without adding them to the loss."""
    return monitor.metrics(
        adapter,
        adapter_params,
        site_inputs,
        corrected=corrected,
    )


__all__ = [
    "A2_LOSS_KEYS",
    "STABILITY_METRIC_KEYS",
    "collect_stability_metrics",
    "hidden_prediction_loss",
    "residual_smoothness",
    "step_metrics",
]

