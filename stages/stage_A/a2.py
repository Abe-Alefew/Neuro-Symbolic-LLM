"""Stage A2 (local-predictor part): ordinary backprop + PC-inspired auxiliary objectives.

L_A2 = L_task + lambda_kl * L_base-KL + lambda_wd * ||phi||^2 + lambda_hid * L_hidden-pred

* ``L_hidden-pred``: a separate FabricPC feedforward predictor ``P_l`` guesses the next
  hidden state from the residual-corrected state ``h~_l``. The target is stop-gradient:
  - ``corrected_teacher`` (A2 review, Option B, recommended): the next frozen block applied
    to ``h~_l``, i.e. ``result.hidden_state(l + 1)`` of the adapted forward pass;
  - ``base_anchor`` (plan example / review Option A): ``h^base_{l+1}`` from the unmodified
    base forward pass.
* Stability diagnostics (``residual/stability.py``) are measurement only: residual smoothness
  and the settling-step metrics are logged but add nothing to the loss. With no settling the
  step metrics are exactly zero; they are wired now so A3 can feed real trajectories.

``L_contrastive-future`` and ``L_stability`` are not part of this module yet.

Parameters differentiated by ``jax.grad`` are one flat dict: adapter entries
``layer_<l>`` and predictor entries ``pred_layer_<l>``. That keeps the A1 optimizer,
checkpoint and evaluation code unchanged; ``split_params`` separates them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from frozenllm.substrate import FrozenSubstrate, identity_modify
from residual import ResidualAdapter
from residual.stability import StabilityConfig, StabilityMonitor
from residual.local_predictor import (
    LocalPredictorBank,
    PredictorConfig,
    split_params,
)
from residual.a2_losses import collect_stability_metrics, hidden_prediction_loss
from stages.stage_base import checked_kwargs

HIDDEN_TARGETS = ("corrected_teacher", "base_anchor")
HIDDEN_NORMS = ("target_var", "none")
_VAR_FLOOR = 1e-6


def compose_a2_loss(
    model_loss: jax.Array,
    *,
    config: A2Config,
    weight_decay: jax.Array,
    task: jax.Array | None = None,
    kl: jax.Array | None = None,
    hidden: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Combine the trainable A2 loss terms and return their components."""
    total = model_loss + config.lambda_wd * weight_decay
    terms: dict[str, jax.Array] = {
        "task": model_loss if task is None else task,
        "wd": weight_decay,
    }
    if kl is not None:
        terms["kl"] = kl
    if hidden is not None:
        total = total + config.lambda_hid * hidden
        terms["hid"] = hidden
    return total, terms


@dataclass(frozen=True)
class A2Config:
    lambda_kl: float = 0.0
    lambda_wd: float = 0.0
    lambda_hid: float = 0.1
    hidden_target: str = "corrected_teacher"
    # target_var: MSE / mean per-feature target variance (scale-free across layers);
    # none: plain MSE.
    hidden_norm: str = "target_var"
    # Position 0 of GPT-2 carries massive activations that dominate an MSE.
    skip_first_token: bool = True
    # Predictor learning rate = training.learning_rate * this multiplier.
    predictor_lr_mult: float = 1.0
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)

    def __post_init__(self) -> None:
        for name in ("lambda_kl", "lambda_wd", "lambda_hid"):
            value = getattr(self, name)
            if not isinstance(value, int | float) or value < 0:
                raise ValueError(f"{name} must be a number >= 0, got {value!r}")
        if self.hidden_target not in HIDDEN_TARGETS:
            raise ValueError(
                f"hidden_target must be one of {HIDDEN_TARGETS}, got {self.hidden_target!r}"
            )
        if self.hidden_norm not in HIDDEN_NORMS:
            raise ValueError(
                f"hidden_norm must be one of {HIDDEN_NORMS}, got {self.hidden_norm!r}"
            )
        if not self.predictor_lr_mult > 0:
            raise ValueError(
                f"predictor_lr_mult must be > 0, got {self.predictor_lr_mult!r}"
            )
        if not isinstance(self.skip_first_token, bool):
            raise ValueError(
                f"skip_first_token must be a boolean, got {self.skip_first_token!r}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> A2Config:
        kwargs = checked_kwargs(cls, data)
        if "predictor" in kwargs:
            kwargs["predictor"] = PredictorConfig.from_dict(kwargs["predictor"] or {})
        if "stability" in kwargs:
            kwargs["stability"] = StabilityConfig.from_dict(kwargs["stability"] or {})
        return cls(**kwargs)

    @property
    def uses_hidden_pred(self) -> bool:
        return self.lambda_hid > 0


def make_a2_optimizer(
    config: A2Config, learning_rate: float
) -> optax.GradientTransformation:
    """Adam over the combined PyTree; the predictor group can use its own step size."""
    if config.predictor_lr_mult == 1.0:
        return optax.adam(learning_rate)

    def labels(params: Mapping[str, Any]) -> dict[str, Any]:
        _, pred = split_params(params)
        return {
            k: jax.tree_util.tree_map(lambda _: "pred" if k in pred else "adapter", v)
            for k, v in params.items()
        }

    return optax.multi_transform(
        {
            "adapter": optax.adam(learning_rate),
            "pred": optax.adam(learning_rate * config.predictor_lr_mult),
        },
        labels,
    )


def _intercept_sites(
    adapter: ResidualAdapter, pred_layers: Sequence[int]
) -> tuple[int, ...]:
    # Adapted sites, the predictor input sites, and the blocks that follow them.
    return tuple(
        sorted(set(adapter.layers) | set(pred_layers) | {i + 1 for i in pred_layers})
    )


def make_a2_objective(
    bank: LocalPredictorBank | None, monitor: StabilityMonitor | None
) -> Callable[..., tuple[jax.Array, dict[str, jax.Array]]]:
    """Bind the predictor bank and stability monitor into a stage objective.

    The returned function matches ``stages.stage_base.Objective``; ``params`` is the
    combined adapter + predictor dict.
    """

    def a2_objective(
        params: Mapping[str, Any],
        *,
        substrate: FrozenSubstrate,
        adapter: ResidualAdapter,
        config: A2Config,
        input_ids: jax.Array,
        labels: jax.Array | None = None,
        base: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        adapter_p, pred_p = split_params(params)
        target_labels = input_ids if labels is None else labels
        use_hid = config.uses_hidden_pred
        if use_hid and bank is None:
            raise ValueError("lambda_hid > 0 needs a LocalPredictorBank.")
        pred_layers: tuple[int, ...] = bank.layers if (use_hid and bank) else ()
        anchor = use_hid and config.hidden_target == "base_anchor"
        monitor_on = monitor is not None and monitor.config.enabled

        # Adapted forward: logits plus the hidden states every term needs.
        result = substrate.run_with_interception(
            input_ids,
            modify_fn=adapter.modify_fn(adapter_p),
            intercept_layers=_intercept_sites(adapter, pred_layers),
        )
        adapted = result.logits

        # One base forward serves both the KL term and the base-anchor target.
        need_base = config.lambda_kl > 0 or base is not None or anchor
        base_result = None
        if need_base:
            anchor_blocks = tuple(sorted({i + 1 for i in pred_layers})) if anchor else ()
            base_result = substrate.run_with_interception(
                input_ids, modify_fn=identity_modify, intercept_layers=anchor_blocks
            )
            base_lg = (
                jax.lax.stop_gradient(base_result.logits)
                if base is None
                else jax.lax.stop_gradient(base)
            )
            model_loss, comps = substrate.compute_loss(
                adapted,
                target_labels,
                losses=("CE", "KL"),
                base_logits=base_lg,
                weights={"KL": config.lambda_kl},
                return_components=True,
            )
            task, kl = comps["ce"], comps["kl"]
        else:
            task = substrate.compute_loss(adapted, target_labels, losses=("CE",))
            model_loss, kl = task, None

        wd = ResidualAdapter.l2_norm_sq(adapter_p)
        hid = None
        terms: dict[str, jax.Array] = {}

        if use_hid and bank is not None:
            losses, rels, copies = [], [], []
            for i in pred_layers:
                u = adapter.apply(adapter_p, result.hidden_state(i), i)  # h~_i
                y = (
                    base_result.hidden_state(i + 1)
                    if anchor and base_result is not None
                    else result.hidden_state(i + 1)
                )
                stats = hidden_prediction_loss(
                    bank.predict(pred_p, i, u), y, u, config
                )
                losses.append(stats["loss"])
                rels.append(stats["rel"])
                copies.append(stats["copy_rel"])
                terms[f"hid_rel_l{i}"] = stats["rel"]
            hid = jnp.mean(jnp.stack(losses))
            terms["hid_rel"] = jnp.mean(jnp.stack(rels))
            terms["hid_copy_rel"] = jnp.mean(jnp.stack(copies))

        total, loss_terms = compose_a2_loss(
            model_loss,
            config=config,
            weight_decay=wd,
            task=task,
            kl=kl,
            hidden=hid,
        )
        terms = loss_terms | {
            key: value for key, value in terms.items() if key not in loss_terms
        }

        if monitor_on and monitor is not None:
            # Metrics only (stop-gradient inside); the loss above is untouched.
            terms.update(
                collect_stability_metrics(
                    monitor,
                    adapter,
                    adapter_p,
                    {i: result.hidden_state(i) for i in monitor.layers},
                )
            )

        return total, {"total": total, **terms}

    return a2_objective


def a2_diagnostics(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    bank: LocalPredictorBank,
    config: A2Config,
    params: Mapping[str, Any],
    eval_batches: Sequence[jax.Array],
    *,
    jit: bool = True,
) -> dict[str, float]:
    """Held-out predictor diagnostics on the current adapted model.

    Per predictor site ``l`` (relative errors are MSE / mean target variance):
    ``pred_rel_l``  -- the predictor against the corrected-teacher target (Option B),
    ``pred_rel_base_l`` -- against the base-trajectory target (Option A),
    ``copy_rel_l``  -- trivial ``y ~ u`` baseline against the Option B target,
    ``teacher_shift_l`` -- how far the corrected-state next block moved away from the base
    trajectory (a large value means Options A and B are very different targets).
    """
    adapter_p, pred_p = split_params(params)
    sites = _intercept_sites(adapter, bank.layers)
    next_blocks = tuple(sorted({i + 1 for i in bank.layers}))

    def step(p_adapter: Mapping[str, Any], p_pred: Mapping[str, Any], ids: jax.Array):
        res = substrate.run_with_interception(
            ids, modify_fn=adapter.modify_fn(p_adapter), intercept_layers=sites
        )
        base_res = substrate.run_with_interception(
            ids, modify_fn=identity_modify, intercept_layers=next_blocks
        )
        out: dict[str, jax.Array] = {}
        for i in bank.layers:
            u = adapter.apply(p_adapter, res.hidden_state(i), i)
            guess = bank.predict(p_pred, i, u)
            teacher = res.hidden_state(i + 1)
            base_next = base_res.hidden_state(i + 1)
            on_teacher = hidden_prediction_loss(guess, teacher, u, config)
            on_base = hidden_prediction_loss(guess, base_next, u, config)
            shift = hidden_prediction_loss(teacher, base_next, teacher, config)
            out[f"pred_rel_l{i}"] = on_teacher["rel"]
            out[f"copy_rel_l{i}"] = on_teacher["copy_rel"]
            out[f"pred_rel_base_l{i}"] = on_base["rel"]
            out[f"teacher_shift_l{i}"] = shift["rel"]
        return out

    fn = jax.jit(step) if jit else step
    rows = [{k: float(v) for k, v in fn(adapter_p, pred_p, ids).items()} for ids in eval_batches]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    summary["pred_rel_mean"] = float(
        np.mean([summary[f"pred_rel_l{i}"] for i in bank.layers])
    )
    summary["copy_rel_mean"] = float(
        np.mean([summary[f"copy_rel_l{i}"] for i in bank.layers])
    )

    return summary


__all__ = [
    "HIDDEN_NORMS",
    "HIDDEN_TARGETS",
    "A2Config",
    "a2_diagnostics",
    "collect_stability_metrics",
    "compose_a2_loss",
    "hidden_prediction_loss",
    "make_a2_objective",
    "make_a2_optimizer",
]
