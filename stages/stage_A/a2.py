"""Stage A2: the PC-inspired auxiliary objective on top of the A1 residual adapter.

    L_A2 = L_task + lambda_kl * KL(p_F0 || p_F0+R) + lambda_wd * ||phi||^2   (A1, unchanged)
         + lambda_cp   * L_cpc                                               (contrastive future)
         + lambda_stab * (w_anch * L_zanch + w_hep * L_hep-nudge)            (stability)
         + lambda_hid  * L_hid                                               (hidden prediction)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from frozenllm.substrate import FrozenSubstrate, identity_modify
from residual import AdapterParams, ResidualAdapter
from residual.a2_losses import (
    anchor_loss,
    collect_stability_metrics,
    contrastive_future_loss,
    hep_nudge_loss,
    hidden_prediction_loss,
    output_error,
)
from residual.cpc_heads import BACKENDS as CPC_BACKENDS
from residual.cpc_heads import (
    CPCParams,
    a_key,
    init_cpc_heads,
    q_key,
    split_cpc,
)
from residual.fpc_energy import BACKENDS as ENERGY_BACKENDS
from residual.hep import (
    KINDS,
    TRAINABLE_KINDS,
    HEPParams,
    highway_for,
    init_hep,
    split_hep,
)
from residual.local_predictor import (
    LocalPredictorBank,
    PredictorConfig,
    PredictorParams,
    split_params,
)
from residual.stability import StabilityConfig as MonitorConfig
from residual.stability import StabilityMonitor
from stages.stage_A.a1 import A1Config, a1_objective
from stages.stage_base import base_logits, checked_kwargs

# =============================================================================
# 1. Configuration
# =============================================================================


POSITIVES = ("base", "corrected")
HIDDEN_TARGETS = ("corrected_teacher", "base_anchor")


@dataclass(frozen=True)
class CPCConfig:
    """``objective.cpc`` section of the A2 YAML."""

    # lambda_cp; 0 switches the term (and its heads) off.
    weight: float = 0.0
    # Future offsets k (the paper: "such as 1, 2 or 4"; start with (1,)): the
    # positive is the code k positions ahead. One predictor A_{l,k} per offset.
    offsets: tuple[int, ...] = (1,)
    # m: width of the CPC code.
    dim: int = 32
    tau: float = 0.1
    # "base": positive from the unmodified model h^base; "corrected": from h~.
    positive: str = "base"
    # Deterministic stride subsampling of anchor positions; null keeps all.
    max_anchors_per_seq: int | None = 256
    # Std of the noise added to the identity predictor A_{l,k}.
    a_init_noise: float = 0.01
    # "fabricpc": the heads q and A run as FabricPC nodes; "jax": pure-JAX reference.
    heads: str = "fabricpc"

    def __post_init__(self) -> None:
        if not isinstance(self.weight, int | float) or self.weight < 0:
            raise ValueError(f"weight must be a number >= 0, got {self.weight!r}")
        offsets = self.offsets
        if isinstance(offsets, int) and not isinstance(offsets, bool):
            offsets = (offsets,)
        offsets = tuple(offsets) if isinstance(offsets, list | tuple) else offsets
        if (
            not isinstance(offsets, tuple)
            or not offsets
            or any(
                not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in offsets
            )
            or len(set(offsets)) != len(offsets)
        ):
            raise ValueError(
                f"offsets must be distinct integers >= 1, got {self.offsets!r}"
            )
        object.__setattr__(self, "offsets", offsets)
        if self.heads not in CPC_BACKENDS:
            raise ValueError(
                f"heads must be one of {list(CPC_BACKENDS)}, got {self.heads!r}"
            )
        if not isinstance(self.dim, int) or isinstance(self.dim, bool) or self.dim < 1:
            raise ValueError(f"dim must be an integer >= 1, got {self.dim!r}")
        if not self.tau > 0:
            raise ValueError(f"tau must be > 0, got {self.tau!r}")
        if self.positive not in POSITIVES:
            raise ValueError(
                f"positive must be one of {list(POSITIVES)}, got {self.positive!r}"
            )
        if self.max_anchors_per_seq is not None and (
            not isinstance(self.max_anchors_per_seq, int)
            or isinstance(self.max_anchors_per_seq, bool)
            or self.max_anchors_per_seq < 1
        ):
            raise ValueError(
                "max_anchors_per_seq must be an integer >= 1 or null, "
                f"got {self.max_anchors_per_seq!r}"
            )
        if not self.a_init_noise >= 0:
            raise ValueError(f"a_init_noise must be >= 0, got {self.a_init_noise!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CPCConfig:
        data = dict(data)
        if "offset" in data:  # single-offset spelling
            if "offsets" in data:
                raise ValueError("give either 'offset' or 'offsets', not both")
            data["offsets"] = (data.pop("offset"),)
        return cls(**checked_kwargs(cls, data))

    @property
    def enabled(self) -> bool:
        return self.weight > 0

    @property
    def needs_base_states(self) -> bool:
        return self.enabled and self.positive == "base"


NORMALIZATIONS = ("relative", "none")


def _number(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ValueError(f"{name} must be a number >= 0, got {value!r}")


@dataclass(frozen=True)
class HEPConfig:
    """``objective.stab.hep`` section of the A2 YAML."""

    # null switches HEP off (no nudge metric, no HEP loss).
    kind: str | None = None
    # Rank of the low-rank / fixed-random highway.
    rank: int = 16
    # alpha_y: gain of the nudge (the program sweeps it).
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.kind is not None and self.kind not in KINDS:
            raise ValueError(
                f"kind must be null or one of {list(KINDS)}, got {self.kind!r}"
            )
        if (
            not isinstance(self.rank, int)
            or isinstance(self.rank, bool)
            or self.rank < 1
        ):
            raise ValueError(f"rank must be an integer >= 1, got {self.rank!r}")
        _number("alpha", self.alpha)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HEPConfig:
        return cls(**checked_kwargs(cls, data))

    @property
    def trainable(self) -> bool:
        return self.kind in TRAINABLE_KINDS


@dataclass(frozen=True)
class StabilityConfig:
    """``objective.stab`` section of the A2 YAML."""

    # lambda_stab; 0 switches the whole term off.
    weight: float = 0.0
    w_anch: float = 1.0
    # 0: HEP nudge is only logged. > 0 needs a trainable highway (dense / lowrank).
    w_hep: float = 0.0
    normalize: str = "relative"
    # "fabricpc": the squared errors are FabricPC node energies (IdentityNode for the
    # anchor, Linear node(s) for the highway); "jax": plain arithmetic (test oracle).
    backend: str = "fabricpc"
    hep: HEPConfig = field(default_factory=HEPConfig)

    def __post_init__(self) -> None:
        for name in ("weight", "w_anch", "w_hep"):
            _number(name, getattr(self, name))
        if self.normalize not in NORMALIZATIONS:
            raise ValueError(
                f"normalize must be one of {list(NORMALIZATIONS)}, got {self.normalize!r}"
            )
        if self.backend not in ENERGY_BACKENDS:
            raise ValueError(
                f"backend must be one of {list(ENERGY_BACKENDS)}, got {self.backend!r}"
            )
        if self.w_hep > 0 and self.hep.kind is None:
            raise ValueError("w_hep > 0 needs hep.kind (dense or lowrank).")
        if self.w_hep > 0 and not self.hep.trainable:
            raise ValueError(
                f"w_hep > 0 with hep.kind={self.hep.kind!r}: the highway is fixed and "
                "eps is under stop-gradient, so the term has no gradient. Set w_hep = 0 "
                "(the nudge norm is still logged) or use kind dense / lowrank."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StabilityConfig:
        raw = dict(checked_kwargs(cls, data))
        raw["hep"] = HEPConfig.from_dict(raw.get("hep") or {})
        return cls(**raw)

    @property
    def enabled(self) -> bool:
        return self.weight > 0

    @property
    def needs_base_states(self) -> bool:
        return self.enabled and self.w_anch > 0

    @property
    def needs_output_error(self) -> bool:
        return self.enabled and self.hep.kind is not None


HIDDEN_NORMS = ("target_var", "none")


@dataclass(frozen=True)
class HiddenConfig:
    """``objective.hid`` local predictor/hidden prediction configuration."""

    # lambda_hid; 0 switches the term (and the predictors) off.
    weight: float = 0.0
    # Compatibility spelling for ``weight`` when declared inside ``hid``.
    lambda_hid: float | None = None
    hidden_target: str = "corrected_teacher"
    predictor_lr_mult: float = 1.0
    # "target_var": divide the squared error by the target variance (scale-free);
    # "none": plain mean squared error.
    hidden_norm: str = "target_var"
    # The first token has no left context (an attention sink with a huge norm in GPT-2).
    skip_first_token: bool = True
    predictor: PredictorConfig = field(default_factory=PredictorConfig)

    def __post_init__(self) -> None:
        _number("weight", self.weight)
        if self.lambda_hid is not None:
            _number("lambda_hid", self.lambda_hid)
            if self.weight != 0.0 and self.weight != self.lambda_hid:
                raise ValueError("hid.weight and hid.lambda_hid must match")
            if self.weight == 0.0:
                object.__setattr__(self, "weight", self.lambda_hid)
        if self.hidden_target not in HIDDEN_TARGETS:
            raise ValueError(
                f"hidden_target must be one of {list(HIDDEN_TARGETS)}, "
                f"got {self.hidden_target!r}"
            )
        if not isinstance(self.predictor_lr_mult, int | float) or (
            isinstance(self.predictor_lr_mult, bool)
        ) or self.predictor_lr_mult <= 0:
            raise ValueError(
                "predictor_lr_mult must be a number > 0, "
                f"got {self.predictor_lr_mult!r}"
            )
        if self.hidden_norm not in HIDDEN_NORMS:
            raise ValueError(
                f"hidden_norm must be one of {list(HIDDEN_NORMS)}, got {self.hidden_norm!r}"
            )
        if not isinstance(self.skip_first_token, bool):
            raise ValueError(
                f"skip_first_token must be a boolean, got {self.skip_first_token!r}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HiddenConfig:
        raw = dict(data)
        if "lambda_hid" in raw:
            if "weight" in raw and raw["weight"] != raw["lambda_hid"]:
                raise ValueError("hid.weight and hid.lambda_hid must match")
            raw["weight"] = raw["lambda_hid"]
        raw = dict(checked_kwargs(cls, raw))
        raw["predictor"] = PredictorConfig.from_dict(raw.get("predictor") or {})
        return cls(**raw)

    @property
    def enabled(self) -> bool:
        return self.weight > 0

    @property
    def coupled(self) -> bool:
        """Whether the predictor error reaches the adapter (``detach_input`` false)."""
        return not self.predictor.detach_input


@dataclass(frozen=True)
class A2Config:
    """A2 objective configuration.

    ``hid`` is the canonical local-predictor/hidden-prediction configuration.
    Local predictor/hidden prediction settings are grouped under ``hid``.
    """

    # A1 terms, same names and meaning as A1Config.
    lambda_kl: float = 0.0
    lambda_wd: float = 0.0
    # Seed of the auxiliary heads and of a fixed-random highway (the adapter has its own).
    seed: int = 0
    cpc: CPCConfig = field(default_factory=CPCConfig)
    stab: StabilityConfig = field(default_factory=StabilityConfig)
    hid: HiddenConfig = field(default_factory=HiddenConfig)
    # Stop-gradient stability metrics (residual smoothness, step metrics); never a loss.
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

    def __post_init__(self) -> None:
        _number("lambda_kl", self.lambda_kl)
        _number("lambda_wd", self.lambda_wd)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")
        if self.stab.hep.kind == "tied":
            raise ValueError(
                "hep.kind='tied' is not wired into a2_objective yet (it needs the frozen "
                "LM-head matrix); use dense, lowrank or fixed_random."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> A2Config:
        raw = dict(data)
        # Accept the historical flat spelling while keeping nested sections
        # canonical for new configurations.
        legacy_hid = {
            key: raw.pop(key)
            for key in (
                "hidden_norm",
                "skip_first_token",
                "predictor",
            )
            if key in raw
        }
        legacy_monitor = raw.pop("stability", None)
        if legacy_monitor is not None and "monitor" in raw:
            raise ValueError("give either 'stability' or 'monitor', not both")
        if legacy_monitor is not None:
            raw["monitor"] = legacy_monitor
        if legacy_hid:
            hid = dict(raw.get("hid") or {})
            overlap = sorted(set(hid) & set(legacy_hid))
            if overlap:
                raise ValueError(
                    "legacy local-predictor keys overlap with objective.hid: "
                    f"{overlap}"
                )
            hid.update(legacy_hid)
            raw["hid"] = hid
        raw = dict(checked_kwargs(cls, raw))
        raw["cpc"] = CPCConfig.from_dict(raw.get("cpc") or {})
        raw["stab"] = StabilityConfig.from_dict(raw.get("stab") or {})
        raw["hid"] = HiddenConfig.from_dict(raw.get("hid") or {})
        raw["monitor"] = MonitorConfig.from_dict(raw.get("monitor") or {})
        return cls(**raw)

    @property
    def needs_base_states(self) -> bool:
        """Whether the objective needs the base model's hidden states ``h^base``."""
        return (
            self.cpc.needs_base_states
            or self.stab.needs_base_states
            or (self.hid.enabled and self.hid.hidden_target == "base_anchor")
        )

    @property
    def aux_enabled(self) -> bool:
        """Whether any auxiliary *loss* is on."""
        return self.cpc.enabled or self.stab.enabled or self.hid.enabled

    @property
    def needs_corrected(self) -> bool:
        """Whether the corrected states ``h~_l`` are needed (a loss or the monitor)."""
        return self.aux_enabled or self.monitor.enabled


# =============================================================================
# 2. Parameters
# =============================================================================

# The flat tree is two levels deep (``RunWriter.save_params`` requires it).
ADAPTER_PREFIX = "layer_"  # layer_key(l) == f"{ADAPTER_PREFIX}{l}"


def adapter_params(params: Mapping[str, Mapping[str, jax.Array]]) -> AdapterParams:
    """The ``layer_<l>`` entries: what ``ResidualAdapter`` and the A1 weight decay see."""
    return {k: dict(v) for k, v in params.items() if k.startswith(ADAPTER_PREFIX)}


def init_a2_params(config: A2Config, adapter: ResidualAdapter) -> dict[str, Any]:
    """The adapter parameters plus the heads the enabled terms need.

    With every auxiliary weight at 0 this is exactly ``adapter.init_params()``.
    """
    d = adapter.hidden_size
    layers = tuple(adapter.layers)
    params: dict[str, Any] = dict(adapter.init_params())
    heads = {
        **init_cpc_params(config.cpc, config.seed, layers, d),
        **init_hep_params(
            config.stab, config.seed, layers, adapter.architecture.vocab_size, d
        ),
    }
    clash = sorted(set(heads) & set(params))
    if clash:
        raise ValueError(f"head parameters clash with the adapter keys: {clash}")
    params.update(heads)
    if config.hid.enabled:
        pred = predictor_bank(config.hid, adapter).init_params(config.seed)
        clash = sorted(set(pred) & set(params))
        if clash:
            raise ValueError(f"predictor parameters clash with the other keys: {clash}")
        params.update(pred)
    return params


_BANKS: dict[tuple[Any, ...], LocalPredictorBank] = {}


def predictor_bank(
    config: HiddenConfig, adapter: ResidualAdapter
) -> LocalPredictorBank:
    """The predictors ``P_l`` of ``config`` for ``adapter``'s sites (built once, then reused).
    """
    p = config.predictor
    arch = adapter.architecture
    key = (
        p.ff_dim,
        p.out_init_std,
        p.predict_delta,
        p.detach_input,
        None if p.layers is None else tuple(p.layers),
        p.seed,
        arch.hidden_size,
        arch.num_layers,
        tuple(adapter.layers),
    )
    if key not in _BANKS:
        bank = LocalPredictorBank(p, arch, tuple(adapter.layers))
        stray = sorted(set(bank.layers) - set(adapter.layers))
        if stray:
            raise ValueError(
                f"Predictor sites must be adapted layers; {stray} are not in "
                f"{list(adapter.layers)}."
            )
        _BANKS[key] = bank
    return _BANKS[key]


def predictor_params(params: Mapping[str, Mapping[str, jax.Array]]) -> PredictorParams:
    """The ``pred_layer_<l>`` entries of a full A2 parameter tree."""
    return split_params(params)[1]


def param_counts(
    params: Mapping[str, Mapping[str, jax.Array]],
) -> dict[str, int]:
    """Trainable parameters by group: ``adapter``, ``cpc``, ``hep``, ``pred`` and ``total``."""

    def count(tree: Mapping[str, Mapping[str, jax.Array]]) -> int:
        return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(tree))

    groups = {
        "adapter": count(adapter_params(params)),
        "cpc": count(split_cpc(params)),
        "hep": count(split_hep(params)),
        "pred": count(predictor_params(params)),
    }
    other = count(params) - sum(groups.values())
    if other:
        raise ValueError(f"{other} parameters belong to no known group.")
    return {**groups, "total": sum(groups.values())}


def init_cpc_params(
    config: CPCConfig, seed: int, layers: tuple[int, ...], hidden_size: int
) -> CPCParams:
    """Heads for ``layers``; empty when the term is disabled."""
    if not config.enabled:
        return {}
    return init_cpc_heads(
        seed,
        layers,
        hidden_size,
        config.dim,
        config.offsets,
        config.a_init_noise,
    )


def init_hep_params(
    config: StabilityConfig,
    seed: int,
    layers: tuple[int, ...],
    vocab_size: int,
    hidden_size: int,
) -> HEPParams:
    """Trainable highway parameters; empty unless HEP is enabled with a trainable kind."""
    if not (config.enabled and config.hep.trainable):
        return {}
    assert config.hep.kind is not None
    return init_hep(
        config.hep.kind, seed, layers, vocab_size, hidden_size, config.hep.rank
    )


# =============================================================================
# 3. Terms (pure functions of hidden-state dictionaries ``{layer: [B, T, d]}``)
# =============================================================================


def token_mask(labels: jax.Array | None, ignore_index: int = -100) -> jax.Array | None:
    """``[B, T]`` validity of the tokens from the labels: ``labels != ignore_index``.

    ``None`` when there are no labels (every token is then treated as real). Note
    that tokens whose label is ignored for the task loss (a prompt, say) are also
    left out of the contrastive term; pass an explicit ``mask`` to ``cpc_term``
    to keep them.
    """
    if labels is None:
        return None
    return labels != ignore_index


def cpc_term(
    params: Mapping[str, Mapping[str, jax.Array]],
    *,
    config: CPCConfig,
    corrected: Mapping[int, jax.Array],
    base: Mapping[int, jax.Array] | None = None,
    mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Unweighted ``L_cpc`` (mean over sites ``(l, k)``) and its diagnostics.

    Args:
        params: a tree holding the heads ``cpcq_<l>`` / ``cpca_<l>_<k>``.
        corrected: ``{layer: h~_l}``, each ``[..., B, T, d]``, carrying gradient.
        base: ``{layer: h^base_l}``; required when ``config.positive == "base"``.
        mask: optional ``[B, T]`` token validity (``False`` = padding); see
            :func:`token_mask`. Positions outside it are never anchors, positives
            or negatives.
    """
    if not corrected:
        raise ValueError("cpc_term needs at least one site.")
    if config.positive == "base" and base is None:
        raise ValueError("positive='base' requires the base hidden states.")
    losses = []
    metrics: dict[str, list[jax.Array]] = {}
    for layer, anchors in corrected.items():
        if config.positive == "base":
            assert base is not None
            if layer not in base:
                raise KeyError(f"missing base hidden state for layer {layer}.")
            targets = base[layer]
        else:
            targets = anchors
        for offset in config.offsets:
            loss, m = contrastive_future_loss(
                params[q_key(layer)],
                params[a_key(layer, offset)],
                anchors,
                targets,
                offset=offset,
                tau=config.tau,
                max_anchors_per_seq=config.max_anchors_per_seq,
                mask=mask,
                backend=config.heads,
            )
            losses.append(loss)
            for name, value in m.items():
                metrics.setdefault(name, []).append(value)
    reduced = {name: jnp.mean(jnp.stack(vals)) for name, vals in metrics.items()}
    return jnp.mean(jnp.stack(losses)), reduced


def stability_term(
    params: Mapping[str, Mapping[str, jax.Array]],
    *,
    config: StabilityConfig,
    corrected: Mapping[int, jax.Array],
    base: Mapping[int, jax.Array] | None = None,
    logits: jax.Array | None = None,
    labels: jax.Array | None = None,
    seed: int = 0,
    tied: jax.Array | None = None,
    mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """``w_anch * L_zanch + w_hep * L_hep-nudge`` (without ``lambda_stab``) and diagnostics.

    Args:
        params: a tree holding the trainable highway entries ``hep_<l>`` (if any).
        corrected: ``{layer: h~_l}``, each ``[..., B, T, d]``, carrying gradient.
        base: ``{layer: h^base_l}``; required when ``w_anch > 0``.
        logits, labels: adapted logits ``[B, T, V]`` and the labels (the inputs when
            training on plain text); required when HEP is configured. ``-100`` labels
            are ignored.
        seed: seed of the ``fixed_random`` highway.
        tied: ``[V, d]`` matrix for ``hep.kind == "tied"``.
        mask: optional ``[B, T]`` token validity (``False`` = padding) for the anchor
            term, e.g. ``token_mask(labels)``. The HEP term masks by the
            ignored labels on its own.
    """
    if not corrected:
        raise ValueError("stability_term needs at least one site.")
    total = jnp.zeros((), jnp.float32)
    metrics: dict[str, jax.Array] = {}

    if config.w_anch > 0:
        if base is None:
            raise ValueError("w_anch > 0 requires the base hidden states.")
        losses, parts = [], []
        for layer, h in corrected.items():
            if layer not in base:
                raise KeyError(f"missing base hidden state for layer {layer}.")
            loss, m = anchor_loss(
                h,
                base[layer],
                normalize=config.normalize,
                mask=mask,
                backend=config.backend,
            )
            losses.append(loss)
            parts.append(m)
        l_anch = jnp.mean(jnp.stack(losses))
        total = total + config.w_anch * l_anch
        metrics["stab_anch"] = l_anch
        for name in parts[0]:
            metrics[name] = jnp.mean(jnp.stack([m[name] for m in parts]))

    if config.hep.kind is not None:
        if logits is None or labels is None:
            raise ValueError("HEP needs the adapted logits and the labels.")
        eps, eps_mask = output_error(logits, labels)
        n_pos = eps.shape[-2]
        losses, parts = [], []
        for layer, h in corrected.items():
            v = highway_for(
                config.hep.kind,
                layer,
                params=params,
                seed=seed,
                d_out=eps.shape[-1],
                d_hidden=h.shape[-1],
                rank=config.hep.rank,
                tied=tied,
            )
            # eps at position t is the error of the prediction made at t.
            loss, m = hep_nudge_loss(
                eps,
                v,
                config.hep.alpha,
                reference=h[..., :n_pos, :],
                mask=eps_mask,
                backend=config.backend,
            )
            losses.append(loss)
            parts.append(m)
        l_hep = jnp.mean(jnp.stack(losses))
        if config.w_hep > 0:
            total = total + config.w_hep * l_hep
        metrics["stab_hep"] = l_hep
        for name in parts[0]:
            metrics[name] = jnp.mean(jnp.stack([m[name] for m in parts]))

    return total, metrics


def hid_term(
    params: Mapping[str, Mapping[str, jax.Array]],
    *,
    config: HiddenConfig,
    bank: LocalPredictorBank,
    corrected: Mapping[int, jax.Array],
    targets: Mapping[int, jax.Array],
    mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Unweighted ``L_hid`` (mean over the predictor sites) and its diagnostics.

    Args:
        params: a tree holding the predictors ``pred_layer_<l>``.
        bank: the :class:`LocalPredictorBank` that built them.
        corrected: ``{layer: h~_l}``, each ``[B, T, d]`` (the predictor input).
        targets: ``{layer: h_{l+1}}``, the output of the next frozen block for the
            adapted pass; stop-gradient is applied here.
        mask: optional ``[B, T]`` token validity (``False`` = padding).

    ``hid_rel`` is the squared error relative to the target variance and
    ``hid_copy_rel`` the same for the trivial predictor ``h~_l`` itself: a predictor
    that beats copying has ``hid_rel < hid_copy_rel``.
    """
    losses = []
    parts: list[dict[str, jax.Array]] = []
    for layer in bank.layers:
        if layer not in corrected:
            raise KeyError(f"missing corrected hidden state for layer {layer}.")
        if layer not in targets:
            raise KeyError(f"missing next-block hidden state for layer {layer}.")
        source = corrected[layer]
        pred = bank.predict(params, layer, source)
        out = hidden_prediction_loss(pred, targets[layer], source, config, mask=mask)
        losses.append(out["loss"])
        parts.append(out)
    metrics = {
        f"hid_{name}": jnp.mean(jnp.stack([p[name] for p in parts]))
        for name in ("mse", "rel", "copy_rel")
    }
    return jnp.mean(jnp.stack(losses)), metrics


# =============================================================================
# 4. Objective
# =============================================================================


def a2_objective(
    params: Mapping[str, Mapping[str, jax.Array]],
    *,
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    config: A2Config,
    input_ids: jax.Array,
    labels: jax.Array | None = None,
    base: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """``(total, terms)`` of the A2 objective; the ``Objective`` contract of ``run_sequential``.

    ``params`` is the whole flat tree of :func:`init_a2_params`. ``base`` is optional
    precomputed base *logits* (as in A1). ``terms`` holds ``total``, ``task``,
    ``wd`` (and ``kl`` when computed), plus ``cpc`` / ``stab`` / ``hid`` and their
    diagnostics for the enabled terms (``cpc``, ``stab`` and ``hid`` are logged *before*
    their lambda), and the monitor metrics when ``monitor.mode`` is ``diagnostic``.

    With every auxiliary weight 0 the result equals ``a1_objective``; the monitor never
    changes ``total`` or a gradient.
    """
    ap = adapter_params(params)
    bank = predictor_bank(config.hid, adapter) if config.hid.enabled else None
    # The hidden-prediction target of site l is the output of block l+1, so that block
    # is intercepted as well (a layer without adapter parameters is passed through).
    intercept = tuple(adapter.layers)
    if bank is not None:
        intercept = tuple(sorted(set(intercept) | {i + 1 for i in bank.layers}))
    run = substrate.run_with_interception(
        input_ids, modify_fn=adapter.modify_fn(ap), intercept_layers=intercept
    )
    adapted = run.logits
    target_labels = input_ids if labels is None else labels
    wd = ResidualAdapter.l2_norm_sq(ap)

    need_states = config.needs_base_states
    need_kl = config.lambda_kl > 0 or base is not None
    base_states: dict[int, jax.Array] | None = None
    base_lg: jax.Array | None = None
    if need_states:
        # One extra forward gives both the base states and (if not given) base logits.
        base_run = substrate.run_with_interception(
            input_ids,
            modify_fn=identity_modify,
            intercept_layers=intercept,
        )
        base_states = {
            layer: jax.lax.stop_gradient(base_run.hidden_state(layer))
            for layer in intercept
        }
        base_lg = jax.lax.stop_gradient(base_run.logits if base is None else base)
    elif need_kl:
        base_lg = (
            base_logits(substrate, input_ids)
            if base is None
            else jax.lax.stop_gradient(base)
        )

    terms: dict[str, jax.Array]
    if base_lg is None:  # no KL: skip it rather than log NaN (strict JSON rejects it)
        task = substrate.compute_loss(adapted, target_labels, losses=("CE",))
        total = task
        terms = {"task": task}
    else:
        total, components = substrate.compute_loss(
            adapted,
            target_labels,
            losses=("CE", "KL"),
            base_logits=base_lg,
            weights={"KL": config.lambda_kl},
            return_components=True,
        )
        task = components["ce"]
        terms = {"task": task, "kl": components["kl"]}
    total = total + config.lambda_wd * wd
    terms["wd"] = wd

    if config.needs_corrected:
        # h~_l = R applied to the pristine block output (what the next block receives).
        sites = {layer: run.hidden_state(layer) for layer in adapter.layers}
        corrected = {layer: adapter.apply(ap, h, layer) for layer, h in sites.items()}
        mask = token_mask(labels)
        if config.monitor.enabled:
            terms.update(
                collect_stability_metrics(
                    StabilityMonitor(config.monitor, adapter.layers),
                    adapter,
                    ap,
                    sites,
                    corrected=corrected,
                )
            )
        if config.cpc.enabled:
            l_cpc, m = cpc_term(
                params,
                config=config.cpc,
                corrected=corrected,
                base=base_states,
                mask=mask,
            )
            total = total + config.cpc.weight * l_cpc
            terms["cpc"] = l_cpc
            terms.update(m)
        if config.stab.enabled:
            l_stab, m = stability_term(
                params,
                config=config.stab,
                corrected=corrected,
                base=base_states,
                logits=adapted,
                labels=target_labels,
                seed=config.seed,
                mask=mask,
            )
            total = total + config.stab.weight * l_stab
            terms["stab"] = l_stab
            terms.update(m)
        if config.hid.enabled:
            assert bank is not None
            if config.hid.hidden_target == "base_anchor":
                assert base_states is not None
                targets = {
                    layer: base_states[layer + 1]
                    for layer in bank.layers
                }
            else:
                targets = {
                    layer: jax.lax.stop_gradient(run.hidden_state(layer + 1))
                    for layer in bank.layers
                }
            l_hid, m = hid_term(
                params,
                config=config.hid,
                bank=bank,
                corrected=corrected,
                targets=targets,
                mask=mask,
            )
            total = total + config.hid.weight * l_hid
            terms["hid"] = l_hid
            terms.update(m)

    return total, {"total": total, **terms}


# =============================================================================
# 5. Ablation arms
# =============================================================================

# Weights used when an arm switches a term on and the config's own weight is 0.
ARM_DEFAULT_WEIGHTS = {"cpc": 0.05, "stab": 0.1, "hid": 0.1}

# Arms the implemented terms allow. Each arm only chooses *which* terms run and their
# variant; lambda_kl, lambda_wd, the data, optimiser and seeds are shared, so contrasts
# against ``a1_mode`` isolate one term. ``monitor`` is measurement-only and is never
# switched by an arm: set ``objective.monitor.mode`` to log it in any arm.
ARMS = (
    "a1_mode",  # no auxiliary term: the A1 objective
    "stab_anch",  # anchor only
    "stab_hep",  # anchor + trainable low-rank HEP highway
    "cpc_base",  # contrastive future, positive from the base state
    "cpc_corrected",  # contrastive future, positive from the corrected state
    "cpc_multi",  # contrastive future, offsets 1, 2, 4
    "hid_detached",  # hidden prediction, predictor input detached: trains P_l only (control)
    "hid_coupled",  # hidden prediction, predictor error reaches the adapter
    "full",  # cpc_base + stab_anch + hid_coupled
)


def _weight(section_weight: float, term: str) -> float:
    return section_weight if section_weight > 0 else ARM_DEFAULT_WEIGHTS[term]


def apply_arm(config: A2Config, arm: str) -> A2Config:
    """``config`` with the terms of ablation arm ``arm`` on and all others off.

    The lambdas of the config are kept (a zero weight falls back to
    ``ARM_DEFAULT_WEIGHTS``), so ``objective.cpc.weight`` / ``objective.stab.weight``
    in the YAML tune the arms.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose one of {list(ARMS)}")
    off_cpc = replace(config.cpc, weight=0.0)
    off_stab = replace(config.stab, weight=0.0, w_hep=0.0, hep=HEPConfig())
    off_hid = replace(config.hid, weight=0.0)

    def cpc_on(**over: Any) -> CPCConfig:
        return replace(config.cpc, weight=_weight(config.cpc.weight, "cpc"), **over)

    def anchor_on() -> StabilityConfig:
        # Keep a fixed highway of the config as a logged metric; a trainable one would
        # allocate parameters that get no gradient without w_hep.
        hep = config.stab.hep if not config.stab.hep.trainable else HEPConfig()
        return replace(
            config.stab,
            weight=_weight(config.stab.weight, "stab"),
            w_anch=1.0,
            w_hep=0.0,
            hep=hep,
        )

    def hid_on(detach_input: bool) -> HiddenConfig:
        predictor = replace(config.hid.predictor, detach_input=detach_input)
        return replace(
            config.hid, weight=_weight(config.hid.weight, "hid"), predictor=predictor
        )

    cpc, stab, hid = off_cpc, off_stab, off_hid
    if arm == "stab_anch":
        stab = anchor_on()
    elif arm == "stab_hep":
        base = anchor_on()
        kind = config.stab.hep.kind if config.stab.hep.trainable else "lowrank"
        w_hep = config.stab.w_hep if config.stab.w_hep > 0 else 0.1
        stab = replace(base, w_hep=w_hep, hep=replace(config.stab.hep, kind=kind))
    elif arm == "cpc_base":
        cpc = cpc_on(positive="base")
    elif arm == "cpc_corrected":
        cpc = cpc_on(positive="corrected")
    elif arm == "cpc_multi":
        cpc = cpc_on(positive="base", offsets=(1, 2, 4))
    elif arm == "hid_detached":
        hid = hid_on(detach_input=True)
    elif arm == "hid_coupled":
        hid = hid_on(detach_input=False)
    elif arm == "full":
        cpc, stab, hid = (
            cpc_on(positive="base"),
            anchor_on(),
            hid_on(detach_input=False),
        )
    return replace(config, cpc=cpc, stab=stab, hid=hid)


# =============================================================================
# 6. Audit: checkpoints you can inspect
# =============================================================================


@dataclass(frozen=True)
class Check:
    """One audit checkpoint. ``ok`` is ``None`` for information that has no pass/fail."""

    name: str
    ok: bool | None
    value: Any
    detail: str = ""


def _f(x: Any) -> float:
    return float(x)


def _group_norms(tree: Mapping[str, Mapping[str, jax.Array]]) -> dict[str, float]:
    """L2 norm of ``tree`` per parameter group (``adapter`` / ``cpc`` / ``hep`` / ``pred``)."""
    out = {}
    for name, sub in (
        ("adapter", adapter_params(tree)),
        ("cpc", split_cpc(tree)),
        ("hep", split_hep(tree)),
        ("pred", predictor_params(tree)),
    ):
        sq = sum(
            (jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(sub)),
            start=jnp.zeros((), jnp.float32),
        )
        out[name] = _f(jnp.sqrt(sq))
    return out


def a2_audit(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    config: A2Config,
    input_ids: jax.Array,
    *,
    probe_scale: float = 0.05,
) -> list[Check]:
    """Run the A2 checkpoints once on ``input_ids`` and return them.

    """
    checks: list[Check] = []

    def add(name: str, ok: bool | None, value: Any, detail: str = "") -> None:
        checks.append(Check(name, ok, value, detail))

    params0 = init_a2_params(config, adapter)
    key = jax.random.PRNGKey(config.seed + 1)
    probe = {k: dict(v) for k, v in params0.items()}
    for i, k in enumerate(adapter_params(params0)):
        shape = probe[k]["B"].shape
        probe[k]["B"] = probe_scale * jax.random.normal(
            jax.random.fold_in(key, i), shape, jnp.float32
        )

    def run(p: Any, cfg: A2Config = config, **kw: Any):
        return a2_objective(
            p,
            substrate=substrate,
            adapter=adapter,
            config=cfg,
            input_ids=input_ids,
            **kw,
        )

    # 1. The parameter tree has exactly the groups the config asks for.
    counts = param_counts(params0)
    want_cpc = config.cpc.enabled
    want_hep = config.stab.enabled and config.stab.hep.trainable
    want_pred = config.hid.enabled
    add(
        "param_groups",
        (counts["cpc"] > 0) == want_cpc
        and (counts["hep"] > 0) == want_hep
        and (counts["pred"] > 0) == want_pred,
        counts,
        f"expected cpc heads: {want_cpc}, trainable highway: {want_hep}, "
        f"hidden predictors: {want_pred}",
    )

    # 2. The frozen model is untouched before we start.
    add("frozen_before", bool(substrate.params_unchanged()), None)

    # 3. At init the adapted model equals the base model.
    base = base_logits(substrate, input_ids)
    adapted0 = substrate.run_with_interception(
        input_ids,
        modify_fn=adapter.modify_fn(adapter_params(params0)),
        intercept_layers=adapter.layers,
    ).logits
    diff = _f(jnp.max(jnp.abs(adapted0 - base)))
    add("init_equals_base", diff < 1e-4, diff, "max |adapted logits - base logits|")

    # 4. Every term is finite, at init and at the probe.
    for label, p in (("init", params0), ("probe", probe)):
        _, terms = run(p)
        bad = sorted(k for k, v in terms.items() if not bool(jnp.isfinite(v)))
        add(f"finite_{label}", not bad, {k: _f(v) for k, v in terms.items()}, str(bad))

    # 5. total = the weighted sum of the logged terms.
    total, t = run(probe)
    expected = (
        t["task"]
        + config.lambda_kl * t.get("kl", 0.0)
        + config.lambda_wd * t["wd"]
        + config.cpc.weight * t.get("cpc", 0.0)
        + config.stab.weight * t.get("stab", 0.0)
        + config.hid.weight * t.get("hid", 0.0)
    )
    err = abs(_f(total) - _f(expected)) / max(abs(_f(total)), 1e-12)
    add("total_decomposition", err < 1e-4, err, "relative error of total vs the sum")

    # 6. With the auxiliary weights at 0, A2 is A1 (values and gradients).
    off = replace(
        config,
        cpc=replace(config.cpc, weight=0.0),
        stab=replace(config.stab, weight=0.0),
        hid=replace(config.hid, weight=0.0),
    )
    a1_cfg = A1Config(lambda_kl=config.lambda_kl, lambda_wd=config.lambda_wd)
    ap = adapter_params(probe)

    def a1_total(p: Any) -> Any:
        return a1_objective(
            p,
            substrate=substrate,
            adapter=adapter,
            config=a1_cfg,
            input_ids=input_ids,
        )[0]

    t_a1, t_a2 = a1_total(ap), run(probe, off)[0]
    g_a1 = jax.grad(a1_total)(ap)
    g_a2 = adapter_params(jax.grad(lambda p: run(p, off)[0])(probe))
    gdiff = max(
        _f(jnp.max(jnp.abs(x - y)))
        for x, y in zip(
            jax.tree_util.tree_leaves(g_a1),
            jax.tree_util.tree_leaves(g_a2),
            strict=True,
        )
    )
    add(
        "a1_parity",
        abs(_f(t_a1) - _f(t_a2)) < 1e-5 * max(1.0, abs(_f(t_a1))) and gdiff < 1e-6,
        {"total_diff": abs(_f(t_a1) - _f(t_a2)), "grad_max_diff": gdiff},
        "aux weights 0: A2 total and adapter gradient vs a1_objective",
    )

    # 6b. The monitor is measurement-only: it changes neither the total nor a gradient.
    if config.monitor.enabled:
        quiet = replace(config, monitor=replace(config.monitor, mode="off"))
        t_on, t_off = run(probe)[0], run(probe, quiet)[0]
        g_on = jax.grad(lambda p: run(p)[0])(probe)
        g_off = jax.grad(lambda p: run(p, quiet)[0])(probe)
        mdiff = max(
            _f(jnp.max(jnp.abs(x - y)))
            for x, y in zip(
                jax.tree_util.tree_leaves(g_on),
                jax.tree_util.tree_leaves(g_off),
                strict=True,
            )
        )
        add(
            "monitor_measurement_only",
            abs(_f(t_on) - _f(t_off)) < 1e-6 * max(1.0, abs(_f(t_off)))
            and mdiff < 1e-6,
            {"total_diff": abs(_f(t_on) - _f(t_off)), "grad_max_diff": mdiff},
            "monitor on vs off: total and full gradient",
        )

    # 7. Gradient routing: which term reaches which parameter group.
    _, probe_terms = run(probe)
    routed = {"task": ("task", {"adapter"})}
    if config.cpc.enabled:
        routed["cpc"] = ("cpc", {"adapter", "cpc"})
    if config.hid.enabled:
        # detach_input: only P_l learns (negative control); otherwise the adapter does too.
        routed["hid"] = ("hid", {"adapter", "pred"} if config.hid.coupled else {"pred"})
    if "stab_anch" in probe_terms:
        routed["stab_anch"] = ("stab_anch", {"adapter"})
    if "stab_hep" in probe_terms:
        routed["stab_hep"] = (
            "stab_hep",
            {"hep"} if config.stab.hep.trainable else set(),
        )
    norms: dict[str, dict[str, float]] = {}
    for label, (term, expected_groups) in routed.items():
        grad = jax.grad(lambda p, term=term: run(p)[1][term])(probe)
        norms[label] = _group_norms(grad)
        got = {g for g, v in norms[label].items() if v > 0.0}
        add(
            f"grad_routing_{label}",
            got == expected_groups,
            norms[label],
            f"gradient reaches {sorted(got)}, expected {sorted(expected_groups)}",
        )

    # 8. Information: size of each term's gradient on the adapter relative to the task's
    #    (the plan aims at roughly 0.1 to 1 for each auxiliary term).
    task_norm = norms["task"]["adapter"]
    weights = {
        "cpc": config.cpc.weight,
        "stab_anch": config.stab.weight * config.stab.w_anch,
        "hid": config.hid.weight if config.hid.coupled else 0.0,
    }
    for label, weight in weights.items():
        if label in norms and task_norm > 0 and weight > 0:
            ratio = weight * norms[label]["adapter"] / task_norm
            add(
                f"grad_ratio_{label}",
                None,
                ratio,
                "lambda * |grad_adapter(term)| / |grad_adapter(task)|; target ~0.1-1",
            )

    # 8b. Information: does the predictor beat copying its input? (rel < copy_rel)
    if config.hid.enabled:
        add(
            "hid_vs_copy",
            None,
            {
                "hid_rel": _f(probe_terms["hid_rel"]),
                "copy_rel": _f(probe_terms["hid_copy_rel"]),
            },
            "target-variance-normalised error of P_l(h~) vs the copy baseline h~ (untrained P_l)",
        )

    # 9. jit equals eager.
    eager = _f(run(probe)[0])
    jitted = _f(jax.jit(lambda p: run(p)[0])(probe))
    add(
        "jit_equals_eager",
        abs(eager - jitted) <= 1e-4 * max(1.0, abs(eager)),
        {"eager": eager, "jit": jitted},
    )

    # 10. Ignored labels (-100) keep every term finite.
    if config.needs_corrected:
        half = input_ids.shape[1] // 2
        labels = input_ids.at[:, half:].set(-100)
        _, masked = run(probe, labels=labels)
        bad = sorted(k for k, v in masked.items() if not bool(jnp.isfinite(v)))
        add("finite_with_ignored_labels", not bad, None, str(bad))

    # 11. The frozen model is still untouched.
    add("frozen_after", bool(substrate.params_unchanged()), None)
    return checks


def audit_ok(checks: list[Check]) -> bool:
    """``True`` when no checkpoint failed (information-only checks never fail)."""
    return all(c.ok is not False for c in checks)


def audit_to_dict(checks: list[Check]) -> dict[str, Any]:
    """JSON-serialisable form of an audit, for ``audit.json``."""
    return {
        "ok": audit_ok(checks),
        "checks": [
            {"name": c.name, "ok": c.ok, "value": c.value, "detail": c.detail}
            for c in checks
        ],
    }


def format_audit(checks: list[Check]) -> str:
    """A readable table of the audit: ``PASS`` / ``FAIL`` / ``info`` per checkpoint."""
    lines = []
    for c in checks:
        status = "info" if c.ok is None else ("PASS" if c.ok else "FAIL")
        value = ""
        if isinstance(c.value, float):
            value = f"{c.value:.4g}"
        elif isinstance(c.value, dict):
            value = " ".join(
                f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in c.value.items()
                if not isinstance(v, dict)
            )
        lines.append(f"[{status:4s}] {c.name:28s} {value}")
        if c.ok is False and c.detail:
            lines.append(f"         -> {c.detail}")
    return "\n".join(lines)


__all__ = [
    "ARMS",
    "Check",
    "ARM_DEFAULT_WEIGHTS",
    "A2Config",
    "ADAPTER_PREFIX",
    "CPCConfig",
    "HEPConfig",
    "HIDDEN_NORMS",
    "HiddenConfig",
    "MonitorConfig",
    "POSITIVES",
    "StabilityConfig",
    "a2_objective",
    "adapter_params",
    "a2_audit",
    "apply_arm",
    "audit_ok",
    "audit_to_dict",
    "cpc_term",
    "init_a2_params",
    "init_cpc_params",
    "format_audit",
    "hid_term",
    "init_hep_params",
    "param_counts",
    "predictor_bank",
    "predictor_params",
    "stability_term",
    "token_mask",
]