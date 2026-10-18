"""Contrastive-future (CPC) heads for Stage A2: a projection ``q_l`` and a per-offset predictor ``A_{l,k}``.


    c_t = q_l(h~_t) = LN(h~_t) @ W        W: [d, m]
    z_t = A_{l,k}(c_t) = c_t @ M          M: [m, m]   (predicts the code k steps ahead)

The two matrix products are FabricPC nodes (the default ``backend="fabricpc"``):
``q`` is a bias-free ``Linear`` node fed by the parameter-free LayerNorm, and ``A`` is
a bias-free ``Linear`` node. Each runs as a tiny two-node FabricPC graph whose
feed-forward value is the head's output; there is no inference/settling, so
gradients come from ordinary autodiff of the loss, exactly as with
``backend="jax"``, the pure-JAX reference used as the test oracle. The weights
stay in the flat A2 parameter tree (below); a FabricPC ``GraphParams`` is built
from them at call time, so nothing FabricPC-specific is stored or checkpointed.

Parameters are stored in the flat, prefixed, two-level layout shared with the
adapter tree, so ``RunWriter.save_params`` and ``ResidualAdapter.modify_fn``
(which ignores unknown keys) work unchanged:

    {"cpcq_7": {"W": f32[d, m]}, "cpca_7_1": {"M": f32[m, m]}, ...}
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.topology import Edge
from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_graph_state
from fabricpc.nodes import Linear
from fabricpc.training.trainer import build_clamps

CPCParams = dict[str, dict[str, jax.Array]]

Q_PREFIX = "cpcq_"
A_PREFIX = "cpca_"

# Separates the head PRNG stream from the adapter's ``fold_in(PRNGKey(seed), layer)``.
_STREAM = 0xA2
_KIND_Q = 1
_KIND_A = 2
_LN_EPS = 1e-5

BACKENDS = ("fabricpc", "jax")


def q_key(layer_idx: int) -> str:
    return f"{Q_PREFIX}{layer_idx}"


def a_key(layer_idx: int, offset: int) -> str:
    return f"{A_PREFIX}{layer_idx}_{offset}"


def init_cpc_heads(
    seed: int,
    layers: Iterable[int],
    hidden_size: int,
    dim: int,
    offsets: Iterable[int] = (1,),
    a_noise_std: float = 0.01,
) -> CPCParams:
    """Initial heads for every ``layer`` and ``offset``.

    ``W ~ N(0, 1/d)``. ``M = I + N(0, a_noise_std^2)``: never zero (a zero head
    has no gradient to the projection) and different for every offset. The
    stream is derived from ``seed`` but is disjoint from the adapter's.
    """
    if hidden_size < 1 or dim < 1:
        raise ValueError(f"hidden_size and dim must be >= 1, got {hidden_size}, {dim}")
    if not a_noise_std >= 0:
        raise ValueError(f"a_noise_std must be >= 0, got {a_noise_std!r}")
    offsets = tuple(offsets)
    if not offsets or any(
        not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in offsets
    ):
        raise ValueError(f"offsets must be integers >= 1, got {offsets!r}")

    base = jax.random.fold_in(jax.random.PRNGKey(seed), _STREAM)
    params: CPCParams = {}
    for layer in sorted(set(layers)):
        site = jax.random.fold_in(base, layer)
        kq = jax.random.fold_in(site, _KIND_Q)
        params[q_key(layer)] = {
            "W": jax.random.normal(kq, (hidden_size, dim), dtype=jnp.float32)
            / jnp.sqrt(jnp.float32(hidden_size))
        }
        for k in offsets:
            ka = jax.random.fold_in(jax.random.fold_in(site, _KIND_A), k)
            noise = a_noise_std * jax.random.normal(ka, (dim, dim), dtype=jnp.float32)
            params[a_key(layer, k)] = {"M": jnp.eye(dim, dtype=jnp.float32) + noise}
    return params


def _layer_norm(x: jax.Array) -> jax.Array:
    # Parameter-free, float32: hidden states of pretrained LMs have a few huge
    # outlier dimensions that would otherwise dominate a random projection.
    x = x.astype(jnp.float32)
    mu = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + _LN_EPS)


def _check_backend(backend: str) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {list(BACKENDS)}, got {backend!r}")


def project_reference(q_params: Mapping[str, jax.Array], h: jax.Array) -> jax.Array:
    """Pure-JAX ``q_l``: ``LN(h) @ W``. The oracle for the FabricPC head."""
    return _layer_norm(h) @ q_params["W"]


def predict_future_reference(
    a_params: Mapping[str, jax.Array], c: jax.Array
) -> jax.Array:
    """Pure-JAX ``A_{l,k}``: ``c @ M``. The oracle for the FabricPC head."""
    return c @ a_params["M"]


# --- FabricPC heads ---------------------------------------------------------------------------
# Tokens are folded into the batch axis, so a node has shape ``(features,)`` and the
# graphs do not depend on the sequence length. The graphs have no target and no
# inference step: only the feed-forward value ``z_mu`` of the output node is read.

_Q_EDGE = "x->q:in"
_A_EDGE = "c->A:in"


@cache
def _linear_head(src: str, dst: str, d_in: int, d_out: int) -> Any:
    source = Linear(name=src, shape=(d_in,))
    head = Linear(name=dst, shape=(d_out,), use_bias=False)
    return graph(
        nodes=[source, head],
        edges=[Edge(source=source, target=head.slot("in"))],
        task_map=TaskMap(x=source, y=head),
        inference=InferenceSGD(eta_infer=0.0, infer_steps=0),
    )


def _run_head(
    structure: Any, src: str, dst: str, edge: str, weight: jax.Array, x: jax.Array
) -> jax.Array:
    if weight.ndim != 2:
        raise ValueError(f"head weight must be a matrix, got shape {weight.shape}")
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    params = GraphParams(
        nodes={
            src: NodeParams({}, {}),
            dst: NodeParams({edge: weight.astype(jnp.float32)}, {}),
        }
    )
    clamps = build_clamps({"x": flat}, structure, clamp_target=False)
    state = initialize_graph_state(
        structure, flat.shape[0], jax.random.PRNGKey(0), clamps=clamps, params=params
    )
    return state.nodes[dst].z_mu.reshape(*lead, -1)


def project(
    q_params: Mapping[str, jax.Array], h: jax.Array, backend: str = "fabricpc"
) -> jax.Array:
    """``q_l``: ``[..., d] -> [..., m]`` (float32)."""
    _check_backend(backend)
    if backend == "jax":
        return project_reference(q_params, h)
    w = q_params["W"]
    structure = _linear_head("x", "q", int(w.shape[0]), int(w.shape[1]))
    return _run_head(structure, "x", "q", _Q_EDGE, w, _layer_norm(h))


def predict_future(
    a_params: Mapping[str, jax.Array], c: jax.Array, backend: str = "fabricpc"
) -> jax.Array:
    """``A_{l,k}``: ``[..., m] -> [..., m]``, the predicted code ``k`` steps ahead."""
    _check_backend(backend)
    if backend == "jax":
        return predict_future_reference(a_params, c)
    m = a_params["M"]
    structure = _linear_head("c", "A", int(m.shape[0]), int(m.shape[1]))
    return _run_head(structure, "c", "A", _A_EDGE, m, c.astype(jnp.float32))


def split_cpc(params: Mapping[str, Mapping[str, jax.Array]]) -> CPCParams:
    """The CPC head entries of a full A2 parameter tree."""
    return {k: dict(v) for k, v in params.items() if k.startswith((Q_PREFIX, A_PREFIX))}


__all__ = [
    "A_PREFIX",
    "BACKENDS",
    "CPCParams",
    "Q_PREFIX",
    "a_key",
    "init_cpc_heads",
    "predict_future",
    "predict_future_reference",
    "project",
    "project_reference",
    "q_key",
    "split_cpc",
]