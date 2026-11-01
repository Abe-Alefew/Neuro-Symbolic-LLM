"""Stage A2 Contrastive Predictive Coding (CPC) tests.

Consolidates:
- InfoNCE contrastive future loss math, temperature scaling, stride, and padding masks.
- CPC query and action projection heads, parameter shapes, and FabricPC node representations.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from fabricpc.graph import initialize_params
from fabricpc.nodes import Linear

from losses.a2_loss import anchor_positions, contrastive_future_loss
from residual import cpc_heads
from residual.cpc_heads import (
    a_key,
    init_cpc_heads,
    predict_future,
    predict_future_reference,
    project,
    project_reference,
    q_key,
    split_cpc,
)

# --- Contrastive Future Loss (losses/a2_loss.py) -----------------------------------

D = M = 16


def heads(seed: int = 0, d: int = D, m: int = M, k: int = 1):
    p = init_cpc_heads(seed, [3], d, m, offsets=(k,), a_noise_std=0.05)
    return p[q_key(3)], p[a_key(3, k)]


def randn(*shape: int, seed: int = 0) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(seed), shape)


_jit_loss = jax.jit(
    contrastive_future_loss,
    static_argnames=("offset", "tau", "max_anchors_per_seq"),
)


def loss_fn(q, a, anchors, targets, **kw):
    return _jit_loss(q, a, anchors, targets, **kw)


# --- an independent, loop-based reference of the definition -----------------------------


def reference_loss(q, a, anchors, targets, *, offset, tau, max_anchors=None):
    """Literal per-anchor implementation; positive branch under stop_gradient."""
    *lead, batch, seq_len, d = anchors.shape
    anc = anchors.reshape(-1, batch, seq_len, d)
    tgt = targets.reshape(-1, batch, seq_len, d)
    n = seq_len - offset
    stride = 1 if max_anchors is None else math.ceil(n / max_anchors)
    items = []  # (sequence index, predicted code, positive code)
    for s in range(anc.shape[0]):
        for b in range(batch):
            for t in range(0, n, stride):
                z = predict_future(a, project(q, anc[s, b, t]))
                y = jax.lax.stop_gradient(project(q, tgt[s, b, t + offset]))
                z = z / jnp.sqrt(jnp.sum(z * z) + 1e-8)
                y = y / jnp.sqrt(jnp.sum(y * y) + 1e-8)
                items.append((b, z, y))
    losses = []
    for bi, zi, yi in items:
        logits = [zi @ yi / tau]
        logits += [zi @ yj / tau for j, (bj, _, yj) in enumerate(items) if bj != bi]
        losses.append(jax.nn.logsumexp(jnp.stack(logits)) - logits[0])
    return jnp.mean(jnp.stack(losses))


@pytest.mark.parametrize(
    "shape, offset, max_anchors",
    [
        ((3, 5, D), 1, None),
        ((3, 6, D), 3, None),
        ((3, 9, D), 1, 3),  # stride > 1
        ((2, 3, 4, D), 2, None),  # leading step axis S = 2
    ],
)
def test_matches_reference_value_and_gradients(shape, offset, max_anchors) -> None:
    q, a = heads(k=offset)
    anchors, targets = randn(*shape, seed=1), randn(*shape, seed=2)
    kw = {"offset": offset, "tau": 0.2}

    def mine(q, a, anchors):
        return loss_fn(q, a, anchors, targets, max_anchors_per_seq=max_anchors, **kw)[0]

    def ref(q, a, anchors):
        return reference_loss(q, a, anchors, targets, max_anchors=max_anchors, **kw)

    v1, g1 = jax.jit(jax.value_and_grad(mine, argnums=(0, 1, 2)))(q, a, anchors)
    v2, g2 = jax.jit(jax.value_and_grad(ref, argnums=(0, 1, 2)))(q, a, anchors)
    np.testing.assert_allclose(v1, v2, rtol=1e-5)
    # Equal gradients also prove the positive branch is under stop_gradient.
    for x, y in zip(
        jax.tree_util.tree_leaves(g1), jax.tree_util.tree_leaves(g2), strict=False
    ):
        np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-6)


def test_aligned_positives_give_near_zero_loss_and_perfect_retrieval() -> None:
    q = {"W": jnp.eye(D)}
    a = {"M": jnp.eye(D)}
    seq = randn(4, 6, D, seed=3)
    offset = 1
    # The anchor at t has exactly the content of the target at t + offset.
    anchors = jnp.concatenate([seq[:, offset:], seq[:, :offset]], axis=1)
    loss, m = loss_fn(q, a, anchors, seq, offset=offset, tau=0.05)
    assert float(loss) < 1e-2
    assert float(m["cpc_top1"]) == 1.0
    assert float(m["cpc_pos_sim"]) == pytest.approx(1.0, abs=1e-4)
    assert float(m["cpc_neg_sim"]) < 0.9


@pytest.mark.parametrize("batch, steps", [(3, 1), (4, 1), (3, 2)])
def test_uninformative_codes_give_log_n_loss(batch, steps) -> None:
    q, a = heads()
    # Constant states: LayerNorm maps them to 0, so every similarity is 0.
    shape = (steps, batch, 5, D) if steps > 1 else (batch, 5, D)
    x = jnp.full(shape, 2.0)
    loss, m = loss_fn(q, a, x, x, offset=1, tau=0.1)
    n_pos = 4
    expected = math.log(1 + (batch - 1) * n_pos * steps)
    assert float(loss) == pytest.approx(expected, rel=1e-5)
    assert float(m["cpc_valid"]) == 1.0


def test_other_positions_of_the_same_sequence_are_not_negatives() -> None:
    q = {"W": jnp.eye(D)}
    a = {"M": jnp.eye(D)}
    batch, seq_len = 4, 9
    per_seq = randn(batch, 1, D, seed=4)
    x = jnp.broadcast_to(per_seq, (batch, seq_len, D))  # identical at every position
    loss, m = loss_fn(q, a, x, x, offset=1, tau=0.05)
    n_pos = seq_len - 1
    # If same-sequence entries counted as negatives, ties with the positive would force
    # loss >= log(n_pos). Masked, only the (random) other sequences compete.
    assert float(loss) < 0.25 * math.log(n_pos)
    assert float(m["cpc_top1"]) == 1.0


@pytest.mark.parametrize("offset", [1, 3])
def test_positions_beyond_the_sequence_are_ignored(offset) -> None:
    q, a = heads(k=offset)
    anchors, targets = randn(3, 10, D, seed=5), randn(3, 10, D, seed=6)
    base, _ = loss_fn(q, a, anchors, targets, offset=offset)
    # Last `offset` anchor positions have no future: must not matter.
    a2 = anchors.at[:, -offset:].set(randn(3, offset, D, seed=7))
    np.testing.assert_array_equal(base, loss_fn(q, a, a2, targets, offset=offset)[0])
    # First `offset` target positions are nobody's future: must not matter.
    t2 = targets.at[:, :offset].set(randn(3, offset, D, seed=8))
    np.testing.assert_array_equal(base, loss_fn(q, a, anchors, t2, offset=offset)[0])


def test_single_sequence_has_no_negatives() -> None:
    q, a = heads()
    for shape in ((1, 6, D), (3, 1, 6, D)):
        anchors, targets = randn(*shape, seed=9), randn(*shape, seed=10)

        def f(q, a, anchors, targets=targets):
            return loss_fn(q, a, anchors, targets, offset=1)

        (loss, m), grads = jax.value_and_grad(f, argnums=(0, 1, 2), has_aux=True)(
            q, a, anchors
        )
        assert float(loss) == 0.0
        assert float(m["cpc_valid"]) == 0.0
        assert all(
            np.isfinite(g).all() and not np.any(g)
            for g in jax.tree_util.tree_leaves(grads)
        )


def test_same_sequence_at_other_leading_steps_is_not_a_negative() -> None:
    q = {"W": jnp.eye(D)}
    a = {"M": jnp.eye(D)}
    steps, batch, seq_len = 3, 4, 6
    per_seq = randn(1, batch, 1, D, seed=11)
    x = jnp.broadcast_to(
        per_seq, (steps, batch, seq_len, D)
    )  # same at every step/position
    loss, m = loss_fn(q, a, x, x, offset=1, tau=0.05)
    # Counted as negatives, the steps * (seq_len - 1) - 1 copies of the sequence would tie
    # with the positive and push the loss to >= log(steps * n_pos).
    assert float(loss) < 0.25 * math.log(steps * (seq_len - 1))
    assert float(m["cpc_top1"]) == 1.0
    assert float(m["cpc_n_anchors"]) == steps * batch * (seq_len - 1)


def test_extra_negative_copies_raise_the_loss() -> None:
    # Duplicating the batch along a leading axis adds copies of *other* sequences'
    # positives, i.e. more negatives: the loss must grow (so the mask is not too greedy).
    q, a = heads()
    anchors, targets = randn(4, 7, D, seed=11), randn(4, 7, D, seed=12)
    base, _ = loss_fn(q, a, anchors, targets, offset=1, tau=0.2)
    rep = lambda x: jnp.stack([x, x, x])  # noqa: E731
    stacked, _ = loss_fn(q, a, rep(anchors), rep(targets), offset=1, tau=0.2)
    assert float(stacked) > float(base)


def test_gradient_audit() -> None:
    q, a = heads()
    anchors, targets = randn(4, 8, D, seed=13), randn(4, 8, D, seed=14)

    def f(q, a, anchors, targets):
        return loss_fn(q, a, anchors, targets, offset=1, tau=0.1)[0]

    gq, ga, g_anchor, g_target = jax.grad(f, argnums=(0, 1, 2, 3))(
        q, a, anchors, targets
    )
    assert not np.any(g_target), "the positive must not receive gradient"
    assert np.abs(gq["W"]).max() > 0 and np.abs(ga["M"]).max() > 0
    assert np.abs(g_anchor).max() > 0
    # No gradient into anchor positions that have no future.
    assert not np.any(g_anchor[:, -1])
    for g in (gq["W"], ga["M"], g_anchor):
        assert np.isfinite(g).all()


def test_anchor_positions() -> None:
    assert anchor_positions(5, None) == (0, 1, 2, 3, 4)
    assert anchor_positions(127, 256) == tuple(range(127))
    pos = anchor_positions(127, 50)  # stride 3
    assert pos[0] == 0 and pos[1] - pos[0] == 3 and len(pos) <= 50
    assert pos == anchor_positions(127, 50)  # deterministic
    assert len(anchor_positions(1000, 256)) <= 256
    with pytest.raises(ValueError):
        anchor_positions(0, None)
    with pytest.raises(ValueError):
        anchor_positions(5, 0)


def test_max_anchors_limits_the_anchor_count() -> None:
    q, a = heads()
    x = randn(4, 40, D, seed=15)
    _, m_all = loss_fn(q, a, x, x, offset=1)
    _, m_cap = loss_fn(q, a, x, x, offset=1, max_anchors_per_seq=10)
    assert float(m_all["cpc_n_anchors"]) == 4 * 39
    assert float(m_cap["cpc_n_anchors"]) <= 4 * 10


def test_numerically_stable_for_tiny_temperature_and_low_precision() -> None:
    q, a = heads()
    anchors, targets = randn(4, 8, D, seed=16), randn(4, 8, D, seed=17)

    def f(q, a, anchors):
        return loss_fn(q, a, anchors, targets, offset=1, tau=1e-3)[0]

    v, g = jax.value_and_grad(f, argnums=(0, 1, 2))(q, a, anchors)
    assert np.isfinite(v)
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(g))

    f32, _ = loss_fn(q, a, anchors, targets, offset=1, tau=0.2)
    bf16, _ = loss_fn(
        q,
        a,
        anchors.astype(jnp.bfloat16),
        targets.astype(jnp.bfloat16),
        offset=1,
        tau=0.2,
    )
    assert bf16.dtype == jnp.float32 and np.isfinite(bf16)
    np.testing.assert_allclose(bf16, f32, rtol=5e-2)


def test_jit_matches_eager() -> None:
    q, a = heads()
    anchors, targets = randn(4, 8, D, seed=18), randn(4, 8, D, seed=19)

    def f(q, a, anchors, targets):
        return loss_fn(q, a, anchors, targets, offset=2, tau=0.1, max_anchors_per_seq=4)

    (e, em), eg = jax.value_and_grad(f, argnums=(0, 1), has_aux=True)(
        q, a, anchors, targets
    )
    (j, jm), jg = jax.jit(jax.value_and_grad(f, argnums=(0, 1), has_aux=True))(
        q, a, anchors, targets
    )
    np.testing.assert_allclose(e, j, rtol=1e-5)
    for x, y in zip(
        jax.tree_util.tree_leaves(eg), jax.tree_util.tree_leaves(jg), strict=False
    ):
        np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-6)
    assert set(em) == set(jm)


@pytest.mark.parametrize(
    "kwargs, shape, match",
    [
        ({"offset": 0}, (3, 6, D), "offset"),
        ({"offset": -1}, (3, 6, D), "offset"),
        ({"offset": True}, (3, 6, D), "offset"),
        ({"offset": 1.0}, (3, 6, D), "offset"),
        ({"offset": 6}, (3, 6, D), "seq_len"),
        ({"tau": 0.0}, (3, 6, D), "tau"),
        ({"tau": -0.1}, (3, 6, D), "tau"),
        ({"max_anchors_per_seq": 0}, (3, 6, D), "max_anchors"),
        ({}, (6, D), "expected"),
    ],
)
def test_validation(kwargs, shape, match) -> None:
    q, a = heads()
    x = randn(*shape)
    with pytest.raises(ValueError, match=match):
        loss_fn(q, a, x, x, **kwargs)


def test_shape_mismatch_is_rejected() -> None:
    q, a = heads()
    with pytest.raises(ValueError, match="equal shapes"):
        loss_fn(q, a, randn(3, 6, D), randn(3, 7, D))


def test_heads_learn_to_predict_the_future_of_a_linear_process() -> None:
    """The loss is actually minimisable: trained heads retrieve the true future."""
    d, m, batch, seq_len = 16, 8, 16, 12
    rng = np.random.default_rng(0)
    rot = np.linalg.qr(rng.normal(size=(d, d)))[0]  # h_{t+1} = R h_t + noise
    h = np.zeros((batch, seq_len, d), np.float32)
    h[:, 0] = rng.normal(size=(batch, d))
    for t in range(1, seq_len):
        h[:, t] = h[:, t - 1] @ rot.T + 0.05 * rng.normal(size=(batch, d))
    x = jnp.asarray(h)

    params = init_cpc_heads(0, [3], d, m)
    params = {"q": params[q_key(3)], "a": params[a_key(3, 1)]}

    def f(p):
        return loss_fn(p["q"], p["a"], x, x, offset=1, tau=0.1)

    opt = optax.adam(2e-2)
    state = opt.init(params)
    step = jax.jit(jax.value_and_grad(f, has_aux=True))
    (first, m0), _ = step(params)
    for _ in range(300):
        (_, _), grads = step(params)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    (last, m1), _ = step(params)

    chance = 1 / (1 + (batch - 1) * (seq_len - 1))
    assert float(last) < 0.5 * float(first)
    assert float(m1["cpc_top1"]) > max(10 * chance, 2 * float(m0["cpc_top1"]))


# ----------------------------------------------------------------------
# Padding mask and head backend
# ----------------------------------------------------------------------

TAU = 0.2


def _heads(k: int = 1, seed: int = 0):
    p = init_cpc_heads(seed, [3], D, M, offsets=(k,), a_noise_std=0.1)
    return p[q_key(3)], p[a_key(3, k)]


def _x(*shape: int, seed: int = 0) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(seed), shape)


_loss = jax.jit(
    contrastive_future_loss,
    static_argnames=("offset", "tau", "max_anchors_per_seq", "backend"),
)


def _right_padded(lengths, seq_len: int) -> jax.Array:
    return jnp.arange(seq_len)[None, :] < jnp.asarray(lengths)[:, None]


def reference_masked(q, a, anchors, targets, mask, *, offset, tau):
    """Literal loop: valid anchors need t and t+k real; negatives are the positives of
    the valid anchors of *other* sequences; anchors without any negative are skipped."""
    batch, seq_len, _ = anchors.shape
    items = []
    for b in range(batch):
        for t in range(seq_len - offset):
            if bool(mask[b, t]) and bool(mask[b, t + offset]):
                z = predict_future(a, project(q, anchors[b, t], "jax"), "jax")
                y = jax.lax.stop_gradient(project(q, targets[b, t + offset], "jax"))
                z = z / jnp.sqrt(jnp.sum(z * z) + 1e-8)
                y = y / jnp.sqrt(jnp.sum(y * y) + 1e-8)
                items.append((b, z, y))
    losses = []
    for bi, zi, yi in items:
        neg = [zi @ yj / tau for bj, _, yj in items if bj != bi]
        if not neg:
            continue
        logits = jnp.stack([zi @ yi / tau, *neg])
        losses.append(jax.nn.logsumexp(logits) - logits[0])
    return jnp.mean(jnp.stack(losses)) if losses else jnp.zeros(())


def _leaves(tree):
    return jax.tree_util.tree_leaves(tree)


# --- mask semantics --------------------------------------------------------------------------


def test_all_true_mask_equals_no_mask() -> None:
    q, a = _heads()
    anc, tgt = _x(3, 7, D, seed=1), _x(3, 7, D, seed=2)
    ones = jnp.ones((3, 7), dtype=bool)

    def f(mask):
        return jax.value_and_grad(
            lambda q, a, anc: _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)[0],
            argnums=(0, 1, 2),
        )(q, a, anc)

    (v0, g0), (v1, g1) = f(None), f(ones)
    np.testing.assert_allclose(v0, v1, rtol=1e-5, atol=1e-7)
    for x, y in zip(_leaves(g0), _leaves(g1), strict=True):
        np.testing.assert_allclose(x, y, atol=1e-7)
    m0 = _loss(q, a, anc, tgt, offset=1, tau=TAU)[1]
    m1 = _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=ones)[1]
    for k in m0:
        np.testing.assert_allclose(m0[k], m1[k], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "mask",
    [
        _right_padded([9, 5, 7], 9),  # right padding
        jnp.asarray(  # a hole in the middle of a sequence, plus a short one
            [[1, 1, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0, 0], [1] * 9], bool
        ),
    ],
)
@pytest.mark.parametrize("offset", [1, 2])
def test_matches_a_literal_reference_with_padding(mask, offset) -> None:
    q, a = _heads(k=offset)
    anc, tgt = _x(3, 9, D, seed=3), _x(3, 9, D, seed=4)

    def mine(q, a, anc):
        return _loss(q, a, anc, tgt, offset=offset, tau=TAU, mask=mask)[0]

    def ref(q, a, anc):
        return reference_masked(q, a, anc, tgt, mask, offset=offset, tau=TAU)

    v1, g1 = jax.value_and_grad(mine, argnums=(0, 1, 2))(q, a, anc)
    v2, g2 = jax.value_and_grad(ref, argnums=(0, 1, 2))(q, a, anc)
    np.testing.assert_allclose(v1, v2, rtol=1e-5)
    for x, y in zip(_leaves(g1), _leaves(g2), strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-6)


def test_masked_positions_have_no_influence_and_no_gradient() -> None:
    q, a = _heads()
    mask = _right_padded([9, 5, 7], 9)
    anc, tgt = _x(3, 9, D, seed=5), _x(3, 9, D, seed=6)
    # Different random content on the padding only.
    junk = jnp.where(mask[..., None], 0.0, _x(3, 9, D, seed=32) * 100.0)

    def f(anc, tgt):
        return _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)[0]

    v0 = f(anc, tgt)
    v1 = f(anc + junk, tgt + junk)
    np.testing.assert_allclose(v0, v1, rtol=1e-5, atol=1e-7)
    g_anc, g_tgt = jax.grad(f, argnums=(0, 1))(anc, tgt)
    assert not np.any(g_anc[~mask])  # padding receives no gradient
    assert bool(jnp.all(jnp.isfinite(g_anc)))
    assert not np.any(g_tgt)  # positives are stop-gradient


def test_a_position_only_counts_when_both_t_and_t_plus_k_are_real() -> None:
    q, a = _heads()
    anc, tgt = _x(3, 8, D, seed=7), _x(3, 8, D, seed=8)
    # Only token 4 of sequence 0 is masked: anchors (0, 4) and (0, 3) disappear.
    mask = jnp.ones((3, 8), bool).at[0, 4].set(False)

    def f(anc, tgt, mask):
        return _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)

    _, m = f(anc, tgt, mask)
    assert float(m["cpc_n_anchors"]) == 3 * 7 - 2
    # Changing the content of the masked token changes nothing.
    # (random content: a constant shift would be removed by the head's LayerNorm)
    anc2 = anc.at[0, 4].set(_x(D, seed=30) * 5.0)
    tgt2 = tgt.at[0, 4].set(_x(D, seed=31) * 5.0)
    np.testing.assert_allclose(
        f(anc, tgt, mask)[0], f(anc2, tgt2, mask)[0], rtol=1e-5, atol=1e-7
    )
    # ... but without the mask it does.
    assert abs(float(f(anc, tgt, None)[0]) - float(f(anc2, tgt2, None)[0])) > 1e-3


def test_a_fully_masked_sequence_equals_dropping_it() -> None:
    q, a = _heads()
    anc, tgt = _x(3, 7, D, seed=9), _x(3, 7, D, seed=10)
    mask = jnp.asarray([[1] * 7, [1] * 7, [0] * 7], bool)
    masked, m = _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)
    dropped, md = _loss(q, a, anc[:2], tgt[:2], offset=1, tau=TAU)
    np.testing.assert_allclose(masked, dropped, rtol=1e-5, atol=1e-7)
    for k in ("cpc_top1", "cpc_pos_sim", "cpc_neg_sim", "cpc_n_anchors"):
        np.testing.assert_allclose(m[k], md[k], rtol=1e-5, atol=1e-7)


def test_padded_negatives_are_not_used() -> None:
    """Moving a padded sequence's content must not change the other sequences' loss."""
    q, a = _heads()
    anc, tgt = _x(3, 7, D, seed=11), _x(3, 7, D, seed=12)
    mask = _right_padded([7, 7, 3], 7)
    base = _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)[0]
    tgt_p = tgt.at[2, 3:].set(_x(4, D, seed=13) * 10.0)  # only the padded tail
    np.testing.assert_allclose(
        base,
        _loss(q, a, anc, tgt_p, offset=1, tau=TAU, mask=mask)[0],
        rtol=1e-5,
        atol=1e-7,
    )


def test_without_any_valid_negative_the_loss_is_zero_and_flagged_invalid() -> None:
    q, a = _heads()
    anc, tgt = _x(3, 6, D, seed=14), _x(3, 6, D, seed=15)
    only_first = jnp.asarray([[1] * 6, [0] * 6, [0] * 6], bool)
    nothing = jnp.zeros((3, 6), bool)
    for mask in (only_first, nothing):
        loss, m = _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)
        assert float(loss) == 0.0
        assert float(m["cpc_valid"]) == 0.0 and float(m["cpc_n_anchors"]) == 0.0
        assert all(bool(jnp.isfinite(v)) for v in m.values())

        def loss_of(q, a, anc, mask=mask):
            return _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)[0]

        g = jax.grad(loss_of, argnums=(0, 1, 2))(q, a, anc)
        assert all(not np.any(x) and bool(jnp.all(jnp.isfinite(x))) for x in _leaves(g))


def test_metrics_only_count_valid_anchors() -> None:
    q = {"W": jnp.eye(D)}
    a = {"M": jnp.eye(D)}
    seq = _x(4, 6, D, seed=16)
    anchors = jnp.concatenate([seq[:, 1:], seq[:, :1]], axis=1)  # aligned for k=1
    mask = _right_padded([6, 6, 4, 6], 6)
    # Garble the padded tail so that, were it counted, retrieval would fail.
    anchors = jnp.where(mask[..., None], anchors, _x(4, 6, D, seed=17))
    loss, m = _loss(q, a, anchors, seq, offset=1, tau=0.05, mask=mask)
    assert float(m["cpc_top1"]) == 1.0
    assert float(m["cpc_pos_sim"]) == pytest.approx(1.0, abs=1e-4)
    assert float(loss) < 1e-2


def test_step_axis_accepts_a_shared_or_a_per_step_mask() -> None:
    q, a = _heads()
    anc, tgt = _x(2, 3, 7, D, seed=18), _x(2, 3, 7, D, seed=19)
    mask = _right_padded([7, 4, 6], 7)
    shared = _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)[0]
    per_step = _loss(
        q, a, anc, tgt, offset=1, tau=TAU, mask=jnp.broadcast_to(mask, (2, 3, 7))
    )[0]
    np.testing.assert_allclose(shared, per_step, rtol=1e-5, atol=1e-7)


def test_integer_masks_are_accepted() -> None:
    q, a = _heads()
    anc, tgt = _x(3, 7, D, seed=20), _x(3, 7, D, seed=21)
    mask = _right_padded([7, 5, 6], 7)
    np.testing.assert_allclose(
        _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)[0],
        _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask.astype(jnp.int32))[0],
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize("shape", [(3,), (7, 3), (2, 7), (3, 8), (2, 3, 7)])
def test_wrongly_shaped_mask_is_rejected(shape) -> None:
    q, a = _heads()
    anc, tgt = _x(3, 7, D, seed=22), _x(3, 7, D, seed=23)
    with pytest.raises(ValueError, match="mask"):
        contrastive_future_loss(
            q, a, anc, tgt, offset=1, tau=TAU, mask=jnp.ones(shape, bool)
        )


def test_jit_with_a_traced_mask() -> None:
    q, a = _heads()
    anc, tgt = _x(3, 7, D, seed=24), _x(3, 7, D, seed=25)
    mask = _right_padded([7, 4, 6], 7)
    eager = contrastive_future_loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)[0]
    jitted = _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask)[0]
    np.testing.assert_allclose(eager, jitted, rtol=1e-5)


def test_max_anchors_stride_still_applies_with_a_mask() -> None:
    q, a = _heads()
    anc, tgt = _x(3, 13, D, seed=26), _x(3, 13, D, seed=27)
    mask = _right_padded([13, 13, 13], 13)
    _, m = _loss(q, a, anc, tgt, offset=1, tau=TAU, mask=mask, max_anchors_per_seq=4)
    assert float(m["cpc_n_anchors"]) == 3 * 4  # 12 positions, stride 3 -> 4 each


# --- CPC Projection & Action Heads (residual/cpc_heads.py) ---------------------------

D_HEADS, M_HEADS = 24, 8


def test_keys_and_shapes() -> None:
    params = init_cpc_heads(0, [7, 9], D_HEADS, M_HEADS, offsets=(1, 2))
    assert set(params) == {
        q_key(7),
        q_key(9),
        a_key(7, 1),
        a_key(7, 2),
        a_key(9, 1),
        a_key(9, 2),
    }
    assert q_key(7) == "cpcq_7" and a_key(7, 2) == "cpca_7_2"
    for layer in (7, 9):
        assert params[q_key(layer)]["W"].shape == (D_HEADS, M_HEADS)
        for k in (1, 2):
            assert params[a_key(layer, k)]["M"].shape == (M_HEADS, M_HEADS)
    # Two-level layout required by RunWriter.save_params.
    assert all(set(v) <= {"W", "M"} for v in params.values())
    assert all(v.dtype == jnp.float32 for p in params.values() for v in p.values())


def test_layers_are_deduplicated_and_order_independent() -> None:
    a = init_cpc_heads(0, [9, 7, 7], D_HEADS, M_HEADS)
    b = init_cpc_heads(0, [7, 9], D_HEADS, M_HEADS)
    assert set(a) == set(b)
    for key in a:
        for name in a[key]:
            np.testing.assert_array_equal(a[key][name], b[key][name])


def test_deterministic_in_seed() -> None:
    a, b, c = (init_cpc_heads(s, [3], D_HEADS, M_HEADS) for s in (0, 0, 1))
    np.testing.assert_array_equal(a["cpcq_3"]["W"], b["cpcq_3"]["W"])
    assert not np.array_equal(a["cpcq_3"]["W"], c["cpcq_3"]["W"])


def test_q_is_random_and_has_unit_scale() -> None:
    w = np.asarray(init_cpc_heads(0, [3], 256, 64)["cpcq_3"]["W"])
    assert np.abs(w).max() > 0
    # N(0, 1/d): each column has norm ~ sqrt(m/d)... check the entry std instead.
    assert w.std() == pytest.approx(1 / np.sqrt(256), rel=0.1)


def test_a_is_identity_plus_noise_never_zero_and_differs_per_offset() -> None:
    p = init_cpc_heads(0, [3], D_HEADS, M_HEADS, offsets=(1, 2), a_noise_std=0.01)
    a1, a2 = (np.asarray(p[a_key(3, k)]["M"]) for k in (1, 2))
    assert np.abs(a1).max() > 0
    np.testing.assert_allclose(a1, np.eye(M_HEADS), atol=0.1)
    assert not np.array_equal(a1, np.eye(M_HEADS))  # noise present
    assert not np.array_equal(a1, a2)  # offsets not tied


def test_zero_noise_gives_exact_identity() -> None:
    p = init_cpc_heads(0, [3], D_HEADS, M_HEADS, a_noise_std=0.0)
    np.testing.assert_array_equal(p[a_key(3, 1)]["M"], np.eye(M_HEADS))


def test_layers_do_not_share_weights() -> None:
    p = init_cpc_heads(0, [3, 4], D_HEADS, M_HEADS)
    assert not np.array_equal(p["cpcq_3"]["W"], p["cpcq_4"]["W"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 0},
        {"dim": 0},
        {"a_noise_std": -1.0},
        {"offsets": ()},
        {"offsets": (0,)},
        {"offsets": (True,)},
    ],
)
def test_init_validation(kwargs: dict) -> None:
    args = {"hidden_size": D_HEADS, "dim": M} | kwargs
    with pytest.raises(ValueError):
        init_cpc_heads(0, [3], **args)


def test_project_shape_dtype_and_layernorm_invariance() -> None:
    q = init_cpc_heads(0, [3], D_HEADS, M_HEADS)["cpcq_3"]
    h = jax.random.normal(jax.random.PRNGKey(0), (2, 5, D_HEADS))
    c = project(q, h)
    assert c.shape == (2, 5, M_HEADS) and c.dtype == jnp.float32
    # LayerNorm: invariant to a positive rescale and a shift of each state.
    np.testing.assert_allclose(project(q, 3.0 * h + 5.0), c, atol=2e-3)
    # float32 output for bf16 input.
    assert project(q, h.astype(jnp.bfloat16)).dtype == jnp.float32


def test_project_is_linear_after_layernorm() -> None:
    q = init_cpc_heads(0, [3], D_HEADS, M_HEADS)["cpcq_3"]
    h = jax.random.normal(jax.random.PRNGKey(1), (4, D_HEADS))
    expected = (
        (h - h.mean(-1, keepdims=True)) / jnp.sqrt(h.var(-1, keepdims=True) + 1e-5)
    ) @ q["W"]
    np.testing.assert_allclose(project(q, h), expected, atol=1e-5)


def test_predict_future_is_a_matmul() -> None:
    p = init_cpc_heads(0, [3], D_HEADS, M_HEADS)
    c = jax.random.normal(jax.random.PRNGKey(2), (2, 5, M_HEADS))
    out = predict_future(p[a_key(3, 1)], c)
    assert out.shape == c.shape
    np.testing.assert_allclose(out, c @ p[a_key(3, 1)]["M"], atol=1e-6)


def test_split_cpc_selects_only_head_entries() -> None:
    tree = {
        "layer_3": {"A": jnp.ones((2, 2)), "B": jnp.ones((2, 2))},
        "pred_3": {"W": jnp.ones((2, 2))},
        **init_cpc_heads(0, [3], D_HEADS, M_HEADS),
    }
    assert set(split_cpc(tree)) == {"cpcq_3", "cpca_3_1"}
    assert split_cpc({"layer_3": tree["layer_3"]}) == {}


# ----------------------------------------------------------------------
# FabricPC backend against the pure-JAX reference
# ----------------------------------------------------------------------


def _cpc_heads_pair(seed: int = 0):
    p = init_cpc_heads(seed, [3], D_HEADS, M_HEADS, offsets=(1,), a_noise_std=0.1)
    return p[q_key(3)], p[a_key(3, 1)]


@pytest.mark.parametrize("shape", [(7, D_HEADS), (3, 5, D_HEADS), (2, 3, 5, D_HEADS)])
def test_projection_matches_the_reference(shape) -> None:
    q, _ = _cpc_heads_pair()
    h = _x(*shape, seed=1)
    out = project(q, h)
    assert out.shape == (*shape[:-1], M_HEADS) and out.dtype == jnp.float32
    np.testing.assert_allclose(out, project_reference(q, h), atol=1e-5)
    np.testing.assert_allclose(out, project(q, h, "jax"), atol=1e-5)


@pytest.mark.parametrize("shape", [(7, M_HEADS), (3, 5, M_HEADS), (2, 3, 5, M_HEADS)])
def test_predictor_matches_the_reference(shape) -> None:
    _, a = _cpc_heads_pair()
    c = _x(*shape, seed=2)
    out = predict_future(a, c)
    assert out.shape == shape
    np.testing.assert_allclose(out, predict_future_reference(a, c), atol=1e-5)


def test_gradients_match_the_reference_for_weights_and_input() -> None:
    q, a = _cpc_heads_pair()
    h, w = _x(3, 5, D_HEADS, seed=3), _x(3, 5, M_HEADS, seed=4)

    def loss(backend):
        def f(q, a, h):
            z = predict_future(a, project(q, h, backend), backend)
            return jnp.sum(z * w)

        return f

    g_f = jax.grad(loss("fabricpc"), argnums=(0, 1, 2))(q, a, h)
    g_j = jax.grad(loss("jax"), argnums=(0, 1, 2))(q, a, h)
    for x, y in zip(
        jax.tree_util.tree_leaves(g_f), jax.tree_util.tree_leaves(g_j), strict=True
    ):
        assert float(jnp.abs(y).max()) > 0
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5)


def test_jit_matches_eager_and_does_not_retrace_per_call() -> None:
    q, a = _cpc_heads_pair()
    h = _x(2, 6, D_HEADS, seed=5)
    traces = []

    def f(q, a, h):
        traces.append(1)
        return predict_future(a, project(q, h))

    jf = jax.jit(f)
    np.testing.assert_allclose(jf(q, a, h), f(q, a, h), atol=1e-5)
    n = len(traces)
    jf(q, a, h + 1.0)
    assert len(traces) == n  # same shapes -> compiled once


def test_low_precision_input_is_computed_in_float32() -> None:
    q, a = _cpc_heads_pair()
    h = _x(2, 4, D_HEADS, seed=6)
    out = predict_future(a, project(q, h.astype(jnp.bfloat16)))
    assert out.dtype == jnp.float32 and bool(jnp.all(jnp.isfinite(out)))
    np.testing.assert_allclose(out, predict_future(a, project(q, h)), atol=0.1)


def test_layer_norm_makes_the_projection_scale_and_shift_invariant() -> None:
    q, _ = _cpc_heads_pair()
    h = _x(4, D_HEADS, seed=7)
    np.testing.assert_allclose(project(q, 3.0 * h + 5.0), project(q, h), atol=2e-3)


def test_the_head_is_a_bias_free_linear_node_with_the_expected_edge_key() -> None:
    """Guards the weight-key convention the bridge relies on."""
    for src, dst, edge, d_in, d_out in (
        ("x", "q", "x->q:in", D_HEADS, M_HEADS),
        ("c", "A", "c->A:in", M_HEADS, M_HEADS),
    ):
        structure = cpc_heads._linear_head(src, dst, d_in, d_out)
        assert isinstance(structure.nodes[dst], Linear)  # a FabricPC Linear node
        assert structure.nodes[dst].shape == (d_out,)
        params = initialize_params(structure, jax.random.PRNGKey(0))
        assert set(params.nodes) == {src, dst}
        assert set(params.nodes[dst].weights) == {edge}
        assert params.nodes[dst].weights[edge].shape == (d_in, d_out)
        assert params.nodes[dst].biases == {}  # bias-free, like the reference
        assert params.nodes[src].weights == {}


def test_the_graphs_are_built_once_per_shape() -> None:
    assert cpc_heads._linear_head("x", "q", D_HEADS, M_HEADS) is cpc_heads._linear_head(
        "x", "q", D_HEADS, M
    )
    assert cpc_heads._linear_head(
        "x", "q", D_HEADS, M_HEADS
    ) is not cpc_heads._linear_head("x", "q", D_HEADS, M_HEADS + 1)


def test_stored_parameters_stay_plain_arrays_in_the_flat_tree() -> None:
    p = init_cpc_heads(0, [3, 4], D_HEADS, M_HEADS, offsets=(1, 2))
    assert all(isinstance(v, dict) for v in p.values())
    assert all(isinstance(w, jax.Array) for v in p.values() for w in v.values())
    assert set(p) == {
        q_key(3),
        q_key(4),
        a_key(3, 1),
        a_key(3, 2),
        a_key(4, 1),
        a_key(4, 2),
    }


def test_a_zero_predictor_gives_a_zero_prediction() -> None:
    _, a = _cpc_heads_pair()
    zero = {"M": jnp.zeros_like(a["M"])}
    assert not np.any(predict_future(zero, _x(3, M_HEADS, seed=8)))


def test_non_matrix_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="matrix"):
        project({"W": jnp.ones((D_HEADS, M_HEADS, 1))}, _x(2, D_HEADS))
