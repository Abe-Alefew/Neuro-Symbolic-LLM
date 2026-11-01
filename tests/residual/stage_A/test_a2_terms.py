"""Stage A2 term math tests (without through-model execution).

Covers pure loss math for CPC, Anchor/HEP, Residual Smoothness, Step Loss,
and Hidden Prediction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from losses.a2_loss import (
    anchor_loss,
    hidden_prediction_loss,
)
from residual.cpc_heads import a_key, init_cpc_heads, project, q_key
from stages.stage_A.a2 import (
    CPCConfig,
    HEPConfig,
    HiddenConfig,
    StabLossConfig,
    cpc_term,
    hid_term,
    init_a2_params,
    init_cpc_params,
    predictor_bank,
    residual_smoothness_loss,
    stability_term,
    step_loss,
    token_mask,
)
from tests.residual.stage_A.conftest import (
    BATCH,
    LAYERS,
    SEQ,
    make_cfg,
    rand,
    with_random_predictors,
)


def _states(seed: int, layers=(5, 6), d: int = 16) -> dict[int, jax.Array]:
    keys = jax.random.split(jax.random.PRNGKey(seed), len(layers))
    return {
        layer: jax.random.normal(k, (3, 8, d))
        for layer, k in zip(layers, keys, strict=True)
    }


def _cfg_stab(hep_kind: str | None = None, **kw: object) -> StabLossConfig:
    hep = HEPConfig(kind=hep_kind, rank=4, alpha=1.0)
    return StabLossConfig(weight=1.0, hep=hep, **kw)  # type: ignore[arg-type]


def reference_smoothness(
    r: np.ndarray, h: np.ndarray, mask: np.ndarray | None
) -> tuple[float, float]:
    dr = r[:, 1:] - r[:, :-1]
    dh = h[:, 1:] - h[:, :-1]
    dr_sq = np.sum(dr**2, axis=-1)
    dh_sq = np.sum(dh**2, axis=-1)
    if mask is None:
        return float(dr_sq.mean()), float(dh_sq.mean())
    pair_mask = mask[:, 1:] & mask[:, :-1]
    n = max(float(pair_mask.sum()), 1.0)
    return float(np.sum(dr_sq * pair_mask) / n), float(np.sum(dh_sq * pair_mask) / n)


# --- CPC Term Math ----------------------------------------------------------------------------


def test_cpc_term_is_mean_over_sites_of_pure_loss() -> None:
    cfg = CPCConfig(weight=1.0, offsets=(2,), dim=8, tau=0.2, positive="base")
    params = init_cpc_params(cfg, 0, (5, 6), 16)
    corrected, base = _states(1), _states(2)
    loss, metrics = cpc_term(params, config=cfg, corrected=corrected, base=base)

    from losses.a2_loss import contrastive_future_loss

    per_site = [
        float(
            contrastive_future_loss(
                params[q_key(layer)],
                params[a_key(layer, 2)],
                corrected[layer],
                base[layer],
                offset=2,
                tau=0.2,
                max_anchors_per_seq=cfg.max_anchors_per_seq,
            )[0]
        )
        for layer in (5, 6)
    ]
    assert float(loss) == pytest.approx(float(np.mean(per_site)), rel=1e-5)
    for layer in (5, 6):
        assert f"cpc/site_{layer}" in metrics


def test_cpc_positive_base_reads_targets_from_base_and_corrected_from_anchors() -> None:
    corrected, base = _states(1), _states(2)
    cfg_b = CPCConfig(weight=1.0, dim=8, positive="base")
    cfg_c = CPCConfig(weight=1.0, dim=8, positive="corrected")
    params = init_cpc_params(cfg_b, 0, (5, 6), 16)
    lb, _ = cpc_term(params, config=cfg_b, corrected=corrected, base=base)
    lc, _ = cpc_term(params, config=cfg_c, corrected=corrected, base=base)
    assert not np.isclose(float(lb), float(lc))


def test_cpc_term_input_errors() -> None:
    cfg = CPCConfig(weight=1.0, dim=8)
    params = init_cpc_params(cfg, 0, (5, 6), 16)
    corrected = _states(1)
    with pytest.raises(ValueError, match="at least one site"):
        cpc_term(params, config=cfg, corrected={}, base={})
    with pytest.raises(ValueError, match="same sites"):
        cpc_term(
            params,
            config=cfg,
            corrected={5: corrected[5]},
            base={5: corrected[5], 6: corrected[6]},
        )
    with pytest.raises(ValueError, match="missing head"):
        cpc_term(
            {q_key(5): params[q_key(5)]},
            config=cfg,
            corrected={5: corrected[5]},
            base={5: corrected[5]},
        )


def test_cpc_anchor_codes_are_causal(substrate, adapter, input_ids) -> None:
    heads = init_cpc_heads(0, adapter.layers, substrate.architecture.hidden_size, 8)
    t0 = 7
    changed = input_ids.at[:, t0 + 1 :].set(0)
    ref_run = substrate.run_with_interception(input_ids, adapter.interceptor({}))
    alt_run = substrate.run_with_interception(changed, adapter.interceptor({}))
    for layer in adapter.layers:
        q_ref = project(ref_run.intercepted_states[layer], heads[f"cpcq_{layer}"])
        q_alt = project(alt_run.intercepted_states[layer], heads[f"cpcq_{layer}"])
        np.testing.assert_array_equal(q_ref[:, : t0 + 1, :], q_alt[:, : t0 + 1, :])


def test_cpc_term_is_mean_over_sites_and_offsets() -> None:
    cfg = CPCConfig(weight=1.0, offsets=(1, 2), dim=8, tau=0.2, positive="base")
    params = init_cpc_params(cfg, 0, (5, 6), 16)
    corrected, base = _states(1), _states(2)
    loss, metrics = cpc_term(params, config=cfg, corrected=corrected, base=base)
    for off in (1, 2):
        assert f"cpc/off_{off}" in metrics
    for layer in (5, 6):
        assert f"cpc/site_{layer}" in metrics


def test_cpc_offset_longer_than_sequence_is_error() -> None:
    cfg = CPCConfig(weight=1.0, offsets=(1, 8), dim=8)
    params = init_cpc_params(cfg, 0, (5, 6), 16)
    with pytest.raises(ValueError, match="offset"):
        cpc_term(params, config=cfg, corrected=_states(1), base=_states(2))


def test_token_mask_from_labels() -> None:
    labels = jnp.asarray([[3, 4, -100, -100], [1, -100, 2, 5]])
    np.testing.assert_array_equal(
        token_mask(labels),
        [[True, True, False, False], [True, False, True, True]],
    )
    assert token_mask(None) is None
    np.testing.assert_array_equal(
        token_mask(labels[:, :2]), [[True, True], [True, True]]
    )


def test_cpc_term_ignores_padded_positions() -> None:
    cfg = CPCConfig(weight=1.0, offsets=(1, 2), dim=8, tau=0.2, positive="base")
    params = init_cpc_params(cfg, 0, (5, 6), 16)
    corrected, base = _states(1), _states(2)
    mask = jnp.arange(8)[None, :] < jnp.asarray([8, 5, 6])[:, None]
    l_clean, _ = cpc_term(params, config=cfg, corrected=corrected, base=base, mask=mask)
    corrupted = {
        k: jnp.where(
            mask[..., None],
            v,
            100.0 * jax.random.normal(jax.random.PRNGKey(9), v.shape),
        )
        for k, v in corrected.items()
    }
    l_junk, _ = cpc_term(params, config=cfg, corrected=corrupted, base=base, mask=mask)
    assert float(l_clean) == pytest.approx(float(l_junk), abs=1e-5)


# --- Stability Term Math (Anchor & HEP) --------------------------------------------------------


def test_anchor_term_is_the_weighted_mean_over_sites() -> None:
    cfg = _cfg_stab(w_anch=2.0)
    corrected, base = _states(1), _states(2)
    loss, m = stability_term({}, config=cfg, corrected=corrected, base=base)
    per = [float(anchor_loss(corrected[layer], base[layer])[0]) for layer in (5, 6)]
    assert float(loss) == pytest.approx(2.0 * float(np.mean(per)), rel=1e-5)
    for layer in (5, 6):
        assert f"stab/site_{layer}_anch" in m
    assert "stab/site_5_hep" not in m


def test_w_anch_zero_skips_the_anchor_and_needs_no_base() -> None:
    loss, m = stability_term({}, config=_cfg_stab(w_anch=0.0), corrected=_states(1))
    assert float(loss) == 0.0 and "stab_anch" not in m


def test_hep_metric_is_logged_but_not_added_when_w_hep_is_zero() -> None:
    corrected, base = _states(1), _states(2)
    logits = jax.random.normal(jax.random.PRNGKey(3), (3, 8, 20))
    labels = jax.random.randint(jax.random.PRNGKey(4), (3, 8), 0, 20)
    tied = jax.random.normal(jax.random.PRNGKey(5), (20, 16))
    cfg = _cfg_stab("tied", w_anch=1.0, w_hep=0.0)
    loss, m = stability_term(
        {},
        config=cfg,
        corrected=corrected,
        base=base,
        logits=logits,
        labels=labels,
        tied=tied,
    )
    assert float(m["stab_hep"]) == 0.0
    assert float(m["stab_anch"]) == float(loss)
    assert float(m["stab/hep_nudge"]) > 0.0


def test_stability_term_input_errors() -> None:
    corrected = _states(1)
    with pytest.raises(ValueError, match="at least one site"):
        stability_term({}, config=_cfg_stab(), corrected={}, base={})
    with pytest.raises(ValueError, match="base states"):
        stability_term({}, config=_cfg_stab(w_anch=1.0), corrected=corrected, base=None)


def test_padding_is_ignored_by_the_anchor_term() -> None:
    cfg = StabLossConfig(weight=1.0)
    corrected, base = _states(1), _states(2)
    mask = jnp.arange(8)[None, :] < jnp.asarray([8, 5, 6])[:, None]
    junk = jnp.where(
        mask[..., None],
        0.0,
        100.0 * jax.random.normal(jax.random.PRNGKey(9), corrected[5].shape),
    )
    corrupted = {k: v + junk for k, v in corrected.items()}
    clean, _ = stability_term({}, config=cfg, corrected=corrected, base=base, mask=mask)
    dirty, _ = stability_term({}, config=cfg, corrected=corrupted, base=base, mask=mask)
    assert float(clean) == pytest.approx(float(dirty), rel=1e-5)


def test_the_loss_terms_vanish_when_their_weights_are_zero() -> None:
    sites = {k: rand(2, 6, 8, seed=100 + k) for k in (5, 6)}
    corrected = {k: rand(2, 6, 8, seed=k) for k in sites}
    base = {k: rand(2, 6, 8, seed=100 + k) for k in sites}
    cfg = StabLossConfig(weight=1.0, w_anch=0.0)
    total, m = stability_term(
        {}, config=cfg, corrected=corrected, sites=sites, base=base
    )
    assert float(total) == 0.0 and "stab_anch" not in m


def test_the_term_is_the_weighted_sum_of_all_four_parts() -> None:
    sites = {k: rand(2, 6, 8, seed=100 + k) for k in (5, 6)}
    corrected = {k: rand(2, 6, 8, seed=k) for k in sites}
    base = {k: rand(2, 6, 8, seed=100 + k) for k in sites}
    cfg = StabLossConfig(weight=1.0, w_anch=0.7, w_smooth=0.4, w_step=2.0)
    total, m = stability_term(
        {}, config=cfg, corrected=corrected, base=base, sites=sites
    )
    parts = (
        0.7 * float(m["stab_anch"])
        + 0.4 * float(m["stab_smooth"])
        + 2.0 * float(m["stab_step"])
    )
    assert float(total) == pytest.approx(parts, rel=1e-5)


def test_the_step_term_uses_a_real_trajectory_when_given_one() -> None:
    sites = {k: rand(2, 6, 8, seed=100 + k) for k in (5, 6)}
    corrected = {k: rand(2, 6, 8, seed=k) for k in sites}
    cfg = StabLossConfig(weight=1.0, w_anch=0.0, w_step=1.0)
    single, _ = stability_term({}, config=cfg, corrected=corrected, sites=sites)
    assert float(single) == 0.0
    traj = {
        k: jnp.stack([corrected[k], corrected[k] + 0.1, corrected[k] + 0.2])
        for k in sites
    }
    with_traj, m = stability_term(
        {}, config=cfg, corrected=corrected, trajectories=traj, sites=sites
    )
    assert float(with_traj) > 0.0
    assert float(m["stab_step"]) == float(with_traj)


# --- Smoothness & Step Term Math --------------------------------------------------------------


@pytest.mark.parametrize("backend", ["fabricpc", "jax"])
@pytest.mark.parametrize("use_mask", [False, True])
def test_smoothness_matches_a_numpy_reference(backend: str, use_mask: bool) -> None:
    r, h = rand(2, 7, 8, seed=1), rand(2, 7, 8, seed=2)
    mask = np.array([[1, 1, 1, 1, 0, 0, 0], [1] * 7], bool) if use_mask else None
    res, inp = reference_smoothness(np.asarray(r), np.asarray(h), mask)
    jmask = jnp.asarray(mask) if mask is not None else None
    loss, m = residual_smoothness_loss(
        r, h, mask=jmask, normalize="none", backend=backend
    )
    assert float(loss) == pytest.approx(res, rel=1e-5)
    loss_rel, m_rel = residual_smoothness_loss(
        r, h, mask=jmask, normalize="input", backend=backend
    )
    assert float(loss_rel) == pytest.approx(res / (inp + 1e-6), rel=1e-5)


def test_a_constant_correction_is_perfectly_smooth() -> None:
    r = jnp.broadcast_to(rand(2, 1, 8), (2, 6, 8))
    loss, _ = residual_smoothness_loss(r, rand(2, 6, 8, seed=5))
    assert float(loss) == pytest.approx(0.0, abs=1e-8)


def test_smoothness_is_scale_free_when_relative() -> None:
    r, h = rand(2, 6, 8, seed=6), rand(2, 6, 8, seed=7)
    a = residual_smoothness_loss(r, h)[0]
    b = residual_smoothness_loss(3.0 * r, 3.0 * h)[0]
    assert float(a) == pytest.approx(float(b), rel=1e-4)
    c = residual_smoothness_loss(3.0 * r, h)[0]
    assert float(c) == pytest.approx(9.0 * float(a), rel=1e-4)


def test_the_input_jitter_is_a_constant_for_the_gradient() -> None:
    r, h = rand(2, 6, 8, seed=8), rand(2, 6, 8, seed=9)
    g = jax.grad(lambda x: residual_smoothness_loss(r, x)[0])(h)
    assert float(jnp.max(jnp.abs(g))) == 0.0
    assert (
        float(
            jnp.max(jnp.abs(jax.grad(lambda x: residual_smoothness_loss(x, h)[0])(r)))
        )
        > 0.0
    )


def test_masked_out_tokens_have_no_influence() -> None:
    r, h = rand(2, 6, 8, seed=10), rand(2, 6, 8, seed=11)
    mask = jnp.asarray([[1, 1, 1, 0, 0, 0], [1] * 6], bool)
    g = jax.grad(lambda x: residual_smoothness_loss(x, h, mask=mask)[0])(r)
    assert float(jnp.max(jnp.abs(g[0, 4:]))) == 0.0
    assert float(jnp.max(jnp.abs(g[0, :3]))) > 0.0


def test_a_single_token_has_no_pairs() -> None:
    r = rand(2, 1, 8)
    loss, m = residual_smoothness_loss(r, rand(2, 1, 8, seed=1))
    assert float(loss) == 0.0 and float(m["smooth_ratio"]) == 0.0
    g = jax.grad(lambda x: residual_smoothness_loss(x, r)[0])(r)
    assert float(jnp.max(jnp.abs(g))) == 0.0


def test_smoothness_input_errors() -> None:
    with pytest.raises(ValueError, match="normalize"):
        residual_smoothness_loss(rand(2, 4, 8), rand(2, 4, 8), normalize="mean")
    with pytest.raises(ValueError, match="share a shape"):
        residual_smoothness_loss(rand(2, 4, 8), rand(2, 5, 8))


@pytest.mark.parametrize("backend", ["fabricpc", "jax"])
def test_step_loss_matches_a_numpy_reference(backend: str) -> None:
    z = rand(4, 2, 5, 8, seed=12)
    zn = np.asarray(z)
    raw = (np.sum((zn[1:] - zn[:-1]) ** 2, axis=-1).sum(axis=0)).mean()
    scale = np.mean(np.sum(zn[0] ** 2, axis=-1))
    none, m = step_loss(z, normalize="none", backend=backend)
    assert float(none) == pytest.approx(float(raw), rel=1e-5)
    rel, _ = step_loss(z, normalize="initial", backend=backend)
    assert float(rel) == pytest.approx(float(raw / (scale + 1e-6)), rel=1e-5)


def test_step_loss_is_exactly_zero_without_settling() -> None:
    z = rand(1, 2, 5, 8)
    loss, m = step_loss(z)
    assert float(loss) == 0.0 and float(m["step_sq"]) == 0.0
    g = jax.grad(lambda x: step_loss(x)[0])(z)
    assert float(jnp.max(jnp.abs(g))) == 0.0


def test_step_loss_is_zero_for_a_fixed_point_and_grows_with_the_steps() -> None:
    z0 = rand(1, 2, 5, 8, seed=14)
    fixed = jnp.concatenate([z0, z0, z0])
    assert float(step_loss(fixed)[0]) == 0.0
    moving = jnp.concatenate([z0, z0 + 0.1, z0 + 0.2])
    assert float(step_loss(moving, normalize="none")[0]) > 0.0


def test_step_loss_input_errors() -> None:
    with pytest.raises(ValueError, match="trajectory"):
        step_loss(rand(2, 5, 8))
    with pytest.raises(ValueError, match="normalize"):
        step_loss(rand(2, 2, 5, 8), normalize="x")


# --- Hidden Prediction Term Math --------------------------------------------------------------


def test_hid_term_is_the_mean_of_the_site_losses(substrate, adapter) -> None:
    cfg = make_cfg(hid=0.1, detach=False)
    params = with_random_predictors(init_a2_params(cfg, adapter))
    bank = predictor_bank(cfg, adapter, SEQ)
    rng = np.random.default_rng(3)
    d = substrate.architecture.hidden_size
    adapted = {
        layer: jnp.asarray(rng.normal(size=(BATCH, SEQ, d)), jnp.float32)
        for layer in LAYERS
    }
    targets = {
        LAYERS[0]: adapted[LAYERS[1]],
        LAYERS[1]: jnp.asarray(rng.normal(size=(BATCH, SEQ, d)), jnp.float32),
    }
    total, metrics = hid_term(
        params, bank=bank, config=cfg, corrected=adapted, targets=targets
    )
    per = []
    for layer in LAYERS:
        u = adapted[layer]
        y = targets[layer]
        p = bank.predict(params, layer, u)
        sub = hidden_prediction_loss(p, y, u, cfg.hid)
        per.append(float(sub["loss"]))
        assert f"hid/mse_{layer}" in metrics
    assert float(total) == pytest.approx(float(np.mean(per)), rel=1e-5)


def test_hid_term_needs_every_site(substrate, adapter) -> None:
    cfg = make_cfg(hid=0.1)
    params = init_a2_params(cfg, adapter)
    bank = predictor_bank(cfg, adapter, SEQ)
    d = substrate.architecture.hidden_size
    z = jnp.zeros((BATCH, SEQ, d))
    with pytest.raises(KeyError, match="corrected"):
        hid_term(
            params,
            bank=bank,
            config=cfg,
            corrected={LAYERS[0]: z},
            targets={LAYERS[0]: z, LAYERS[1]: z},
        )
    with pytest.raises(KeyError, match="targets"):
        hid_term(
            params,
            bank=bank,
            config=cfg,
            corrected={LAYERS[0]: z, LAYERS[1]: z},
            targets={LAYERS[0]: z},
        )


def test_hidden_prediction_loss_math() -> None:
    cfg = HiddenConfig(skip_first_token=False, hidden_norm="target_var")
    d = 16
    pred = rand(2, 6, d, seed=20)
    target = pred  # exact match
    source = rand(2, 6, d, seed=21)
    out = hidden_prediction_loss(pred, target, source, cfg)
    assert float(out["loss"]) == pytest.approx(0.0, abs=1e-7)
    assert float(out["mse"]) == pytest.approx(0.0, abs=1e-7)
    assert float(out["copy_mse"]) > 0.0

    # With mask
    mask = jnp.asarray([[True, True, True, False, False, False], [True] * 6])
    corrupted_pred = pred.at[0, 4:].set(999.0)
    out_masked = hidden_prediction_loss(corrupted_pred, target, source, cfg, mask=mask)
    assert float(out_masked["loss"]) == pytest.approx(0.0, abs=1e-7)
