"""Stage A2 objective on the 12-layer GPT-2 fixture: targets, gradient routing, training."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from frozenllm.substrate import FrozenSubstrate, identity_modify
from residual import ResidualAdapter, ResidualConfig, layer_key
from residual.stability import StabilityConfig, StabilityMonitor
from residual.local_predictor import (
    LocalPredictorBank,
    PredictorConfig,
    merge_params,
    split_params,
)
from stages.results import RunWriter, load_params
from stages.stage_A.a2 import (
    A2Config,
    a2_diagnostics,
    make_a2_objective,
    make_a2_optimizer,
)
from stages.stage_base import Domain, TrainingConfig, load_configs, run_sequential
from tests.frozenllm.conftest import BATCH, NUM_TOKENS, SEQ, make_substrate

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def substrate() -> FrozenSubstrate:
    sub: FrozenSubstrate = make_substrate("gpt2")[1]
    return sub


@pytest.fixture(scope="module")
def adapter(substrate: FrozenSubstrate) -> ResidualAdapter:
    return ResidualAdapter(ResidualConfig(rank=4), substrate.architecture)


@pytest.fixture(scope="module")
def ids() -> jax.Array:
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.integers(0, NUM_TOKENS, (BATCH, SEQ)), dtype=jnp.int32)


def _random_b(params, scale=0.05):
    rng = np.random.default_rng(1)
    return {
        k: {"A": v["A"], "B": jnp.asarray(scale * rng.normal(size=v["B"].shape), jnp.float32)}
        for k, v in params.items()
    }


def _setup(substrate, adapter, **obj_kw):
    pred_kw = obj_kw.pop("predictor", {})
    stab_kw = obj_kw.pop("stability", {})
    cfg = A2Config(
        predictor=PredictorConfig(ff_dim=16, **pred_kw),
        stability=StabilityConfig(**stab_kw),
        **obj_kw,
    )
    bank = LocalPredictorBank(cfg.predictor, substrate.architecture, SEQ, adapter.layers)
    mon = StabilityMonitor(cfg.stability, adapter.layers)  
    params = merge_params(_random_b(adapter.init_params()), bank.init_params())
    return cfg, bank, mon, params


def _grad(substrate, adapter, cfg, bank, mon, params, ids):
    fn = make_a2_objective(bank, mon)
    return jax.grad(
        lambda p: fn(p, substrate=substrate, adapter=adapter, config=cfg, input_ids=ids)[0]
    )(params)


def _abs_sum(tree):
    return float(sum(jnp.abs(x).sum() for x in jax.tree_util.tree_leaves(tree)))


# ── hidden-state capture / target semantics ──────────────────────────────────


def test_teacher_target_is_next_block_before_its_own_correction(substrate, adapter, ids) -> None:
    params = _random_b(adapter.init_params())
    layer = adapter.layers[0]
    sites = (layer, layer + 1)
    run = lambda p: substrate.run_with_interception(  # noqa: E731
        ids, modify_fn=adapter.modify_fn(p), intercept_layers=sites
    )
    base = substrate.run_with_interception(ids, modify_fn=identity_modify, intercept_layers=sites)
    res = run(params)
    # Shapes the predictor sees: [B, S, D] for input and target.
    d = substrate.architecture.hidden_size
    assert res.hidden_state(layer).shape == res.hidden_state(layer + 1).shape == (BATCH, SEQ, d)
    # The correction at `layer` has reached the next block's output ...
    assert not jnp.allclose(res.hidden_state(layer + 1), base.hidden_state(layer + 1), atol=1e-6)
    # ... but the next layer's own residual has not (it is recorded before modify_fn).
    bumped = {**params, layer_key(layer + 1): {**params[layer_key(layer + 1)],
              "B": params[layer_key(layer + 1)]["B"] + 1.0}}
    assert jnp.allclose(run(bumped).hidden_state(layer + 1), res.hidden_state(layer + 1), atol=1e-6)
    # Zero residual: adapted == base trajectory, so Options A and B coincide at init.
    zero = run(adapter.init_params())
    assert jnp.allclose(zero.hidden_state(layer + 1), base.hidden_state(layer + 1), atol=1e-5)


# ── gradient routing ─────────────────────────────────────────────────────────


def test_detached_input_keeps_hidden_loss_out_of_the_adapter(substrate, adapter, ids) -> None:
    cfg, bank, mon, params = _setup(substrate, adapter, lambda_hid=1.0)
    g_det = _grad(substrate, adapter, cfg, bank, mon, params, ids)
    g_off = _grad(substrate, adapter, replace(cfg, lambda_hid=0.0), None, mon, params, ids)
    a_det, p_det = split_params(g_det)
    a_off, _ = split_params(g_off)
    for x, y in zip(jax.tree_util.tree_leaves(a_det), jax.tree_util.tree_leaves(a_off), strict=True):
        assert jnp.allclose(x, y, atol=1e-6)
    assert _abs_sum(p_det) > 0


def test_attached_input_sends_hidden_loss_into_the_adapter(substrate, adapter, ids) -> None:
    cfg, bank, mon, params = _setup(
        substrate, adapter, lambda_hid=1.0, predictor={"detach_input": False}
    )
    g = _grad(substrate, adapter, cfg, bank, mon, params, ids)
    g0 = _grad(substrate, adapter, replace(cfg, lambda_hid=0.0), None, mon, params, ids)
    diff = jax.tree_util.tree_map(lambda a, b: jnp.abs(a - b).sum(), split_params(g)[0], split_params(g0)[0])
    assert _abs_sum(diff) > 1e-6


def test_zero_lambda_hid_gives_no_predictor_gradient(substrate, adapter, ids) -> None:
    cfg, bank, mon, params = _setup(substrate, adapter, lambda_hid=0.0)
    fn = make_a2_objective(bank, mon)
    _, terms = fn(params, substrate=substrate, adapter=adapter, config=cfg, input_ids=ids)
    assert "hid" not in terms
    g = _grad(substrate, adapter, cfg, bank, mon, params, ids)
    assert _abs_sum(split_params(g)[1]) == 0.0


# ── targets and terms ────────────────────────────────────────────────────────


def test_base_anchor_logs_kl_and_matches_teacher_at_init(substrate, adapter, ids) -> None:
    fn_terms = {}
    for target in ("corrected_teacher", "base_anchor"):
        cfg, bank, mon, params = _setup(substrate, adapter, lambda_hid=0.5, hidden_target=target)
        params = merge_params(adapter.init_params(), split_params(params)[1])  # B = 0
        _, terms = make_a2_objective(bank, mon)(
            params, substrate=substrate, adapter=adapter, config=cfg, input_ids=ids
        )
        fn_terms[target] = terms
    assert "kl" in fn_terms["base_anchor"] and "kl" not in fn_terms["corrected_teacher"]
    assert float(fn_terms["base_anchor"]["kl"]) < 1e-6
    assert float(fn_terms["base_anchor"]["hid"]) == pytest.approx(
        float(fn_terms["corrected_teacher"]["hid"]), rel=1e-4
    )


def test_stability_never_changes_the_loss_or_gradients(substrate, adapter, ids) -> None:
    base_cfg, bank, _, params = _setup(substrate, adapter, lambda_hid=0.0)
    off = StabilityMonitor(StabilityConfig(), adapter.layers)
    on = StabilityMonitor(StabilityConfig(mode="diagnostic"), adapter.layers)
    g_off = _grad(substrate, adapter, base_cfg, None, off, params, ids)
    g_on = _grad(substrate, adapter, base_cfg, None, on, params, ids)
    for x, y in zip(jax.tree_util.tree_leaves(g_off), jax.tree_util.tree_leaves(g_on), strict=True):
        assert jnp.allclose(x, y, atol=1e-7)
    loss = lambda m: make_a2_objective(None, m)(  # noqa: E731
        params, substrate=substrate, adapter=adapter, config=base_cfg, input_ids=ids)[0]
    assert float(loss(off)) == float(loss(on))


def test_stability_metrics_are_logged_and_steps_are_zero(substrate, adapter, ids) -> None:
    cfg, bank, mon, params = _setup(
        substrate, adapter, lambda_hid=0.1, stability={"mode": "diagnostic"}
    )
    _, terms = make_a2_objective(bank, mon)(
        params, substrate=substrate, adapter=adapter, config=cfg, input_ids=ids
    )
    for key in ("total", "task", "wd", "hid", "res_smooth", "input_smooth", "res_vs_input",
                "step_loss", "step_noncontract", "step_last_norm"):
        assert key in terms and bool(jnp.isfinite(terms[key]))
    assert float(terms["res_smooth"]) > 0  # random B => the correction varies along the sequence
    assert float(terms["step_loss"]) == float(terms["step_noncontract"]) == 0.0  # no settling
    off_cfg, off_bank, off_mon, off_params = _setup(substrate, adapter, lambda_hid=0.1)
    _, off_terms = make_a2_objective(off_bank, off_mon)(
        off_params, substrate=substrate, adapter=adapter, config=off_cfg, input_ids=ids
    )
    assert "res_smooth" not in off_terms and "step_loss" not in off_terms


def test_objective_is_jittable(substrate, adapter, ids) -> None:
    cfg, bank, mon, params = _setup(substrate, adapter, lambda_hid=0.1, stability={"mode": "diagnostic"})
    fn = make_a2_objective(bank, mon)
    step = jax.jit(jax.value_and_grad(
        lambda p, x: fn(p, substrate=substrate, adapter=adapter, config=cfg, input_ids=x), has_aux=True))
    (loss, _), grads = step(params, ids)
    assert bool(jnp.isfinite(loss))
    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(params)


# ── end-to-end training loop ─────────────────────────────────────────────────


def test_training_moves_predictor_and_adapter_not_the_base(substrate, adapter, tmp_path) -> None:
    cfg, bank, mon, params = _setup(
        substrate, adapter, lambda_hid=1.0, predictor_lr_mult=3.0,
        predictor={"detach_input": False}, stability={"mode": "diagnostic"},
    )
    params = merge_params(adapter.init_params(), split_params(params)[1])
    rng = np.random.default_rng(3)
    blocks = rng.integers(0, NUM_TOKENS, (8, SEQ)).astype(np.int32)
    domain = Domain(name="toy", train_blocks=blocks,
                    eval_set=(jnp.asarray(blocks[:BATCH]),), epochs=2)
    training = TrainingConfig(seq_len=SEQ, batch_size=BATCH, epochs=2, learning_rate=1e-2,
                              log_every=1, output_dir=str(tmp_path))
    before = jax.tree_util.tree_map(np.asarray, params)
    with RunWriter(tmp_path, "run") as writer:
        new, summary = run_sequential(
            substrate, adapter, params, [domain], make_a2_objective(bank, mon), cfg,
            training, writer, optimizer=make_a2_optimizer(cfg, training.learning_rate),
        )
        ckpt = writer.dir / "checkpoints" / "task_0_toy.npz"
    a_new, p_new = split_params(new)
    a_old, p_old = split_params(before)
    moved = lambda x, y: _abs_sum(jax.tree_util.tree_map(lambda a, b: jnp.abs(a - b).sum(), x, y))  # noqa: E731
    assert moved(a_new, a_old) > 0 and moved(p_new, p_old) > 0
    assert substrate.verify_frozen()["params_unchanged"]
    # Predictor weights survive the checkpoint round trip next to the adapter's.
    loaded = load_params(ckpt)
    assert set(loaded) == set(new)
    for k in new:
        for kk in new[k]:
            assert np.allclose(loaded[k][kk], np.asarray(new[k][kk]))
    assert summary["steps"] == 8  # 4 batches x 2 epochs

    diag = a2_diagnostics(substrate, adapter, bank, cfg, new, domain.eval_set)
    for i in bank.layers:
        assert {f"pred_rel_l{i}", f"copy_rel_l{i}", f"pred_rel_base_l{i}", f"teacher_shift_l{i}"} <= set(diag)


def test_yaml_loads_into_a2_config() -> None:
    cfg = load_configs(REPO_ROOT / "configs" / "stage_A" / "a2_pc_inspired.yaml", A2Config)
    obj = cfg.objective
    assert obj.hidden_target == "corrected_teacher" and obj.predictor.detach_input is True
    assert obj.stability.mode == "off" and obj.predictor.ff_dim == 256
    assert cfg.to_dict()["objective"]["predictor"]["ff_dim"] == 256


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="hidden_target"):
        A2Config(hidden_target="nope")
    with pytest.raises(ValueError, match="Unknown A2Config"):
        A2Config.from_dict({"bogus": 1})
    with pytest.raises(ValueError, match="Unknown StabilityConfig"):
        A2Config.from_dict({"stability": {"bogus": 1}})
