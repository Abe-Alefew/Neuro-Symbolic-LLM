"""Consolidated configuration tests for Stage A2.

Covers CPCConfig, HEPConfig, StabLossConfig, HiddenConfig, PredictorConfig,
MonitorConfig, and the unified A2Config, including defaults, validation,
serialization, flags, and all ablation arms.
"""

from __future__ import annotations

from typing import Any

import pytest

from stages.stage_A.a2 import (
    ARMS,
    A2Config,
    CPCConfig,
    HEPConfig,
    HiddenConfig,
    MonitorConfig,
    StabLossConfig,
    apply_arm,
)
from stages.stage_base import load_configs
from tests.residual.stage_A.conftest import FF, SHIPPED, PredictorConfig

# --- Defaults & Flags -------------------------------------------------------------------------


def test_a2_defaults_are_the_a1_baseline() -> None:
    cfg = A2Config()
    assert (cfg.lambda_kl, cfg.lambda_wd, cfg.seed) == (0.0, 0.0, 0)
    assert cfg.cpc == CPCConfig()
    assert cfg.stab == StabLossConfig()
    assert cfg.hid == HiddenConfig()
    assert cfg.monitor == MonitorConfig()
    assert not cfg.aux_enabled
    assert not cfg.needs_base_states
    assert not cfg.needs_base_targets
    assert not cfg.needs_corrected
    assert not cfg.uses_hidden_pred


@pytest.mark.parametrize(
    "config,aux,base_states,base_targets,corrected,uses_hid",
    [
        (A2Config(), False, False, False, False, False),
        (
            A2Config(cpc=CPCConfig(weight=0.1, positive="base")),
            True,
            True,
            False,
            True,
            False,
        ),
        (
            A2Config(cpc=CPCConfig(weight=0.1, positive="corrected")),
            True,
            False,
            False,
            True,
            False,
        ),
        (
            A2Config(stab=StabLossConfig(weight=0.1, w_anch=1.0)),
            True,
            True,
            False,
            True,
            False,
        ),
        (
            A2Config(stab=StabLossConfig(weight=0.1, w_anch=0.0, w_smooth=1.0)),
            True,
            False,
            False,
            True,
            False,
        ),
        (
            A2Config(hid=HiddenConfig(weight=0.1, hidden_target="corrected_teacher")),
            True,
            False,
            False,
            True,
            True,
        ),
        (
            A2Config(hid=HiddenConfig(weight=0.1, hidden_target="base_anchor")),
            True,
            True,
            True,
            True,
            True,
        ),
        (
            A2Config(monitor=MonitorConfig(mode="diagnostic")),
            False,
            False,
            False,
            True,
            False,
        ),
    ],
)
def test_a2_objective_flags(
    config: A2Config,
    aux: bool,
    base_states: bool,
    base_targets: bool,
    corrected: bool,
    uses_hid: bool,
) -> None:
    assert config.aux_enabled == aux
    assert config.needs_base_states == base_states
    assert config.needs_base_targets == base_targets
    assert config.needs_corrected == corrected
    assert config.uses_hidden_pred == uses_hid


# --- Section Defaults & Validations -----------------------------------------------------------


def test_cpc_defaults_and_validation() -> None:
    c = CPCConfig()
    assert not c.enabled and c.weight == 0.0
    assert (c.offsets, c.dim, c.tau, c.positive, c.heads) == (
        (1,),
        32,
        0.1,
        "base",
        "fabricpc",
    )

    # Invalid cases
    with pytest.raises(ValueError, match="weight"):
        CPCConfig(weight=-0.1)
    with pytest.raises(ValueError, match="offsets"):
        CPCConfig(offsets=(0,))
    with pytest.raises(ValueError, match="offsets"):
        CPCConfig(offsets=(1, 1))
    with pytest.raises(ValueError, match="dim"):
        CPCConfig(dim=0)
    with pytest.raises(ValueError, match="tau"):
        CPCConfig(tau=0.0)
    with pytest.raises(ValueError, match="positive"):
        CPCConfig(positive="unknown")
    with pytest.raises(ValueError, match="heads"):
        CPCConfig(heads="invalid")


def test_stability_and_hep_validation() -> None:
    s = StabLossConfig()
    assert not s.enabled and s.weight == 0.0
    assert (s.w_anch, s.w_hep, s.w_smooth, s.w_step) == (1.0, 0.0, 0.0, 0.0)
    assert (s.normalize, s.backend) == ("relative", "fabricpc")
    assert s.hep == HEPConfig()

    with pytest.raises(ValueError, match="weight"):
        StabLossConfig(weight=-0.1)
    with pytest.raises(ValueError, match="normalize"):
        StabLossConfig(normalize="zscore")
    with pytest.raises(ValueError, match="backend"):
        StabLossConfig(backend="torch")
    with pytest.raises(ValueError, match="w_hep > 0 needs hep.kind"):
        StabLossConfig(weight=0.1, w_hep=1.0, hep=HEPConfig(kind=None))
    with pytest.raises(ValueError, match="w_hep > 0 with hep.kind='fixed_random'"):
        StabLossConfig(weight=0.1, w_hep=1.0, hep=HEPConfig(kind="fixed_random"))

    h = HEPConfig()
    assert h.kind is None and h.rank == 16 and h.alpha == 1.0 and not h.trainable
    assert HEPConfig(kind="dense").trainable
    assert HEPConfig(kind="lowrank").trainable
    assert not HEPConfig(kind="fixed_random").trainable

    with pytest.raises(ValueError, match="kind"):
        HEPConfig(kind="convolutional")
    with pytest.raises(ValueError, match="rank"):
        HEPConfig(rank=0)
    with pytest.raises(ValueError, match="alpha"):
        HEPConfig(alpha=-0.5)


def test_hidden_and_predictor_validation() -> None:
    h = HiddenConfig()
    assert not h.enabled and h.weight == 0.0
    assert (
        h.hidden_target,
        h.hidden_norm,
        h.skip_first_token,
        h.predictor_lr_mult,
    ) == (
        "corrected_teacher",
        "target_var",
        True,
        1.0,
    )
    assert h.predictor == PredictorConfig()

    with pytest.raises(ValueError, match="weight"):
        HiddenConfig(weight=-1.0)
    with pytest.raises(ValueError, match="hidden_target"):
        HiddenConfig(hidden_target="latent")
    with pytest.raises(ValueError, match="hidden_norm"):
        HiddenConfig(hidden_norm="l2")
    with pytest.raises(ValueError, match="predictor_lr_mult"):
        HiddenConfig(predictor_lr_mult=0.0)
    with pytest.raises(ValueError, match="skip_first_token"):
        HiddenConfig(skip_first_token="yes")  # type: ignore[arg-type]

    p = PredictorConfig(ff_dim=FF)
    assert p.ff_dim == FF and p.detach_input is True and p.predict_delta is False
    with pytest.raises(ValueError, match="ff_dim"):
        PredictorConfig(ff_dim=0)
    with pytest.raises(ValueError, match="out_init_std"):
        PredictorConfig(out_init_std=-0.1)


# --- Serialization & Dictionaries -------------------------------------------------------------


def test_from_dict_and_round_trip() -> None:
    data = {
        "lambda_kl": 0.1,
        "lambda_wd": 0.01,
        "seed": 42,
        "cpc": {"weight": 0.2, "offsets": [1, 2], "dim": 16},
        "stab": {"weight": 0.3, "w_smooth": 0.5, "hep": {"kind": "lowrank", "rank": 8}},
        "hid": {"weight": 0.4, "predictor": {"ff_dim": 32, "detach_input": False}},
        "monitor": {"mode": "diagnostic"},
    }
    cfg = A2Config.from_dict(data)
    assert cfg.lambda_kl == 0.1 and cfg.lambda_wd == 0.01 and cfg.seed == 42
    assert cfg.cpc.weight == 0.2 and cfg.cpc.offsets == (1, 2) and cfg.cpc.dim == 16
    assert (
        cfg.stab.weight == 0.3 and cfg.stab.w_smooth == 0.5 and cfg.stab.hep.rank == 8
    )
    assert (
        cfg.hid.weight == 0.4
        and cfg.hid.predictor.ff_dim == 32
        and not cfg.hid.predictor.detach_input
    )
    assert cfg.monitor.mode == "diagnostic"

    # Single offset alias in cpc
    cpc_alias = CPCConfig.from_dict({"weight": 0.1, "offset": 4})
    assert cpc_alias.offsets == (4,)
    with pytest.raises(ValueError, match="give either 'offset' or 'offsets'"):
        CPCConfig.from_dict({"offset": 1, "offsets": [1, 2]})

    # Legacy flat hid aliases in A2Config
    legacy = A2Config.from_dict({"lambda_hid": 0.5, "predictor_lr_mult": 2.0})
    assert legacy.hid.weight == 0.5 and legacy.hid.predictor_lr_mult == 2.0


@pytest.mark.parametrize(
    "bad_dict",
    [
        {"unknown_root_key": 1},
        {"cpc": {"non_existent": 1}},
        {"stab": {"wrong": True}},
        {"hid": {"bogus": "value"}},
        {"monitor": {"bad_mode": "extra"}},
    ],
)
def test_unknown_keys_rejected(bad_dict: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        A2Config.from_dict(bad_dict)


# --- Ablation Arms ----------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ARMS)
def test_all_ablation_arms_are_valid_and_switch_correct_terms(arm: str) -> None:
    base = A2Config()
    applied = apply_arm(base, arm)

    # Defaults apply when weights are 0 in base
    if arm == "a1_mode":
        assert not applied.aux_enabled
    elif arm == "cpc_base":
        assert applied.cpc.enabled and applied.cpc.positive == "base"
        assert not applied.stab.enabled and not applied.hid.enabled
    elif arm == "cpc_corrected":
        assert applied.cpc.enabled and applied.cpc.positive == "corrected"
        assert not applied.stab.enabled and not applied.hid.enabled
    elif arm == "cpc_multi":
        assert applied.cpc.enabled and applied.cpc.offsets == (1, 2, 4)
    elif arm == "stab_anch":
        assert (
            applied.stab.enabled and applied.stab.w_anch > 0 and applied.stab.w_hep == 0
        )
    elif arm == "stab_hep":
        assert (
            applied.stab.enabled
            and applied.stab.w_hep > 0
            and applied.stab.hep.trainable
        )
    elif arm == "stab_smooth":
        assert applied.stab.enabled and applied.stab.w_smooth > 0
    elif arm == "stab_all":
        assert (
            applied.stab.enabled
            and applied.stab.w_hep > 0
            and applied.stab.w_smooth > 0
            and applied.stab.w_step > 0
        )
    elif arm == "hid_detached":
        assert applied.hid.enabled and applied.hid.predictor.detach_input is True
    elif arm == "hid_coupled":
        assert applied.hid.enabled and applied.hid.predictor.detach_input is False
    elif arm == "full":
        assert applied.cpc.enabled and applied.stab.enabled and applied.hid.enabled


def test_invalid_arm_raises_error() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        apply_arm(A2Config(), "non_existent_arm")


def test_shipped_config_loads_cleanly() -> None:
    cfg = load_configs(SHIPPED, A2Config)
    assert cfg.training.model == "gpt2"
    assert cfg.objective.cpc.weight > 0 or cfg.objective.stab.weight > 0
