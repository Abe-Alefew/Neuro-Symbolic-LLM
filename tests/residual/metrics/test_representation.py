"""Unit tests for residual compressibility. No model needed."""

from __future__ import annotations

import pytest

from metrics.representation import (
    compressibility_curve,
    loss_improvement,
    residual_compressibility,
)


def test_loss_improvement_sign() -> None:
    assert float(loss_improvement(3.0, 2.5)) == pytest.approx(0.5)
    assert float(loss_improvement(3.0, 3.2)) == pytest.approx(-0.2)


def test_residual_compressibility() -> None:
    assert float(residual_compressibility(0.4, 0.5)) == pytest.approx(0.8)
    assert float(residual_compressibility(0.0, 0.0, eps=1e-3)) == 0.0


def test_negative_eps_raises() -> None:
    with pytest.raises(ValueError, match="eps"):
        residual_compressibility(0.1, 0.2, eps=-1.0)


def test_compressibility_curve_is_relative_to_largest_rank() -> None:
    curve = compressibility_curve({64: 0.50, 1: 0.10, 16: 0.45}, eps=0.0)
    assert list(curve) == [1, 16, 64]
    assert float(curve[1]) == pytest.approx(0.2)
    assert float(curve[16]) == pytest.approx(0.9)
    assert float(curve[64]) == pytest.approx(1.0)


@pytest.mark.parametrize("deltas", [{}, {0: 0.1, 4: 0.2}])
def test_compressibility_curve_rejects_bad_input(deltas: dict[int, float]) -> None:
    with pytest.raises(ValueError):
        compressibility_curve(deltas)


def test_guarded_compressibility_ok() -> None:
    from metrics.representation import guarded_compressibility

    out = guarded_compressibility({4: 0.2, 16: 0.4})
    assert out["status"] == "ok"
    assert out["reference_rank"] == 16
    assert out["curve"][4] == pytest.approx(0.5)
    assert out["curve"][16] == pytest.approx(1.0)


@pytest.mark.parametrize("ref", [0.0, -0.3, 0.001])
def test_guarded_compressibility_rejects_nonpositive_or_noise_reference(ref) -> None:
    """A3 worse than base at the largest rank must not yield a huge/sign-flipped ratio."""
    from metrics.representation import guarded_compressibility

    out = guarded_compressibility({4: 0.2, 16: ref})
    assert out["status"] == "unreliable_reference"
    assert out["curve"] is None
    assert out["reference_improvement"] == ref


def test_guarded_compressibility_needs_two_ranks_and_flags_negative_ranks() -> None:
    from metrics.representation import guarded_compressibility

    assert guarded_compressibility({16: 0.4})["status"] == "insufficient_ranks"
    out = guarded_compressibility({4: -0.1, 16: 0.4})
    assert out["negative_ranks"] == [4]
    assert out["curve"][4] == pytest.approx(-0.25)
    with pytest.raises(ValueError):
        guarded_compressibility({4: 0.1, 8: 0.2}, min_reference=0.0)
    assert guarded_compressibility({4: float("nan"), 16: 0.4})["status"] == "insufficient_ranks"
