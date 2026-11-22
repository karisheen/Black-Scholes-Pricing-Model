"""Forward pricing identities, limiting cases and PV-unit IV diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import ArrayLike

from quant_pricing.black76 import implied_volatility, price, price_bounds
from quant_pricing.black_scholes import price as spot_price
from quant_pricing.exceptions import (
    ImpliedVolatilityError,
    InputValidationError,
    NoArbitrageError,
)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("rate,yield_", [(0.05, 0.02), (-0.03, 0.08)])
def test_forward_prices_match_spot_bsm(kind: str, rate: float, yield_: float) -> None:
    t = np.array([0.01, 0.5, 3.0])
    s = np.array([[80.0], [100.0], [150.0]])
    f = s * np.exp((rate - yield_) * t)
    actual = price(kind, f, 100.0, t, np.exp(-rate * t), 0.27)
    expected = spot_price(kind, s, 100.0, t, rate, 0.27, yield_)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-13)


def test_atm_matches_independent_normal_integral_and_parity() -> None:
    f, d, t, sigma = 105.0, 0.96, 0.8, 0.25
    expected = d * f * math.erf(sigma * math.sqrt(t) / (2.0 * math.sqrt(2.0)))
    assert price("call", f, f, t, d, sigma) == pytest.approx(expected, rel=1e-13)
    for k in (70.0, 105.0, 140.0):
        call = price("call", f, k, t, d, sigma)
        put = price("put", f, k, t, d, sigma)
        assert call - put == pytest.approx(d * (f - k), abs=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_zero_time_and_volatility_limits_are_continuous(kind: str) -> None:
    forwards = np.array([90.0, 100.0, 110.0])
    intrinsic = (
        np.maximum(forwards - 100.0 if kind == "call" else 100.0 - forwards, 0.0) * 0.93
    )
    np.testing.assert_allclose(price(kind, forwards, 100, 0, 0.93, 0.2), intrinsic)
    np.testing.assert_allclose(price(kind, forwards, 100, 1, 0.93, 0), intrinsic)
    np.testing.assert_allclose(
        price(kind, forwards, 100, 1e-24, 0.93, 0.2), intrinsic, atol=1e-11
    )
    np.testing.assert_allclose(
        price(kind, forwards, 100, 1, 0.93, 1e-12), intrinsic, atol=1e-10
    )
    lower, upper = price_bounds(kind, forwards, 100, 1, 0.93)
    np.testing.assert_allclose(lower, intrinsic)
    np.testing.assert_allclose(upper, 0.93 * (forwards if kind == "call" else 100))


def test_direct_discounting_avoids_rate_mapping_overflow() -> None:
    # -log(D)/T overflows; direct D is well defined and total vol is 0.2.
    value = price("call", 100, 100, 1e-310, 0.5, 2e154)
    expected = 50 * math.erf(0.2 / (2 * math.sqrt(2)))
    assert value == pytest.approx(expected, rel=1e-12)
    # Neither an undiscounted model price nor F/K may overflow first.
    assert price("call", 1e300, 1e-300, 1, 1e-300, 0.2) == pytest.approx(1.0)
    assert price("put", 1e-300, 1e300, 1, 1e-300, 0.2) == pytest.approx(1.0)
    assert price("call", 100, 100, 1e308, 0.9, 1e308) == pytest.approx(90.0)


@pytest.mark.parametrize(
    "arguments",
    [
        (0, 100, 1, 1, 0.2),
        (100, -1, 1, 1, 0.2),
        (100, 100, -1, 1, 0.2),
        (100, 100, 1, 0, 0.2),
        (100, 100, 1, 1, -0.2),
        (float("nan"), 100, 1, 1, 0.2),
        (100, 100, 1, float("inf"), 0.2),
        (1e308, 100, 1, 1e308, 0.2),
        ([100, 110], [90, 100, 110], 1, 1, 0.2),
    ],
)
def test_reject_invalid_inputs(arguments: tuple[ArrayLike, ...]) -> None:
    with pytest.raises(InputValidationError):
        price("call", *arguments)


def test_reject_invalid_option_type() -> None:
    with pytest.raises(InputValidationError, match="option_type"):
        price("straddle", 100, 100, 1, 1, 0.2)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("d", [1e-100, 0.7, 1.1, 1e100])
def test_iv_roundtrip_and_all_diagnostics_remain_pv(kind: str, d: float) -> None:
    target = float(price(kind, 103, 100, 0.7, d, 0.37))
    result = implied_volatility(kind, target, 103, 100, 0.7, d)
    assert result.volatility == pytest.approx(0.37, abs=1e-9)
    assert result.model_price == float(price(kind, 103, 100, 0.7, d, result.volatility))
    assert result.target_price == target
    assert result.residual == result.model_price - target
    assert result.price_scale == pytest.approx(d * (103 if kind == "call" else 100))
    assert result.normalized_residual == pytest.approx(
        result.residual / result.price_scale
    )
    assert result.bracket[0] <= result.volatility <= result.bracket[1]
    assert result.converged and result.method == "bisection"


@pytest.mark.parametrize("kind", ["call", "put"])
def test_iv_bounds_and_infinite_volatility(kind: str) -> None:
    lower, upper = price_bounds(kind, 110, 100, 1, 0.9)
    result = implied_volatility(kind, float(lower), 110, 100, 1, 0.9)
    assert result.volatility == 0
    assert result.termination_reason == "lower_bound"
    assert result.iterations == 0
    with pytest.raises(NoArbitrageError, match="outside bounds"):
        implied_volatility(kind, float(lower) - 1, 110, 100, 1, 0.9)
    with pytest.raises(NoArbitrageError, match="outside bounds"):
        implied_volatility(kind, float(upper) + 1, 110, 100, 1, 0.9)
    with pytest.raises(ImpliedVolatilityError, match="infinite"):
        implied_volatility(kind, float(upper), 110, 100, 1, 0.9)


def test_iv_bracketing_expansion_and_explicit_failures() -> None:
    target = float(price("call", 100, 100, 1, 0.95, 2.1))
    result = implied_volatility("call", target, 100, 100, 1, 0.95)
    assert result.volatility == pytest.approx(2.1)
    with pytest.raises(ImpliedVolatilityError, match="bracket"):
        implied_volatility("call", target, 100, 100, 1, 0.95, max_volatility=0.1)
    with pytest.raises(ImpliedVolatilityError, match="did not converge"):
        implied_volatility("call", target, 100, 100, 1, 0.95, max_iterations=1)
    result = implied_volatility(
        "call", target, 100, 100, 1, 0.95, volatility_tolerance=1
    )
    assert result.termination_reason == "volatility_bracket"
    exact_target = float(price("call", 100, 100, 1, 0.95, 0.2))
    exact = implied_volatility("call", exact_target, 100, 100, 1, 0.95)
    assert exact.volatility == 0.2
    assert exact.termination_reason == "price_residual"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_volatility": 0},
        {"relative_price_tolerance": 0},
        {"volatility_tolerance": -1},
        {"max_iterations": 0},
        {"max_iterations": True},
        {"max_iterations": 1.5},
        {"max_volatility": float("inf")},
    ],
)
def test_iv_invalid_solver_controls(kwargs: dict[str, float]) -> None:
    with pytest.raises(InputValidationError):
        implied_volatility("call", 5, 100, 100, 1, 0.95, **kwargs)  # type: ignore[arg-type]


def test_iv_rejects_expiry_and_nonscalar_target() -> None:
    with pytest.raises(InputValidationError, match="time_to_expiry"):
        implied_volatility("call", 0, 100, 100, 0, 1)
    with pytest.raises(InputValidationError, match="real scalar"):
        implied_volatility("call", [5, 6], 100, 100, 1, 1)  # type: ignore[arg-type]
