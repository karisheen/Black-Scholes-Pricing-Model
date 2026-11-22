"""Tests for safeguarded Black-Scholes-Merton implied volatility."""

from __future__ import annotations

import math

import pytest

from quant_pricing.black_scholes import OptionType, price, price_bounds
from quant_pricing.exceptions import (
    ImpliedVolatilityError,
    InputValidationError,
    NoArbitrageError,
)
from quant_pricing.implied_volatility import implied_volatility


@pytest.mark.parametrize("option_type", list(OptionType))
@pytest.mark.parametrize(
    ("spot", "strike", "time", "rate", "volatility", "dividend_yield"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.2, 0.0),
        (50.0, 100.0, 0.05, -0.02, 0.7, 0.01),
        (150.0, 80.0, 5.0, 0.08, 0.12, 0.04),
        (100.0, 130.0, 15.0, -0.01, 1.5, 0.03),
    ],
)
def test_recovers_volatility_across_regimes(
    option_type: OptionType,
    spot: float,
    strike: float,
    time: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
) -> None:
    target = price(option_type, spot, strike, time, rate, volatility, dividend_yield)
    assert isinstance(target, float)

    result = implied_volatility(
        option_type,
        target,
        spot,
        strike,
        time,
        rate,
        dividend_yield,
        initial_guess=0.3,
        price_tolerance=1e-11,
        relative_price_tolerance=0.0,
        volatility_tolerance=1e-12,
    )

    assert result.converged
    # Deep out-of-the-money prices have low vega, so price-space convergence
    # cannot imply the same volatility-space precision as near-the-money cases.
    assert result.volatility == pytest.approx(volatility, rel=5e-8, abs=2e-9)
    assert abs(result.residual) <= 1e-9
    assert result.iterations <= 100
    assert result.method == "safeguarded-newton"
    assert math.isfinite(result.normalized_residual)


def test_intrinsic_bound_maps_to_zero_volatility() -> None:
    lower, _ = price_bounds("call", 120.0, 100.0, 2.0, 0.03, 0.01)
    assert isinstance(lower, float)

    result = implied_volatility("call", lower, 120.0, 100.0, 2.0, 0.03, 0.01)

    assert result.volatility == 0.0
    assert result.iterations == 0
    assert result.residual == 0.0
    assert result.bracket == (0.0, 0.0)
    assert result.termination_reason == "lower_bound"


@pytest.mark.parametrize(
    ("option_type", "target"),
    [
        ("call", -0.01),
        ("call", 100.01),
        ("put", -0.01),
        ("put", 100.01 * math.exp(-0.02)),
    ],
)
def test_rejects_prices_outside_no_arbitrage_bounds(
    option_type: str, target: float
) -> None:
    with pytest.raises(NoArbitrageError, match="no-arbitrage bounds"):
        implied_volatility(
            option_type,
            target,
            100.0,
            100.0,
            1.0,
            0.02,
        )


@pytest.mark.parametrize("option_type", list(OptionType))
def test_upper_bound_requires_infinite_volatility(
    option_type: OptionType,
) -> None:
    _, upper = price_bounds(option_type, 100.0, 90.0, 2.0, 0.03, 0.01)
    assert isinstance(upper, float)

    with pytest.raises(ImpliedVolatilityError, match="infinite volatility"):
        implied_volatility(option_type, upper, 100.0, 90.0, 2.0, 0.03, 0.01)


def test_reports_failure_when_maximum_volatility_cannot_bracket_root() -> None:
    target = price("put", 100.0, 100.0, 1.0, 0.02, 3.0)
    assert isinstance(target, float)

    with pytest.raises(ImpliedVolatilityError, match="could not bracket"):
        implied_volatility(
            "put",
            target,
            100.0,
            100.0,
            1.0,
            0.02,
            max_volatility=1.0,
        )


def test_bisection_safeguard_converges_from_poor_guess() -> None:
    target = price("call", 65.0, 100.0, 0.2, -0.01, 0.9, 0.03)
    assert isinstance(target, float)

    result = implied_volatility(
        "call",
        target,
        65.0,
        100.0,
        0.2,
        -0.01,
        0.03,
        initial_guess=5.0,
    )

    assert result.volatility == pytest.approx(0.9, abs=1e-9)
    assert result.bracket[0] <= result.volatility <= result.bracket[1]


def test_iteration_limit_failure_includes_diagnostics() -> None:
    target = price("call", 65.0, 100.0, 0.2, -0.01, 0.9)
    assert isinstance(target, float)

    with pytest.raises(
        ImpliedVolatilityError,
        match=r"did not converge in 1 iterations.*price residual=.*bracket=",
    ):
        implied_volatility(
            "call",
            target,
            65.0,
            100.0,
            0.2,
            -0.01,
            initial_guess=4.0,
            max_iterations=1,
            price_tolerance=1e-15,
            relative_price_tolerance=0.0,
            volatility_tolerance=1e-15,
        )


def test_candidate_evaluated_on_final_iteration_can_converge() -> None:
    result = implied_volatility(
        "call",
        10.450583572185565,
        100.0,
        100.0,
        1.0,
        0.05,
        initial_guess=0.3,
        max_iterations=29,
        price_tolerance=1e-12,
        relative_price_tolerance=0.0,
        volatility_tolerance=1e-15,
    )

    assert result.volatility == pytest.approx(0.2, abs=1e-14)
    assert abs(result.residual) <= 1e-12
    assert result.iterations == 29


def test_volatility_bracket_tolerance_is_a_valid_stopping_criterion() -> None:
    target = price("call", 100.0, 100.0, 1.0, 0.02, 0.4)
    assert isinstance(target, float)

    result = implied_volatility(
        "call",
        target,
        100.0,
        100.0,
        1.0,
        0.02,
        volatility_tolerance=0.25,
        price_tolerance=1e-15,
        relative_price_tolerance=0.0,
    )

    assert result.converged
    assert result.bracket[1] - result.bracket[0] <= 0.25
    assert result.bracket[0] <= result.volatility <= result.bracket[1]


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12, 1e20])
def test_implied_volatility_is_invariant_to_monetary_units(scale: float) -> None:
    # Independently evaluated with the standard-library erf normal CDF for
    # S=K=1, T=1, r=2%, q=0, and sigma=30%.
    normalized_reference_price = 0.12821581392691417

    result = implied_volatility(
        "call",
        normalized_reference_price * scale,
        scale,
        scale,
        1.0,
        0.02,
    )

    assert result.volatility == pytest.approx(0.3, abs=2e-11)
    assert abs(result.normalized_residual) <= 2e-12
    assert result.price_scale == pytest.approx(scale)


def test_external_reference_price_recovers_known_volatility() -> None:
    result = implied_volatility(
        "call",
        10.450583572185565,
        100.0,
        100.0,
        1.0,
        0.05,
        relative_price_tolerance=1e-13,
        volatility_tolerance=1e-13,
    )

    assert result.volatility == pytest.approx(0.2, abs=1e-12)
    assert result.termination_reason == "price_residual"


def test_at_least_one_price_tolerance_must_be_positive() -> None:
    with pytest.raises(InputValidationError, match="at least one"):
        implied_volatility(
            "call",
            10.0,
            100.0,
            100.0,
            1.0,
            0.02,
            price_tolerance=0.0,
            relative_price_tolerance=0.0,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("target_price", "invalid", "target_price must be a real scalar"),
        ("target_price", math.nan, "target_price must be finite"),
        ("time_to_expiry", 0.0, "volatility is undefined at expiry"),
        ("initial_guess", 0.0, "initial_guess must be strictly positive"),
        ("max_volatility", -1.0, "max_volatility must be strictly positive"),
        ("price_tolerance", -1.0, "price_tolerance cannot be negative"),
        (
            "relative_price_tolerance",
            -1.0,
            "relative_price_tolerance cannot be negative",
        ),
        (
            "volatility_tolerance",
            math.inf,
            "volatility_tolerance must be finite",
        ),
        (
            "volatility_tolerance",
            -1.0,
            "volatility_tolerance must be strictly positive",
        ),
        ("max_iterations", True, "max_iterations must be an integer"),
        ("max_iterations", 0, "max_iterations must be strictly positive"),
    ],
)
def test_rejects_invalid_solver_inputs(
    keyword: str, value: object, message: str
) -> None:
    arguments: dict[str, object] = {
        "option_type": "call",
        "target_price": 10.0,
        "spot": 100.0,
        "strike": 100.0,
        "time_to_expiry": 1.0,
        "rate": 0.02,
    }
    arguments[keyword] = value

    with pytest.raises(InputValidationError, match=message):
        implied_volatility(**arguments)  # type: ignore[arg-type]
