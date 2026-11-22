"""Numerical and invariant tests for Black-Scholes-Merton analytics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_pricing.black_scholes import (
    Greeks,
    OptionType,
    d1_d2,
    greeks,
    price,
    price_bounds,
    value_and_greeks,
)
from quant_pricing.exceptions import InputValidationError

SPOT = 100.0
STRIKE = 100.0
TIME = 1.0
RATE = 0.05
VOLATILITY = 0.2


@pytest.mark.parametrize(
    (
        "option_type",
        "expected_price",
        "expected_delta",
        "expected_theta",
        "expected_rho",
    ),
    [
        (
            "call",
            10.450583572185565,
            0.6368306511756191,
            -6.414027546438197,
            53.232481545376345,
        ),
        (
            "put",
            5.573526022256971,
            -0.3631693488243809,
            -1.657880423934626,
            -41.89046090469506,
        ),
    ],
)
def test_hull_reference_values(
    option_type: str,
    expected_price: float,
    expected_delta: float,
    expected_theta: float,
    expected_rho: float,
) -> None:
    """Regression values for the canonical S=K=100 one-year example."""

    result = value_and_greeks(option_type, SPOT, STRIKE, TIME, RATE, VOLATILITY)

    assert result.price == pytest.approx(expected_price, abs=1e-12)
    assert result.greeks.delta == pytest.approx(expected_delta, abs=1e-12)
    assert result.greeks.gamma == pytest.approx(0.018762017345846895, abs=1e-14)
    assert result.greeks.vega == pytest.approx(37.52403469169379, abs=1e-12)
    assert result.greeks.theta == pytest.approx(expected_theta, abs=1e-12)
    assert result.greeks.rho == pytest.approx(expected_rho, abs=1e-12)


def test_put_call_parity_with_dividend_yield() -> None:
    spot = np.array([60.0, 100.0, 140.0])
    strike = 105.0
    time = np.array([0.05, 1.0, 5.0])
    rate = -0.01
    dividend_yield = 0.025
    volatility = 0.35

    call = price("call", spot, strike, time, rate, volatility, dividend_yield)
    put = price("put", spot, strike, time, rate, volatility, dividend_yield)
    parity = spot * np.exp(-dividend_yield * time) - strike * np.exp(-rate * time)

    np.testing.assert_allclose(call - put, parity, rtol=2e-14, atol=2e-14)


def test_call_put_greek_relationships() -> None:
    dividend_yield = 0.02
    call = greeks("call", 110.0, 95.0, 1.7, 0.04, 0.31, dividend_yield)
    put = greeks("put", 110.0, 95.0, 1.7, 0.04, 0.31, dividend_yield)

    assert call.gamma == pytest.approx(put.gamma, abs=1e-15)
    assert call.vega == pytest.approx(put.vega, abs=1e-14)
    assert call.delta - put.delta == pytest.approx(
        math.exp(-dividend_yield * 1.7), abs=1e-14
    )
    assert call.rho - put.rho == pytest.approx(
        95.0 * 1.7 * math.exp(-0.04 * 1.7), abs=1e-12
    )
    assert call.theta - put.theta == pytest.approx(
        dividend_yield * 110.0 * math.exp(-dividend_yield * 1.7)
        - 0.04 * 95.0 * math.exp(-0.04 * 1.7),
        abs=1e-12,
    )


def test_analytic_greeks_match_central_finite_differences() -> None:
    option_type = OptionType.PUT
    spot = 103.0
    strike = 97.0
    time = 0.8
    rate = 0.03
    volatility = 0.24
    dividend_yield = 0.015
    risk = greeks(option_type, spot, strike, time, rate, volatility, dividend_yield)

    spot_step = 1e-4 * spot
    price_up = price(
        option_type,
        spot + spot_step,
        strike,
        time,
        rate,
        volatility,
        dividend_yield,
    )
    price_mid = price(option_type, spot, strike, time, rate, volatility, dividend_yield)
    price_down = price(
        option_type,
        spot - spot_step,
        strike,
        time,
        rate,
        volatility,
        dividend_yield,
    )
    finite_delta = (price_up - price_down) / (2.0 * spot_step)
    finite_gamma = (price_up - 2.0 * price_mid + price_down) / spot_step**2

    parameter_step = 1e-5
    finite_vega = (
        price(
            option_type,
            spot,
            strike,
            time,
            rate,
            volatility + parameter_step,
            dividend_yield,
        )
        - price(
            option_type,
            spot,
            strike,
            time,
            rate,
            volatility - parameter_step,
            dividend_yield,
        )
    ) / (2.0 * parameter_step)
    finite_rho = (
        price(
            option_type,
            spot,
            strike,
            time,
            rate + parameter_step,
            volatility,
            dividend_yield,
        )
        - price(
            option_type,
            spot,
            strike,
            time,
            rate - parameter_step,
            volatility,
            dividend_yield,
        )
    ) / (2.0 * parameter_step)
    finite_theta = -(
        price(
            option_type,
            spot,
            strike,
            time + parameter_step,
            rate,
            volatility,
            dividend_yield,
        )
        - price(
            option_type,
            spot,
            strike,
            time - parameter_step,
            rate,
            volatility,
            dividend_yield,
        )
    ) / (2.0 * parameter_step)

    assert risk.delta == pytest.approx(finite_delta, rel=2e-7)
    assert risk.gamma == pytest.approx(finite_gamma, rel=2e-6)
    assert risk.vega == pytest.approx(finite_vega, rel=2e-9)
    assert risk.rho == pytest.approx(finite_rho, rel=2e-9)
    assert risk.theta == pytest.approx(finite_theta, rel=2e-9)


def test_expiry_returns_intrinsic_and_documented_boundary_greeks() -> None:
    spots = np.array([90.0, 100.0, 110.0])

    call = price("call", spots, 100.0, 0.0, -0.2, 0.0, 0.5)
    put = price("put", spots, 100.0, 0.0, -0.2, 0.0, 0.5)
    call_risk = greeks("call", spots, 100.0, 0.0, -0.2, 0.0, 0.5)
    put_risk = greeks("put", spots, 100.0, 0.0, -0.2, 0.0, 0.5)

    np.testing.assert_array_equal(call, [0.0, 0.0, 10.0])
    np.testing.assert_array_equal(put, [10.0, 0.0, 0.0])
    np.testing.assert_array_equal(call_risk.delta, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(put_risk.delta, [-1.0, -0.5, 0.0])
    np.testing.assert_array_equal(call_risk.gamma, 0.0)
    np.testing.assert_array_equal(call_risk.vega, 0.0)
    np.testing.assert_array_equal(call_risk.theta, 0.0)
    np.testing.assert_array_equal(call_risk.rho, 0.0)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_zero_volatility_is_discounted_deterministic_payoff(
    option_type: str,
) -> None:
    spot = 120.0
    strike = 100.0
    time = 2.0
    rate = 0.03
    dividend_yield = 0.01
    discounted_spot = spot * math.exp(-dividend_yield * time)
    discounted_strike = strike * math.exp(-rate * time)
    expected = (
        max(discounted_spot - discounted_strike, 0.0)
        if option_type == "call"
        else max(discounted_strike - discounted_spot, 0.0)
    )

    assert price(
        option_type, spot, strike, time, rate, 0.0, dividend_yield
    ) == pytest.approx(expected, abs=1e-12)


def test_zero_volatility_put_boundary_greeks_are_finite() -> None:
    strike = 100.0
    time = 2.0
    rate = 0.03
    dividend_yield = 0.01
    spot = strike * math.exp((dividend_yield - rate) * time)

    result = greeks("put", spot, strike, time, rate, 0.0, dividend_yield)

    assert result.delta == pytest.approx(-0.5 * math.exp(-dividend_yield * time))
    assert result.gamma == 0.0
    assert result.vega == 0.0
    assert result.rho == pytest.approx(-0.5 * strike * time * math.exp(-rate * time))


def test_prices_respect_no_arbitrage_bounds_across_grid() -> None:
    spots = np.geomspace(20.0, 500.0, 17)
    strikes = np.geomspace(30.0, 300.0, 17)
    times = np.geomspace(1e-8, 30.0, 17)
    volatilities = np.geomspace(1e-8, 5.0, 17)

    for option_type in OptionType:
        values = price(option_type, spots, strikes, times, -0.03, volatilities, 0.04)
        lower, upper = price_bounds(option_type, spots, strikes, times, -0.03, 0.04)
        assert np.all(np.isfinite(values))
        assert np.all(values >= lower - 1e-13)
        assert np.all(values <= upper + 1e-13)


@pytest.mark.parametrize("option_type", list(OptionType))
def test_prices_are_monotone_in_volatility_and_maturity(
    option_type: OptionType,
) -> None:
    volatilities = np.linspace(0.0, 2.0, 101)
    times = np.linspace(0.0, 5.0, 101)

    by_volatility = price(option_type, 100.0, 110.0, 1.0, 0.0, volatilities)
    by_maturity = price(option_type, 100.0, 110.0, times, 0.0, 0.3)

    assert np.all(np.diff(by_volatility) >= -1e-13)
    assert np.all(np.diff(by_maturity) >= -1e-13)


def test_prices_have_expected_spot_and_strike_monotonicity() -> None:
    grid = np.linspace(50.0, 150.0, 101)

    calls_by_spot = price("call", grid, 100.0, 1.0, 0.02, 0.3)
    puts_by_spot = price("put", grid, 100.0, 1.0, 0.02, 0.3)
    calls_by_strike = price("call", 100.0, grid, 1.0, 0.02, 0.3)
    puts_by_strike = price("put", 100.0, grid, 1.0, 0.02, 0.3)

    assert np.all(np.diff(calls_by_spot) >= 0.0)
    assert np.all(np.diff(puts_by_spot) <= 0.0)
    assert np.all(np.diff(calls_by_strike) <= 0.0)
    assert np.all(np.diff(puts_by_strike) >= 0.0)


def test_randomized_put_call_parity_and_positive_gamma() -> None:
    generator = np.random.default_rng(20260709)
    size = 1_000
    spots = np.exp(generator.uniform(np.log(1.0), np.log(1_000.0), size))
    strikes = np.exp(generator.uniform(np.log(1.0), np.log(1_000.0), size))
    times = generator.uniform(1e-6, 30.0, size)
    rates = generator.uniform(-0.1, 0.2, size)
    yields = generator.uniform(-0.05, 0.15, size)
    volatilities = generator.uniform(1e-5, 3.0, size)

    calls = price("call", spots, strikes, times, rates, volatilities, yields)
    puts = price("put", spots, strikes, times, rates, volatilities, yields)
    call_risk = greeks("call", spots, strikes, times, rates, volatilities, yields)
    parity = spots * np.exp(-yields * times) - strikes * np.exp(-rates * times)

    np.testing.assert_allclose(calls - puts, parity, rtol=2e-12, atol=2e-12)
    assert np.all(call_risk.gamma >= 0.0)
    assert np.all(call_risk.vega >= 0.0)


def test_vectorized_inputs_broadcast_and_match_scalar_prices() -> None:
    spots = np.array([[80.0], [100.0], [120.0]])
    strikes = np.array([90.0, 100.0, 110.0, 120.0])
    values = price("call", spots, strikes, 0.75, 0.02, 0.25)
    risks = greeks("call", spots, strikes, 0.75, 0.02, 0.25)

    assert isinstance(values, np.ndarray)
    assert values.shape == (3, 4)
    assert isinstance(risks.delta, np.ndarray)
    assert risks.delta.shape == (3, 4)
    for row, spot in enumerate(spots[:, 0]):
        for column, strike in enumerate(strikes):
            assert values[row, column] == pytest.approx(
                price("call", spot, strike, 0.75, 0.02, 0.25), abs=1e-14
            )


def test_scalar_inputs_return_python_floats() -> None:
    value = price("call", SPOT, STRIKE, TIME, RATE, VOLATILITY)
    risk = greeks("call", SPOT, STRIKE, TIME, RATE, VOLATILITY)

    assert isinstance(value, float)
    assert isinstance(risk, Greeks)
    assert all(
        isinstance(metric, float)
        for metric in (risk.delta, risk.gamma, risk.vega, risk.theta, risk.rho)
    )


def test_standardized_terms_match_reference() -> None:
    first, second = d1_d2(SPOT, STRIKE, TIME, RATE, VOLATILITY)

    assert first == pytest.approx(0.35, abs=1e-15)
    assert second == pytest.approx(0.15, abs=1e-15)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("spot", 0.0, "spot must be strictly positive"),
        ("strike", -1.0, "strike must be strictly positive"),
        ("time_to_expiry", -1e-9, "time_to_expiry cannot be negative"),
        ("volatility", -0.1, "volatility cannot be negative"),
        ("rate", math.inf, "rate must contain only finite values"),
        ("dividend_yield", math.nan, "dividend_yield must contain only finite values"),
    ],
)
def test_invalid_numeric_inputs_raise_clear_errors(
    field: str, value: float, message: str
) -> None:
    arguments = {
        "spot": SPOT,
        "strike": STRIKE,
        "time_to_expiry": TIME,
        "rate": RATE,
        "volatility": VOLATILITY,
        "dividend_yield": 0.0,
    }
    arguments[field] = value

    with pytest.raises(InputValidationError, match=message):
        price("call", **arguments)


def test_invalid_option_type_and_shapes_raise_clear_errors() -> None:
    with pytest.raises(InputValidationError, match="option_type"):
        price("straddle", SPOT, STRIKE, TIME, RATE, VOLATILITY)
    with pytest.raises(InputValidationError, match="cannot be broadcast"):
        price(
            "call",
            np.ones((2, 3)),
            np.ones((4,)),
            TIME,
            RATE,
            VOLATILITY,
        )


def test_non_numeric_and_overflowing_discount_inputs_raise_clear_errors() -> None:
    with pytest.raises(InputValidationError, match="spot must contain real numbers"):
        price("call", "not-a-number", STRIKE, TIME, RATE, VOLATILITY)
    with pytest.raises(InputValidationError, match="discount exponent exceeds"):
        price("call", SPOT, STRIKE, 1e308, 1e308, VOLATILITY)
    with pytest.raises(InputValidationError, match="rate produces"):
        price("call", SPOT, STRIKE, 1.0, -1e3, VOLATILITY)
    with pytest.raises(InputValidationError, match="dividend_yield produces"):
        price("call", SPOT, STRIKE, 1.0, RATE, VOLATILITY, -1e3)
    with pytest.raises(InputValidationError, match="discounted strike exceeds"):
        price("put", SPOT, STRIKE, 1.0, -709.0, VOLATILITY)
    with pytest.raises(InputValidationError, match="discounted spot exceeds"):
        price("call", SPOT, STRIKE, 1.0, RATE, VOLATILITY, -709.0)


def test_complex_inputs_are_rejected_without_lossy_conversion() -> None:
    with pytest.raises(InputValidationError, match="spot must contain real numbers"):
        price(
            "call",
            np.array([100.0 + 1.0j]),
            STRIKE,
            TIME,
            RATE,
            VOLATILITY,
        )


@pytest.mark.parametrize(
    ("time_to_expiry", "volatility", "message"),
    [
        (0.0, VOLATILITY, "undefined at expiry"),
        (TIME, 0.0, "undefined at zero volatility"),
    ],
)
def test_d1_d2_rejects_singular_boundaries(
    time_to_expiry: float, volatility: float, message: str
) -> None:
    with pytest.raises(InputValidationError, match=message):
        d1_d2(SPOT, STRIKE, time_to_expiry, RATE, volatility)


def test_near_expiry_and_extreme_tail_prices_remain_nonnegative_and_finite() -> None:
    near_expiry = price("call", 100.0, 100.0, 1e-14, 0.05, 0.2)
    deep_out_of_money = price("call", 1.0, 1e6, 1.0, 0.01, 0.2)
    deep_in_the_money = price("put", 1.0, 1e6, 30.0, -0.05, 3.0)

    assert 0.0 < near_expiry < 1e-5
    assert math.isfinite(deep_out_of_money)
    assert deep_out_of_money >= 0.0
    assert math.isfinite(deep_in_the_money)
    assert deep_in_the_money >= 0.0


def test_extreme_total_variance_uses_finite_limiting_values_without_warnings() -> None:
    with np.errstate(all="raise"):
        first, second = d1_d2(100.0, 100.0, 1e200, 0.0, 1e200)
        call = price("call", 100.0, 100.0, 1e200, 0.0, 1e200)
        put = price("put", 100.0, 100.0, 1e200, 0.0, 1e200)
        risk = greeks("call", 100.0, 100.0, 1e200, 0.0, 1e200)

    assert first > 0.0
    assert second < 0.0
    assert call == pytest.approx(100.0)
    assert put == pytest.approx(100.0)
    assert all(
        math.isfinite(metric)
        for metric in (risk.delta, risk.gamma, risk.vega, risk.theta, risk.rho)
    )


def test_underflowed_total_volatility_uses_deterministic_limit() -> None:
    with np.errstate(all="raise"):
        first, second = d1_d2(100.0, 100.0, 1e-300, 0.0, 1e-200)
        call = price("call", 100.0, 100.0, 1e-300, 0.0, 1e-200)
        put = price("put", 100.0, 100.0, 1e-300, 0.0, 1e-200)
        risk = greeks("call", 100.0, 100.0, 1e-300, 0.0, 1e-200)

    assert first == 0.0
    assert second == 0.0
    assert call == 0.0
    assert put == 0.0
    assert risk.delta == 0.5
    assert risk.gamma == 0.0
    assert risk.vega == 0.0
