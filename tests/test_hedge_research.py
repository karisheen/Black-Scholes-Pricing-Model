"""Economic/statistical checks on fixed, common-random-number synthetic markets.

Assertions concern controlled ensembles, never monotone outcomes on every path.
The generating volatility is not estimated and fed back into hedge decisions.
"""

from dataclasses import replace
from math import erf, exp, log, sqrt

import numpy as np
import pytest

from quant_pricing.hedge_paths import GBMConfig, generate_gbm
from quant_pricing.hedging import (
    HedgeContract,
    HedgeCosts,
    HedgePolicy,
    prepare_market,
    simulate_hedge,
)

# Fixed before examining results: no seed search, filtering, or adaptive sampling.
SEED = 519021
PATHS = 8192
STEPS = 64
MATURITY = 0.25
VOLATILITY = 0.2


def _cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _call_moments(
    spot: float, strike: float, rate: float, volatility: float, maturity: float
) -> tuple[float, float]:
    """Closed-form first/variance moments of discounted lognormal call payoff."""
    width = volatility * sqrt(maturity)
    d2 = (log(spot / strike) + (rate - 0.5 * volatility**2) * maturity) / width
    price = spot * _cdf(d2 + width) - strike * exp(-rate * maturity) * _cdf(d2)
    second_moment = exp(-2.0 * rate * maturity) * (
        spot**2 * exp((2.0 * rate + volatility**2) * maturity) * _cdf(d2 + 2.0 * width)
        - 2.0 * strike * spot * exp(rate * maturity) * _cdf(d2 + width)
        + strike**2 * _cdf(d2)
    )
    return price, second_moment - price**2


def _paths(volatility: float = VOLATILITY, rate: float = 0.0) -> np.ndarray:
    return generate_gbm(
        GBMConfig(100.0, rate, volatility, MATURITY, STEPS, PATHS, SEED)
    )


def test_discounted_no_hedge_expectation_matches_analytic_lognormal_moments() -> None:
    rate = 0.03
    premium, variance = _call_moments(100.0, 100.0, rate, VOLATILITY, MATURITY)
    contract = HedgeContract(
        "call", 100.0, MATURITY, VOLATILITY, rate=rate, premium=premium
    )
    spots = _paths(rate=rate)
    market = prepare_market(contract, spots)
    result = simulate_hedge(contract, market, HedgePolicy("none"), HedgeCosts())
    discounted_payoff = exp(-rate * MATURITY) * np.maximum(spots[-1] - 100.0, 0.0)
    np.testing.assert_allclose(result.hedge_error, premium - discounted_payoff)
    assert abs(float(np.mean(result.hedge_error))) < 5.0 * sqrt(variance / PATHS)
    # The independent analytic price, not the production entry mark, sets E[P&L]=0.
    assert market.marks[0, 0] == pytest.approx(premium, abs=1e-12)


@pytest.mark.parametrize(
    "policy",
    [
        HedgePolicy("periodic", every=1),
        HedgePolicy("periodic", every=8),
        HedgePolicy("threshold", threshold=0.1),
    ],
)
def test_predictable_self_financing_hedges_preserve_risk_neutral_expectation(
    policy: HedgePolicy,
) -> None:
    rate = -0.02
    premium, _ = _call_moments(100.0, 100.0, rate, VOLATILITY, MATURITY)
    contract = HedgeContract(
        "call",
        100.0,
        MATURITY,
        VOLATILITY,
        rate=rate,
        premium=premium,
        initial_cash=13.0,
        initial_inventory=-0.3,
    )
    result = simulate_hedge(
        contract, prepare_market(contract, _paths(rate=rate)), policy, HedgeCosts()
    )
    # Initial stock is marked at S0; subtract its pre-entry wealth, not zero.
    error = result.hedge_error
    standard_error = float(np.std(error, ddof=1)) / sqrt(PATHS)
    assert abs(float(np.mean(error))) < 5.0 * standard_error
    initial_wealth = 13.0 - 30.0
    np.testing.assert_allclose(
        error,
        exp(-rate * MATURITY) * (result.final_pnl + initial_wealth) - initial_wealth,
        atol=1e-12,
    )


def test_finer_hedging_reduces_ensemble_dispersion_not_every_path_error() -> None:
    contract = HedgeContract("call", 100.0, MATURITY, VOLATILITY)
    market = prepare_market(contract, _paths())
    policies = [
        HedgePolicy("none"),
        HedgePolicy("periodic", every=16),
        HedgePolicy("periodic", every=4),
        HedgePolicy("periodic", every=1),
    ]
    results = [
        simulate_hedge(contract, market, policy, HedgeCosts()) for policy in policies
    ]
    dispersions = [float(np.std(item.hedge_error, ddof=1)) for item in results]
    assert dispersions[1] < 0.65 * dispersions[0]
    assert dispersions[2] < 0.8 * dispersions[1]
    assert dispersions[3] < 0.8 * dispersions[2]
    for result in results:
        standard_error = float(np.std(result.hedge_error, ddof=1)) / sqrt(PATHS)
        assert abs(float(np.mean(result.hedge_error))) < 5.0 * standard_error
    # A real hedging trade-off: finer rebalancing has greater average turnover.
    assert float(np.mean(results[-1].turnover)) > float(np.mean(results[1].turnover))


def test_model_vs_diffusion_volatility_effect_uses_common_random_numbers() -> None:
    contract = HedgeContract("call", 100.0, MATURITY, VOLATILITY)
    errors = []
    for diffusion_volatility in (0.12, 0.20, 0.32):
        market = prepare_market(contract, _paths(diffusion_volatility))
        result = simulate_hedge(contract, market, HedgePolicy("periodic"), HedgeCosts())
        errors.append(result.hedge_error)
    low, matched, high = errors
    # For a short convex payoff, selling model variance above realized diffusion
    # variance gives positive expected gross P&L; reversing mismatch reverses it.
    assert float(np.mean(low)) > 5.0 * float(np.std(low, ddof=1)) / sqrt(PATHS)
    assert float(np.mean(high)) < -5.0 * float(np.std(high, ddof=1)) / sqrt(PATHS)
    for paired_difference in (low - matched, matched - high):
        standard_error = float(np.std(paired_difference, ddof=1)) / sqrt(PATHS)
        assert float(np.mean(paired_difference)) > 5.0 * standard_error


@pytest.mark.parametrize(
    "policy",
    [
        HedgePolicy("none"),
        HedgePolicy("periodic", every=4),
        HedgePolicy("threshold", threshold=0.1),
    ],
)
def test_paired_cost_drag_includes_funding_not_just_fee_subtraction(
    policy: HedgePolicy,
) -> None:
    rate = 0.05
    contract = HedgeContract("put", 100.0, MATURITY, VOLATILITY, rate=rate)
    market = prepare_market(contract, _paths(rate=rate))
    zero = simulate_hedge(contract, market, policy, HedgeCosts())
    costly = simulate_hedge(contract, market, policy, HedgeCosts(2.0, 3.0, 4.0))
    explicit_cost = costly.fees + costly.spread_cost + costly.slippage_cost
    financing_difference = costly.financing - zero.financing
    np.testing.assert_allclose(
        costly.final_pnl - zero.final_pnl,
        financing_difference - explicit_cost,
        atol=2e-12,
    )
    np.testing.assert_array_equal(costly.option_pnl, zero.option_pnl)
    np.testing.assert_array_equal(costly.hedge_pnl, zero.hedge_pnl)
    np.testing.assert_array_equal(costly.trade_count, zero.trade_count)
    assert np.all(financing_difference <= 1e-12)
    if policy.kind != "none":
        assert float(np.mean(financing_difference)) < 0.0
        assert float(np.mean(costly.final_pnl - zero.final_pnl)) < 0.0
    else:
        np.testing.assert_array_equal(costly.final_pnl, zero.final_pnl)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_vectorized_attribution_and_long_short_mirror_on_random_paths(
    kind: str,
) -> None:
    contract = HedgeContract(
        kind,
        100.0,
        MATURITY,
        VOLATILITY,
        rate=-0.01,
        dividend_yield=0.02,
        quantity=-2.0,
        multiplier=3.0,
        initial_cash=7.0,
        initial_inventory=0.4,
    )
    market = prepare_market(contract, _paths())
    policy = HedgePolicy("threshold", threshold=0.3)
    short = simulate_hedge(contract, market, policy, HedgeCosts())
    long = simulate_hedge(
        replace(contract, quantity=2.0, initial_cash=-7.0, initial_inventory=-0.4),
        market,
        policy,
        HedgeCosts(),
    )
    for name in (
        "final_pnl",
        "hedge_error",
        "option_pnl",
        "hedge_pnl",
        "financing",
        "dividends",
    ):
        np.testing.assert_allclose(
            getattr(long, name), -getattr(short, name), atol=1e-12
        )
    np.testing.assert_array_equal(long.trade_count, short.trade_count)
    np.testing.assert_allclose(long.turnover, short.turnover, atol=1e-12)
    costly = simulate_hedge(contract, market, policy, HedgeCosts(1.0, 2.0, 3.0))
    np.testing.assert_allclose(
        costly.final_pnl,
        costly.option_pnl
        + costly.hedge_pnl
        + costly.financing
        + costly.dividends
        - costly.fees
        - costly.spread_cost
        - costly.slippage_cost,
        atol=2e-12,
    )
    assert np.all(costly.max_drawdown >= 0.0)
    assert np.all(costly.mean_abs_residual_delta <= 0.3 + 1e-12)


def test_public_matrix_results_equal_independent_single_path_runs() -> None:
    # Three different histories expose accidental cross-path averages/broadcasts.
    spots = np.array(
        [
            [100.0, 100.0, 100.0],
            [90.0, 100.0, 120.0],
            [110.0, 95.0, 80.0],
            [120.0, 90.0, 105.0],
        ]
    )
    contract = HedgeContract(
        "put",
        102.0,
        0.3,
        0.25,
        rate=-0.03,
        dividend_yield=0.02,
        initial_cash=12.0,
        initial_inventory=-0.4,
    )
    policy = HedgePolicy("periodic", every=2)
    costs = HedgeCosts(2.0, 1.0, 3.0)
    matrix = simulate_hedge(
        contract, prepare_market(contract, spots), policy, costs, trace_path=1
    )
    for index in range(3):
        scalar = simulate_hedge(
            contract,
            prepare_market(contract, spots[:, index : index + 1]),
            policy,
            costs,
            trace_path=0,
        )
        for name in (
            "final_pnl",
            "hedge_error",
            "turnover",
            "fees",
            "spread_cost",
            "slippage_cost",
            "financing",
            "dividends",
            "option_pnl",
            "hedge_pnl",
            "mean_abs_residual_delta",
            "max_drawdown",
            "trade_count",
        ):
            assert getattr(matrix, name)[index] == pytest.approx(
                getattr(scalar, name)[0]
            )
        if index == 1:
            assert matrix.ledger == scalar.ledger
