"""Independent economic regressions for the synthetic self-financing ledger."""

from dataclasses import replace
from math import erf, exp, expm1, log, sqrt

import numpy as np
import pytest

from quant_pricing.exceptions import InputValidationError
from quant_pricing.hedging import (
    HedgeContract,
    HedgeCosts,
    HedgePolicy,
    prepare_market,
    simulate_hedge,
)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _price_delta(
    contract: HedgeContract, spot: float, remaining: float
) -> tuple[float, float]:
    """Scalar textbook oracle, independent of the production BSM routines."""
    sign = 1.0 if contract.kind == "call" else -1.0
    if remaining == 0.0:
        return max(sign * (spot - contract.strike), 0.0), 0.0
    forward = spot * exp((contract.rate - contract.dividend_yield) * remaining)
    discount = exp(-contract.rate * remaining)
    carry = exp(-contract.dividend_yield * remaining)
    if contract.model_volatility == 0.0:
        active = sign * (forward - contract.strike) > 0.0
        return (
            discount * max(sign * (forward - contract.strike), 0.0),
            sign * carry if active else 0.0,
        )
    width = contract.model_volatility * sqrt(remaining)
    d1 = log(forward / contract.strike) / width + 0.5 * width
    d2 = d1 - width
    return (
        sign
        * (
            spot * carry * _normal_cdf(sign * d1)
            - contract.strike * discount * _normal_cdf(sign * d2)
        ),
        sign * carry * _normal_cdf(sign * d1),
    )


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("quantity", [-1.0, 1.0])
@pytest.mark.parametrize("terminal", [80.0, 120.0])
def test_one_interval_execution_and_settlement_are_hand_computable(
    kind: str, quantity: float, terminal: float
) -> None:
    contract = HedgeContract(kind, 100.0, 0.5, 0.2, quantity=quantity, premium=7.0)
    costs = HedgeCosts(half_spread_bps=10.0, slippage_bps=20.0, fee_bps=30.0)
    result = simulate_hedge(
        contract,
        prepare_market(contract, np.array([[100.0], [terminal]])),
        HedgePolicy("periodic"),
        costs,
        trace_path=0,
    )
    delta = _price_delta(contract, 100.0, 0.5)[1]
    hedge = -quantity * delta
    entry_price = 100.0 * (1.0 + np.sign(hedge) * 0.003)
    close_price = terminal * (1.0 - np.sign(hedge) * 0.003)
    entry_fee = abs(hedge * entry_price) * 0.003
    close_fee = abs(hedge * close_price) * 0.003
    intrinsic = max((terminal - 100.0) * (1 if kind == "call" else -1), 0)
    expected_cash = (
        -quantity * 7.0
        - hedge * entry_price
        - entry_fee
        + quantity * intrinsic
        + hedge * close_price
        - close_fee
    )
    opening, closing = result.ledger
    assert opening.hedge_trade == pytest.approx(hedge)
    assert opening.execution_price == pytest.approx(entry_price)
    assert opening.cash == pytest.approx(
        -quantity * 7.0 - hedge * entry_price - entry_fee
    )
    assert closing.hedge_trade == pytest.approx(-hedge)
    assert closing.execution_price == pytest.approx(close_price)
    assert closing.cash == pytest.approx(expected_cash)
    assert result.final_pnl[0] == pytest.approx(expected_cash)
    assert result.fees[0] == pytest.approx(entry_fee + close_fee)
    assert result.spread_cost[0] == pytest.approx(abs(hedge) * (100 + terminal) * 0.001)
    assert result.slippage_cost[0] == pytest.approx(
        abs(hedge) * (100 + terminal) * 0.002
    )
    assert result.turnover[0] == pytest.approx(
        abs(hedge * entry_price) + abs(hedge * close_price)
    )
    assert result.option_pnl[0] == pytest.approx(quantity * (intrinsic - 7.0))
    assert result.hedge_pnl[0] == pytest.approx(hedge * (terminal - 100.0))
    assert result.trade_count[0] == 2
    assert closing.option_units == closing.hedge_inventory == 0.0
    assert closing.residual_delta == closing.unrealized_pnl == 0.0
    assert closing.realized_pnl == pytest.approx(expected_cash)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_average_cost_add_reduce_and_reopen_with_initial_inventory(
    direction: float,
) -> None:
    # At zero volatility these strictly non-ATM deltas are exactly 0 or 1.
    # Hold 1/4 share at 90; add 3/4 at 120 => basis 112.5; sell at 80
    # realizes -32.5. Reopen at 140, close at 150 => gross stock P&L -22.5.
    contract = HedgeContract(
        "call",
        100.0,
        1.0,
        0.0,
        quantity=-direction,
        premium=5.0,
        initial_cash=7.0,
        initial_inventory=0.25 * direction,
    )
    spots = np.array([[90.0], [120.0], [130.0], [80.0], [140.0], [150.0]])
    result = simulate_hedge(
        contract,
        prepare_market(contract, spots),
        HedgePolicy("threshold", threshold=0.3),
        HedgeCosts(),
        trace_path=0,
    )
    expected_inventory = [0.25, 1.0, 1.0, 0.0, 1.0, 0.0]
    expected_basis = [90.0, 112.5, 112.5, 0.0, 140.0, 0.0]
    stock_realized = [0.0, 0.0, 0.0, -32.5, -32.5, -22.5]
    stock_unrealized = [0.0, 7.5, 17.5, 0.0, 0.0, 0.0]
    option_unrealized = [5.0, -15.0, -25.0, 5.0, -35.0, 0.0]
    initial_wealth = 7.0 + direction * 22.5
    for index, row in enumerate(result.ledger):
        assert row.hedge_inventory == pytest.approx(
            direction * expected_inventory[index]
        )
        assert row.average_cost == pytest.approx(expected_basis[index])
        assert row.stock_realized_pnl == pytest.approx(
            direction * stock_realized[index]
        )
        assert row.stock_unrealized_pnl == pytest.approx(
            direction * stock_unrealized[index]
        )
        option_realized = -45.0 if index == 5 else 0.0
        assert row.realized_pnl == pytest.approx(
            direction * (stock_realized[index] + option_realized)
        )
        assert row.unrealized_pnl == pytest.approx(
            direction * (stock_unrealized[index] + option_unrealized[index])
        )
        assert row.realized_pnl + row.unrealized_pnl == pytest.approx(
            row.equity - initial_wealth
        )
    assert result.final_pnl[0] == pytest.approx(-67.5 * direction)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_opening_trade_can_cross_preexisting_stock_through_zero(
    direction: float,
) -> None:
    contract = HedgeContract(
        "call",
        100.0,
        1.0,
        0.0,
        quantity=-direction,
        initial_inventory=-2.0 * direction,
        premium=20.0,
    )
    result = simulate_hedge(
        contract,
        prepare_market(contract, np.array([[120.0], [130.0], [90.0]])),
        HedgePolicy("periodic"),
        HedgeCosts(),
        trace_path=0,
    )
    assert result.ledger[0].hedge_trade == 3.0 * direction
    assert result.ledger[0].average_cost == 120.0
    assert result.ledger[0].stock_realized_pnl == 0.0
    assert result.ledger[1].stock_unrealized_pnl == 10.0 * direction
    assert result.ledger[-1].stock_realized_pnl == -30.0 * direction
    assert result.final_pnl[0] == pytest.approx(-10.0 * direction)


@pytest.mark.parametrize("rate", [-0.1, 0.1])
@pytest.mark.parametrize("inventory", [-2.0, 2.0])
def test_no_hedge_interest_and_left_point_dividends_include_initial_endowment(
    rate: float, inventory: float
) -> None:
    contract = HedgeContract(
        "put",
        100.0,
        1.0,
        0.2,
        rate=rate,
        dividend_yield=0.08,
        quantity=-2.0,
        multiplier=3.0,
        premium=4.0,
        initial_cash=10.0,
        initial_inventory=inventory,
    )
    result = simulate_hedge(
        contract,
        prepare_market(contract, np.array([[100.0], [110.0], [80.0]])),
        HedgePolicy("none"),
        HedgeCosts(),
        trace_path=0,
    )
    cash0 = 10.0 + 6.0 * 4.0
    interest1 = cash0 * expm1(rate * 0.5)
    dividend1 = inventory * 100.0 * 0.08 * 0.5
    cash1 = cash0 + interest1 + dividend1
    interest2 = cash1 * expm1(rate * 0.5)
    dividend2 = inventory * 110.0 * 0.08 * 0.5
    terminal = cash1 + interest2 + dividend2 - 6.0 * 20.0 + inventory * 80.0
    initial_wealth = 10.0 + inventory * 100.0
    assert result.ledger[0].hedge_trade == result.ledger[1].hedge_trade == 0.0
    assert result.ledger[0].cash_interest == result.ledger[0].dividend_credit == 0.0
    assert result.ledger[1].cash_interest == pytest.approx(interest1)
    assert result.ledger[1].dividend_credit == pytest.approx(dividend1)
    assert result.ledger[1].cash == pytest.approx(cash1)
    assert result.ledger[2].cash_interest == pytest.approx(interest2)
    assert result.ledger[2].dividend_credit == pytest.approx(dividend2)
    assert result.ledger[2].hedge_trade == -inventory
    assert result.financing[0] == pytest.approx(interest1 + interest2)
    assert result.dividends[0] == pytest.approx(dividend1 + dividend2)
    assert result.final_pnl[0] == pytest.approx(terminal - initial_wealth)
    assert result.hedge_error[0] == pytest.approx(
        exp(-rate) * terminal - initial_wealth
    )
    assert result.hedge_pnl[0] == pytest.approx(inventory * (80.0 - 100.0))
    assert result.trade_count[0] == 1
    assert result.ledger[-1].realized_pnl == pytest.approx(result.final_pnl[0])


def test_periodic_rebalances_at_zero_and_threshold_is_strict() -> None:
    contract = HedgeContract("call", 100.0, 1.0, 0.0, premium=20.0)
    market = prepare_market(contract, np.array([[120.0], [90.0], [130.0], [110.0]]))
    periodic = simulate_hedge(
        contract, market, HedgePolicy("periodic", every=2), HedgeCosts(), trace_path=0
    )
    equal = simulate_hedge(
        contract,
        market,
        HedgePolicy("threshold", threshold=1.0),
        HedgeCosts(),
        trace_path=0,
    )
    below = simulate_hedge(
        contract,
        market,
        HedgePolicy("threshold", threshold=np.nextafter(1.0, 0.0)),
        HedgeCosts(),
        trace_path=0,
    )
    assert [row.hedge_inventory for row in periodic.ledger] == [1.0, 1.0, 1.0, 0.0]
    assert [row.hedge_trade for row in equal.ledger] == [0.0, 0.0, 0.0, 0.0]
    assert [row.hedge_trade for row in below.ledger] == [1.0, -1.0, 1.0, -1.0]
    assert equal.mean_abs_residual_delta[0] == pytest.approx(2.0 / 3.0)
    assert periodic.mean_abs_residual_delta[0] == pytest.approx(1.0 / 3.0)


def test_fair_entry_delta_uses_independent_normal_cdf_and_no_lookahead() -> None:
    contract = HedgeContract(
        "put",
        103.0,
        0.75,
        0.27,
        rate=-0.02,
        dividend_yield=0.04,
        quantity=-3.0,
        multiplier=2.0,
    )
    first = np.array([[100.0], [104.0], [98.0], [125.0], [130.0]])
    second = np.array([[100.0], [104.0], [98.0], [70.0], [60.0]])
    traces = [
        simulate_hedge(
            contract,
            prepare_market(contract, spots),
            HedgePolicy("threshold", threshold=0.2),
            HedgeCosts(),
            trace_path=0,
        ).ledger
        for spots in (first, second)
    ]
    price, delta = _price_delta(contract, 100.0, 0.75)
    assert traces[0][0].option_mark == pytest.approx(price, abs=1e-12)
    assert traces[0][0].option_delta == pytest.approx(delta, abs=1e-12)
    assert traces[0][0].hedge_inventory == pytest.approx(6.0 * delta)
    assert traces[0][0].equity == pytest.approx(0.0, abs=1e-12)
    assert traces[0][:3] == traces[1][:3]


def test_quantity_multiplier_and_threshold_scale_all_cash_and_exposure() -> None:
    base = HedgeContract(
        "call",
        100.0,
        0.4,
        0.2,
        rate=0.03,
        dividend_yield=0.01,
        premium=6.0,
        initial_cash=8.0,
        initial_inventory=-0.2,
    )
    scaled = replace(base, multiplier=7.0, initial_cash=56.0, initial_inventory=-1.4)
    market = prepare_market(base, np.array([[100.0], [110.0], [90.0], [105.0]]))
    costs = HedgeCosts(2.0, 3.0, 4.0)
    small = simulate_hedge(
        base, market, HedgePolicy("threshold", threshold=0.1), costs, trace_path=0
    )
    large = simulate_hedge(
        scaled, market, HedgePolicy("threshold", threshold=0.7), costs, trace_path=0
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
    ):
        np.testing.assert_allclose(
            getattr(large, name), 7.0 * getattr(small, name), atol=1e-12
        )
    np.testing.assert_array_equal(large.trade_count, small.trade_count)
    for row in large.ledger:
        assert row.realized_pnl + row.unrealized_pnl == pytest.approx(
            row.equity - (56.0 - 140.0)
        )


def test_costs_are_realized_once_and_opening_loss_enters_drawdown() -> None:
    contract = HedgeContract("call", 100.0, 1.0, 0.0, premium=20.0)
    result = simulate_hedge(
        contract,
        prepare_market(contract, np.array([[120.0], [120.0]])),
        HedgePolicy("periodic"),
        HedgeCosts(10.0, 20.0, 30.0),
        trace_path=0,
    )
    opening, closing = result.ledger
    opening_cost = 120.0 * 0.003 + 120.36 * 0.003
    closing_cost = 120.0 * 0.003 + 119.64 * 0.003
    assert opening.realized_pnl == pytest.approx(-opening_cost)
    assert opening.unrealized_pnl == pytest.approx(0.0)
    assert result.final_pnl[0] == pytest.approx(-opening_cost - closing_cost)
    assert result.max_drawdown[0] == pytest.approx(opening_cost + closing_cost)
    assert closing.realized_pnl == pytest.approx(-opening_cost - closing_cost)
    for name in ("fees", "spread_cost", "slippage_cost", "turnover"):
        assert getattr(result, name)[0] == pytest.approx(
            sum(getattr(row, name) for row in result.ledger)
        )


def test_dividend_and_interest_use_positions_before_each_rebalance() -> None:
    contract = HedgeContract(
        "call", 50.0, 1.0, 0.0, rate=-0.05, dividend_yield=0.08, premium=55.0
    )
    result = simulate_hedge(
        contract,
        prepare_market(contract, np.array([[100.0], [110.0], [80.0]])),
        HedgePolicy("periodic"),
        HedgeCosts(),
        trace_path=0,
    )
    hedge0, hedge1 = exp(-0.08), exp(-0.04)
    cash0 = 55.0 - 100.0 * hedge0
    interest1 = cash0 * expm1(-0.05 * 0.5)
    dividend1 = hedge0 * 100.0 * 0.08 * 0.5
    cash1 = cash0 + interest1 + dividend1 - (hedge1 - hedge0) * 110.0
    interest2 = cash1 * expm1(-0.05 * 0.5)
    dividend2 = hedge1 * 110.0 * 0.08 * 0.5
    assert result.ledger[0].hedge_inventory == pytest.approx(hedge0)
    assert result.ledger[1].hedge_inventory == pytest.approx(hedge1)
    assert result.ledger[1].cash == pytest.approx(cash1)
    assert result.ledger[1].cash_interest == pytest.approx(interest1)
    assert result.ledger[2].cash_interest == pytest.approx(interest2)
    assert result.ledger[1].dividend_credit == pytest.approx(dividend1)
    assert result.ledger[2].dividend_credit == pytest.approx(dividend2)
    assert result.hedge_pnl[0] == pytest.approx(10.0 * hedge0 - 30.0 * hedge1)
    assert result.final_pnl[0] == pytest.approx(
        cash1 + interest2 + dividend2 - 30.0 + hedge1 * 80.0
    )


def test_no_hedge_zero_inventory_has_no_execution_or_trace_by_default() -> None:
    contract = HedgeContract("put", 100.0, 1.0, 0.2, premium=9.0)
    result = simulate_hedge(
        contract,
        prepare_market(contract, np.array([[100.0], [80.0]])),
        HedgePolicy("none"),
        HedgeCosts(10.0, 10.0, 10.0),
    )
    assert result.ledger == ()
    assert result.final_pnl[0] == pytest.approx(-11.0)
    for name in ("hedge_pnl", "turnover", "fees", "spread_cost", "slippage_cost"):
        assert getattr(result, name)[0] == 0.0
    assert result.trade_count[0] == 0


@pytest.mark.parametrize("remaining", [1e-12, 1e-8])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_near_expiry_atm_market_stays_finite_and_settles_exactly(
    remaining: float, kind: str
) -> None:
    contract = HedgeContract(kind, 100.0, remaining, 0.2, rate=-0.02)
    result = simulate_hedge(
        contract,
        prepare_market(contract, np.array([[100.0], [100.0], [100.01]])),
        HedgePolicy("periodic"),
        HedgeCosts(),
        trace_path=0,
    )
    for row in result.ledger:
        assert all(
            np.isfinite(value)
            for value in (
                row.cash,
                row.equity,
                row.option_mark,
                row.option_delta,
                row.realized_pnl,
                row.unrealized_pnl,
            )
        )
    expected = 0.01 if kind == "call" else 0.0
    assert result.ledger[-1].option_mark == pytest.approx(expected)
    assert result.ledger[-1].option_units == 0.0
    assert result.ledger[-1].hedge_inventory == 0.0


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "straddle"},
        {"strike": 0.0},
        {"strike": float("nan")},
        {"maturity": 0.0},
        {"maturity": float("inf")},
        {"model_volatility": -0.1},
        {"rate": float("nan")},
        {"dividend_yield": float("inf")},
        {"quantity": float("nan")},
        {"multiplier": 0.0},
        {"multiplier": -1.0},
        {"premium": -1.0},
        {"premium": float("inf")},
        {"initial_cash": float("nan")},
        {"initial_inventory": float("inf")},
    ],
)
def test_invalid_contract_inputs_fail_loudly(changes: dict[str, object]) -> None:
    with pytest.raises(InputValidationError):
        HedgeContract(
            **{
                "kind": "call",
                "strike": 100.0,
                "maturity": 1.0,
                "model_volatility": 0.2,
                **changes,
            }
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "unknown"},
        {"every": 0},
        {"every": 1.5},
        {"every": True},
        {"threshold": -1.0},
        {"threshold": float("nan")},
    ],
)
def test_invalid_policy_inputs_fail_loudly(changes: dict[str, object]) -> None:
    with pytest.raises(InputValidationError):
        HedgePolicy(**{"kind": "periodic", **changes})


@pytest.mark.parametrize("name", ["half_spread_bps", "slippage_bps", "fee_bps"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_cost_inputs_fail_loudly(name: str, value: float) -> None:
    with pytest.raises(InputValidationError):
        HedgeCosts(**{name: value})


@pytest.mark.parametrize(
    "spots",
    [
        np.array([100.0, 110.0]),
        np.array([[100.0]]),
        np.empty((2, 0)),
        np.array([[100.0], [0.0]]),
        np.array([[100.0], [-1.0]]),
        np.array([[100.0], [np.nan]]),
        np.ma.array([[100.0], [111.0]], mask=[[False], [True]]),
        np.array([[100.0], [np.inf]]),
    ],
)
def test_invalid_market_fails_loudly(spots: np.ndarray) -> None:
    with pytest.raises(InputValidationError):
        prepare_market(HedgeContract("call", 100.0, 1.0, 0.2), spots)


def test_market_owns_frozen_data_and_rejects_pricing_mismatch() -> None:
    contract = HedgeContract("call", 100.0, 1.0, 0.2)
    spots = np.array([[100.0, 100.0], [90.0, 110.0]])
    market = prepare_market(contract, spots)
    spots[1, 0] = 1.0
    assert market.spots[1, 0] == 90.0
    for array in (market.spots, market.marks, market.deltas, market.times):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0.0
    with pytest.raises(InputValidationError):
        simulate_hedge(
            replace(contract, strike=101.0), market, HedgePolicy("none"), HedgeCosts()
        )
    for index in (-1, 2):
        with pytest.raises(InputValidationError):
            simulate_hedge(
                contract, market, HedgePolicy("none"), HedgeCosts(), trace_path=index
            )


def test_replacing_market_spots_reprices_and_cannot_inject_future_deltas() -> None:
    contract = HedgeContract("put", 100.0, 1.0, 0.2)
    market = prepare_market(contract, [[100.0], [115.0]])
    revised = replace(market, spots=np.array([[100.0], [70.0]]))
    assert revised.deltas[0, 0] == market.deltas[0, 0]
    assert revised.marks[-1, 0] == 30.0
    assert market.marks[-1, 0] == 0.0
    with pytest.raises((TypeError, ValueError)):
        replace(market, deltas=revised.deltas)
    with pytest.raises((TypeError, ValueError)):
        replace(market, marks=revised.marks)


def test_periodic_cadence_rebalances_at_scheduled_step_not_intervening_steps() -> None:
    contract = HedgeContract("call", 100.0, 1.0, 0.0, premium=20.0)
    spots = np.array([[120.0], [130.0], [90.0], [110.0], [120.0]])
    result = simulate_hedge(
        contract,
        prepare_market(contract, spots),
        HedgePolicy("periodic", every=2),
        HedgeCosts(),
        trace_path=0,
    )
    assert [row.hedge_trade for row in result.ledger] == [1.0, 0.0, -1.0, 0.0, 0.0]
    assert [row.hedge_inventory for row in result.ledger] == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert result.hedge_pnl[0] == pytest.approx(-30.0)
    assert result.final_pnl[0] == pytest.approx(-30.0)
    assert result.trade_count[0] == 2


def test_nonfinite_economic_outputs_raise_instead_of_silent_fill() -> None:
    with pytest.raises(InputValidationError):
        contract = HedgeContract(
            "call", 100.0, 1.0, 0.2, quantity=-1e308, multiplier=10.0
        )
        market = prepare_market(contract, np.array([[100.0], [110.0]]))
        simulate_hedge(contract, market, HedgePolicy("periodic"), HedgeCosts())
