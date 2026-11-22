"""Economic units, analytic derivatives, scenario repricing and reconciled P&L."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import log

import pytest

from quant_pricing.black_scholes import price
from quant_pricing.exceptions import InputValidationError
from quant_pricing.market import OptionContract, Portfolio, Position, Underlying
from quant_pricing.risk import (
    Exposure,
    RiskInputs,
    RiskLimitError,
    RiskLimits,
    RiskScenario,
    Trade,
    advanced_greeks,
    attribute_pnl,
    check_limits,
    scenario_portfolio,
    value_portfolio,
)

NOW = datetime(2026, 1, 2, tzinfo=timezone.utc)
USD = Underlying("ABC", "USD")
OTHER = Underlying("XYZ", "USD")
EUR = Underlying("EU", "EUR")


def contract(**changes: object) -> OptionContract:
    base = OptionContract(
        "ABC-C100",
        USD,
        "call",
        100.0,
        NOW + timedelta(days=365),
        "european",
        "cash",
        100.0,
        "USD",
        "USD",
        "per_unit",
    )
    return replace(base, **changes)


class Smile:
    def volatility(self, log_moneyness: float, time_to_expiry: float) -> float:
        return 0.25 - 0.06 * log_moneyness + 0.01 * time_to_expiry


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("spot", [75.0, 103.0, 140.0])
def test_higher_greeks_are_absolute_partials(kind: str, spot: float) -> None:
    args = dict(
        option_type=kind,
        spot=spot,
        strike=100.0,
        time_to_expiry=0.7,
        rate=0.04,
        volatility=0.27,
        dividend_yield=0.025,
    )
    base = advanced_greeks(**args)
    h = 1e-5
    vol_up = advanced_greeks(**(args | {"volatility": 0.27 + h}))
    vol_down = advanced_greeks(**(args | {"volatility": 0.27 - h}))
    spot_up = advanced_greeks(**(args | {"spot": spot + h}))
    spot_down = advanced_greeks(**(args | {"spot": spot - h}))
    later = advanced_greeks(**(args | {"time_to_expiry": 0.7 - h}))
    earlier = advanced_greeks(**(args | {"time_to_expiry": 0.7 + h}))
    assert base.vanna == pytest.approx(
        (vol_up.delta - vol_down.delta) / (2 * h), rel=2e-7, abs=1e-9
    )
    assert base.vanna == pytest.approx(
        (spot_up.vega - spot_down.vega) / (2 * h), rel=2e-7, abs=1e-9
    )
    assert base.volga == pytest.approx(
        (vol_up.vega - vol_down.vega) / (2 * h), rel=2e-7, abs=1e-8
    )
    assert base.vomma == base.volga
    assert base.charm == pytest.approx(
        (later.delta - earlier.delta) / (2 * h), rel=2e-7, abs=1e-9
    )
    assert base.speed == pytest.approx(
        (spot_up.gamma - spot_down.gamma) / (2 * h), rel=2e-7, abs=1e-9
    )
    assert base.color == pytest.approx(
        (later.gamma - earlier.gamma) / (2 * h), rel=2e-7, abs=1e-9
    )
    assert base.theta == pytest.approx(
        (later.price - earlier.price) / (2 * h), rel=2e-7
    )


def test_call_put_higher_risk_parity_and_calendar_signs() -> None:
    call = advanced_greeks("call", 100, 100, 1, 0, 0.2)
    put = advanced_greeks("put", 100, 100, 1, 0, 0.2)
    assert call.charm < 0
    assert call.color > 0
    assert call.speed < 0
    assert call.volga < 0  # ATM-forward d1*d2 is negative, not abs(vomma).
    for field in ("vanna", "volga", "charm", "speed", "color"):
        assert getattr(call, field) == getattr(put, field)
    call = advanced_greeks("call", 100, 100, 0.6, -0.01, 0.2, 0.03)
    put = advanced_greeks("put", 100, 100, 0.6, -0.01, 0.2, 0.03)
    assert call.charm - put.charm == pytest.approx(0.03 * (call.delta - put.delta))


@pytest.mark.parametrize(
    "field,value",
    [
        ("spot", 0),
        ("strike", -1),
        ("time_to_expiry", 0),
        ("time_to_expiry", -0.1),
        ("volatility", 0),
        ("volatility", -0.1),
        ("rate", float("nan")),
        ("dividend_yield", float("inf")),
        ("spot", True),
        ("spot", "100"),
    ],
)
def test_undefined_or_invalid_greek_inputs_rejected(field: str, value: object) -> None:
    args = dict(
        option_type="call",
        spot=100,
        strike=100,
        time_to_expiry=1,
        rate=0.03,
        volatility=0.2,
        dividend_yield=0,
    )
    args[field] = value
    with pytest.raises(InputValidationError):
        advanced_greeks(**args)


def test_numerical_extremes_remain_explicit() -> None:
    deep = advanced_greeks("call", 100, 1e-100, 1, 0, 0.2)
    assert deep.delta == 1
    assert deep.gamma == 0
    assert deep.vanna == 0
    with pytest.raises(InputValidationError):
        advanced_greeks("call", 100, 100, 1, 0, 1e200)


def test_signed_multiplied_currency_safe_group_aggregation() -> None:
    first = contract()
    second = contract(
        symbol="XYZ-P110",
        underlying=OTHER,
        strike=110,
        option_type="put",
        expiry=NOW + timedelta(days=180),
        multiplier=10,
    )
    euro = contract(
        symbol="EU-C100",
        underlying=EUR,
        quote_currency="EUR",
        settlement_currency="EUR",
        multiplier=1,
    )
    positions = (
        Position(first, 3),
        Position(first, -1),
        Position(second, -4),
        Position(euro, 7),
    )
    inputs = {
        USD: RiskInputs(100, 0.2, 0.03, 0.01),
        OTHER: RiskInputs(120, 0.35, -0.01),
        EUR: RiskInputs(90, 0.1),
    }
    report = value_portfolio(Portfolio(positions), NOW, inputs, vol_buckets=(0.2, 0.3))
    assert set(report.totals) == {"USD", "EUR"}
    assert len(report.by_underlying) == 3
    assert len(report.by_expiry) == 3
    assert len(report.by_strike) == 3
    assert report.positions[0].vol_bucket == 1
    assert report.positions[2].vol_bucket == 2
    assert report.positions[3].vol_bucket == 0
    assert report.by_vol_bucket[("USD", 1)] == report.by_underlying[USD]
    assert report.by_expiry[("USD", first.expiry)] == report.by_underlying[USD]
    assert report.by_strike[("USD", USD, 100)] == report.by_underlying[USD]
    row = report.positions[0]
    assert row.exposure.value == pytest.approx(
        300 * float(price("call", 100, 100, 1, 0.03, 0.2, 0.01))
    )
    assert row.exposure.delta == pytest.approx(300 * 100 * row.greeks.delta)
    assert row.exposure.gamma == pytest.approx(300 * 100**2 * row.greeks.gamma)
    assert row.exposure.vanna == pytest.approx(300 * 100 * row.greeks.vanna)
    assert row.exposure.speed == pytest.approx(300 * 100**3 * row.greeks.speed)
    assert row.exposure.color == pytest.approx(300 * 100**2 * row.greeks.color)
    assert report.totals["USD"] == (
        report.by_underlying[USD] + report.by_underlying[OTHER]
    )
    assert report.totals["USD"].gross_notional == 4 * 100 * 100 + 4 * 10 * 120
    assert report.totals["USD"].gross_contracts == 8
    with pytest.raises(TypeError):
        report.totals["USD"] = Exposure()
    assert value_portfolio(Portfolio(()), NOW, {}).totals == {}
    per_contract = replace(first, premium_basis="per_contract")
    assert (
        value_portfolio(Portfolio((Position(per_contract, 3),)), NOW, inputs).totals[
            "USD"
        ]
        == row.exposure
    )


def test_scenario_reprices_sticky_forward_moneyness_smile() -> None:
    option = contract(strike=108)
    portfolio = Portfolio((Position(option, -2.5),))
    inputs = {USD: RiskInputs(100, Smile(), 0.03, 0.01)}
    shock = RiskScenario(
        spot_return=0.1,
        volatility_shift=0.02,
        elapsed_years=0.25,
        rate_shift=-0.01,
        skew_shift=0.03,
        curvature_shift=0.08,
    )
    report = scenario_portfolio(portfolio, NOW, inputs, shock)
    row = report.positions[0]
    k = log(108 / 110) - (0.02 - 0.01) * 0.75
    sigma = Smile().volatility(k, 0.75) + 0.02 + 0.03 * k + 0.04 * k**2
    expected = advanced_greeks("call", 110, 108, 0.75, 0.02, sigma, 0.01)
    assert row.spot == pytest.approx(110)
    assert row.time_to_expiry == 0.75
    assert row.volatility == pytest.approx(sigma)
    assert row.greeks.price == pytest.approx(expected.price)
    assert report.totals["USD"].value == pytest.approx(-250 * expected.price)
    assert scenario_portfolio(portfolio, NOW, inputs, RiskScenario()) == (
        value_portfolio(portfolio, NOW, inputs)
    )
    flat = scenario_portfolio(
        portfolio,
        NOW,
        {USD: RiskInputs(100, 0.2)},
        RiskScenario(volatility_shift=0.05),
    )
    assert flat.positions[0].volatility == 0.25


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spot_return": -1},
        {"elapsed_years": -1},
        {"rate_shift": float("nan")},
        {"skew_shift": float("inf")},
    ],
)
def test_invalid_scenarios(kwargs: dict[str, float]) -> None:
    with pytest.raises(InputValidationError):
        RiskScenario(**kwargs)


@pytest.mark.parametrize("buckets", [(0.3, 0.2), (0.2, 0.2), (0,), (float("nan"),)])
def test_bad_vol_buckets(buckets: tuple[float, ...]) -> None:
    with pytest.raises(InputValidationError):
        value_portfolio(Portfolio(()), NOW, {}, vol_buckets=buckets)


@pytest.mark.parametrize(
    "changes",
    [
        {"exercise_style": "american"},
        {"quote_currency": "EUR"},
        {"settlement_currency": "EUR"},
    ],
)
def test_unsupported_contract_risks(changes: dict[str, str]) -> None:
    with pytest.raises(InputValidationError, match="single-currency European"):
        value_portfolio(Portfolio((Position(contract(**changes), 1),)), NOW, {})


def test_missing_expired_and_bad_market_inputs() -> None:
    portfolio = Portfolio((Position(contract(), 1),))
    with pytest.raises(InputValidationError, match="missing risk inputs"):
        value_portfolio(portfolio, NOW, {})
    for bad_vol in (0, -0.2, True, object()):
        with pytest.raises(InputValidationError):
            RiskInputs(100, bad_vol)
    for market_args in ((0, 0.2), (100, 0.2, float("nan"))):
        with pytest.raises(InputValidationError):
            RiskInputs(*market_args)
    inputs = {USD: RiskInputs(100, 0.2)}
    with pytest.raises(InputValidationError, match="remaining maturity"):
        scenario_portfolio(portfolio, NOW, inputs, RiskScenario(elapsed_years=1))
    with pytest.raises(InputValidationError, match="volatility"):
        scenario_portfolio(portfolio, NOW, inputs, RiskScenario(volatility_shift=-0.3))
    with pytest.raises(InputValidationError):
        value_portfolio(portfolio, NOW + timedelta(days=366), inputs)
    with pytest.raises(InputValidationError):
        value_portfolio(portfolio, NOW.replace(tzinfo=None), inputs)


def test_deterministic_pnl_attribution_trades_and_cash_reconcile() -> None:
    call = contract()
    put = contract(symbol="ABC-P105", strike=105, option_type="put", multiplier=10)
    portfolio = Portfolio((Position(call, 2), Position(put, -3)))
    start_inputs = {USD: RiskInputs(100, 0.2, 0.03, 0.01)}
    end_inputs = {USD: RiskInputs(102, 0.22, 0.032, 0.01)}
    end = NOW + timedelta(days=7)
    buy = Trade(call, 1, 9, fees=2)
    sell = Trade(replace(put, premium_basis="per_contract"), -2, 120, fees=1)
    result = attribute_pnl(
        portfolio,
        NOW,
        end,
        start_inputs,
        end_inputs,
        trades=(buy, sell),
        carry_cash={"USD": -7, "EUR": 3},
    )
    opening = value_portfolio(portfolio, NOW, start_inputs)
    final_positions = (
        *portfolio.positions,
        Position(buy.contract, 1),
        Position(sell.contract, -2),
    )
    final = value_portfolio(Portfolio(final_positions), end, end_inputs)
    pnl = result["USD"]
    assert buy.cash == -902
    assert sell.cash == 239
    assert pnl.trade_cash == -663
    assert pnl.final_value == pytest.approx(final.totals["USD"].value)
    assert pnl.total_pnl == pytest.approx(
        final.totals["USD"].value - opening.totals["USD"].value - 663 - 7
    )
    assert pnl.delta == pytest.approx(opening.totals["USD"].delta * 0.02)
    assert pnl.gamma == pytest.approx(0.5 * opening.totals["USD"].gamma * 0.02**2)
    assert pnl.vega == pytest.approx(opening.totals["USD"].vega * 0.02)
    assert pnl.theta + pnl.carry == pytest.approx(
        opening.totals["USD"].theta * 7 / 365 - 7
    )
    assert pnl.total_pnl == pytest.approx(
        pnl.delta
        + pnl.gamma
        + pnl.vega
        + pnl.theta
        + pnl.carry
        + pnl.trades
        + pnl.residual
    )
    assert abs(pnl.residual) > 0.01
    assert result["EUR"].total_pnl == 3
    assert result["EUR"].carry == 3
    assert result["EUR"].residual == 0
    assert result == attribute_pnl(
        portfolio,
        NOW,
        end,
        start_inputs,
        end_inputs,
        trades=(buy, sell),
        carry_cash={"USD": -7, "EUR": 3},
    )


def test_pnl_no_change_trades_only_and_empty() -> None:
    option = contract()
    portfolio = Portfolio((Position(option, -1),))
    inputs = {USD: RiskInputs(100, Smile())}
    pnl = attribute_pnl(portfolio, NOW, NOW, inputs, inputs)["USD"]
    assert pnl.total_pnl == pnl.residual == pnl.trades == pnl.delta == 0
    assert attribute_pnl(Portfolio(()), NOW, NOW, {}, {}) == {}
    trade = Trade(option, -1, 10)
    pnl = attribute_pnl(Portfolio(()), NOW, NOW, {}, inputs, trades=(trade,))["USD"]
    assert pnl.total_pnl == pnl.trades
    assert pnl.residual == 0
    assert pnl.trade_cash == 1000
    with pytest.raises(InputValidationError):
        attribute_pnl(
            portfolio,
            NOW,
            NOW,
            inputs,
            inputs,
            carry_cash={"USD": float("nan")},
        )
    with pytest.raises(InputValidationError):
        attribute_pnl(portfolio, NOW, NOW - timedelta(days=1), inputs, inputs)


@pytest.mark.parametrize(
    "field,value",
    [("price", -1), ("fees", -1), ("contracts", float("inf"))],
)
def test_trade_validation(field: str, value: float) -> None:
    kwargs = {"contracts": 1, "price": 10, "fees": 0}
    kwargs[field] = value
    with pytest.raises(InputValidationError):
        Trade(contract(), **kwargs)


@pytest.mark.parametrize("sign", [-1, 1])
def test_limits_use_absolute_exposure_with_actionable_failures(sign: int) -> None:
    report = value_portfolio(
        Portfolio((Position(contract(), sign * 3),)), NOW, {USD: RiskInputs(100, 0.2)}
    )
    limits = RiskLimits(2, 1000, 1, 1, 1)
    breaches = check_limits(report, limits)
    assert [breach.metric for breach in breaches] == [
        "position",
        "gross_notional",
        "delta",
        "gamma",
        "vega",
    ]
    assert all(breach.actual > breach.limit for breach in breaches)
    assert "reduce absolute exposure" in breaches[0].message
    assert "ABC-C100" in breaches[0].scope
    with pytest.raises(RiskLimitError, match="exceeds") as error:
        check_limits(report, limits, raise_on_breach=True)
    assert error.value.breaches == breaches
    assert check_limits(report, RiskLimits(), raise_on_breach=True) == ()
    exposure = report.totals["USD"]
    equal = RiskLimits(
        3,
        exposure.gross_notional,
        abs(exposure.delta),
        abs(exposure.gamma),
        abs(exposure.vega),
    )
    assert check_limits(report, equal) == ()


def test_net_position_and_gross_notional_limits_have_distinct_economics() -> None:
    option = contract()
    report = value_portfolio(
        Portfolio((Position(option, 3), Position(option, -3))),
        NOW,
        {USD: RiskInputs(100, 0.2)},
    )
    breaches = check_limits(report, RiskLimits(0, 1, 0, 0, 0))
    assert len(breaches) == 1
    assert breaches[0].metric == "gross_notional"
    assert breaches[0].actual == 60000
    assert report.totals["USD"].value == 0
    assert check_limits(value_portfolio(Portfolio(()), NOW, {}), RiskLimits(0)) == ()


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_limit_values(value: float) -> None:
    with pytest.raises(InputValidationError):
        RiskLimits(max_vega=value)


def test_source_and_scenario_numerical_boundaries() -> None:
    class BadSmile:
        def volatility(self, log_moneyness: float, time_to_expiry: float) -> float:
            return float("nan")

    portfolio = Portfolio((Position(contract(), 1),))
    with pytest.raises(InputValidationError, match="source volatility"):
        value_portfolio(portfolio, NOW, {USD: RiskInputs(100, BadSmile())})
    with pytest.raises(InputValidationError, match="scenario spot"):
        scenario_portfolio(
            portfolio,
            NOW,
            {USD: RiskInputs(1e308, 0.2)},
            RiskScenario(spot_return=1e308),
        )
    with pytest.raises(InputValidationError, match="scenario rate"):
        scenario_portfolio(
            portfolio,
            NOW,
            {USD: RiskInputs(100, 0.2, 1e308)},
            RiskScenario(rate_shift=1e308),
        )


def test_limits_do_not_offset_currencies() -> None:
    dollar = contract()
    euro = contract(
        symbol="EU-C100",
        underlying=EUR,
        quote_currency="EUR",
        settlement_currency="EUR",
    )
    report = value_portfolio(
        Portfolio((Position(dollar, 1), Position(euro, -1))),
        NOW,
        {USD: RiskInputs(100, 0.2), EUR: RiskInputs(100, 0.2)},
    )
    breaches = check_limits(report, RiskLimits(max_vega=0))
    assert {breach.scope for breach in breaches} == {"USD", "EUR"}
    assert len(breaches) == 2
