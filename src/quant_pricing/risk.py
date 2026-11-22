"""Scalar BSM risks, explicit-market portfolios, scenarios and P&L attribution.

All volatility/rate changes are absolute decimals, never percent/vol points.
Time uses ACT/365F years. Calendar derivatives advance time (decrease maturity).
Full higher-order risk requires strictly positive maturity and volatility; expiry
and deterministic exercise-boundary limits are deliberately rejected rather than
presented as ordinary finite derivatives. Only linear, single-currency European
contracts are supported. No quotes, FX, curves or market data are fetched.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime
from itertools import pairwise
from math import exp, isfinite, log, pi, sqrt
from numbers import Real
from types import MappingProxyType
from typing import Protocol

from quant_pricing.black_scholes import OptionType, value_and_greeks
from quant_pricing.conventions import utc_timestamp, year_fraction
from quant_pricing.exceptions import InputValidationError
from quant_pricing.market import OptionContract, Portfolio, Position, Underlying


class VolatilitySource(Protocol):
    """Absolute annual volatility at k=log(K/F), ACT/365F maturity T."""

    def volatility(self, log_moneyness: float, time_to_expiry: float) -> float: ...


def _finite(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise InputValidationError(f"{name} must be finite")


def _positive(value: float, name: str) -> None:
    _finite(value, name)
    if value <= 0:
        raise InputValidationError(f"{name} must be strictly positive")


@dataclass(frozen=True, slots=True)
class RiskGreeks:
    """Per underlying unit, in quote currency C and spot unit S.

    delta: C/S; gamma: C/S²; vega: C/absolute vol; theta: C/year;
    rho: C/absolute rate; vanna: C/(S*vol); volga (vomma): C/vol²;
    charm: C/(S*calendar year); speed: C/S³;
    color: C/(S²*calendar year). No automatic 0.01 or day conversion.
    """

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float
    volga: float
    charm: float
    speed: float
    color: float

    @property
    def vomma(self) -> float:
        """Standard synonym for volga, with identical absolute-vol units."""
        return self.volga


def advanced_greeks(
    option_type: OptionType | str,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> RiskGreeks:
    """Analytic partial derivatives holding all other BSM inputs fixed.

    These are fixed-vol partials, not smile-adjusted total derivatives. Charm
    and color are calendar-time derivatives of delta and gamma respectively.
    Undefined expiry/zero-vol higher derivatives are explicitly unsupported.
    """
    for name, value in (
        ("spot", spot),
        ("strike", strike),
        ("time_to_expiry", time_to_expiry),
        ("volatility", volatility),
    ):
        _positive(value, name)
    _finite(rate, "rate")
    _finite(dividend_yield, "dividend_yield")
    valuation = value_and_greeks(
        option_type, spot, strike, time_to_expiry, rate, volatility, dividend_yield
    )
    base = valuation.greeks
    try:
        root_time = sqrt(time_to_expiry)
        width = volatility * root_time
        drift = rate - dividend_yield + 0.5 * volatility**2
        d1 = (log(spot) - log(strike) + drift * time_to_expiry) / width
        d2 = d1 - width
        density = exp(-dividend_yield * time_to_expiry - 0.5 * d1**2)
        density /= sqrt(2.0 * pi)
        d1_time = drift / width - d1 / (2.0 * time_to_expiry)
        result = RiskGreeks(
            price=float(valuation.price),
            delta=float(base.delta),
            gamma=float(base.gamma),
            vega=float(base.vega),
            theta=float(base.theta),
            rho=float(base.rho),
            vanna=-density * d2 / volatility,
            volga=float(base.vega) * d1 * d2 / volatility,
            charm=dividend_yield * float(base.delta) - density * d1_time,
            speed=-float(base.gamma) / spot * (1.0 + d1 / width),
            color=float(base.gamma)
            * (dividend_yield + 0.5 / time_to_expiry + d1 * d1_time),
        )
    except (OverflowError, ZeroDivisionError) as error:
        raise InputValidationError("risk inputs exceed numerical range") from error
    for field in fields(result):
        _finite(getattr(result, field.name), field.name)
    return result


@dataclass(frozen=True, slots=True)
class RiskInputs:
    """Explicit spot, continuously compounded rates and scalar/smile volatility."""

    spot: float
    volatility: float | VolatilitySource
    rate: float = 0.0
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        _positive(self.spot, "spot")
        _finite(self.rate, "rate")
        _finite(self.dividend_yield, "dividend_yield")
        if isinstance(self.volatility, (int, float)):
            _positive(self.volatility, "volatility")
        elif not callable(getattr(self.volatility, "volatility", None)):
            raise InputValidationError(
                "volatility must be a scalar or VolatilitySource"
            )


@dataclass(frozen=True, slots=True)
class RiskScenario:
    """Parallel shocks across positions; spot_return is fractional, others absolute.

    Sticky forward moneyness: preserve the source function, querying it at the
    shocked k=log(K/S)-(r-q)T and remaining T. Add volatility_shift + skew_shift*k +
    curvature_shift*k²/2. A scalar source is a flat smile. Elapsed years cannot
    move through expiry. Rate shifts leave the explicit dividend yield fixed.
    """

    spot_return: float = 0.0
    volatility_shift: float = 0.0
    elapsed_years: float = 0.0
    rate_shift: float = 0.0
    skew_shift: float = 0.0
    curvature_shift: float = 0.0

    def __post_init__(self) -> None:
        for field in fields(self):
            _finite(getattr(self, field.name), field.name)
        if self.spot_return <= -1 or self.elapsed_years < 0:
            raise InputValidationError("spot_return must exceed -1; elapsed_years >= 0")


@dataclass(frozen=True, slots=True)
class Exposure:
    """Signed monetary risks, safely additive only within one currency.

    delta=S*dV/dS and gamma=S²*d²V/dS² are currency per fractional spot
    change and its square (the gamma P&L term still needs 1/2). vanna and
    charm are multiplied by S, speed by S³, color by S². Other risks retain
    RiskGreeks units. Gross notional is sum(abs(quantity)*multiplier*spot),
    not option premium or delta-adjusted notional. gross_contracts is unnetted.
    """

    value: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    vanna: float = 0.0
    volga: float = 0.0
    charm: float = 0.0
    speed: float = 0.0
    color: float = 0.0
    gross_notional: float = 0.0
    gross_contracts: float = 0.0

    def __add__(self, other: Exposure) -> Exposure:
        return Exposure(
            **{
                field.name: getattr(self, field.name) + getattr(other, field.name)
                for field in fields(self)
            }
        )


@dataclass(frozen=True, slots=True)
class PositionRisk:
    position: Position
    spot: float
    time_to_expiry: float
    volatility: float
    greeks: RiskGreeks
    exposure: Exposure
    vol_bucket: int


@dataclass(frozen=True, slots=True)
class RiskReport:
    """Currency-separated totals; bucket i has bounds [edge[i-1], edge[i])."""

    as_of: datetime
    scenario: RiskScenario
    positions: tuple[PositionRisk, ...]
    totals: Mapping[str, Exposure]
    by_underlying: Mapping[Underlying, Exposure]
    by_expiry: Mapping[tuple[str, datetime], Exposure]
    by_strike: Mapping[tuple[str, Underlying, float], Exposure]
    by_vol_bucket: Mapping[tuple[str, int], Exposure]
    vol_buckets: tuple[float, ...]


def _position_risk(
    position: Position,
    as_of: datetime,
    inputs: Mapping[Underlying, RiskInputs],
    scenario: RiskScenario,
    buckets: tuple[float, ...],
) -> PositionRisk:
    contract = position.contract
    currency = contract.underlying.currency
    if contract.exercise_style != "european" or (
        contract.quote_currency != currency or contract.settlement_currency != currency
    ):
        raise InputValidationError("risk requires single-currency European contracts")
    try:
        market = inputs[contract.underlying]
    except KeyError as error:
        raise InputValidationError(
            f"missing risk inputs for {contract.underlying.symbol} ({currency})"
        ) from error
    maturity = year_fraction(as_of, contract.expiry) - scenario.elapsed_years
    _positive(maturity, "remaining maturity")
    spot = market.spot * (1.0 + scenario.spot_return)
    rate = market.rate + scenario.rate_shift
    _positive(spot, "scenario spot")
    _finite(rate, "scenario rate")
    k = log(contract.strike) - log(spot) - (rate - market.dividend_yield) * maturity
    source = market.volatility
    volatility = (
        float(source)
        if isinstance(source, (int, float))
        else source.volatility(k, maturity)
    )
    _positive(volatility, "source volatility")
    volatility += (
        scenario.volatility_shift
        + scenario.skew_shift * k
        + 0.5 * scenario.curvature_shift * k * k
    )
    greek = advanced_greeks(
        contract.option_type,
        spot,
        contract.strike,
        maturity,
        rate,
        volatility,
        market.dividend_yield,
    )
    units = position.underlying_units
    exposure = Exposure(
        value=units * greek.price,
        delta=units * spot * greek.delta,
        gamma=units * spot**2 * greek.gamma,
        vega=units * greek.vega,
        theta=units * greek.theta,
        rho=units * greek.rho,
        vanna=units * spot * greek.vanna,
        volga=units * greek.volga,
        charm=units * spot * greek.charm,
        speed=units * spot**3 * greek.speed,
        color=units * spot**2 * greek.color,
        gross_notional=abs(units) * spot,
        gross_contracts=abs(position.contracts),
    )
    for field in fields(exposure):
        _finite(getattr(exposure, field.name), field.name)
    return PositionRisk(
        position,
        spot,
        maturity,
        volatility,
        greek,
        exposure,
        bisect_right(buckets, volatility),
    )


def value_portfolio(
    portfolio: Portfolio,
    as_of: datetime,
    inputs: Mapping[Underlying, RiskInputs],
    *,
    vol_buckets: tuple[float, ...] = (),
) -> RiskReport:
    """Price signed contracts, multiplying every per-unit Greek by the multiplier."""
    return scenario_portfolio(
        portfolio, as_of, inputs, RiskScenario(), vol_buckets=vol_buckets
    )


def scenario_portfolio(
    portfolio: Portfolio,
    as_of: datetime,
    inputs: Mapping[Underlying, RiskInputs],
    scenario: RiskScenario,
    *,
    vol_buckets: tuple[float, ...] = (),
) -> RiskReport:
    """Fully reprice a deterministic input shock; no Taylor approximation."""
    as_of = utc_timestamp(as_of)
    buckets = tuple(vol_buckets)
    for edge in buckets:
        _positive(edge, "vol bucket edge")
    if any(left >= right for left, right in pairwise(buckets)):
        raise InputValidationError("vol bucket edges must be strictly increasing")
    rows = tuple(
        _position_risk(position, as_of, inputs, scenario, buckets)
        for position in portfolio.positions
    )
    totals: dict[str, Exposure] = {}
    underlyings: dict[Underlying, Exposure] = {}
    expiries: dict[tuple[str, datetime], Exposure] = {}
    strikes: dict[tuple[str, Underlying, float], Exposure] = {}
    volatility: dict[tuple[str, int], Exposure] = {}
    for row in rows:
        contract = row.position.contract
        currency = contract.quote_currency
        expiry_key = (currency, contract.expiry)
        strike_key = (currency, contract.underlying, contract.strike)
        vol_key = (currency, row.vol_bucket)
        totals[currency] = totals.get(currency, Exposure()) + row.exposure
        underlyings[contract.underlying] = (
            underlyings.get(contract.underlying, Exposure()) + row.exposure
        )
        expiries[expiry_key] = expiries.get(expiry_key, Exposure()) + row.exposure
        strikes[strike_key] = strikes.get(strike_key, Exposure()) + row.exposure
        volatility[vol_key] = volatility.get(vol_key, Exposure()) + row.exposure
    return RiskReport(
        as_of,
        scenario,
        rows,
        MappingProxyType(totals),
        MappingProxyType(underlyings),
        MappingProxyType(expiries),
        MappingProxyType(strikes),
        MappingProxyType(volatility),
        buckets,
    )


@dataclass(frozen=True, slots=True)
class Trade:
    """End-timestamp execution: signed contracts, price in contract premium_basis.

    Buy quantities are positive and consume cash; sell quantities are negative
    and generate cash. Fees are nonnegative currency cash outflows. Trades are
    marked at the end market, so their P&L is mark minus execution cost and fees.
    """

    contract: OptionContract
    contracts: float
    price: float
    fees: float = 0.0

    def __post_init__(self) -> None:
        _finite(self.contracts, "trade contracts")
        _finite(self.price, "trade price")
        _finite(self.fees, "trade fees")
        if self.price < 0 or self.fees < 0:
            raise InputValidationError("trade price and fees must be nonnegative")

    @property
    def cash(self) -> float:
        return (
            -(
                self.contracts
                * self.contract.multiplier
                * self.contract.premium_per_unit(self.price)
            )
            - self.fees
        )


@dataclass(frozen=True, slots=True)
class PnlAttribution:
    """Currency amounts, reconciled exactly by the explicit residual.

    Delta/Gamma/Vega use opening fixed-vol partials and realized absolute input
    changes; Gamma includes 1/2. Theta is diffusion-only -sigma²*S²*Gamma/2*dt.
    Carry is the remaining opening calendar theta times dt plus supplied signed
    carry cash (e.g. financing), so theta+model carry equals full calendar theta.
    Trades are end marked value plus signed execution cash. Residual contains
    rates, higher orders, interactions, and all approximation error; it is not
    claimed as unexplained trading alpha. Total = final-initial+trade_cash+cash.
    """

    initial_value: float
    final_value: float
    trade_cash: float
    carry_cash: float
    total_pnl: float
    delta: float
    gamma: float
    vega: float
    theta: float
    carry: float
    trades: float
    residual: float


def attribute_pnl(
    portfolio: Portfolio,
    start: datetime,
    end: datetime,
    start_inputs: Mapping[Underlying, RiskInputs],
    end_inputs: Mapping[Underlying, RiskInputs],
    *,
    trades: tuple[Trade, ...] = (),
    carry_cash: Mapping[str, float] | None = None,
) -> Mapping[str, PnlAttribution]:
    """Attribute opening holdings and end-time trades without implicit cash flows."""
    elapsed = year_fraction(start, end)
    opening = value_portfolio(portfolio, start, start_inputs)
    closing = value_portfolio(portfolio, end, end_inputs)
    trade_report = value_portfolio(
        Portfolio(tuple(Position(trade.contract, trade.contracts) for trade in trades)),
        end,
        end_inputs,
    )
    cash = dict(carry_cash or {})
    for amount in cash.values():
        _finite(amount, "carry cash")
    components: dict[str, list[float]] = {}
    for before, after in zip(opening.positions, closing.positions, strict=True):
        currency = before.position.contract.quote_currency
        parts = components.setdefault(currency, [0.0] * 5)
        units = before.position.underlying_units
        ds = after.spot - before.spot
        dv = after.volatility - before.volatility
        diffusion = -0.5 * before.volatility**2 * before.exposure.gamma
        parts[0] += units * before.greeks.delta * ds
        parts[1] += 0.5 * units * before.greeks.gamma * ds**2
        parts[2] += units * before.greeks.vega * dv
        parts[3] += diffusion * elapsed
        parts[4] += (before.exposure.theta - diffusion) * elapsed
    trade_cash: dict[str, float] = {}
    for trade in trades:
        currency = trade.contract.quote_currency
        trade_cash[currency] = trade_cash.get(currency, 0.0) + trade.cash
    result: dict[str, PnlAttribution] = {}
    currencies = sorted(set(opening.totals) | set(trade_report.totals) | set(cash))
    for currency in currencies:
        initial = opening.totals.get(currency, Exposure()).value
        held_final = closing.totals.get(currency, Exposure()).value
        traded_value = trade_report.totals.get(currency, Exposure()).value
        execution_cash = trade_cash.get(currency, 0.0)
        extra_cash = cash.get(currency, 0.0)
        delta, gamma, vega, theta, carry = components.get(currency, [0.0] * 5)
        carry += extra_cash
        trade_pnl = traded_value + execution_cash
        final = held_final + traded_value
        total = final - initial + execution_cash + extra_cash
        explained = delta + gamma + vega + theta + carry + trade_pnl
        result[currency] = PnlAttribution(
            initial,
            final,
            execution_cash,
            extra_cash,
            total,
            delta,
            gamma,
            vega,
            theta,
            carry,
            trade_pnl,
            total - explained,
        )
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Nonnegative absolute limits, applied independently in each currency.

    Position is absolute net contracts per exact contract, not signed long-only.
    Notional is gross; delta/gamma/vega bound absolute net currency totals using
    Exposure units. No offset across currencies is ever permitted.
    """

    max_position: float | None = None
    max_notional: float | None = None
    max_delta: float | None = None
    max_gamma: float | None = None
    max_vega: float | None = None

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                _finite(value, field.name)
                if value < 0:
                    raise InputValidationError(f"{field.name} must be nonnegative")


@dataclass(frozen=True, slots=True)
class LimitBreach:
    metric: str
    scope: str
    actual: float
    limit: float

    @property
    def message(self) -> str:
        return (
            f"{self.scope}: |{self.metric}|={self.actual:g} exceeds {self.limit:g}; "
            f"reduce absolute exposure by at least {self.actual - self.limit:g}"
        )


class RiskLimitError(InputValidationError):
    """Actionable breaches retained alongside the human-readable explanation."""

    def __init__(self, breaches: tuple[LimitBreach, ...]) -> None:
        self.breaches = breaches
        super().__init__("; ".join(breach.message for breach in breaches))


def check_limits(
    report: RiskReport, limits: RiskLimits, *, raise_on_breach: bool = False
) -> tuple[LimitBreach, ...]:
    """Return every violation in deterministic order; equality is within limit."""
    breaches: list[LimitBreach] = []
    contracts: dict[OptionContract, float] = {}
    for row in report.positions:
        position = row.position
        contracts[position.contract] = (
            contracts.get(position.contract, 0.0) + position.contracts
        )
    if limits.max_position is not None:
        for contract, quantity in contracts.items():
            if abs(quantity) > limits.max_position:
                breaches.append(
                    LimitBreach(
                        "position",
                        f"{contract.symbol} ({contract.quote_currency})",
                        abs(quantity),
                        limits.max_position,
                    )
                )
    for currency, exposure in report.totals.items():
        for metric, limit in (
            ("gross_notional", limits.max_notional),
            ("delta", limits.max_delta),
            ("gamma", limits.max_gamma),
            ("vega", limits.max_vega),
        ):
            value = abs(getattr(exposure, metric))
            if limit is not None and value > limit:
                breaches.append(LimitBreach(metric, currency, value, limit))
    result = tuple(breaches)
    if raise_on_breach and result:
        raise RiskLimitError(result)
    return result
