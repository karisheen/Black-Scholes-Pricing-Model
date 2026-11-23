"""Indicative BBO butterflies, separate from midpoint/IV surface diagnostics.

Only declared adjacent strikes at one exact expiry are checked. Costs are per
underlying unit of the short body (USDC per BTC for BTC_USDC). Displayed size is
indicative: fees, simultaneous fills, size steps and financing are not assured.
This is neither a full-chain nor a different-calendar arbitrage claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import fsum, isfinite, ulp
from typing import Literal

from quant_pricing.black_scholes import OptionType
from quant_pricing.exceptions import InputValidationError
from quant_pricing.market import Forward, OptionContract, SnapshotQuote

Side = Literal["buy", "sell"]


@dataclass(frozen=True, slots=True)
class ButterflyOptionLeg:
    symbol: str
    option_type: str
    side: Side
    signed_weight: float
    price: float
    displayed_size: float
    multiplier: float


@dataclass(frozen=True, slots=True)
class ButterflyFuturesHedge:
    symbol: str
    side: Side
    signed_weight: float
    price: float
    displayed_size: float


@dataclass(frozen=True, slots=True)
class ButterflyPackage:
    option_legs: tuple[ButterflyOptionLeg, ...]
    futures_hedge: ButterflyFuturesHedge | None
    option_entry_cost: float
    parity_adjustment: float
    net_entry_cost: float
    displayed_size_cap: float
    negative_cost_candidate: bool
    cost_tolerance: float


@dataclass(frozen=True, slots=True)
class ButterflyCheck:
    strikes: tuple[float, float, float]
    weights: tuple[float, float, float]
    combinations_checked: int
    combinations_skipped: int
    flags: tuple[str, ...]
    best_package: ButterflyPackage | None
    negative_cost_candidate: bool | None


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isfinite(value) or value <= 0:
        raise InputValidationError(f"{name} must be finite and positive")


def _issues(row: SnapshotQuote, side: Side) -> tuple[str, ...]:
    issues: list[str] = list(row.flags)
    if row.sequence_gaps:
        issues.append("sequence_gap")
    quote = row.quote
    if quote is None:
        issues.append("missing")
    else:
        if not all(isfinite(p) and p >= 0 for p in (quote.bid, quote.ask)):
            issues.append("invalid_price")
        elif quote.bid > quote.ask:
            issues.append("crossed")
        if isinstance(row.instrument, Forward) and min(quote.bid, quote.ask) <= 0:
            issues.append("invalid_price")
        if not all(isfinite(s) and s >= 0 for s in (quote.bid_size, quote.ask_size)):
            issues.append("invalid_size")
        size = quote.ask_size if side == "buy" else quote.bid_size
        if size == 0:
            issues.append(f"empty_{'ask' if side == 'buy' else 'bid'}_size")
    return tuple(f"{row.instrument.symbol}:{issue}" for issue in dict.fromkeys(issues))


def _package(
    legs: tuple[SnapshotQuote, ...],
    weights: tuple[Fraction, Fraction, Fraction],
    future: SnapshotQuote,
    discount_factor: float,
) -> tuple[ButterflyPackage | None, tuple[str, ...]]:
    option_legs: list[ButterflyOptionLeg] = []
    costs: list[float] = []
    caps: list[float] = []
    put_weight = Fraction(0)
    put_strikes = Fraction(0)
    for row, exact_weight in zip(legs, weights, strict=True):
        contract = row.instrument
        assert isinstance(contract, OptionContract)
        side: Side = "buy" if exact_weight > 0 else "sell"
        issues = _issues(row, side)
        if issues:
            return None, issues
        quote = row.quote
        assert quote is not None
        weight = float(exact_weight)
        if weight == 0:
            return None, ("unrepresentable_weight",)
        raw_price = quote.ask if side == "buy" else quote.bid
        price = contract.premium_per_unit(raw_price)
        size = quote.ask_size if side == "buy" else quote.bid_size
        costs.append(weight * price)
        caps.append(size * contract.multiplier / abs(weight))
        option_legs.append(
            ButterflyOptionLeg(
                contract.symbol,
                contract.option_type.value,
                side,
                weight,
                price,
                size,
                contract.multiplier,
            )
        )
        if contract.option_type == OptionType.PUT:
            put_weight += exact_weight
            put_strikes += exact_weight * Fraction(float(contract.strike))

    hedge = None
    forward_term = 0.0
    if put_weight:
        side = "buy" if put_weight > 0 else "sell"
        issues = _issues(future, side)
        if issues:
            return None, issues
        future_quote = future.quote
        assert future_quote is not None
        hedge_weight = float(put_weight)
        if hedge_weight == 0:
            return None, ("unrepresentable_weight",)
        forward_price = future_quote.ask if side == "buy" else future_quote.bid
        forward_size = future_quote.ask_size if side == "buy" else future_quote.bid_size
        hedge = ButterflyFuturesHedge(
            future.instrument.symbol, side, hedge_weight, forward_price, forward_size
        )
        forward_term = hedge_weight * forward_price
        caps.append(forward_size / abs(hedge_weight))

    strike_term = float(put_strikes)
    if not all(isfinite(value) for value in (*costs, forward_term, strike_term)):
        return None, ("nonfinite_calculation",)
    try:
        option_cost = fsum(costs)
        adjustment = discount_factor * fsum((forward_term, -strike_term))
        cost = fsum((option_cost, adjustment))
    except (OverflowError, ValueError):
        return None, ("nonfinite_calculation",)
    # Absolute floor plus a cancellation-aware allowance of 32 floating ULPs.
    scale = max(
        *(abs(value) for value in costs),
        abs(discount_factor * forward_term),
        abs(discount_factor * strike_term),
    )
    tolerance = max(1e-10, 32 * ulp(scale))
    cap = min(caps)
    values = (option_cost, adjustment, cost, cap, tolerance)
    if not all(isfinite(value) for value in values) or cap <= 0:
        return None, ("nonfinite_calculation",)
    return (
        ButterflyPackage(
            tuple(option_legs),
            hedge,
            option_cost,
            adjustment,
            cost,
            cap,
            cost < -tolerance,
            tolerance,
        ),
        (),
    )


def check_quoted_butterflies(
    rows: tuple[SnapshotQuote, ...],
    future: SnapshotQuote,
    discount_factor: float,
) -> tuple[ButterflyCheck, ...]:
    """Check every call/put package on adjacent *declared* strike triples.

    Snapshot freshness/availability flags are authoritative; this function has
    no clock, and must receive point-in-time snapshots, not unfiltered events.
    Invalid books and sequence gaps are excluded, never replaced by marks or
    IVs. Only the traded side needs strictly positive displayed quantity.

    Wings have weights (K3-K2)/(K3-K1), (K2-K1)/(K3-K1); the body is -1.
    Convert each signed put through C=P+D*(F-K), net the futures hedge exactly,
    then cross its ask for positive weight or bid for negative weight. A missing
    futures book therefore does not prevent all-call or all-put packages.
    Option premium bases are normalized per underlying unit; option sizes are
    contracts and futures sizes are underlying units. No other scales or FX
    conversion are inferred. A candidate means cost < -cost_tolerance, before
    fees and fill risk, not a guaranteed executable or financed profit.
    """
    _positive(discount_factor, "discount_factor")
    forward = future.instrument
    if not isinstance(forward, Forward):
        raise InputValidationError("future must be a matching dated Forward")
    by_strike: dict[float, dict[OptionType, SnapshotQuote]] = {}
    for row in rows:
        contract = row.instrument
        if not isinstance(contract, OptionContract):
            raise InputValidationError("butterfly rows must be OptionContract rows")
        if contract.option_type not in (OptionType.CALL, OptionType.PUT):
            raise InputValidationError("butterfly option type must be call or put")
        _positive(contract.strike, "strike")
        _positive(contract.multiplier, "multiplier")
        if contract.exercise_style != "european" or contract.settlement_style != "cash":
            raise InputValidationError("butterflies require European cash options")
        if not (
            contract.quote_currency
            == contract.settlement_currency
            == contract.underlying.currency
        ):
            raise InputValidationError("butterflies require one currency without FX")
        if (contract.underlying, contract.expiry) != (
            forward.underlying,
            forward.expiry,
        ):
            raise InputValidationError(
                "options and future must match underlying/expiry"
            )
        choices = by_strike.setdefault(float(contract.strike), {})
        if contract.option_type in choices:
            raise InputValidationError("duplicate option type and strike")
        choices[contract.option_type] = row

    strikes = sorted(by_strike)
    checks: list[ButterflyCheck] = []
    for low, middle, high in zip(strikes, strikes[1:], strikes[2:], strict=False):
        a, b, c = Fraction(low), Fraction(middle), Fraction(high)
        exact_weights = ((c - b) / (c - a), Fraction(-1), (b - a) / (c - a))
        best: ButterflyPackage | None = None
        checked = 0
        flags: list[str] = []
        for kinds in product((OptionType.CALL, OptionType.PUT), repeat=3):
            selected: list[SnapshotQuote] = []
            for strike, kind in zip((low, middle, high), kinds, strict=True):
                leg = by_strike[strike].get(kind)
                if leg is None:
                    flags.append(f"{strike:g}:{kind.value}:undeclared")
                else:
                    selected.append(leg)
            if len(selected) != 3:
                continue
            package, issues = _package(
                tuple(selected), exact_weights, future, discount_factor
            )
            flags.extend(issues)
            if package is not None:
                checked += 1
                if best is None or package.net_entry_cost < best.net_entry_cost:
                    best = package
        if checked < 8:
            flags.append("partial_support" if checked else "no_usable_combinations")
        checks.append(
            ButterflyCheck(
                (low, middle, high),
                (float(exact_weights[0]), -1.0, float(exact_weights[2])),
                checked,
                8 - checked,
                tuple(dict.fromkeys(flags)),
                best,
                None if best is None else best.negative_cost_candidate,
            )
        )
    return tuple(checks)
