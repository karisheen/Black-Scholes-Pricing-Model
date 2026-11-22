"""Executable put/call parity intervals from independent observed option sides.

All premiums and residuals are in PV currency per underlying unit. A bid/ask
interval is not a midpoint observation. Theoretical forwards and independently
quoted forwards are separate comparisons; neither creates a second market quote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypeAlias

from quant_pricing.black_scholes import OptionType
from quant_pricing.exceptions import InputValidationError
from quant_pricing.market import (
    Forward,
    OptionContract,
    SnapshotQuote,
    _positive,
    _quote_flags,
    _utc,
)

Interval: TypeAlias = tuple[float, float]


@dataclass(frozen=True, slots=True)
class CarryAssumptions:
    """Conditional carry, with exactly one independently known yield.

    F = spot * exp((funding_rate - dividend_yield - borrow_yield) * T).
    Borrow yield means a stock-lending/specialness yield reducing the forward,
    not a funding spread added to the funding rate. Supply borrow_yield=0 to
    infer dividends assuming no borrow effect; supply known dividend_yield to
    infer borrow. The funding rate and horizon are explicit caller assumptions,
    not values inferred from an option midpoint or the discount factor.
    """

    spot: float
    funding_rate: float
    time_to_expiry: float
    dividend_yield: float | None = None
    borrow_yield: float | None = None

    def __post_init__(self) -> None:
        _positive(self.spot, "spot")
        _positive(self.time_to_expiry, "carry time_to_expiry")
        if not math.isfinite(self.funding_rate):
            raise InputValidationError("funding_rate must be finite")
        if (self.dividend_yield is None) == (self.borrow_yield is None):
            raise InputValidationError(
                "supply exactly one known dividend or borrow yield"
            )
        known = (
            self.dividend_yield
            if self.dividend_yield is not None
            else self.borrow_yield
        )
        assert known is not None
        if not math.isfinite(known):
            raise InputValidationError("known carry yield must be finite")


@dataclass(frozen=True, slots=True)
class ParityDiagnostic:
    """Auditable intervals; unusable evidence has flags and no derived interval.

    A residual interval containing zero is consistent with the benchmark after
    spreads; otherwise its endpoints bound the signed discrepancy. This is not
    an arbitrage claim: transaction costs, funding and executable sizes matter.
    """

    option_flags: tuple[str, ...]
    forward_flags: tuple[str, ...]
    premium_difference: Interval | None
    implied_forward: Interval | None
    parity_residual: Interval | None
    independent_forward_residual: Interval | None
    implied_dividend_yield: Interval | None
    implied_borrow_yield: Interval | None


def _contract(row: SnapshotQuote, kind: OptionType) -> OptionContract:
    contract = row.instrument
    if not isinstance(contract, OptionContract) or contract.option_type != kind:
        raise InputValidationError(f"expected a {kind.value} OptionContract")
    if contract.exercise_style != "european" or contract.settlement_style != "cash":
        raise InputValidationError("parity requires European cash-settled options")
    if not (
        contract.quote_currency
        == contract.settlement_currency
        == contract.underlying.currency
    ):
        raise InputValidationError("parity requires one currency without implicit FX")
    return contract


def _flags(
    row: SnapshotQuote, label: str, as_of: datetime, max_age: timedelta
) -> tuple[str, ...]:
    flags: list[str] = list(row.flags)
    if row.sequence_gaps:
        flags.append("sequence_gap")
    quote = row.quote
    if quote is not None:
        flags.extend(_quote_flags(quote, as_of, max_age))
        if quote.quote_time > as_of or quote.available_at > as_of:
            flags.append("future")
        if quote.bid_size == 0 or quote.ask_size == 0:
            flags.append("empty_size")
    return tuple(f"{label}:{flag}" for flag in dict.fromkeys(flags))


def _benchmark(value: float | Interval) -> Interval:
    low, high = value if isinstance(value, tuple) else (value, value)
    _positive(low, "forward lower bound")
    _positive(high, "forward upper bound")
    if low > high:
        raise InputValidationError("forward interval must be ordered")
    return low, high


def _residual(
    premium: Interval, forward: Interval, strike: float, d: float
) -> Interval:
    return premium[0] - d * (forward[1] - strike), premium[1] - d * (
        forward[0] - strike
    )


def diagnose_parity(
    call: SnapshotQuote,
    put: SnapshotQuote,
    *,
    as_of: datetime,
    max_age: timedelta,
    discount_factor: float,
    forward: float | Interval | None = None,
    forward_quote: SnapshotQuote | None = None,
    carry: CarryAssumptions | None = None,
) -> ParityDiagnostic:
    """Diagnose a matched pair without inventing a second observation.

    The two option contracts must match underlying, exact expiry and strike.
    Premium bases and multipliers may differ: each premium is normalized using
    its own declared contract units. Positive displayed size is required, but
    depth is not modeled. Cross-source observations are permitted if individually
    eligible at ``as_of``. Existing and recomputed quality flags, sequence gaps,
    stale observations and either future timestamp suppress affected evidence.

    F_bid = K + (C_bid - P_ask)/D; F_ask = K + (C_ask - P_bid)/D.
    Residuals compare executable premium-difference intervals to D*(F-K).
    ``forward`` is a theoretical scalar/interval benchmark; ``forward_quote``
    is an independently observed, matched dated Forward, never a synthetic quote.
    Carry inference additionally needs explicit ``CarryAssumptions`` and positive
    implied-forward endpoints; otherwise carry is undefined and flagged.
    """
    cutoff = _utc(as_of, "as_of")
    if max_age < timedelta(0):
        raise InputValidationError("max_age cannot be negative")
    _positive(discount_factor, "discount_factor")
    c, p = _contract(call, OptionType.CALL), _contract(put, OptionType.PUT)
    if (c.underlying, c.expiry, c.strike) != (p.underlying, p.expiry, p.strike):
        raise InputValidationError(
            "call and put must match underlying, expiry and strike"
        )
    if c.expiry < cutoff:
        raise InputValidationError("option expiry precedes as_of")
    benchmark = None if forward is None else _benchmark(forward)
    option_flags = _flags(call, "call", cutoff, max_age) + _flags(
        put, "put", cutoff, max_age
    )
    forward_flags: tuple[str, ...] = ()
    observed_forward: Interval | None = None
    if forward_quote is not None:
        instrument = forward_quote.instrument
        if not isinstance(instrument, Forward) or (
            instrument.underlying,
            instrument.expiry,
        ) != (c.underlying, c.expiry):
            raise InputValidationError(
                "independent forward must match underlying/expiry"
            )
        forward_flags = _flags(forward_quote, "forward", cutoff, max_age)
        if not forward_flags:
            assert forward_quote.quote is not None
            observed_forward = (
                forward_quote.quote.bid,
                forward_quote.quote.ask,
            )
    if option_flags:
        return ParityDiagnostic(
            option_flags, forward_flags, None, None, None, None, None, None
        )
    assert call.quote is not None and put.quote is not None
    premium = (
        c.premium_per_unit(call.quote.bid) - p.premium_per_unit(put.quote.ask),
        c.premium_per_unit(call.quote.ask) - p.premium_per_unit(put.quote.bid),
    )
    implied = (
        c.strike + premium[0] / discount_factor,
        c.strike + premium[1] / discount_factor,
    )
    if not all(math.isfinite(value) for value in (*premium, *implied)):
        raise InputValidationError("parity interval exceeds float range")
    dividend: Interval | None = None
    borrow: Interval | None = None
    if carry is not None:
        if implied[0] <= 0.0:
            option_flags += ("carry:nonpositive_forward",)
        else:
            total_yield = tuple(
                carry.funding_rate
                - (math.log(value) - math.log(carry.spot)) / carry.time_to_expiry
                for value in reversed(implied)
            )
            if carry.dividend_yield is not None:
                borrow = (
                    total_yield[0] - carry.dividend_yield,
                    total_yield[1] - carry.dividend_yield,
                )
            else:
                assert carry.borrow_yield is not None
                dividend = (
                    total_yield[0] - carry.borrow_yield,
                    total_yield[1] - carry.borrow_yield,
                )
    return ParityDiagnostic(
        option_flags=option_flags,
        forward_flags=forward_flags,
        premium_difference=premium,
        implied_forward=implied,
        parity_residual=(
            None
            if benchmark is None
            else _residual(premium, benchmark, c.strike, discount_factor)
        ),
        independent_forward_residual=(
            None
            if observed_forward is None
            else _residual(premium, observed_forward, c.strike, discount_factor)
        ),
        implied_dividend_yield=dividend,
        implied_borrow_yield=borrow,
    )
