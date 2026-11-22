"""Immutable instruments, auditable quotes, and point-in-time frozen snapshots.

Prices retain their declared currency; multipliers change units, never currency.
Quote sizes are contracts for options and underlying units otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import pairwise
from math import isfinite
from numbers import Real
from typing import Literal, Protocol, TypeAlias

from quant_pricing.black_scholes import OptionType
from quant_pricing.exceptions import InputValidationError

QuoteFlag: TypeAlias = Literal[
    "missing", "stale", "crossed", "invalid_price", "invalid_size"
]


def _text(value: str, name: str) -> None:
    if not value.strip():
        raise InputValidationError(f"{name} must be a nonempty string")


def _utc(value: datetime, name: str) -> datetime:
    if value.utcoffset() is None:
        raise InputValidationError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and isfinite(value)


def _positive(value: float, name: str) -> None:
    if not _finite(value) or value <= 0:
        raise InputValidationError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class Underlying:
    """A spot asset priced in its declared currency."""

    symbol: str
    currency: str

    def __post_init__(self) -> None:
        _text(self.symbol, "symbol")
        _text(self.currency, "currency")


@dataclass(frozen=True, slots=True)
class Forward:
    """A dated underlying instrument, without a curve or pricing convention."""

    symbol: str
    underlying: Underlying
    expiry: datetime

    def __post_init__(self) -> None:
        _text(self.symbol, "symbol")
        object.__setattr__(self, "expiry", _utc(self.expiry, "expiry"))


@dataclass(frozen=True, slots=True)
class OptionContract:
    """Explicit linear contract units; no implicit FX or inverse-payoff support."""

    symbol: str
    underlying: Underlying
    option_type: OptionType
    strike: float
    expiry: datetime
    exercise_style: Literal["european", "american"]
    settlement_style: Literal["cash", "physical"]
    multiplier: float
    quote_currency: str
    settlement_currency: str
    premium_basis: Literal["per_unit", "per_contract"]

    def __post_init__(self) -> None:
        _text(self.symbol, "symbol")
        _positive(self.strike, "strike")
        object.__setattr__(self, "expiry", _utc(self.expiry, "expiry"))
        if self.exercise_style not in ("european", "american"):
            raise InputValidationError("exercise_style must be european or american")
        if self.settlement_style not in ("cash", "physical"):
            raise InputValidationError("settlement_style must be cash or physical")
        _positive(self.multiplier, "multiplier")
        _text(self.quote_currency, "quote_currency")
        _text(self.settlement_currency, "settlement_currency")
        if self.premium_basis not in ("per_unit", "per_contract"):
            raise InputValidationError("premium_basis must be per_unit or per_contract")

    def premium_per_unit(self, amount: float) -> float:
        """Normalize premium units, retaining the quote currency unchanged."""
        return (
            amount / self.multiplier if self.premium_basis == "per_contract" else amount
        )


Instrument: TypeAlias = Underlying | Forward | OptionContract


@dataclass(frozen=True, slots=True)
class Quote:
    """Raw observed sides, including invalid prices/sizes, retained for audit.

    quote_time is the event time; available_at is when the event became knowable.
    Neither timestamp alone establishes point-in-time eligibility.
    """

    instrument: Instrument
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    quote_time: datetime
    available_at: datetime
    source: str
    source_sequence: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "quote_time", _utc(self.quote_time, "quote_time"))
        object.__setattr__(
            self, "available_at", _utc(self.available_at, "available_at")
        )
        _text(self.source, "source")
        if self.source_sequence < 0:
            raise InputValidationError("source_sequence must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class SequenceGap:
    """Inclusive missing interval bounded by two eligible observed events.

    Sequences are assumed contiguous per source and instrument, without resets.
    No expected initial sequence or unseen trailing interval is inferred.
    """

    first_missing: int
    last_missing: int
    before: Quote
    after: Quote


@dataclass(frozen=True, slots=True)
class SnapshotQuote:
    """A selected raw quote and its exclusion flags; empty flags mean usable."""

    instrument: Instrument
    quote: Quote | None
    flags: tuple[QuoteFlag, ...]
    sequence_gaps: tuple[SequenceGap, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "flags", tuple(self.flags))
        object.__setattr__(self, "sequence_gaps", tuple(self.sequence_gaps))
        if self.quote is None:
            if self.flags != ("missing",):
                raise InputValidationError(
                    "an absent quote must have only the missing flag"
                )
        elif self.quote.instrument != self.instrument:
            raise InputValidationError("quote must match the snapshot instrument")
        elif "missing" in self.flags:
            raise InputValidationError("a present quote cannot have the missing flag")


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """One source at one UTC instant, with deterministic instrument row order."""

    as_of: datetime
    source: str
    rows: tuple[SnapshotQuote, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", _utc(self.as_of, "as_of"))
        _text(self.source, "source")
        object.__setattr__(self, "rows", tuple(self.rows))
        symbols: set[str] = set()
        for row in self.rows:
            if row.instrument.symbol in symbols:
                raise InputValidationError("snapshot instrument symbols must be unique")
            symbols.add(row.instrument.symbol)
            if row.quote is not None:
                if row.quote.source != self.source:
                    raise InputValidationError(
                        "snapshot quotes must use its declared source"
                    )
                if (
                    row.quote.quote_time > self.as_of
                    or row.quote.available_at > self.as_of
                ):
                    raise InputValidationError("snapshot cannot contain future quotes")

    @property
    def complete(self) -> bool:
        """Whether every declared instrument is usable and has no observed gap."""
        return all(not row.flags and not row.sequence_gaps for row in self.rows)


class SnapshotProvider(Protocol):
    """An injectable source of point-in-time market snapshots."""

    def snapshot(self, as_of: datetime) -> MarketSnapshot:
        """Return the declared universe using only information known by as_of."""
        ...


def _same_sides(left: Quote, right: Quote) -> bool:
    # NaNs remain invalid but repeated copies of the same bad event are idempotent.
    return all(
        (a == b and isinstance(a, bool) == isinstance(b, bool)) or (a != a and b != b)
        for a, b in (
            (left.bid, right.bid),
            (left.ask, right.ask),
            (left.bid_size, right.bid_size),
            (left.ask_size, right.ask_size),
        )
    )


def _quote_flags(
    quote: Quote, as_of: datetime, max_age: timedelta
) -> tuple[QuoteFlag, ...]:
    flags: list[QuoteFlag] = []
    if as_of - quote.quote_time > max_age:
        flags.append("stale")
    finite_prices = _finite(quote.bid) and _finite(quote.ask)
    if finite_prices and quote.bid > quote.ask:
        flags.append("crossed")
    if not finite_prices or quote.bid < 0 or quote.ask < 0:
        flags.append("invalid_price")
    if (
        not _finite(quote.bid_size)
        or not _finite(quote.ask_size)
        or quote.bid_size < 0
        or quote.ask_size < 0
    ):
        flags.append("invalid_size")
    return tuple(flags)


@dataclass(frozen=True, slots=True)
class FrozenQuoteProvider:
    """Select latest eligible events, never falling back from unusable quotes.

    Event precedence is (quote_time, source_sequence, available_at). Conflicting
    ties are rejected only once eligible; other sources never participate.
    """

    instruments: tuple[Instrument, ...]
    quotes: tuple[Quote, ...]
    source: str
    max_age: timedelta

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruments", tuple(self.instruments))
        object.__setattr__(self, "quotes", tuple(self.quotes))
        _text(self.source, "source")
        if self.max_age < timedelta(0):
            raise InputValidationError("max_age must be a nonnegative timedelta")
        universe: dict[str, Instrument] = {}
        for instrument in self.instruments:
            if instrument.symbol in universe:
                raise InputValidationError("provider instrument symbols must be unique")
            universe[instrument.symbol] = instrument
        for quote in self.quotes:
            if (
                quote.source == self.source
                and universe.get(quote.instrument.symbol) != quote.instrument
            ):
                raise InputValidationError(
                    "quote instrument is not in the declared universe"
                )

    def snapshot(self, as_of: datetime) -> MarketSnapshot:
        as_of = _utc(as_of, "as_of")
        selected: dict[str, Quote] = {}
        seen: dict[tuple[str, datetime, int, datetime], Quote] = {}
        observed: dict[str, dict[int, Quote]] = {}
        for quote in self.quotes:
            if (
                quote.source != self.source
                or quote.quote_time > as_of
                or quote.available_at > as_of
            ):
                continue
            rank = (quote.quote_time, quote.source_sequence, quote.available_at)
            key = (quote.instrument.symbol, *rank)
            duplicate = seen.get(key)
            if duplicate is not None:
                if not _same_sides(quote, duplicate):
                    raise InputValidationError(
                        f"conflicting eligible quotes for {quote.instrument.symbol}"
                    )
                continue
            seen[key] = quote
            per_sequence = observed.setdefault(quote.instrument.symbol, {})
            previous_sequence = per_sequence.get(quote.source_sequence)
            if previous_sequence is None or (
                quote.quote_time,
                quote.available_at,
            ) < (previous_sequence.quote_time, previous_sequence.available_at):
                per_sequence[quote.source_sequence] = quote
            previous = selected.get(quote.instrument.symbol)
            if previous is None or rank > (
                previous.quote_time,
                previous.source_sequence,
                previous.available_at,
            ):
                selected[quote.instrument.symbol] = quote
        rows: list[SnapshotQuote] = []
        for instrument in self.instruments:
            chosen = selected.get(instrument.symbol)
            flags: tuple[QuoteFlag, ...] = (
                ("missing",)
                if chosen is None
                else _quote_flags(chosen, as_of, self.max_age)
            )
            sequences = sorted(
                observed.get(instrument.symbol, {}).values(),
                key=lambda quote: quote.source_sequence,
            )
            gaps = tuple(
                SequenceGap(
                    before.source_sequence + 1,
                    after.source_sequence - 1,
                    before,
                    after,
                )
                for before, after in pairwise(sequences)
                if after.source_sequence > before.source_sequence + 1
            )
            rows.append(SnapshotQuote(instrument, chosen, flags, gaps))
        return MarketSnapshot(as_of, self.source, tuple(rows))


@dataclass(frozen=True, slots=True)
class Position:
    """Signed option contracts; multiplier converts exposure to underlying units."""

    contract: OptionContract
    contracts: float

    def __post_init__(self) -> None:
        if not _finite(self.contracts) or not isfinite(self.underlying_units):
            raise InputValidationError("contracts and underlying units must be finite")

    @property
    def underlying_units(self) -> float:
        return self.contracts * self.contract.multiplier


@dataclass(frozen=True, slots=True)
class Portfolio:
    """An immutable ordered collection of positions; no implicit netting."""

    positions: tuple[Position, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", tuple(self.positions))
