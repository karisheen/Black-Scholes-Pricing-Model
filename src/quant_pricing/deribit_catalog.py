"""Pure local selection from a complete historical Tardis /symbols envelope."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from fractions import Fraction
from math import isfinite
from numbers import Real
from typing import cast

from quant_pricing.exceptions import InputValidationError

_FUTURE = re.compile(r"BTC_USDC-([1-9]|[0-2][0-9]|3[01])([A-Z]{3})([0-9]{2})")
_MONTHS = (
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
)
_AVAILABLE_SINCE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
    r"(?:T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,9})?(?:Z|\+00:00))?"
)
_SCOPE = ("option", "BTC", "USDC")


@dataclass(frozen=True, slots=True)
class CatalogStrikeChoice:
    """Ascending selected strikes and counts before the requested-size cutoff.

    ``paired_strikes_available`` counts eligible call/put strike pairs.
    ``excluded_same_or_future_day`` counts unique matching option symbols, not
    strikes, whose first archive day has not fully elapsed at the UTC anchor.
    """

    strikes: tuple[int, ...]
    paired_strikes_available: int
    excluded_same_or_future_day: int


def _validate_future(symbol: str) -> None:
    match = _FUTURE.fullmatch(symbol) if isinstance(symbol, str) else None
    if match is None:
        raise InputValidationError("future_symbol must be a dated BTC_USDC future")
    try:
        date(2000 + int(match[3]), _MONTHS.index(match[2]) + 1, int(match[1]))
    except ValueError as exc:
        raise InputValidationError("future_symbol has an invalid expiry date") from exc


def _availability(value: object) -> tuple[date, str]:
    if not isinstance(value, str) or _AVAILABLE_SINCE.fullmatch(value) is None:
        raise InputValidationError("availableSince must be a UTC ISO timestamp or date")
    normalized = value.removesuffix("Z").removesuffix("+00:00")
    if "T" not in normalized:
        normalized += "T00:00:00"
    try:
        first_day = datetime.fromisoformat(normalized.partition(".")[0]).date()
    except ValueError as exc:
        raise InputValidationError(
            "availableSince has an invalid UTC date/time"
        ) from exc
    # Preserve sub-microsecond distinctions when detecting conflicting duplicates.
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return first_day, normalized


def _forward_fraction(value: object) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError("forward must be finite and positive")
    try:
        number = float(value)
    except OverflowError as exc:
        raise InputValidationError("forward must be finite and positive") from exc
    if not isfinite(number) or number <= 0:
        raise InputValidationError("forward must be finite and positive")
    return Fraction(str(number))


def choose_catalog_strikes(
    catalog: object,
    future_symbol: str,
    as_of: datetime,
    forward: float,
    *,
    count: int = 9,
) -> CatalogStrikeChoice:
    """Choose paired strikes by absolute log-moneyness, lower strike on ties.

    The input is decoded local /symbols JSON, with a ``symbols`` list and no
    non-null ``nextCursor``. Only exact Deribit BTC/USDC option metadata and
    canonical ``<future_symbol>-<positive integer>-C/P`` identifiers qualify.
    ``availableSince`` describes archive availability, not exchange listing:
    its UTC calendar day must be strictly before the anchor's UTC day. Date-only
    availability is interpreted as UTC. ``availableTo`` is deliberately ignored
    to avoid hindsight filtering, and no option quotes or marks are consumed.

    Repeated identical symbols are counted once; conflicting scope or availability
    for a matching symbol raises ``InputValidationError``. Fewer than ``count``
    pairs, including zero, produce a shortlist rather than manufactured strikes.
    """
    _validate_future(future_symbol)
    if not isinstance(as_of, datetime) or as_of.utcoffset() is None:
        raise InputValidationError("as_of must be a timezone-aware datetime")
    anchor_day = as_of.astimezone(timezone.utc).date()
    anchor_price = _forward_fraction(forward)
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise InputValidationError("count must be a positive integer")
    if not isinstance(catalog, dict) or not isinstance(catalog.get("symbols"), list):
        raise InputValidationError(
            "catalog must be a /symbols object with a symbols list"
        )
    if catalog.get("nextCursor") is not None:
        raise InputValidationError(
            "catalog is incomplete: nextCursor must be null/absent"
        )

    option = re.compile(re.escape(future_symbol) + r"-([1-9][0-9]*)-([CP])")
    scopes: dict[str, tuple[str, ...]] = {}
    availability: dict[str, tuple[date, str]] = {}
    legs: dict[int, set[str]] = {}
    excluded = 0
    for row in catalog["symbols"]:
        if not isinstance(row, dict):
            raise InputValidationError("each catalog symbol must be an object")
        fields = tuple(
            row.get(key)
            for key in ("exchange", "id", "type", "baseCurrency", "quoteCurrency")
        )
        if any(not isinstance(field, str) or not field for field in fields):
            raise InputValidationError("catalog symbol identity/scope must be strings")
        exchange, symbol, kind, base, quote = cast(
            tuple[str, str, str, str, str], fields
        )
        if exchange != "deribit":
            continue
        match = option.fullmatch(symbol)
        if match is None:
            continue
        scope = (kind, base, quote)
        if symbol in scopes and scopes[symbol] != scope:
            raise InputValidationError(f"conflicting duplicate scope for {symbol}")
        scopes[symbol] = scope
        if scope != _SCOPE:
            continue
        since = _availability(row.get("availableSince"))
        if symbol in availability:
            if availability[symbol] != since:
                raise InputValidationError(
                    f"conflicting duplicate availability: {symbol}"
                )
            continue
        availability[symbol] = since
        if since[0] >= anchor_day:
            excluded += 1
            continue
        legs.setdefault(int(match[1]), set()).add(match[2])

    paired = [strike for strike, kinds in legs.items() if kinds == {"C", "P"}]

    def distance(strike: int) -> tuple[Fraction, int]:
        # max(K/F, F/K) ranks exactly as abs(log(K/F)), without tie rounding.
        return max(strike / anchor_price, anchor_price / strike), strike

    nearest = sorted(paired, key=distance)[:count]
    return CatalogStrikeChoice(tuple(sorted(nearest)), len(paired), excluded)
