"""Offline, point-in-time implied volatility for a frozen synthetic option chain."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

from quant_pricing.black_scholes import OptionType
from quant_pricing.conventions import year_fraction
from quant_pricing.exceptions import PricingError
from quant_pricing.implied_volatility import implied_volatility
from quant_pricing.market import (
    FrozenQuoteProvider,
    Instrument,
    MarketSnapshot,
    OptionContract,
    Quote,
    SnapshotQuote,
    Underlying,
)
from quant_pricing.parity import CarryAssumptions, diagnose_parity

_SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0


@dataclass(frozen=True, slots=True)
class ChainOption:
    """Contract and the explicit payoff conventions not modeled by the domain."""

    contract: OptionContract
    strike_currency: str
    payoff_model: str


@dataclass(frozen=True, slots=True)
class FrozenChainFixture:
    """Declared synthetic universe, quotes, and valuation assumptions."""

    underlying: Underlying
    options: tuple[ChainOption, ...]
    provider: FrozenQuoteProvider
    rate: float
    dividend_yield: float


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return cast("dict[str, object]", value)


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return cast("list[object]", value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _number(value: object, name: str, *, finite: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if finite and not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _timestamp(value: object, name: str) -> datetime:
    text = _text(value, name)
    result = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return result.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _json_number(value: float) -> float | str:
    # Invalid raw observations remain auditable without emitting nonstandard JSON.
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


def load_fixture(path: str | Path) -> FrozenChainFixture:
    """Read version-one synthetic JSON; no network, pricing or clock access."""

    return _load_data(_object(json.loads(Path(path).read_bytes()), "fixture"))


def _load_data(data: dict[str, object]) -> FrozenChainFixture:
    if _integer(data.get("schema_version"), "schema_version") != 1:
        raise ValueError("only fixture schema_version 1 is supported")
    if data.get("data_kind") != "frozen_synthetic":
        raise ValueError("only frozen_synthetic fixtures are supported")
    underlying_data = _object(data.get("underlying"), "underlying")
    underlying = Underlying(
        symbol=_text(underlying_data.get("symbol"), "underlying.symbol"),
        currency=_text(underlying_data.get("currency"), "underlying.currency"),
    )
    instruments: dict[str, Instrument] = {underlying.symbol: underlying}
    options: list[ChainOption] = []
    for value in _array(data.get("options"), "options"):
        item = _object(value, "option")
        # OptionContract validates the allowed string values.
        exercise = _text(item.get("exercise_style"), "exercise_style")
        settlement = _text(item.get("settlement_style"), "settlement_style")
        basis = _text(item.get("premium_basis"), "premium_basis")
        contract = OptionContract(
            symbol=_text(item.get("symbol"), "option.symbol"),
            underlying=underlying,
            option_type=OptionType.parse(_text(item.get("option_type"), "option_type")),
            strike=_number(item.get("strike"), "strike"),
            expiry=_timestamp(item.get("expiry"), "expiry"),
            exercise_style=cast("Literal['european', 'american']", exercise),
            settlement_style=cast("Literal['cash', 'physical']", settlement),
            multiplier=_number(item.get("multiplier"), "multiplier"),
            quote_currency=_text(item.get("quote_currency"), "quote_currency"),
            settlement_currency=_text(
                item.get("settlement_currency"), "settlement_currency"
            ),
            premium_basis=cast("Literal['per_unit', 'per_contract']", basis),
        )
        if contract.symbol in instruments:
            raise ValueError(f"duplicate instrument symbol: {contract.symbol}")
        instruments[contract.symbol] = contract
        options.append(
            ChainOption(
                contract,
                _text(item.get("strike_currency"), "strike_currency"),
                _text(item.get("payoff_model"), "payoff_model"),
            )
        )
    quotes: list[Quote] = []
    for value in _array(data.get("quotes"), "quotes"):
        item = _object(value, "quote")
        symbol = _text(item.get("symbol"), "quote.symbol")
        if symbol not in instruments:
            raise ValueError(f"unknown quoted instrument: {symbol}")
        quote = Quote(
            instrument=instruments[symbol],
            bid=_number(item.get("bid"), "bid", finite=False),
            ask=_number(item.get("ask"), "ask", finite=False),
            bid_size=_number(item.get("bid_size"), "bid_size", finite=False),
            ask_size=_number(item.get("ask_size"), "ask_size", finite=False),
            quote_time=_timestamp(item.get("quote_time"), "quote_time"),
            available_at=_timestamp(item.get("available_at"), "available_at"),
            source=_text(item.get("source"), "quote.source"),
            source_sequence=_integer(item.get("source_sequence"), "source_sequence"),
        )
        quotes.append(quote)
    source = _text(data.get("source"), "source")
    max_age = _number(data.get("max_age_seconds"), "max_age_seconds")
    return FrozenChainFixture(
        underlying=underlying,
        options=tuple(options),
        provider=FrozenQuoteProvider(
            instruments=tuple(instruments.values()),
            quotes=tuple(quotes),
            source=source,
            max_age=timedelta(seconds=max_age),
        ),
        rate=_number(data.get("rate"), "rate"),
        dividend_yield=_number(data.get("dividend_yield"), "dividend_yield"),
    )


def _provenance(quote: Quote) -> dict[str, object]:
    return {
        "source": quote.source,
        "source_sequence": quote.source_sequence,
        "quote_time": _iso(quote.quote_time),
        "available_at": _iso(quote.available_at),
    }


def _observation(row: SnapshotQuote, fixture: FrozenChainFixture) -> dict[str, object]:
    quote = row.quote
    instrument = row.instrument
    result: dict[str, object] = {
        "symbol": instrument.symbol,
        "quality_flags": list(row.flags),
        "sequence_gaps": [
            {
                "first_missing": gap.first_missing,
                "last_missing": gap.last_missing,
                "before": _provenance(gap.before),
                "after": _provenance(gap.after),
            }
            for gap in row.sequence_gaps
        ],
        "provenance": None,
        "raw": None,
        "normalized": None,
    }
    if quote is None:
        return result
    result["provenance"] = _provenance(quote)
    if isinstance(instrument, OptionContract):
        currency = instrument.quote_currency
        basis = instrument.premium_basis
        size_unit = "contracts"
        normalized_bid = instrument.premium_per_unit(quote.bid)
        normalized_ask = instrument.premium_per_unit(quote.ask)
    else:
        currency = fixture.underlying.currency
        basis = "per_unit"
        size_unit = "underlying_units"
        normalized_bid = quote.bid
        normalized_ask = quote.ask
    result["raw"] = {
        "bid": _json_number(quote.bid),
        "ask": _json_number(quote.ask),
        "bid_size": _json_number(quote.bid_size),
        "ask_size": _json_number(quote.ask_size),
        "currency": currency,
        "premium_basis": basis,
        "size_unit": size_unit,
    }
    result["normalized"] = {
        "bid": _json_number(normalized_bid),
        "ask": _json_number(normalized_ask),
        "currency": currency,
        "premium_basis": "per_unit",
    }
    return result


def _failed_side(kind: str, message: str) -> dict[str, object]:
    return {"iv": None, "error": {"type": kind, "message": message}}


def _solve_side(
    contract: OptionContract,
    premium: float,
    spot: float,
    time_to_expiry: float,
    fixture: FrozenChainFixture,
) -> dict[str, object]:
    try:
        result = implied_volatility(
            contract.option_type,
            contract.premium_per_unit(premium),
            spot,
            contract.strike,
            time_to_expiry,
            fixture.rate,
            fixture.dividend_yield,
        )
    except PricingError as exc:
        return _failed_side(type(exc).__name__, str(exc))
    return {
        "iv": result.volatility,
        "error": None,
        "iterations": result.iterations,
        "residual": result.residual,
        "termination_reason": result.termination_reason,
    }


def _parity_reports(
    fixture: FrozenChainFixture,
    as_of: datetime,
    rows: dict[str, SnapshotQuote],
    spot: float | None,
) -> list[dict[str, object]]:
    pairs: dict[tuple[datetime, float], list[OptionContract]] = {}
    for spec in fixture.options:
        contract = spec.contract
        if (
            spec.payoff_model == "linear"
            and spec.strike_currency == fixture.underlying.currency
            and contract.expiry > as_of
        ):
            pairs.setdefault((contract.expiry, contract.strike), []).append(contract)
    reports: list[dict[str, object]] = []
    for (expiry, strike), legs in sorted(pairs.items()):
        if len(legs) != 2 or legs[0].option_type is legs[1].option_type:
            continue
        call, put = sorted(legs, key=lambda option: option.option_type.value)
        pair_report: dict[str, object] = {
            "strike": strike,
            "expiry": _iso(expiry),
            "call": call.symbol,
            "put": put.symbol,
        }
        try:
            time = year_fraction(as_of, expiry)
            benchmark = None
            carry = None
            if spot is not None:
                factor = math.exp((fixture.rate - fixture.dividend_yield) * time)
                benchmark = spot * factor
                carry = CarryAssumptions(spot, fixture.rate, time, borrow_yield=0.0)
            result = diagnose_parity(
                rows[call.symbol],
                rows[put.symbol],
                as_of=as_of,
                max_age=fixture.provider.max_age,
                discount_factor=math.exp(-fixture.rate * time),
                forward=benchmark,
                carry=carry,
            )
            pair_report.update(asdict(result), carry_forward_benchmark=benchmark)
        except (PricingError, OverflowError) as exc:
            pair_report.update(
                implied_forward=None,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
        reports.append(pair_report)
    return reports


def chain_report(fixture: FrozenChainFixture, as_of: datetime) -> dict[str, object]:
    """Screen one eligible snapshot and solve each usable option side separately."""

    return _report_snapshot(fixture, fixture.provider.snapshot(as_of))


def _report_snapshot(
    fixture: FrozenChainFixture, snapshot: MarketSnapshot
) -> dict[str, object]:
    rows = {row.instrument.symbol: row for row in snapshot.rows}
    underlying_row = rows[fixture.underlying.symbol]
    underlying_quote = underlying_row.quote
    spot: float | None = None
    if underlying_quote is not None and not (
        underlying_row.flags or underlying_row.sequence_gaps
    ):
        midpoint = underlying_quote.bid / 2.0 + underlying_quote.ask / 2.0
        if midpoint > 0.0:
            spot = midpoint
    underlying_report = _observation(underlying_row, fixture)
    underlying_report["midpoint"] = spot
    option_reports: list[dict[str, object]] = []
    for specification in sorted(fixture.options, key=lambda item: item.contract.symbol):
        contract = specification.contract
        row = rows[contract.symbol]
        report = _observation(row, fixture)
        elapsed_seconds = (contract.expiry - snapshot.as_of).total_seconds()
        time_to_expiry = elapsed_seconds / _SECONDS_PER_YEAR
        report["contract"] = {
            "underlying": contract.underlying.symbol,
            "option_type": contract.option_type.value,
            "strike": contract.strike,
            "strike_currency": specification.strike_currency,
            "expiry": _iso(contract.expiry),
            "exercise_style": contract.exercise_style,
            "settlement_style": contract.settlement_style,
            "payoff_model": specification.payoff_model,
            "multiplier": contract.multiplier,
            "premium_basis": contract.premium_basis,
            "quote_currency": contract.quote_currency,
            "settlement_currency": contract.settlement_currency,
        }
        report["time_to_expiry"] = time_to_expiry
        failure: dict[str, object] | None = None
        if row.quote is None or row.flags or row.sequence_gaps:
            flags = (*row.flags, "sequence_gap") if row.sequence_gaps else row.flags
            failure = _failed_side("SnapshotQualityError", ", ".join(flags))
        elif spot is None:
            failure = _failed_side(
                "UnderlyingUnavailable", "no valid underlying midpoint"
            )
        elif (
            contract.exercise_style != "european"
            or contract.settlement_style != "cash"
            or specification.payoff_model != "linear"
        ):
            failure = _failed_side(
                "UnsupportedModel",
                "only cash-settled linear European options support IV",
            )
        elif not (
            fixture.underlying.currency
            == specification.strike_currency
            == contract.quote_currency
            == contract.settlement_currency
        ):
            failure = _failed_side(
                "CurrencyMismatch",
                "underlying, strike, quote and settlement currencies must match; no FX",
            )
        if failure is not None:
            report["bid"] = failure
            report["ask"] = failure
        else:
            # The checks above establish these two prerequisites for both sides.
            assert row.quote is not None and spot is not None
            report["bid"] = _solve_side(
                contract, row.quote.bid, spot, time_to_expiry, fixture
            )
            report["ask"] = _solve_side(
                contract, row.quote.ask, spot, time_to_expiry, fixture
            )
        option_reports.append(report)
    return {
        "schema_version": 1,
        "as_of": _iso(snapshot.as_of),
        "source": snapshot.source,
        "complete": snapshot.complete,
        "assumptions": {
            "data_kind": "frozen_synthetic",
            "model": "Black-Scholes-Merton",
            "supported_payoff": "cash-settled linear European",
            "spot": "eligible valid underlying bid/ask midpoint",
            "rate": fixture.rate,
            "dividend_yield": fixture.dividend_yield,
            "rate_convention": "flat deterministic continuously compounded decimal",
            "day_count": "ACT/365F",
            "year_seconds": _SECONDS_PER_YEAR,
            "time_to_expiry": "exact elapsed UTC seconds from snapshot as_of",
            "max_age_seconds": fixture.provider.max_age.total_seconds(),
            "stale_rule": "as_of minus quote_time > max_age",
            "eligibility": "quote_time <= as_of and available_at <= as_of",
            "selection": "latest quote time, then sequence, then availability",
            "premium_normalization": "per_contract divided by multiplier",
            "fx_conversion": False,
        },
        "underlying": underlying_report,
        "options": option_reports,
        "parity": _parity_reports(fixture, snapshot.as_of, rows, spot),
    }


def generate_report(path: str | Path, as_of: datetime) -> dict[str, object]:
    """Load a fixture and produce a deterministic, JSON-compatible report."""

    return chain_report(load_fixture(path), as_of)


def _archive_report(
    path: Path, archive: Path, as_of: datetime, *, ingest: bool
) -> dict[str, object]:
    from quant_pricing.replay import ParquetReplayProvider, RawObservation

    raw = path.read_bytes()
    data = _object(json.loads(raw), "fixture")
    fixture = _load_data(data)
    tape = fixture.provider
    provider = ParquetReplayProvider(
        archive, tape.instruments, tape.source, tape.max_age
    )
    if ingest:
        digest = sha256(raw).hexdigest()
        quote_payloads = _array(data["quotes"], "quotes")
        provider.ingest(
            RawObservation(
                quote,
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
                f"sha256:{digest}#quotes/{index}",
            )
            for index, (quote, payload) in enumerate(
                zip(tape.quotes, quote_payloads, strict=True)
            )
        )
    return _report_snapshot(fixture, provider.snapshot(as_of))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the frozen synthetic chain workflow without any live data source."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--as-of", required=True, help="timezone-aware ISO-8601 cutoff")
    parser.add_argument("--archive", type=Path, help="partitioned Parquet archive")
    parser.add_argument(
        "--ingest", action="store_true", help="ingest fixture quotes before replaying"
    )
    arguments = parser.parse_args(argv)
    try:
        cutoff = _timestamp(arguments.as_of, "as_of")
        if arguments.archive is None:
            if arguments.ingest:
                raise ValueError("--ingest requires --archive")
            report = generate_report(arguments.fixture, cutoff)
        else:
            report = _archive_report(
                arguments.fixture, arguments.archive, cutoff, ingest=arguments.ingest
            )
        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    except (ImportError, OSError, ValueError, PricingError) as exc:
        print(f"option-chain: {exc}", file=sys.stderr)
        return 2
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI integration covers entrypoint
    raise SystemExit(main())
