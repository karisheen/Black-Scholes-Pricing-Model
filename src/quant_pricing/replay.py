"""Filesystem quote tapes with atomic raw/normalized Parquet observations.

Version 1 uses ``v1/source=<escaped>/event_date=<UTC YYYY-MM-DD>/*.parquet``.
Each immutable file is one event: its identity is (source, symbol, sequence,
event time). Repeated identical records are ignored; any changed field,
including availability, payload or provenance, conflicts. Corrections therefore
need a new event identity. Sequences are contiguous per source/instrument,
without resets; gaps are inferred only between eligible observed sequences.

Writers cooperate through a POSIX advisory lock at the root. A temporary file
is flushed and fsynced, then atomically renamed and its directory fsynced. Readers
ignore temporary files: a crash exposes either the entire raw+normalized event
or no event. A batch is NOT transactional; retry it after interruption. Leftover
``.pending-*`` files can be removed with writers stopped. Readers see committed
files discovered at the start of a call, not a transaction across a live writer.
No live feeds, clock reads, FX conversions, or trade-book state are involved.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote as escape

from quant_pricing.black_scholes import OptionType
from quant_pricing.exceptions import InputValidationError
from quant_pricing.market import (
    Forward,
    FrozenQuoteProvider,
    Instrument,
    MarketSnapshot,
    OptionContract,
    Quote,
    Underlying,
    _utc,
)


@dataclass(frozen=True, slots=True)
class RawObservation:
    """Unmodified source bytes and locator alongside a parsed, typed quote."""

    quote: Quote
    raw_payload: bytes
    provenance: str


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Committed new events and exact duplicates, including within a batch."""

    inserted: int
    duplicates: int


def _arrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Parquet replay requires quant-pricing[replay]") from exc
    return pa, pq


def _schema() -> Any:
    pa, _ = _arrow()
    # A string value plus explicit type retains integer precision and invalid bools.
    raw_number = pa.struct([("type", pa.string()), ("value", pa.string())])
    return pa.schema(
        [
            ("source", pa.string()),
            ("symbol", pa.string()),
            ("source_sequence", pa.int64()),
            ("quote_time", pa.timestamp("us", tz="UTC")),
            ("available_at", pa.timestamp("us", tz="UTC")),
            ("instrument", pa.string()),
            ("raw_bid", raw_number),
            ("raw_ask", raw_number),
            ("raw_bid_size", raw_number),
            ("raw_ask_size", raw_number),
            ("raw_payload", pa.binary()),
            ("provenance", pa.string()),
            ("normalized_bid", pa.float64()),
            ("normalized_ask", pa.float64()),
            ("premium_basis", pa.string()),
            ("quote_currency", pa.string()),
            ("record_hash", pa.string()),
        ],
        metadata={b"quant_pricing.replay.version": b"1"},
    )


def _instrument_data(instrument: Instrument) -> dict[str, object]:
    if isinstance(instrument, Underlying):
        return {
            "kind": "underlying",
            "symbol": instrument.symbol,
            "currency": instrument.currency,
        }
    data: dict[str, object] = {
        "kind": "forward",
        "symbol": instrument.symbol,
        "underlying": _instrument_data(instrument.underlying),
        "expiry": instrument.expiry.isoformat(),
    }
    if isinstance(instrument, OptionContract):
        data.update(
            kind="option",
            option_type=instrument.option_type.value,
            strike=instrument.strike,
            exercise_style=instrument.exercise_style,
            settlement_style=instrument.settlement_style,
            multiplier=instrument.multiplier,
            quote_currency=instrument.quote_currency,
            settlement_currency=instrument.settlement_currency,
            premium_basis=instrument.premium_basis,
        )
    return data


def _instrument(data: dict[str, Any]) -> Instrument:
    symbol = cast(str, data["symbol"])
    if data["kind"] == "underlying":
        return Underlying(symbol, data["currency"])
    underlying = Underlying(
        data["underlying"]["symbol"], data["underlying"]["currency"]
    )
    expiry = datetime.fromisoformat(data["expiry"])
    if data["kind"] == "forward":
        return Forward(symbol, underlying, expiry)
    return OptionContract(
        symbol,
        underlying,
        OptionType(data["option_type"]),
        data["strike"],
        expiry,
        data["exercise_style"],
        data["settlement_style"],
        data["multiplier"],
        data["quote_currency"],
        data["settlement_currency"],
        data["premium_basis"],
    )


def _raw_number(value: float) -> dict[str, str]:
    if isinstance(value, bool):
        return {"type": "bool", "value": str(value)}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    return {"type": "float", "value": float(value).hex()}


def _number(data: dict[str, str]) -> float:
    if data["type"] == "bool":
        return data["value"] == "True"
    if data["type"] == "int":
        return int(data["value"])
    return float.fromhex(data["value"])


def _row(observation: RawObservation) -> dict[str, Any]:
    quote = observation.quote
    instrument = quote.instrument
    if isinstance(instrument, OptionContract):
        factor = (
            instrument.multiplier if instrument.premium_basis == "per_contract" else 1
        )
        currency = instrument.quote_currency
    else:
        factor = 1
        currency = (
            instrument.currency
            if isinstance(instrument, Underlying)
            else instrument.underlying.currency
        )
    row: dict[str, Any] = {
        "source": quote.source,
        "symbol": instrument.symbol,
        "source_sequence": quote.source_sequence,
        "quote_time": quote.quote_time,
        "available_at": quote.available_at,
        "instrument": json.dumps(_instrument_data(instrument), sort_keys=True),
        "raw_bid": _raw_number(quote.bid),
        "raw_ask": _raw_number(quote.ask),
        "raw_bid_size": _raw_number(quote.bid_size),
        "raw_ask_size": _raw_number(quote.ask_size),
        "raw_payload": observation.raw_payload,
        "provenance": observation.provenance,
        "normalized_bid": math.nan
        if isinstance(quote.bid, bool)
        else quote.bid / factor,
        "normalized_ask": math.nan
        if isinstance(quote.ask, bool)
        else quote.ask / factor,
        "premium_basis": "per_unit",
        "quote_currency": currency,
    }
    fingerprint = {
        **row,
        "quote_time": quote.quote_time.isoformat(),
        "available_at": quote.available_at.isoformat(),
        "raw_payload": observation.raw_payload.hex(),
    }
    row["record_hash"] = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode()
    ).hexdigest()
    return row


def _observation(row: dict[str, Any]) -> RawObservation:
    return RawObservation(
        Quote(
            _instrument(json.loads(row["instrument"])),
            _number(row["raw_bid"]),
            _number(row["raw_ask"]),
            _number(row["raw_bid_size"]),
            _number(row["raw_ask_size"]),
            row["quote_time"],
            row["available_at"],
            row["source"],
            row["source_sequence"],
        ),
        row["raw_payload"],
        row["provenance"],
    )


def _read_row(path: Path) -> dict[str, Any]:
    _, pq = _arrow()
    parquet = pq.ParquetFile(path)
    if not parquet.schema_arrow.equals(_schema(), check_metadata=True):
        raise InputValidationError(f"unsupported replay schema: {path}")
    rows = cast(list[dict[str, Any]], parquet.read().to_pylist())
    if len(rows) != 1:
        raise InputValidationError(
            f"replay files must contain exactly one event: {path}"
        )
    return rows[0]


@dataclass(frozen=True, slots=True)
class ParquetReplayProvider:
    """A declared universe from one isolated source in an append-only local tape.

    Snapshot selection and gaps use BOTH clocks. Only eligible instrument metadata
    is compared with the declared universe; future observations cannot invalidate
    an earlier snapshot. ``observations`` returns all stored events for this source.
    """

    root: Path
    instruments: tuple[Instrument, ...]
    source: str
    max_age: timedelta

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "instruments", tuple(self.instruments))
        FrozenQuoteProvider(self.instruments, (), self.source, self.max_age)

    def _source_path(self) -> Path:
        return self.root / "v1" / f"source={escape(self.source, safe='')}"

    def _path(self, quote: Quote) -> Path:
        identity = json.dumps(
            [
                quote.source,
                quote.instrument.symbol,
                quote.source_sequence,
                quote.quote_time.isoformat(),
            ]
        )
        name = hashlib.sha256(identity.encode()).hexdigest() + ".parquet"
        return self._source_path() / f"event_date={quote.quote_time.date()}" / name

    def ingest(self, observations: Iterable[RawObservation]) -> IngestResult:
        """Commit each event atomically; reject changed fields on an existing ID.

        Source/universe validation is explicit. Readers may declare a subset of
        the stored universe. Never overwrite an event, even a later arrival.
        Version 1 stores source sequences as nonnegative signed 64-bit integers.
        """
        pa, pq = _arrow()
        universe = {instrument.symbol: instrument for instrument in self.instruments}
        inserted = duplicates = 0
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".writer.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            for observation in observations:
                quote = observation.quote
                if (
                    quote.source != self.source
                    or universe.get(quote.instrument.symbol) != quote.instrument
                ):
                    raise InputValidationError("observation source/instrument mismatch")
                if (
                    type(quote.source_sequence) is not int
                    or quote.source_sequence > 2**63 - 1
                ):
                    raise InputValidationError(
                        "source_sequence must fit a nonnegative signed 64-bit integer"
                    )
                row = _row(observation)
                path = self._path(quote)
                if path.exists():
                    if _read_row(path)["record_hash"] != row["record_hash"]:
                        raise InputValidationError(
                            f"conflicting event identity: {quote.instrument.symbol} "
                            f"sequence {quote.source_sequence} at {quote.quote_time}"
                        )
                    duplicates += 1
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        dir=path.parent, prefix=".pending-", delete=False
                    ) as stream:
                        temporary = Path(stream.name)
                        pq.write_table(
                            pa.Table.from_pylist([row], schema=_schema()), stream
                        )
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, path)
                    directory = os.open(path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                inserted += 1
        return IngestResult(inserted, duplicates)

    def _observations(self, as_of: datetime | None) -> tuple[RawObservation, ...]:
        events: list[RawObservation] = []
        symbols = {instrument.symbol for instrument in self.instruments}
        for path in sorted(self._source_path().glob("event_date=*/*.parquet")):
            row = _read_row(path)
            if row["source"] != self.source:
                raise InputValidationError("record source disagrees with its partition")
            if as_of is not None and (
                row["quote_time"] > as_of
                or row["available_at"] > as_of
                or row["symbol"] not in symbols
            ):
                continue
            events.append(_observation(row))
        return tuple(
            sorted(
                events,
                key=lambda event: (
                    event.quote.instrument.symbol,
                    event.quote.quote_time,
                    event.quote.source_sequence,
                    event.quote.available_at,
                ),
            )
        )

    def observations(self) -> tuple[RawObservation, ...]:
        """Return source-isolated raw records in deterministic event order."""
        return self._observations(None)

    def snapshot(self, as_of: datetime) -> MarketSnapshot:
        as_of = _utc(as_of, "as_of")
        quotes = tuple(event.quote for event in self._observations(as_of))
        return FrozenQuoteProvider(
            self.instruments, quotes, self.source, self.max_age
        ).snapshot(as_of)


@dataclass(frozen=True, slots=True)
class SyntheticTick:
    """A scheduled observation, with explicit event time and publication delay."""

    instrument: Instrument
    quote_time: datetime
    bid: float
    ask: float
    bid_size: float = 1.0
    ask_size: float = 1.0
    delay: timedelta = timedelta(0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "quote_time", _utc(self.quote_time, "quote_time"))
        if self.delay < timedelta(0):
            raise InputValidationError("synthetic publication delay cannot be negative")


@dataclass(frozen=True, slots=True)
class SyntheticProvider:
    """Frozen synthetic tape, or deterministic per-instrument scheduled quotes.

    Schedules assign sequence 1 at each instrument's first event, ordered by event
    time, not arrival time. Equal event times for one instrument are ambiguous and
    rejected. All caller input is frozen; replay never samples RNG or wall time.
    """

    instruments: tuple[Instrument, ...]
    quotes: tuple[Quote, ...]
    source: str
    max_age: timedelta

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruments", tuple(self.instruments))
        object.__setattr__(self, "quotes", tuple(self.quotes))
        FrozenQuoteProvider(self.instruments, self.quotes, self.source, self.max_age)

    @classmethod
    def from_schedule(
        cls,
        instruments: tuple[Instrument, ...],
        schedule: tuple[SyntheticTick, ...],
        source: str,
        max_age: timedelta,
    ) -> SyntheticProvider:
        sequences: dict[str, int] = {}
        seen: set[tuple[str, datetime]] = set()
        quotes: list[Quote] = []
        for tick in sorted(
            schedule, key=lambda tick: (tick.instrument.symbol, tick.quote_time)
        ):
            key = (tick.instrument.symbol, tick.quote_time)
            if key in seen:
                raise InputValidationError("ambiguous synthetic instrument/event time")
            seen.add(key)
            sequence = sequences.get(tick.instrument.symbol, 0) + 1
            sequences[tick.instrument.symbol] = sequence
            quotes.append(
                Quote(
                    tick.instrument,
                    tick.bid,
                    tick.ask,
                    tick.bid_size,
                    tick.ask_size,
                    tick.quote_time,
                    tick.quote_time + tick.delay,
                    source,
                    sequence,
                )
            )
        return cls(instruments, tuple(quotes), source, max_age)

    def snapshot(self, as_of: datetime) -> MarketSnapshot:
        return FrozenQuoteProvider(
            self.instruments, self.quotes, self.source, self.max_age
        ).snapshot(as_of)
