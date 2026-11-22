"""Replay a local Tardis native Deribit book capture; no network or pricing."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Literal, TypedDict

from quant_pricing.exceptions import InputValidationError

_NS = 1_000_000_000
_NS_PER_MS = 1_000_000
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_UTC = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?Z")


class BookEvidence(TypedDict):
    line: int
    available_at: str
    exchange_time: str
    change_id: int


class BookState(TypedDict):
    flags: list[str]
    bid: list[str] | None
    ask: list[str] | None
    snapshot: BookEvidence | None
    last_update: BookEvidence | None
    issue: dict[str, object] | None
    snapshots_applied: int
    changes_applied: int
    changes_ignored: int
    sequence_gaps: int


class BookCheckpoint(TypedDict):
    as_of: str
    complete: bool
    books: dict[str, BookState]


def _timestamp(value: str) -> int:
    match = _UTC.fullmatch(value)
    if match is None:
        raise InputValidationError(f"expected UTC timestamp ending in Z: {value!r}")
    elapsed = datetime.fromisoformat(f"{match[1]}+00:00") - _EPOCH
    seconds = elapsed.days * 86_400 + elapsed.seconds
    return seconds * _NS + int((match[2] or "").ljust(9, "0"))


def _iso(value: int) -> str:
    seconds, fraction = divmod(value, _NS)
    stamp = (_EPOCH + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%S")
    return stamp + (f".{fraction:09d}".rstrip("0") if fraction else "") + "Z"


@dataclass(frozen=True, slots=True)
class _Record:
    line: int
    received: int
    payload: str | None


def _records(lines: Iterable[str]) -> Iterator[_Record]:
    previous: int | None = None
    pending_disconnect: int | None = None
    for number, line in enumerate(lines, 1):
        row = line.strip()
        if not row:
            if pending_disconnect is None:
                pending_disconnect = number
            continue
        timestamp, delimiter, payload = row.partition(" ")
        try:
            received = _timestamp(timestamp)
            if not delimiter or (previous is not None and received < previous):
                raise InputValidationError(
                    "missing payload or out-of-order receipt time"
                )
        except ValueError as exc:
            raise InputValidationError(f"capture line {number}: {exc}") from exc
        if pending_disconnect is not None:
            # Tardis does not timestamp blank lines. Use the first possible
            # instant after the previous receipt, or wire order at equal times.
            effective = received if previous is None else min(previous + 1, received)
            yield _Record(pending_disconnect, effective, None)
            pending_disconnect = None
        yield _Record(number, received, payload)
        previous = received
    if pending_disconnect is not None and previous is not None:
        yield _Record(pending_disconnect, previous + 1, None)


_Action = Literal["new", "change", "delete"]


@dataclass(frozen=True, slots=True)
class _Level:
    action: _Action
    price: Decimal
    amount: Decimal


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InputValidationError("expected a nonnegative integer")
    return value


def _levels(value: object) -> Iterator[_Level]:
    if not isinstance(value, list):
        raise InputValidationError("book side must be an array of levels")
    for level in value:
        if not isinstance(level, list) or len(level) != 3:
            raise InputValidationError("expected [action, price, amount]")
        action, price, amount = level
        if action not in ("new", "change", "delete"):
            raise InputValidationError(f"unknown book action: {action!r}")
        if any(type(item) not in (Decimal, int) for item in (price, amount)):
            raise InputValidationError("price and amount must be JSON numbers")
        p, size = Decimal(price), Decimal(amount)
        if not p.is_finite() or not size.is_finite() or p <= 0 or size < 0:
            raise InputValidationError("invalid book price or amount")
        if (size == 0) != (action == "delete"):
            raise InputValidationError("only deletes may have zero amount")
        yield _Level(action, p, size)


@dataclass(frozen=True, slots=True)
class _Evidence:
    line: int
    received: int
    exchange: int
    change_id: int

    def report(self) -> BookEvidence:
        return {
            "line": self.line,
            "available_at": _iso(self.received),
            "exchange_time": _iso(self.exchange),
            "change_id": self.change_id,
        }


@dataclass(slots=True)
class _Book:
    bids: dict[Decimal, Decimal] = field(default_factory=dict)
    asks: dict[Decimal, Decimal] = field(default_factory=dict)
    snapshot: _Evidence | None = None
    last: _Evidence | None = None
    exchange_high_water: int = 0
    reason: str = "awaiting_snapshot"
    issue: dict[str, object] | None = None
    snapshots_applied: int = 0
    changes_applied: int = 0
    changes_ignored: int = 0
    sequence_gaps: int = 0

    def invalidate(self, reason: str, issue: dict[str, object]) -> None:
        self.bids.clear()
        self.asks.clear()
        self.snapshot = self.last = None
        self.reason, self.issue = reason, issue

    def apply(self, data: dict[str, object], record: _Record) -> None:
        kind = data.get("type")
        change_id = _integer(data["change_id"])
        evidence = _Evidence(
            record.line,
            record.received,
            _integer(data["timestamp"]) * _NS_PER_MS,
            change_id,
        )
        if kind == "snapshot":
            self.bids.clear()
            self.asks.clear()
            self.snapshot = evidence
            self.exchange_high_water = evidence.exchange
            self.reason, self.issue = "", None
            self.snapshots_applied += 1
        elif kind == "change":
            predecessor = _integer(data["prev_change_id"])
            if self.last is None:
                self.changes_ignored += 1
                return
            if self.last.change_id != predecessor or change_id <= predecessor:
                self.sequence_gaps += 1
                self.changes_ignored += 1
                self.invalidate(
                    "sequence_gap",
                    {
                        **evidence.report(),
                        "last_applied_change_id": self.last.change_id,
                        "prev_change_id": predecessor,
                    },
                )
                return
            self.changes_applied += 1
        else:
            raise InputValidationError(f"unsupported book message type: {kind!r}")
        for name, book in (("bids", self.bids), ("asks", self.asks)):
            for level in _levels(data[name]):
                if kind == "snapshot" and level.action != "new":
                    raise InputValidationError("snapshot levels must have action 'new'")
                if level.action == "delete":
                    book.pop(level.price, None)
                else:
                    book[level.price] = level.amount
        self.last = evidence
        self.exchange_high_water = max(self.exchange_high_water, evidence.exchange)

    def report(self, cutoff: int, max_age: int) -> BookState:
        flags: list[str] = []
        bid = ask = None
        if self.last is None:
            flags.append(self.reason)
        elif self.exchange_high_water > cutoff:
            # A regressive exchange timestamp does not erase the dependency
            # on a prior, already applied update with a later exchange time.
            flags.append("future_exchange_time")
        else:
            bid = max(self.bids, default=None)
            ask = min(self.asks, default=None)
            if cutoff - self.last.exchange > max_age:
                flags.append("stale")
            if bid is None or ask is None:
                flags.append("one_sided")
            elif bid > ask:
                flags.append("crossed")
        return {
            "flags": flags,
            "bid": None if bid is None else [str(bid), str(self.bids[bid])],
            "ask": None if ask is None else [str(ask), str(self.asks[ask])],
            "snapshot": self.snapshot.report() if self.snapshot else None,
            "last_update": self.last.report() if self.last else None,
            "issue": self.issue,
            "snapshots_applied": self.snapshots_applied,
            "changes_applied": self.changes_applied,
            "changes_ignored": self.changes_ignored,
            "sequence_gaps": self.sequence_gaps,
        }


def _apply(record: _Record, channels: dict[str, tuple[str, _Book]]) -> None:
    if record.payload is None:
        for _, book in channels.values():
            book.invalidate(
                "disconnect",
                {
                    "line": record.line,
                    "conservative_from": _iso(record.received),
                },
            )
        return
    try:
        message = json.loads(
            record.payload, parse_float=Decimal, parse_constant=Decimal
        )
        if message.get("method") != "subscription":
            return
        params = message["params"]
        selected = channels.get(params["channel"])
        if selected is None:
            return
        symbol, book = selected
        data = params["data"]
        if data["instrument_name"] != symbol:
            raise InputValidationError("book channel and payload instrument disagree")
        book.apply(data, record)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(f"capture line {record.line}: {exc}") from exc


def replay(
    lines: Iterable[str],
    symbols: Sequence[str],
    cutoffs: Sequence[str],
    *,
    max_age_ms: int = 30_000,
) -> list[BookCheckpoint]:
    """Project full native snapshots/deltas into deterministic as-of top of book.

    Cutoffs are inclusive UTC timestamps, strictly increasing. Do not reorder
    messages on the exchange clock. A cutoff past the final dated input fails
    rather than treating an unobserved tail as market continuity.
    """
    if not symbols or len(set(symbols)) != len(symbols) or not all(symbols):
        raise InputValidationError("specify nonempty, unique instrument symbols")
    times = [_timestamp(cutoff) for cutoff in cutoffs]
    if not times or sorted(set(times)) != times:
        raise InputValidationError("cutoffs must be strictly increasing")
    max_age = _integer(max_age_ms) * _NS_PER_MS
    channels = {f"book.{symbol}.raw": (symbol, _Book()) for symbol in sorted(symbols)}
    snapshots: list[BookCheckpoint] = []
    index = 0
    last_received: int | None = None

    def project(cutoff: int) -> BookCheckpoint:
        books = {
            symbol: book.report(cutoff, max_age) for symbol, book in channels.values()
        }
        return {
            "as_of": _iso(cutoff),
            "complete": all(not book["flags"] for book in books.values()),
            "books": books,
        }

    for record in _records(lines):
        while times[index] < record.received:
            snapshots.append(project(times[index]))
            index += 1
            if index == len(times):
                return snapshots
        _apply(record, channels)
        if record.payload is not None:
            last_received = record.received
    if last_received is not None and times[index] == last_received:
        snapshots.append(project(last_received))
        index += 1
    if index != len(times):
        raise InputValidationError("cutoff exceeds timestamped capture coverage")
    return snapshots


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="decoded Tardis raw HTTP response")
    parser.add_argument("--symbol", action="append", required=True)
    parser.add_argument(
        "--as-of", action="append", required=True, help="UTC cutoff (Z)"
    )
    parser.add_argument("--max-age-ms", type=int, default=30_000)
    args = parser.parse_args(argv)
    try:
        with args.capture.open(encoding="utf-8") as capture:
            snapshots = replay(
                capture, args.symbol, args.as_of, max_age_ms=args.max_age_ms
            )
    except (OSError, ValueError) as exc:
        print(f"tardis-deribit: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "source": "tardis/deribit",
                "level_fields": ["native_price", "native_amount"],
                "max_age_ms": args.max_age_ms,
                "snapshots": snapshots,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - local CLI entry point
    raise SystemExit(main())
