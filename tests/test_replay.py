"""Economic replay behavior against real partitioned Parquet on local disk."""

from __future__ import annotations

import builtins
import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_pricing.black_scholes import OptionType
from quant_pricing.exceptions import InputValidationError
from quant_pricing.market import (
    Forward,
    FrozenQuoteProvider,
    OptionContract,
    Quote,
    SnapshotProvider,
    Underlying,
)
from quant_pricing.replay import (
    IngestResult,
    ParquetReplayProvider,
    RawObservation,
    SyntheticProvider,
    SyntheticTick,
)

START = datetime(2025, 1, 2, tzinfo=timezone.utc)
SPOT = Underlying("BTC-USD", "USD")
FORWARD = Forward("BTC-FWD", SPOT, START + timedelta(days=30))
OPTION = OptionContract(
    "BTC-C100",
    SPOT,
    OptionType.CALL,
    100.0,
    FORWARD.expiry,
    "european",
    "cash",
    10.0,
    "USD",
    "USD",
    "per_contract",
)
SOURCE = "venue/a=b"
AGE = timedelta(seconds=30)


def quote(sequence: int = 40, **changes: object) -> Quote:
    base = Quote(SPOT, 99.0, 101.0, 1.0, 2.0, START, START, SOURCE, sequence)
    return replace(base, **changes)


def observation(event: Quote, payload: bytes = b"\x00raw\xff") -> RawObservation:
    return RawObservation(event, payload, "capture/session-01:line-7")


def provider(root: Path, instruments: tuple = (SPOT,)) -> ParquetReplayProvider:
    return ParquetReplayProvider(root, instruments, SOURCE, AGE)


def test_raw_typed_and_normalized_fields_share_real_parquet_record(
    tmp_path: Path,
) -> None:
    per_unit = replace(OPTION, symbol="BTC-UNIT", premium_basis="per_unit")
    instruments = (SPOT, FORWARD, OPTION, per_unit)
    events = tuple(
        observation(quote(instrument=instrument, bid=20.0, ask=22.0))
        for instrument in instruments
    )
    store = provider(tmp_path, instruments)
    assert store.ingest(events) == IngestResult(4, 0)
    assert set(store.observations()) == set(events)
    files = sorted(
        tmp_path.glob("v1/source=venue%2Fa%3Db/event_date=2025-01-02/*.parquet")
    )
    assert len(files) == 4
    rows = {}
    for path in files:
        assert path.read_bytes()[:4] == b"PAR1"
        parquet = pq.ParquetFile(path)
        assert parquet.schema_arrow.metadata == {b"quant_pricing.replay.version": b"1"}
        assert parquet.schema_arrow.field("quote_time").type == pa.timestamp(
            "us", "UTC"
        )
        row = parquet.read().to_pylist()[0]
        rows[row["symbol"]] = row
        assert row["raw_payload"] == b"\x00raw\xff"
        assert row["provenance"] == "capture/session-01:line-7"
        assert row["raw_bid"] == {"type": "float", "value": float(20).hex()}
        assert row["quote_currency"] == "USD"
        assert row["premium_basis"] == "per_unit"
    assert rows[OPTION.symbol]["normalized_bid"] == 2.0
    assert rows[OPTION.symbol]["normalized_ask"] == 2.2
    assert rows[per_unit.symbol]["normalized_bid"] == 20.0
    assert rows[FORWARD.symbol]["normalized_bid"] == 20.0
    snapshot = store.snapshot(START)
    assert snapshot.complete
    # Snapshot retains quoted contract amounts; consumers normalize exactly once.
    assert snapshot.rows[2].quote.bid == 20.0
    assert OPTION.premium_per_unit(snapshot.rows[2].quote.bid) == 2.0


def test_utc_event_partition_not_arrival_or_local_date(tmp_path: Path) -> None:
    local = datetime(2025, 1, 1, 20, 0, 0, 123456, tzinfo=timezone(timedelta(hours=-5)))
    event = quote(quote_time=local, available_at=local + timedelta(days=1))
    store = provider(tmp_path)
    store.ingest((observation(event),))
    assert len(list(tmp_path.glob("v1/*/event_date=2025-01-02/*.parquet"))) == 1
    restored = store.observations()[0].quote
    assert restored == event
    assert restored.quote_time.microsecond == 123456
    assert restored.quote_time.tzinfo == timezone.utc
    assert not store.snapshot(restored.quote_time).complete
    assert store.snapshot(restored.available_at).rows[0].quote == event


def test_idempotent_restart_and_conflicting_updates(tmp_path: Path) -> None:
    event = observation(quote())
    store = provider(tmp_path)
    assert store.ingest((event, event)) == IngestResult(1, 1)
    assert provider(tmp_path).ingest((event,)) == IngestResult(0, 1)
    conflicts = (
        replace(event, quote=replace(event.quote, bid=98.0)),
        replace(event, quote=replace(event.quote, available_at=START + AGE)),
        replace(event, raw_payload=b"changed"),
        replace(event, provenance="other capture"),
    )
    for conflict in conflicts:
        with pytest.raises(InputValidationError, match="conflicting event identity"):
            store.ingest((conflict,))
    assert store.observations() == (event,)
    # Event timestamp is part of identity: a reused sequence at another time is
    # distinct, but sequence-reset semantics are deliberately not inferred.
    new_event = observation(quote(quote_time=START + timedelta(seconds=1)))
    assert store.ingest((new_event,)) == IngestResult(1, 0)


def test_both_clocks_gaps_late_fill_and_input_order_independence(
    tmp_path: Path,
) -> None:
    first = quote(40)
    delayed = quote(
        41,
        quote_time=START + timedelta(seconds=1),
        available_at=START + timedelta(seconds=20),
    )
    third = quote(
        42,
        quote_time=START + timedelta(seconds=2),
        available_at=START + timedelta(seconds=2),
        bid=100.0,
    )
    future = quote(50, quote_time=START + timedelta(seconds=100))
    tape = tuple(map(observation, (future, third, delayed, first)))
    left = provider(tmp_path / "left")
    right = provider(tmp_path / "right")
    left.ingest(tape)
    right.ingest(reversed(tape))
    assert left.snapshot(START).complete  # No invented sequence 1..39 gap.
    for seconds in (0, 2, 19, 20, 100):
        cutoff = START + timedelta(seconds=seconds)
        assert left.snapshot(cutoff) == right.snapshot(cutoff)
    at_two = left.snapshot(START + timedelta(seconds=2))
    assert at_two.rows[0].quote == third
    assert not at_two.complete
    (gap,) = at_two.rows[0].sequence_gaps
    assert (gap.first_missing, gap.last_missing) == (41, 41)
    assert (gap.before, gap.after) == (first, third)
    filled = left.snapshot(START + timedelta(seconds=20))
    assert filled.complete
    assert not filled.rows[0].sequence_gaps
    assert filled.rows[0].quote == third  # Late arrival does not replace later event.
    assert left.snapshot(START + timedelta(seconds=100)).rows[0].quote == future


def test_source_isolation_and_declared_subset(tmp_path: Path) -> None:
    store = provider(tmp_path, (SPOT, FORWARD))
    spot = observation(quote())
    store.ingest((spot, observation(quote(instrument=FORWARD))))
    other = ParquetReplayProvider(tmp_path, (SPOT,), "other", AGE)
    other.ingest((observation(quote(900, source="other", bid=1.0)),))
    subset = provider(tmp_path)
    assert subset.snapshot(START).rows[0].quote == spot.quote
    assert subset.snapshot(START).complete
    assert len(subset.observations()) == 2
    assert len(other.observations()) == 1
    for bad in (quote(source="wrong"), quote(instrument=OPTION)):
        with pytest.raises(InputValidationError, match="source/instrument mismatch"):
            store.ingest((observation(bad),))


def test_future_metadata_never_invalidates_past_snapshot(tmp_path: Path) -> None:
    store = provider(tmp_path)
    store.ingest((observation(quote()),))
    changed_spot = replace(SPOT, currency="EUR")
    new = provider(tmp_path, (changed_spot,))
    new.ingest(
        (observation(quote(41, instrument=changed_spot, available_at=START + AGE)),)
    )
    assert store.snapshot(START).complete
    with pytest.raises(InputValidationError, match="declared universe"):
        store.snapshot(START + AGE)


def test_quality_flags_no_fallback_and_faithful_invalid_types(tmp_path: Path) -> None:
    store = provider(tmp_path, (SPOT, FORWARD, OPTION))
    old = quote(40)
    crossed = quote(41, bid=103.0, ask=101.0, bid_size=-1.0)
    invalid = quote(
        7,
        instrument=FORWARD,
        bid=True,
        ask=math.inf,
        bid_size=2**60 + 1,
        ask_size=math.nan,
    )
    store.ingest(tuple(map(observation, (old, crossed, invalid))))
    assert store.ingest((observation(invalid),)) == IngestResult(0, 1)
    snapshot = store.snapshot(START + AGE + timedelta(microseconds=1))
    assert not snapshot.complete
    assert snapshot.rows[0].quote == crossed
    assert snapshot.rows[0].flags == ("stale", "crossed", "invalid_size")
    assert snapshot.rows[1].flags == ("stale", "invalid_price", "invalid_size")
    assert snapshot.rows[2].flags == ("missing",)
    restored = snapshot.rows[1].quote
    assert restored.bid is True
    assert restored.bid_size == 2**60 + 1
    assert isinstance(restored.bid_size, int)
    assert math.isnan(restored.ask_size)
    raw = next(
        event for event in store.observations() if event.quote.instrument == FORWARD
    )
    assert raw.raw_payload == b"\x00raw\xff"
    for path in tmp_path.glob("v1/*/*/*.parquet"):
        row = pq.ParquetFile(path).read().to_pylist()[0]
        if row["symbol"] == FORWARD.symbol:
            assert math.isnan(row["normalized_bid"])
            assert row["raw_bid"]["type"] == "bool"


def test_atomic_visibility_recovery_and_partial_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = provider(tmp_path)
    first = observation(quote())
    store.ingest((first,))
    second = observation(quote(41))
    import quant_pricing.replay as replay

    def interrupted(*args: object) -> None:
        # Before rename, neither the normalized nor raw second event is visible.
        assert store.observations() == (first,)
        raise OSError("interrupted rename")

    with monkeypatch.context() as patch:
        patch.setattr(replay.os, "replace", interrupted)
        with pytest.raises(OSError, match="interrupted rename"):
            store.ingest((second,))
    assert not list(tmp_path.rglob(".pending-*"))
    partition = next(tmp_path.glob("v1/*/event_date=*"))
    (partition / ".pending-abandoned").write_bytes(b"partial")
    assert store.observations() == (first,)
    third = observation(quote(42))
    conflict = replace(third, raw_payload=b"different")
    with pytest.raises(InputValidationError, match="conflicting event identity"):
        store.ingest((second, third, conflict))
    assert store.ingest((first, second, third)) == IngestResult(0, 3)
    assert store.snapshot(START).complete


def test_schema_version_record_count_and_partition_integrity(tmp_path: Path) -> None:
    store = provider(tmp_path)
    store.ingest((observation(quote()),))
    path = next(tmp_path.glob("v1/*/*/*.parquet"))
    table = pq.ParquetFile(path).read()
    pq.write_table(
        table.replace_schema_metadata({b"quant_pricing.replay.version": b"2"}), path
    )
    with pytest.raises(InputValidationError, match="unsupported replay schema"):
        store.observations()
    pq.write_table(pa.concat_tables([table, table]), path)
    with pytest.raises(InputValidationError, match="exactly one event"):
        store.observations()
    row = table.to_pylist()[0]
    row["source"] = "other"
    pq.write_table(pa.Table.from_pylist([row], schema=table.schema), path)
    with pytest.raises(InputValidationError, match="partition"):
        store.snapshot(START)


def test_optional_dependency_error_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = builtins.__import__

    def without_arrow(name: str, *args: object, **kwargs: object) -> object:
        if name == "pyarrow":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_arrow)
    with pytest.raises(ImportError, match=r"quant-pricing\[replay\]"):
        provider(tmp_path).ingest(())
    # Synthetic operation has no optional dependency.
    assert SyntheticProvider((SPOT,), (quote(),), SOURCE, AGE).snapshot(START).complete


def test_synthetic_schedule_generates_sequences_and_delayed_availability() -> None:
    schedule = (
        SyntheticTick(SPOT, START + timedelta(seconds=2), 102.0, 103.0),
        SyntheticTick(SPOT, START, 99.0, 100.0),
        SyntheticTick(
            SPOT, START + timedelta(seconds=1), 101.0, 102.0, delay=timedelta(seconds=9)
        ),
        SyntheticTick(FORWARD, START, 105.0, 106.0),
    )
    left = SyntheticProvider.from_schedule((SPOT, FORWARD), schedule, SOURCE, AGE)
    right = SyntheticProvider.from_schedule(
        (SPOT, FORWARD),
        tuple(reversed(schedule)),
        SOURCE,
        AGE,
    )
    assert left == right
    provider_contract: SnapshotProvider = left
    before = provider_contract.snapshot(START - timedelta(microseconds=1))
    assert all(row.flags == ("missing",) for row in before.rows)
    snapshot = left.snapshot(START + timedelta(seconds=2))
    assert snapshot.rows[0].quote.source_sequence == 3
    (gap,) = snapshot.rows[0].sequence_gaps
    assert (gap.first_missing, gap.last_missing) == (2, 2)
    assert snapshot.rows[1].quote.source_sequence == 1
    assert left.snapshot(START + timedelta(seconds=10)).complete
    assert left.snapshot(START + timedelta(seconds=10)).rows[0].quote.bid == 102.0


def test_synthetic_frozen_inputs_and_errors() -> None:
    events = [quote()]
    instruments = [SPOT]
    frozen = SyntheticProvider(instruments, events, SOURCE, AGE)
    events.clear()
    instruments.clear()
    assert frozen.snapshot(START).complete
    tick = SyntheticTick(SPOT, START, 1.0, 2.0)
    with pytest.raises(InputValidationError, match="ambiguous synthetic"):
        SyntheticProvider.from_schedule((SPOT,), (tick, tick), SOURCE, AGE)
    with pytest.raises(InputValidationError, match="delay"):
        replace(tick, delay=timedelta(microseconds=-1))
    with pytest.raises(InputValidationError, match="timezone-aware"):
        replace(tick, quote_time=START.replace(tzinfo=None))
    with pytest.raises(InputValidationError, match="nonnegative"):
        SyntheticProvider((SPOT,), (), SOURCE, -AGE)


def test_gap_provenance_is_eligible_deterministic_and_per_instrument() -> None:
    first = quote(100)
    repeated = quote(100, quote_time=START + timedelta(seconds=1))
    end = quote(104, quote_time=START + timedelta(seconds=2))
    unrelated = quote(102, instrument=FORWARD)
    other = quote(101, source="other")
    tape = (repeated, first, end, unrelated, other, first)
    provider = FrozenQuoteProvider((SPOT, FORWARD), tape, SOURCE, AGE)
    cutoff = START + timedelta(seconds=2)
    snapshot = provider.snapshot(cutoff)
    assert snapshot == replace(provider, quotes=tuple(reversed(tape))).snapshot(cutoff)
    (gap,) = snapshot.rows[0].sequence_gaps
    assert (gap.first_missing, gap.last_missing) == (101, 103)
    assert gap.before == first
    assert gap.after == end
    assert not snapshot.rows[1].sequence_gaps
    assert not snapshot.complete


def test_empty_tape_and_snapshot_clock_validation(tmp_path: Path) -> None:
    store = provider(tmp_path)
    assert store.observations() == ()
    assert store.ingest(()) == IngestResult(0, 0)
    assert store.snapshot(START).rows[0].flags == ("missing",)
    with pytest.raises(InputValidationError, match="timezone-aware"):
        store.snapshot(START.replace(tzinfo=None))


@pytest.mark.parametrize("sequence", [True, 1.5, 2**63])
def test_sequence_storage_type_is_never_silently_coerced(
    tmp_path: Path, sequence: int
) -> None:
    with pytest.raises(InputValidationError, match="signed 64-bit integer"):
        provider(tmp_path).ingest((observation(quote(sequence)),))
    assert not list(tmp_path.rglob("*.parquet"))
