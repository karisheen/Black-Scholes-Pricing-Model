"""Point-in-time selection, quote evidence, and contract unit invariants."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from quant_pricing.black_scholes import OptionType
from quant_pricing.market import (
    Forward,
    FrozenQuoteProvider,
    MarketSnapshot,
    OptionContract,
    Portfolio,
    Position,
    Quote,
    SnapshotProvider,
    SnapshotQuote,
    Underlying,
)

AS_OF = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
SPOT = Underlying("BTC-USD", "USD")
FORWARD = Forward("BTC-USD-FWD", SPOT, AS_OF + timedelta(days=30))
OPTION = OptionContract(
    symbol="BTC-USD-C100000",
    underlying=SPOT,
    option_type=OptionType.CALL,
    strike=100_000.0,
    expiry=FORWARD.expiry,
    exercise_style="european",
    settlement_style="cash",
    multiplier=0.1,
    quote_currency="BTC",
    settlement_currency="BTC",
    premium_basis="per_contract",
)


def test_coin_premium_and_signed_positions_have_distinct_units() -> None:
    assert OPTION.premium_per_unit(0.002) == pytest.approx(0.02)
    assert OPTION.quote_currency == OPTION.settlement_currency == "BTC"
    per_unit = replace(OPTION, premium_basis="per_unit")
    assert per_unit.premium_per_unit(0.002) == pytest.approx(0.002)
    short = Position(OPTION, -3.0)
    assert Portfolio((short,)).positions[0].underlying_units == pytest.approx(-0.3)


def test_snapshot_uses_both_clocks_and_does_not_mix_sources() -> None:
    quote = Quote(SPOT, 99_999.0, 100_001.0, 1.0, 2.0, AS_OF, AS_OF, "fixture", 1)
    late = replace(
        quote,
        bid=100_000.0,
        available_at=AS_OF + timedelta(seconds=30),
        source_sequence=2,
    )
    future_dated = replace(
        quote, quote_time=AS_OF + timedelta(seconds=5), source_sequence=3
    )
    other_feed = replace(quote, bid=80_000.0, source="other", source_sequence=100)
    events = (late, quote, future_dated, other_feed, quote)
    provider: SnapshotProvider = FrozenQuoteProvider(
        (SPOT, FORWARD, OPTION), events, "fixture", timedelta(minutes=1)
    )

    same_cutoff = AS_OF.astimezone(timezone(timedelta(hours=-5)))
    snapshot = provider.snapshot(same_cutoff)
    assert snapshot.as_of == AS_OF
    assert snapshot.as_of.utcoffset() == timedelta(0)
    assert [row.quote for row in snapshot.rows] == [quote, None, None]
    assert [row.flags for row in snapshot.rows] == [(), ("missing",), ("missing",)]
    assert snapshot == replace(provider, quotes=tuple(reversed(events))).snapshot(AS_OF)
    assert (
        provider.snapshot(AS_OF + timedelta(seconds=30)).rows[0].quote == future_dated
    )

    only_late = FrozenQuoteProvider(
        (SPOT,), (late, quote), "fixture", timedelta(minutes=1)
    )
    assert only_late.snapshot(AS_OF + timedelta(seconds=29)).rows[0].quote == quote
    assert only_late.snapshot(AS_OF + timedelta(seconds=30)).rows[0].quote == late


def test_snapshot_preserves_bad_quotes_without_falling_back() -> None:
    old = Quote(OPTION, 0.001, 0.002, 3.0, 4.0, AS_OF, AS_OF, "fixture", 1)
    crossed = replace(old, bid=0.003, source_sequence=2)
    invalid_forward = Quote(FORWARD, -1.0, 2.0, -1.0, 4.0, AS_OF, AS_OF, "fixture", 7)
    provider = FrozenQuoteProvider(
        (OPTION, FORWARD),
        (old, crossed, invalid_forward),
        "fixture",
        timedelta(seconds=10),
    )

    option, forward = provider.snapshot(AS_OF + timedelta(seconds=11)).rows
    assert option.quote == crossed
    assert set(option.flags) == {"crossed", "stale"}
    assert forward.quote == invalid_forward
    assert set(forward.flags) == {"invalid_price", "invalid_size", "stale"}
    assert provider.snapshot(AS_OF + timedelta(seconds=10)).rows[0].flags == (
        "crossed",
    )


def test_ambiguous_eligible_quote_order_is_rejected() -> None:
    quote = Quote(SPOT, 99.0, 101.0, 1.0, 2.0, AS_OF, AS_OF, "fixture", 1)
    provider = FrozenQuoteProvider(
        (SPOT,), (quote, replace(quote, bid=98.0)), "fixture", timedelta(minutes=1)
    )
    with pytest.raises(ValueError):
        provider.snapshot(AS_OF)


def test_naive_as_of_is_not_interpreted_in_local_time() -> None:
    provider = FrozenQuoteProvider((SPOT,), (), "fixture", timedelta(minutes=1))
    with pytest.raises(ValueError, match="timezone"):
        provider.snapshot(AS_OF.replace(tzinfo=None))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("symbol", ""),
        ("strike", 0.0),
        ("multiplier", float("inf")),
        ("exercise_style", "unknown"),
        ("settlement_style", "unknown"),
        ("premium_basis", "unknown"),
    ],
)
def test_incomplete_contract_economics_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        replace(OPTION, **{field: value})


def test_position_rejects_nonfinite_contracts_or_exposure() -> None:
    with pytest.raises(ValueError, match="finite"):
        Position(OPTION, float("nan"))
    with pytest.raises(ValueError, match="finite"):
        Position(replace(OPTION, multiplier=1e308), 2.0)


def test_provider_rejects_unknown_instruments_and_ambiguous_symbol_keys() -> None:
    quote = Quote(FORWARD, 99.0, 101.0, 1.0, 2.0, AS_OF, AS_OF, "fixture", 1)
    with pytest.raises(ValueError, match="declared universe"):
        FrozenQuoteProvider((SPOT,), (quote,), "fixture", timedelta(minutes=1))
    with pytest.raises(ValueError, match="symbols must be unique"):
        FrozenQuoteProvider(
            (SPOT, replace(FORWARD, symbol=SPOT.symbol)), (), "fixture", timedelta(0)
        )
    with pytest.raises(ValueError, match="nonnegative"):
        FrozenQuoteProvider((SPOT,), (), "fixture", timedelta(seconds=-1))
    with pytest.raises(ValueError, match="source_sequence"):
        replace(quote, source_sequence=-1)


def test_custom_snapshots_cannot_claim_future_or_other_source_quotes() -> None:
    quote = Quote(SPOT, 99.0, 101.0, 1.0, 2.0, AS_OF, AS_OF, "fixture", 1)
    row = SnapshotQuote(SPOT, quote, ())
    with pytest.raises(ValueError, match="future quotes"):
        MarketSnapshot(AS_OF - timedelta(seconds=1), "fixture", (row,))
    late = replace(quote, available_at=AS_OF + timedelta(seconds=1))
    with pytest.raises(ValueError, match="future quotes"):
        MarketSnapshot(AS_OF, "fixture", (SnapshotQuote(SPOT, late, ()),))
    with pytest.raises(ValueError, match="declared source"):
        MarketSnapshot(AS_OF, "other", (row,))
    with pytest.raises(ValueError, match="symbols must be unique"):
        MarketSnapshot(AS_OF, "fixture", (row, row))


def test_snapshot_rows_cannot_mislabel_missing_or_mismatched_quotes() -> None:
    quote = Quote(SPOT, 99.0, 101.0, 1.0, 2.0, AS_OF, AS_OF, "fixture", 1)
    with pytest.raises(ValueError, match="absent quote"):
        SnapshotQuote(SPOT, None, ())
    with pytest.raises(ValueError, match="match the snapshot instrument"):
        SnapshotQuote(FORWARD, quote, ())
    with pytest.raises(ValueError, match="present quote"):
        SnapshotQuote(SPOT, quote, ("missing",))
