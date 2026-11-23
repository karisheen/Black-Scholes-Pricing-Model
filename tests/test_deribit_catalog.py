"""Consumer-visible historical catalog selection using synthetic metadata only."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from quant_pricing.deribit_catalog import choose_catalog_strikes
from quant_pricing.exceptions import InputValidationError

FUTURE = "BTC_USDC-28JUN24"
ANCHOR = datetime(2024, 6, 1, 12, tzinfo=timezone.utc)


def symbol(strike: int, side: str, **changes: object) -> dict[str, object]:
    return {
        "exchange": "deribit",
        "id": f"{FUTURE}-{strike}-{side}",
        "type": "option",
        "baseCurrency": "BTC",
        "quoteCurrency": "USDC",
        "availableSince": "2024-05-30T00:00:00.000Z",
        **changes,
    }


def pair(strike: int, **changes: object) -> list[dict[str, object]]:
    return [symbol(strike, side, **changes) for side in ("C", "P")]


def test_sparse_catalog_uses_actual_paired_strikes_and_returns_sorted_shortlist() -> (
    None
):
    rows = [
        *pair(57_125),
        *pair(62_350),
        symbol(65_000, "C"),
        *pair(66_777),
        symbol(65_001, "P"),
        *pair(90_123),
    ]
    choice = choose_catalog_strikes({"symbols": rows}, FUTURE, ANCHOR, 65_000, count=3)
    assert choice.strikes == (57_125, 62_350, 66_777)
    assert choice.paired_strikes_available == 4
    assert choice.excluded_same_or_future_day == 0
    assert (
        choose_catalog_strikes(
            {"symbols": rows[::-1], "nextCursor": None}, FUTURE, ANCHOR, 65_000, count=3
        )
        == choice
    )
    assert choose_catalog_strikes(
        {"symbols": rows}, FUTURE, ANCHOR, 65_000
    ).strikes == (57_125, 62_350, 66_777, 90_123)


def test_log_moneyness_not_absolute_strike_distance_and_lower_tie_break() -> None:
    rows = [*pair(50), *pair(200), *pair(160)]
    choice = choose_catalog_strikes({"symbols": rows}, FUTURE, ANCHOR, 100, count=2)
    assert choice.strikes == (50, 160)
    assert choice.paired_strikes_available == 3
    assert choose_catalog_strikes(
        {"symbols": [*pair(60), *pair(150)]}, FUTURE, ANCHOR, 100, count=1
    ).strikes == (150,)


def test_entire_first_archive_day_is_excluded_and_metric_counts_unique_symbols() -> (
    None
):
    rows = [
        *pair(60_000, availableSince="2024-05-31T23:59:59.999999999Z"),
        symbol(61_000, "C", availableSince="2024-05-31"),
        symbol(61_000, "P", availableSince="2024-06-01T00:00:00Z"),
        *pair(62_000, availableSince="2024-06-01"),
        *pair(63_000, availableSince="2024-06-02T00:00:00+00:00"),
        symbol(62_000, "C", availableSince="2024-06-01T00:00:00.000Z"),
    ]
    choice = choose_catalog_strikes({"symbols": rows}, FUTURE, ANCHOR, 62_000)
    assert choice.strikes == (60_000,)
    assert choice.paired_strikes_available == 1
    assert choice.excluded_same_or_future_day == 5
    # Availability advances only when the full first archive UTC day has elapsed.
    tomorrow = datetime(2024, 6, 2, tzinfo=timezone.utc)
    later = choose_catalog_strikes({"symbols": rows}, FUTURE, tomorrow, 62_000)
    assert later.strikes == (60_000, 61_000, 62_000)
    assert later.excluded_same_or_future_day == 2


def test_anchor_day_is_utc_not_local_calendar_date() -> None:
    rows = pair(60_000, availableSince="2024-06-01")
    local = datetime(2024, 6, 2, 1, tzinfo=timezone(timedelta(hours=3)))
    choice = choose_catalog_strikes({"symbols": rows}, FUTURE, local, 60_000)
    assert choice.strikes == ()
    assert choice.excluded_same_or_future_day == 2
    west = datetime(2024, 5, 31, 23, tzinfo=timezone(-timedelta(hours=3)))
    rows = pair(60_000, availableSince="2024-05-31")
    assert choose_catalog_strikes({"symbols": rows}, FUTURE, west, 60_000).strikes == (
        60_000,
    )


def test_available_to_and_quote_mark_fields_never_filter_or_rank() -> None:
    rows = [
        *pair(60_000, availableTo="2024-05-31", markPrice=999_999, bidPrice=None),
        *pair(70_000, availableTo="not a timestamp", markPrice=-1, askPrice=0),
    ]
    assert choose_catalog_strikes(
        {"symbols": rows}, FUTURE, ANCHOR, 60_000, count=1
    ).strikes == (60_000,)


@pytest.mark.parametrize(
    "changes",
    [
        {"exchange": "Deribit"},
        {"exchange": "okx"},
        {"type": "future"},
        {"type": "Option"},
        {"baseCurrency": "btc"},
        {"baseCurrency": "ETH"},
        {"quoteCurrency": "BTC"},
        {"quoteCurrency": "usdc"},
        {"quoteCurrency": "USD"},
    ],
)
def test_exact_metadata_scope_is_required(changes: dict[str, object]) -> None:
    # Malformed availability outside the exact scope is irrelevant.
    rows = pair(60_000, availableSince="irrelevant", **changes)
    choice = choose_catalog_strikes({"symbols": rows}, FUTURE, ANCHOR, 60_000)
    assert choice.strikes == ()
    assert choice.paired_strikes_available == 0
    assert choice.excluded_same_or_future_day == 0


@pytest.mark.parametrize(
    "identifier",
    [
        "BTC-28JUN24-60000-C",
        "BTC_USDC-27JUN24-60000-C",
        f"{FUTURE}-60000.5-C",
        f"{FUTURE}-060000-C",
        f"{FUTURE}-0-C",
        f"{FUTURE}--60000-C",
        f"{FUTURE}-60000-c",
        f"{FUTURE}-60000-X",
        f"{FUTURE}-60000-C-extra",
        f"{FUTURE}-60000-C\n",
    ],
)
def test_symbol_identifier_must_exactly_match_contract_prefix(identifier: str) -> None:
    rows = [symbol(60_000, "C", id=identifier), symbol(60_000, "P")]
    assert (
        choose_catalog_strikes({"symbols": rows}, FUTURE, ANCHOR, 60_000).strikes == ()
    )


def test_equivalent_duplicate_availability_is_deduplicated() -> None:
    rows = [
        *pair(60_000),
        symbol(60_000, "C", availableSince="2024-05-30"),
        symbol(60_000, "P", availableSince="2024-05-30T00:00:00+00:00"),
    ]
    choice = choose_catalog_strikes({"symbols": rows}, FUTURE, ANCHOR, 60_000)
    assert choice.strikes == (60_000,)
    assert choice.paired_strikes_available == 1


@pytest.mark.parametrize(
    "first,second",
    [
        ("2024-05-30", "2024-06-01"),
        ("2024-05-30T00:00:00Z", "2024-05-30T01:00:00Z"),
        ("2024-05-30T00:00:00.000000001Z", "2024-05-30T00:00:00.000000002Z"),
    ],
)
def test_conflicting_duplicate_availability_is_never_optimistically_selected(
    first: str, second: str
) -> None:
    rows = [
        *pair(60_000, availableSince=first),
        symbol(60_000, "C", availableSince=second),
    ]
    for ordered in (rows, rows[::-1]):
        with pytest.raises(InputValidationError, match="conflicting duplicate"):
            choose_catalog_strikes({"symbols": ordered}, FUTURE, ANCHOR, 60_000)


def test_conflicting_duplicate_scope_is_rejected_even_if_one_row_is_out_of_scope() -> (
    None
):
    rows = [*pair(60_000), symbol(60_000, "C", quoteCurrency="BTC")]
    for ordered in (rows, rows[::-1]):
        with pytest.raises(InputValidationError, match="conflicting duplicate scope"):
            choose_catalog_strikes({"symbols": ordered}, FUTURE, ANCHOR, 60_000)


@pytest.mark.parametrize("cursor", ["next-page", "", False, 0, [], {}])
def test_every_non_null_next_cursor_rejects_incomplete_catalog(cursor: object) -> None:
    with pytest.raises(InputValidationError, match="incomplete"):
        choose_catalog_strikes(
            {"symbols": pair(60_000), "nextCursor": cursor}, FUTURE, ANCHOR, 60_000
        )


@pytest.mark.parametrize(
    "catalog",
    [None, [], {}, {"symbols": {}}, {"symbols": [None]}, {"symbols": [{}]}],
)
def test_malformed_catalog_structure_is_rejected(catalog: object) -> None:
    with pytest.raises(InputValidationError):
        choose_catalog_strikes(catalog, FUTURE, ANCHOR, 60_000)


@pytest.mark.parametrize(
    "available",
    [
        None,
        123,
        "",
        "2024-05-30T00:00:00",
        "2024-05-30T00:00:00+01:00",
        "2024-05-30T00:00:00-00:00",
        "2024-05-30 00:00:00Z",
        "2024-02-30",
        "2024-05-30T24:00:00Z",
    ],
)
def test_relevant_availability_must_be_valid_utc(available: object) -> None:
    with pytest.raises(InputValidationError, match="availableSince"):
        choose_catalog_strikes(
            {"symbols": pair(60_000, availableSince=available)}, FUTURE, ANCHOR, 60_000
        )


@pytest.mark.parametrize("forward", [0, -1, float("nan"), float("inf"), True, "60000"])
def test_invalid_forward_is_rejected(forward: Any) -> None:
    with pytest.raises(InputValidationError, match="forward"):
        choose_catalog_strikes({"symbols": []}, FUTURE, ANCHOR, forward)


@pytest.mark.parametrize("count", [0, -1, 1.5, True, "9", None])
def test_invalid_selection_count_is_rejected(count: Any) -> None:
    with pytest.raises(InputValidationError, match="count"):
        choose_catalog_strikes({"symbols": []}, FUTURE, ANCHOR, 60_000, count=count)


@pytest.mark.parametrize("as_of", [datetime(2024, 6, 1), "2024-06-01", None])
def test_anchor_requires_timezone_aware_datetime(as_of: Any) -> None:
    with pytest.raises(InputValidationError, match="as_of"):
        choose_catalog_strikes({"symbols": []}, FUTURE, as_of, 60_000)


@pytest.mark.parametrize(
    "future",
    [
        "BTC-28JUN24",
        "ETH_USDC-28JUN24",
        "BTC_USDC-PERPETUAL",
        "BTC_USDC-31JUN24",
        "BTC_USDC-28XYZ24",
        "",
        None,
    ],
)
def test_future_requires_a_valid_dated_linear_btc_usdc_contract(future: Any) -> None:
    with pytest.raises(InputValidationError, match="future_symbol"):
        choose_catalog_strikes({"symbols": []}, future, ANCHOR, 60_000)


def test_empty_catalog_is_an_empty_shortlist_not_a_selection_error() -> None:
    choice = choose_catalog_strikes({"symbols": []}, FUTURE, ANCHOR, 60_000)
    assert choice.strikes == ()
    assert choice.paired_strikes_available == 0
    assert choice.excluded_same_or_future_day == 0


@pytest.mark.parametrize("forward", [5e-324, 1.7e308])
def test_extreme_positive_forward_preserves_log_distance_ranking(
    forward: float,
) -> None:
    choice = choose_catalog_strikes(
        {"symbols": [*pair(1), *pair(100)]}, FUTURE, ANCHOR, forward, count=1
    )
    assert choice.strikes == ((1,) if forward < 1 else (100,))
