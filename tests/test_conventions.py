"""Exact elapsed time, calendar coverage and exchange wall-clock resolution."""

from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from quant_pricing.conventions import (
    BusinessDayConvention,
    DayCount,
    TradingCalendar,
    utc_timestamp,
    year_fraction,
)

UTC = timezone.utc
NY = ZoneInfo("America/New_York")
CALENDAR = TradingCalendar(
    "XNYS-explicit-2024-sample",
    NY,
    date(2024, 1, 1),
    date(2024, 12, 31),
    frozenset({date(2024, 1, 1), date(2024, 7, 4), date(2024, 11, 28)}),
    early_closes=((date(2024, 11, 29), time(13)),),
)


@pytest.mark.parametrize(
    ("convention", "expected"),
    [
        (DayCount.ACT_365F, 366 / 365),
        (DayCount.ACT_360, 366 / 360),
        (DayCount.ACT_ACT_ISDA, 1.0),
    ],
)
def test_leap_year_day_counts(convention: DayCount, expected: float) -> None:
    assert year_fraction(
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, tzinfo=UTC),
        convention,
    ) == pytest.approx(expected)


def test_isda_splits_calendar_year_and_preserves_intraday_time() -> None:
    start = datetime(2023, 12, 31, 12, tzinfo=UTC)
    end = datetime(2024, 1, 2, 6, tzinfo=UTC)
    assert year_fraction(start, end, DayCount.ACT_ACT_ISDA) == pytest.approx(
        0.5 / 365 + 1.25 / 366
    )
    for convention in DayCount:
        assert year_fraction(start, start, convention) == 0
    assert year_fraction(
        datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 2, 2, tzinfo=UTC),
        DayCount.ACT_ACT_ISDA,
    ) == pytest.approx(31 / 366)


def test_dst_elapsed_time_uses_utc_not_local_subtraction() -> None:
    start = datetime(2024, 3, 9, 16, tzinfo=NY)
    end = datetime(2024, 3, 10, 16, tzinfo=NY)
    assert year_fraction(start, end) == pytest.approx(23 / (24 * 365))
    assert utc_timestamp(start) == datetime(2024, 3, 9, 21, tzinfo=UTC)
    assert year_fraction(
        datetime(2024, 11, 2, 16, tzinfo=NY),
        datetime(2024, 11, 3, 16, tzinfo=NY),
    ) == pytest.approx(25 / (24 * 365))


@pytest.mark.parametrize("value", [datetime(2024, 1, 1), date(2024, 1, 1)])
def test_naive_or_date_only_valuation_is_rejected(value: datetime) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        utc_timestamp(value)


def test_reversed_time_and_unknown_day_count_are_rejected() -> None:
    end = datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="precede"):
        year_fraction(end + timedelta(seconds=1), end)
    with pytest.raises(ValueError):
        year_fraction(end, end, "unknown")  # type: ignore[arg-type]


def test_calendar_holiday_weekend_and_roll_conventions() -> None:
    assert not CALENDAR.is_business_day(date(2024, 7, 4))
    assert not CALENDAR.is_business_day(date(2024, 7, 6))
    assert CALENDAR.is_business_day(date(2024, 7, 5))
    assert CALENDAR.adjust(date(2024, 7, 4)) == date(2024, 7, 5)
    assert CALENDAR.adjust(date(2024, 7, 6), BusinessDayConvention.PRECEDING) == date(
        2024, 7, 5
    )
    assert CALENDAR.adjust(
        date(2024, 8, 31), BusinessDayConvention.MODIFIED_FOLLOWING
    ) == date(2024, 8, 30)
    assert CALENDAR.adjust(
        date(2024, 7, 4), BusinessDayConvention.MODIFIED_FOLLOWING
    ) == date(2024, 7, 5)
    assert CALENDAR.adjust(date(2024, 7, 4), BusinessDayConvention.UNADJUSTED) == date(
        2024, 7, 4
    )


def test_expiry_exact_dst_shift_holiday_and_half_day() -> None:
    assert CALENDAR.resolve_expiry(date(2024, 3, 8)) == datetime(
        2024, 3, 8, 21, tzinfo=UTC
    )
    assert CALENDAR.resolve_expiry(date(2024, 3, 11)) == datetime(
        2024, 3, 11, 20, tzinfo=UTC
    )
    assert CALENDAR.resolve_expiry(date(2024, 7, 4), time(9, 30)) == datetime(
        2024, 7, 3, 13, 30, tzinfo=UTC
    )
    assert CALENDAR.resolve_expiry(date(2024, 11, 29)) == datetime(
        2024, 11, 29, 18, tzinfo=UTC
    )
    assert CALENDAR.resolve_expiry(date(2024, 11, 29), time(16)) == datetime(
        2024, 11, 29, 21, tzinfo=UTC
    )


def test_ambiguous_and_nonexistent_expiry_times_need_explicit_policy() -> None:
    unadjusted = BusinessDayConvention.UNADJUSTED
    with pytest.raises(ValueError, match="nonexistent"):
        CALENDAR.resolve_expiry(date(2024, 3, 10), time(2, 30), unadjusted)
    with pytest.raises(ValueError, match="ambiguous"):
        CALENDAR.resolve_expiry(date(2024, 11, 3), time(1, 30), unadjusted)
    early = CALENDAR.resolve_expiry(date(2024, 11, 3), time(1, 30), unadjusted, fold=0)
    late = CALENDAR.resolve_expiry(date(2024, 11, 3), time(1, 30), unadjusted, fold=1)
    assert early == datetime(2024, 11, 3, 5, 30, tzinfo=UTC)
    assert late - early == timedelta(hours=1)
    with pytest.raises(ValueError, match="explicit time"):
        CALENDAR.resolve_expiry(date(2024, 11, 3), convention=unadjusted)
    with pytest.raises(ValueError, match="local time"):
        CALENDAR.resolve_expiry(date(2024, 11, 4), time(16, tzinfo=UTC))
    with pytest.raises(ValueError, match="fold"):
        CALENDAR.resolve_expiry(date(2024, 11, 4), fold=2)


@pytest.mark.parametrize("day", [date(2023, 12, 31), datetime(2024, 1, 2)])
def test_calendar_rejects_uncovered_dates(day: date) -> None:
    with pytest.raises(ValueError, match="coverage"):
        CALENDAR.is_business_day(day)


def test_roll_cannot_escape_calendar_coverage() -> None:
    with pytest.raises(ValueError, match="coverage"):
        CALENDAR.adjust(date(2024, 1, 1), BusinessDayConvention.PRECEDING)


@pytest.mark.parametrize(
    "changes",
    [
        {"name": " "},
        {"valid_until": date(2023, 12, 31)},
        {"close_time": time(16, tzinfo=UTC)},
        {"holidays": frozenset({date(2025, 1, 1)})},
        {"early_closes": ((date(2024, 7, 4), time(13)),)},
        {"early_closes": ((date(2024, 7, 5), time(17)),)},
        {"early_closes": ((date(2024, 7, 5), time(13, tzinfo=UTC)),)},
        {"early_closes": ((date(2024, 7, 5), time(13)),) * 2},
    ],
)
def test_calendar_rejects_invalid_schedule(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(CALENDAR, **changes)
