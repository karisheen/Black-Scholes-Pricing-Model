"""Explicit UTC clocks, actual-day counts and caller-supplied exchange calendars.

ACT/ACT ISDA splits elapsed UTC time at calendar-year boundaries, dividing each
piece by that year's actual length. Intraday fractions use 86,400-second days;
ACT/365F and ACT/360 use fixed denominators. No civil-time/DST day is silently
assumed to contain 24 hours. Calendars are bounded, explicit schedules, not a
claim of complete exchange holiday coverage.
"""

from __future__ import annotations

from calendar import isleap
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from zoneinfo import ZoneInfo

from quant_pricing.exceptions import InputValidationError


class DayCount(str, Enum):
    ACT_365F = "ACT/365F"
    ACT_360 = "ACT/360"
    ACT_ACT_ISDA = "ACT/ACT ISDA"


def utc_timestamp(value: datetime) -> datetime:
    """Require an exact aware timestamp and normalize it to UTC."""
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise InputValidationError("timestamp must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def year_fraction(
    start: datetime, end: datetime, day_count: DayCount = DayCount.ACT_365F
) -> float:
    """Nonnegative elapsed years; reversed intervals are invalid."""
    start, end = utc_timestamp(start), utc_timestamp(end)
    convention = DayCount(day_count)
    if end < start:
        raise InputValidationError("end must not precede start")
    if convention != DayCount.ACT_ACT_ISDA:
        denominator = 360 if convention == DayCount.ACT_360 else 365
        return (end - start).total_seconds() / (86400 * denominator)
    result = 0.0
    while start < end:
        boundary = (
            datetime(start.year + 1, 1, 1, tzinfo=timezone.utc)
            if start.year < end.year
            else end
        )
        result += (boundary - start).total_seconds() / (
            86400 * (366 if isleap(start.year) else 365)
        )
        start = boundary
    return result


class BusinessDayConvention(str, Enum):
    UNADJUSTED = "unadjusted"
    FOLLOWING = "following"
    PRECEDING = "preceding"
    MODIFIED_FOLLOWING = "modified_following"


@dataclass(frozen=True, slots=True)
class TradingCalendar:
    """Named exchange zone and explicitly bounded holiday/early-close schedule.

    Weekends are Saturday/Sunday. Supply authoritative holidays and half-days
    for the declared coverage. Expiry defaults to session close; a contractual
    local expiry time overrides it. Ambiguous wall times require an explicit
    fold; nonexistent wall times are rejected rather than shifted.
    """

    name: str
    zone: ZoneInfo
    valid_from: date
    valid_until: date
    holidays: frozenset[date] = frozenset()
    close_time: time = time(16)
    early_closes: tuple[tuple[date, time], ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or self.valid_until < self.valid_from:
            raise InputValidationError("calendar needs a name and valid coverage")
        if self.close_time.tzinfo is not None:
            raise InputValidationError("session close must be a local wall time")
        object.__setattr__(self, "holidays", frozenset(self.holidays))
        object.__setattr__(self, "early_closes", tuple(self.early_closes))
        seen: set[date] = set()
        for day in self.holidays:
            self._check_date(day)
        for day, close in self.early_closes:
            if (
                day in seen
                or not self.is_business_day(day)
                or close.tzinfo is not None
                or close >= self.close_time
            ):
                raise InputValidationError("invalid or duplicate early close")
            seen.add(day)

    def _check_date(self, day: date) -> None:
        if isinstance(day, datetime) or not self.valid_from <= day <= self.valid_until:
            raise InputValidationError("date is outside calendar coverage")

    def is_business_day(self, day: date) -> bool:
        self._check_date(day)
        return day.weekday() < 5 and day not in self.holidays

    def adjust(
        self,
        day: date,
        convention: BusinessDayConvention = BusinessDayConvention.FOLLOWING,
    ) -> date:
        """Roll within calendar coverage; never fabricate out-of-range sessions."""
        self._check_date(day)
        rule = BusinessDayConvention(convention)
        if rule == BusinessDayConvention.UNADJUSTED:
            return day
        direction = -1 if rule == BusinessDayConvention.PRECEDING else 1
        adjusted = day
        while not self.is_business_day(adjusted):
            adjusted += timedelta(days=direction)
        if (
            rule == BusinessDayConvention.MODIFIED_FOLLOWING
            and adjusted.month != day.month
        ):
            return self.adjust(day, BusinessDayConvention.PRECEDING)
        return adjusted

    def resolve_expiry(
        self,
        day: date,
        local_time: time | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.PRECEDING,
        *,
        fold: int | None = None,
    ) -> datetime:
        """Resolve a contract date and local wall clock into one UTC instant."""
        day = self.adjust(day, convention)
        if local_time is None:
            if not self.is_business_day(day):
                raise InputValidationError("non-session expiry needs an explicit time")
            local_time = dict(self.early_closes).get(day, self.close_time)
        if local_time.tzinfo is not None or fold not in (None, 0, 1):
            raise InputValidationError("use a local time and fold 0 or 1")
        wall = datetime.combine(day, local_time)
        first = wall.replace(tzinfo=self.zone, fold=0)
        second = wall.replace(tzinfo=self.zone, fold=1)
        chosen = second if fold == 1 else first
        result = chosen.astimezone(timezone.utc)
        if result.astimezone(self.zone).replace(tzinfo=None) != wall:
            raise InputValidationError("nonexistent local expiry time")
        if first.utcoffset() != second.utcoffset() and fold is None:
            raise InputValidationError("ambiguous local expiry time requires fold")
        return result
