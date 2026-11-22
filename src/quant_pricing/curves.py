"""Single-currency log-linear curves and deterministic cash-dividend carry.

Rates are annual continuously compounded decimals. Positive discount factors
may exceed one (negative rates); monotonicity is not imposed. No extrapolation
is supported. Forward knots need an explicit spot knot to cover the anchor.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime
from math import exp, isfinite, log

from quant_pricing.conventions import DayCount, utc_timestamp, year_fraction
from quant_pricing.exceptions import InputValidationError

CurvePoints = tuple[tuple[datetime, float], ...]


def _positive(value: float) -> None:
    if isinstance(value, bool) or not isfinite(value) or value <= 0:
        raise InputValidationError("curve values and spot must be finite and positive")


def _points(as_of: datetime, points: CurvePoints) -> CurvePoints:
    result = tuple((utc_timestamp(expiry), value) for expiry, value in points)
    previous: datetime | None = None
    for expiry, value in result:
        _positive(value)
        if expiry < as_of or (previous is not None and expiry <= previous):
            raise InputValidationError(
                "knots must be ordered, unique and not in the past"
            )
        previous = expiry
    if not result:
        raise InputValidationError("curve needs at least one knot")
    return result


def _interpolate(
    as_of: datetime,
    points: CurvePoints,
    times: tuple[float, ...],
    logs: tuple[float, ...],
    expiry: datetime,
    day_count: DayCount,
) -> float:
    t = year_fraction(as_of, expiry, day_count)
    if t < times[0] or t > times[-1]:
        raise InputValidationError("expiry is outside curve coverage; no extrapolation")
    index = bisect_left(times, t)
    if times[index] == t:
        return points[index][1]
    weight = (t - times[index - 1]) / (times[index] - times[index - 1])
    return exp(logs[index - 1] * (1 - weight) + logs[index] * weight)


@dataclass(frozen=True, slots=True)
class DiscountCurve:
    """Discount factors with an automatic D(as_of)=1 anchor."""

    as_of: datetime
    points: CurvePoints
    day_count: DayCount = DayCount.ACT_365F
    _times: tuple[float, ...] = field(init=False, repr=False)
    _logs: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        anchor = utc_timestamp(self.as_of)
        points = _points(anchor, self.points)
        if points[0][0] == anchor:
            if points[0][1] != 1:
                raise InputValidationError("discount factor at anchor must equal one")
        else:
            points = ((anchor, 1.0), *points)
        object.__setattr__(self, "as_of", anchor)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "day_count", DayCount(self.day_count))
        object.__setattr__(
            self,
            "_times",
            tuple(year_fraction(anchor, d, self.day_count) for d, _ in points),
        )
        object.__setattr__(self, "_logs", tuple(log(v) for _, v in points))

    @classmethod
    def from_zero_rates(
        cls,
        as_of: datetime,
        points: CurvePoints,
        day_count: DayCount = DayCount.ACT_365F,
    ) -> DiscountCurve:
        """Convert annual continuously compounded decimal zero rates."""
        factors: list[tuple[datetime, float]] = []
        for expiry, rate in points:
            if isinstance(rate, bool) or not isfinite(rate):
                raise InputValidationError("zero rates must be finite")
            t = year_fraction(as_of, expiry, day_count)
            if t == 0:
                raise InputValidationError(
                    "zero rate requires a strictly future maturity"
                )
            try:
                factor = exp(-rate * t)
            except OverflowError as exc:
                raise InputValidationError("discount factor overflows") from exc
            factors.append((expiry, factor))
        return cls(as_of, tuple(factors), day_count)

    def discount_factor(self, expiry: datetime) -> float:
        return _interpolate(
            self.as_of, self.points, self._times, self._logs, expiry, self.day_count
        )

    def zero_rate(self, expiry: datetime) -> float:
        """Continuously compounded zero rate; undefined at the anchor."""
        t = year_fraction(self.as_of, expiry, self.day_count)
        if t == 0:
            raise InputValidationError("zero rate is undefined at the curve anchor")
        return -log(self.discount_factor(expiry)) / t


@dataclass(frozen=True, slots=True)
class ForwardCurve:
    """Positive forward prices interpolated in log-price, without extrapolation."""

    as_of: datetime
    points: CurvePoints
    day_count: DayCount = DayCount.ACT_365F
    _times: tuple[float, ...] = field(init=False, repr=False)
    _logs: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        anchor = utc_timestamp(self.as_of)
        points = _points(anchor, self.points)
        object.__setattr__(self, "as_of", anchor)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "day_count", DayCount(self.day_count))
        object.__setattr__(
            self,
            "_times",
            tuple(year_fraction(anchor, d, self.day_count) for d, _ in points),
        )
        object.__setattr__(self, "_logs", tuple(log(v) for _, v in points))

    def forward(self, expiry: datetime) -> float:
        return _interpolate(
            self.as_of, self.points, self._times, self._logs, expiry, self.day_count
        )


@dataclass(frozen=True, slots=True)
class CashDividend:
    """Known cash per underlying unit, with distinct entitlement/payment clocks."""

    ex_date: datetime
    payment_date: datetime
    amount: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "ex_date", utc_timestamp(self.ex_date))
        object.__setattr__(self, "payment_date", utc_timestamp(self.payment_date))
        _positive(self.amount)
        if self.payment_date < self.ex_date:
            raise InputValidationError("dividend payment cannot precede ex-date")


@dataclass(frozen=True, slots=True)
class DividendCurve:
    """Dated cash distributions; ex-date selects entitlement, payment date PV."""

    as_of: datetime
    dividends: tuple[CashDividend, ...]

    def __post_init__(self) -> None:
        anchor = utc_timestamp(self.as_of)
        object.__setattr__(self, "as_of", anchor)
        object.__setattr__(self, "dividends", tuple(self.dividends))
        previous = anchor
        for dividend in self.dividends:
            if dividend.ex_date <= previous:
                raise InputValidationError(
                    "dividend ex-dates must be future and ordered"
                )
            previous = dividend.ex_date

    def present_value(self, expiry: datetime, discount_curve: DiscountCurve) -> float:
        """PV of distributions going ex in (as_of, expiry], including delayed pay."""
        if discount_curve.as_of != self.as_of:
            raise InputValidationError("dividend and discount curve anchors must match")
        expiry = utc_timestamp(expiry)
        year_fraction(self.as_of, expiry)
        return sum(
            dividend.amount * discount_curve.discount_factor(dividend.payment_date)
            for dividend in self.dividends
            if dividend.ex_date <= expiry
        )


def forward_from_spot(
    spot: float,
    expiry: datetime,
    discount_curve: DiscountCurve,
    dividends: DividendCurve | None = None,
) -> float:
    """Cash-and-carry F=(S-PV(dividends))/D, no implicit borrow, FX or yield.

    Dividends are fixed cash flows, not an American exercise approximation.
    """
    _positive(spot)
    pv = 0.0 if dividends is None else dividends.present_value(expiry, discount_curve)
    prepaid = spot - pv
    _positive(prepaid)
    return prepaid / discount_curve.discount_factor(expiry)
