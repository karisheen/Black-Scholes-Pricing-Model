"""Log interpolation, carry identities and dated dividend cash-flow economics."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import exp, log, sqrt

import pytest

from quant_pricing.conventions import DayCount
from quant_pricing.curves import (
    CashDividend,
    DiscountCurve,
    DividendCurve,
    ForwardCurve,
    forward_from_spot,
)

AS_OF = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
YEAR = AS_OF + timedelta(days=365)
TWO_YEARS = AS_OF + timedelta(days=730)


def test_discount_log_interpolation_anchor_and_exact_knots() -> None:
    curve = DiscountCurve(AS_OF, ((YEAR, 0.96), (TWO_YEARS, 0.90)))
    assert curve.points[0] == (AS_OF, 1)
    assert curve.discount_factor(AS_OF) == 1
    assert curve.discount_factor(YEAR) == 0.96
    assert curve.discount_factor(TWO_YEARS) == 0.90
    assert curve.discount_factor(AS_OF + timedelta(days=182.5)) == pytest.approx(
        sqrt(0.96)
    )
    midpoint = AS_OF + timedelta(days=547.5)
    assert curve.discount_factor(midpoint) == pytest.approx(sqrt(0.96 * 0.90))
    assert curve.zero_rate(midpoint) == pytest.approx(-log(sqrt(0.96 * 0.90)) / 1.5)
    assert DiscountCurve(AS_OF, curve.points) == curve


def test_zero_rate_round_trip_negative_rates_and_day_count() -> None:
    curve = DiscountCurve.from_zero_rates(AS_OF, ((YEAR, -0.02), (TWO_YEARS, 0.03)))
    assert curve.discount_factor(YEAR) == pytest.approx(exp(0.02))
    assert curve.zero_rate(YEAR) == pytest.approx(-0.02)
    assert curve.zero_rate(TWO_YEARS) == pytest.approx(0.03)
    act360 = DiscountCurve.from_zero_rates(AS_OF, ((YEAR, 0.05),), DayCount.ACT_360)
    assert act360.discount_factor(YEAR) == pytest.approx(exp(-0.05 * 365 / 360))
    assert act360.zero_rate(YEAR) == pytest.approx(0.05)


def test_canonical_utc_curve_clocks_and_real_second_interpolation() -> None:
    eastern = timezone(timedelta(hours=-5))
    curve = DiscountCurve(
        AS_OF.astimezone(eastern), ((YEAR.astimezone(eastern), 0.96),)
    )
    assert curve.as_of.tzinfo == timezone.utc
    assert curve.points[-1][0].tzinfo == timezone.utc
    assert curve.discount_factor(YEAR.astimezone(eastern)) == 0.96
    assert curve.discount_factor(AS_OF + timedelta(seconds=1)) == pytest.approx(
        exp(log(0.96) / (365 * 86400))
    )


def test_forward_curve_interpolates_prices_in_log_space() -> None:
    curve = ForwardCurve(AS_OF, ((YEAR, 100), (TWO_YEARS, 121)))
    assert curve.forward(YEAR) == 100
    assert curve.forward(TWO_YEARS) == 121
    assert curve.forward(AS_OF + timedelta(days=547.5)) == pytest.approx(110)
    with pytest.raises(ValueError, match="coverage"):
        curve.forward(AS_OF)
    explicit_spot = ForwardCurve(AS_OF, ((AS_OF, 100), (YEAR, 121)))
    assert explicit_spot.forward(AS_OF) == 100
    assert explicit_spot.forward(AS_OF + timedelta(days=182.5)) == pytest.approx(110)


@pytest.mark.parametrize("curve_type", [DiscountCurve, ForwardCurve])
@pytest.mark.parametrize(
    "points",
    [
        (),
        ((YEAR, 0),),
        ((YEAR, -1),),
        ((YEAR, float("nan")),),
        ((YEAR, float("inf")),),
        ((YEAR, True),),
        ((YEAR, 1), (YEAR, 2)),
        ((TWO_YEARS, 1), (YEAR, 2)),
        ((AS_OF - timedelta(seconds=1), 1),),
        ((datetime(2025, 1, 1), 1),),
    ],
)
def test_curves_reject_invalid_knots(
    curve_type: type[DiscountCurve] | type[ForwardCurve],
    points: tuple[tuple[datetime, float], ...],
) -> None:
    with pytest.raises(ValueError):
        curve_type(AS_OF, points)


@pytest.mark.parametrize("curve_type", [DiscountCurve, ForwardCurve])
def test_curves_never_extrapolate(
    curve_type: type[DiscountCurve] | type[ForwardCurve],
) -> None:
    curve = curve_type(AS_OF, ((YEAR, 1),))
    query = curve.discount_factor if isinstance(curve, DiscountCurve) else curve.forward
    for expiry in (AS_OF - timedelta(seconds=1), YEAR + timedelta(seconds=1)):
        with pytest.raises(ValueError):
            query(expiry)


def test_discount_anchor_and_zero_rate_at_zero_are_unambiguous() -> None:
    with pytest.raises(ValueError, match="anchor"):
        DiscountCurve(AS_OF, ((AS_OF, 0.99),))
    with pytest.raises(ValueError, match="undefined"):
        DiscountCurve(AS_OF, ((YEAR, 1),)).zero_rate(AS_OF)
    assert DiscountCurve(AS_OF, ((AS_OF, 1),)).discount_factor(AS_OF) == 1


@pytest.mark.parametrize("rate", [float("nan"), float("inf"), True, -1000, 1000])
def test_invalid_or_unrepresentable_zero_rates_are_rejected(rate: float) -> None:
    with pytest.raises(ValueError):
        DiscountCurve.from_zero_rates(AS_OF, ((YEAR, rate),))


def test_zero_rate_requires_future_expiry() -> None:
    with pytest.raises(ValueError, match="future"):
        DiscountCurve.from_zero_rates(AS_OF, ((AS_OF, 0.01),))


def test_discrete_cash_dividends_use_ex_dates_and_payment_discounting() -> None:
    half = AS_OF + timedelta(days=182.5)
    after_expiry = YEAR + timedelta(days=30)
    discount = DiscountCurve.from_zero_rates(AS_OF, ((TWO_YEARS, 0.05),))
    first = CashDividend(half, half + timedelta(days=10), 2)
    second = CashDividend(YEAR, after_expiry, 3)
    third = CashDividend(YEAR + timedelta(days=1), after_expiry, 4)
    dividends = DividendCurve(AS_OF, (first, second, third))
    expected = 2 * exp(-0.05 * (182.5 + 10) / 365) + 3 * exp(-0.05 * 395 / 365)
    assert dividends.present_value(YEAR, discount) == pytest.approx(expected)
    assert dividends.present_value(half - timedelta(seconds=1), discount) == 0
    assert dividends.present_value(half, discount) == pytest.approx(
        2 * discount.discount_factor(first.payment_date)
    )
    forward = forward_from_spot(100, YEAR, discount, dividends)
    assert forward * discount.discount_factor(YEAR) + expected == pytest.approx(100)
    assert forward_from_spot(100, YEAR, discount) == pytest.approx(100 * exp(0.05))
    assert forward_from_spot(100, AS_OF, discount, dividends) == 100
    assert DividendCurve(AS_OF, ()).present_value(YEAR, discount) == 0


def test_dividend_curve_anchors_and_chronology_cannot_be_mixed() -> None:
    dividend = CashDividend(YEAR, YEAR, 2)
    discount = DiscountCurve(AS_OF, ((TWO_YEARS, 0.9),))
    curve = DividendCurve(AS_OF, (dividend,))
    with pytest.raises(ValueError, match="anchors"):
        curve.present_value(
            YEAR, DiscountCurve(AS_OF + timedelta(days=1), ((YEAR, 1),))
        )
    with pytest.raises(ValueError, match="precede"):
        curve.present_value(AS_OF - timedelta(days=1), discount)
    with pytest.raises(ValueError, match="payment"):
        replace(dividend, payment_date=AS_OF)
    for entries in ((dividend, dividend), (replace(dividend, ex_date=AS_OF),)):
        with pytest.raises(ValueError, match="future and ordered"):
            DividendCurve(AS_OF, entries)
    with pytest.raises(ValueError, match="positive"):
        replace(dividend, amount=0)


def test_forward_carry_requires_positive_prepaid_and_covered_payments() -> None:
    discount = DiscountCurve(AS_OF, ((YEAR, 1),))
    dividends = DividendCurve(AS_OF, (CashDividend(YEAR, YEAR, 100),))
    with pytest.raises(ValueError, match="positive"):
        forward_from_spot(100, YEAR, discount, dividends)
    with pytest.raises(ValueError, match="positive"):
        forward_from_spot(0, YEAR, discount)
    with pytest.raises(ValueError, match="coverage"):
        DividendCurve(
            AS_OF, (CashDividend(YEAR, YEAR + timedelta(days=1), 1),)
        ).present_value(YEAR, discount)
