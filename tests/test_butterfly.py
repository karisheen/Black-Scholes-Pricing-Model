"""Consumer-visible quoted-cost, support and unit regressions (synthetic only)."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from typing import Literal

import pytest

from quant_pricing.black_scholes import OptionType
from quant_pricing.butterfly import (
    ButterflyBook,
    ButterflyCostInputs,
    butterfly_strike_weights,
    check_quoted_butterflies,
    compare_butterfly_costs,
)
from quant_pricing.exceptions import InputValidationError
from quant_pricing.market import (
    Forward,
    Instrument,
    OptionContract,
    Quote,
    SequenceGap,
    SnapshotQuote,
    Underlying,
)

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)
EXPIRY = NOW + timedelta(days=90)
BTC = Underlying("BTC", "USDC")
FUTURE = Forward("BTC-FUTURE", BTC, EXPIRY)


def _row(
    instrument: Instrument,
    bid: float,
    ask: float,
    bid_size: float = 10,
    ask_size: float = 10,
) -> SnapshotQuote:
    return SnapshotQuote(
        instrument,
        Quote(instrument, bid, ask, bid_size, ask_size, NOW, NOW, "synthetic", 1),
        (),
    )


def _option(
    strike: float,
    kind: OptionType,
    bid: float,
    ask: float,
    *,
    bid_size: float = 10,
    ask_size: float = 10,
    multiplier: float = 1,
    premium_basis: Literal["per_unit", "per_contract"] = "per_unit",
) -> SnapshotQuote:
    contract = OptionContract(
        f"BTC-{strike:g}-{kind.value}",
        BTC,
        kind,
        strike,
        EXPIRY,
        "european",
        "cash",
        multiplier,
        "USDC",
        "USDC",
        premium_basis,
    )
    return _row(contract, bid, ask, bid_size, ask_size)


def _chain(center_bid: float = 10, center_ask: float = 12) -> tuple[SnapshotQuote, ...]:
    return (
        _option(90, OptionType.CALL, 14, 16),
        _option(90, OptionType.PUT, 4, 6),
        _option(100, OptionType.CALL, center_bid, center_ask),
        _option(100, OptionType.PUT, center_bid, center_ask),
        _option(110, OptionType.CALL, 2, 4),
        _option(110, OptionType.PUT, 12, 14),
    )


def _missing(row: SnapshotQuote) -> SnapshotQuote:
    return SnapshotQuote(row.instrument, None, ("missing",))


def test_midpoint_gap_disappears_when_crossing_real_option_sides() -> None:
    # Midpoint butterfly = .5*15 - 11 + .5*3 = -2, not a traded cost.
    (check,) = check_quoted_butterflies(_chain(), _row(FUTURE, 99, 101), 1)
    assert check.strikes == (90, 100, 110)
    assert check.weights == (0.5, -1, 0.5)
    assert check.combinations_checked == 8
    assert check.combinations_skipped == 0
    assert check.flags == ()
    assert check.negative_cost_candidate is False
    package = check.best_package
    assert package is not None
    assert package.net_entry_cost == 0
    assert package.futures_hedge is None
    assert [leg.side for leg in package.option_legs] == ["buy", "sell", "buy"]
    assert [leg.price for leg in package.option_legs] == [16, 10, 4]


def test_genuinely_negative_cost_is_indicative_and_json_safe() -> None:
    (check,) = check_quoted_butterflies(_chain(14, 16), _row(FUTURE, 99, 101), 1)
    package = check.best_package
    assert package is not None
    assert check.negative_cost_candidate is True
    assert package.negative_cost_candidate is True
    assert package.net_entry_cost == -4
    assert package.displayed_size_cap == 10
    assert package.cost_tolerance == 1e-10
    encoded = json.loads(json.dumps(asdict(check), allow_nan=False))
    assert encoded["best_package"]["net_entry_cost"] == -4
    assert encoded["strikes"] == [90, 100, 110]


def test_unequal_strikes_short_futures_hedge_uses_bid_and_amount_cap() -> None:
    rows = (
        _option(90, OptionType.CALL, 14, 15, ask_size=6, multiplier=2),
        _option(100, OptionType.PUT, 10, 11, bid_size=20),
        _option(130, OptionType.CALL, 1, 2),
    )
    (check,) = check_quoted_butterflies(rows, _row(FUTURE, 119, 121, 3, 8), 0.9)
    assert check.weights == (0.75, -1, 0.25)
    assert check.combinations_checked == 1
    assert check.combinations_skipped == 7
    package = check.best_package
    assert package is not None
    assert package.option_entry_cost == 1.75
    assert package.parity_adjustment == pytest.approx(-17.1)
    assert package.net_entry_cost == pytest.approx(-15.35)
    assert package.displayed_size_cap == 3
    hedge = package.futures_hedge
    assert hedge is not None
    assert (hedge.symbol, hedge.side, hedge.price) == ("BTC-FUTURE", "sell", 119)
    assert hedge.signed_weight == -1
    assert hedge.displayed_size == 3


def test_positive_net_futures_hedge_uses_ask_and_absolute_quantity() -> None:
    rows = (
        _option(90, OptionType.PUT, 1, 2),
        _option(100, OptionType.CALL, 5, 6),
        _option(130, OptionType.CALL, 1, 2),
    )
    (check,) = check_quoted_butterflies(rows, _row(FUTURE, 119, 121, 9, 1.5), 0.9)
    package = check.best_package
    assert package is not None
    assert package.option_entry_cost == -3
    assert package.parity_adjustment == pytest.approx(20.925)
    assert package.displayed_size_cap == 2
    hedge = package.futures_hedge
    assert hedge is not None
    assert (hedge.side, hedge.price, hedge.signed_weight) == ("buy", 121, 0.75)


def test_unobserved_future_still_checks_exact_zero_hedges() -> None:
    missing_future = SnapshotQuote(FUTURE, None, ("missing",))
    rows = tuple(
        _option(strike, kind, 1, 2)
        for strike in (90.1, 100.3, 130.7)
        for kind in (OptionType.CALL, OptionType.PUT)
    )
    (check,) = check_quoted_butterflies(rows, missing_future, 0.97)
    assert check.combinations_checked == 2
    assert check.combinations_skipped == 6
    assert "BTC-FUTURE:missing" in check.flags
    assert "partial_support" in check.flags
    package = check.best_package
    assert package is not None
    assert package.futures_hedge is None
    assert package.parity_adjustment == 0
    puts = tuple(row for row in rows if row.instrument.option_type == OptionType.PUT)
    (put_check,) = check_quoted_butterflies(puts, missing_future, 0.97)
    assert put_check.combinations_checked == 1
    assert put_check.best_package is not None
    assert put_check.best_package.futures_hedge is None
    assert put_check.best_package.parity_adjustment == 0


def test_declared_missing_strikes_are_not_bridged() -> None:
    rows = tuple(
        _option(strike, kind, 1, 2)
        for strike in (80, 90, 100, 110, 120)
        for kind in (OptionType.CALL, OptionType.PUT)
    )
    rows = tuple(_missing(row) if row.instrument.strike == 100 else row for row in rows)
    checks = check_quoted_butterflies(rows, _row(FUTURE, 99, 101), 1)
    assert [check.strikes for check in checks] == [
        (80, 90, 100),
        (90, 100, 110),
        (100, 110, 120),
    ]
    for check in checks:
        assert check.combinations_checked == 0
        assert check.best_package is None
        assert check.negative_cost_candidate is None
        assert "no_usable_combinations" in check.flags


@pytest.mark.parametrize("flag", ["stale", "crossed", "invalid_price", "invalid_size"])
def test_quality_flagged_rows_cannot_support_packages(flag: str) -> None:
    rows = tuple(replace(row, flags=(flag,)) for row in _chain())
    (check,) = check_quoted_butterflies(rows, _row(FUTURE, 99, 101), 1)
    assert check.combinations_checked == 0
    assert check.negative_cost_candidate is None


def test_sequence_gaps_exclude_rows_without_silently_reusing_old_book() -> None:
    rows = _chain()
    damaged = []
    for row in rows:
        quote = row.quote
        assert quote is not None
        gap = SequenceGap(2, 2, quote, replace(quote, source_sequence=3))
        damaged.append(replace(row, sequence_gaps=(gap,)))
    (check,) = check_quoted_butterflies(tuple(damaged), _row(FUTURE, 99, 101), 1)
    assert check.combinations_checked == 0
    assert any(flag.endswith(":sequence_gap") for flag in check.flags)


def test_only_consumed_sides_need_positive_size_and_multiplier_sets_cap() -> None:
    rows = (
        _option(
            90,
            OptionType.CALL,
            14,
            32,
            bid_size=0,
            ask_size=1,
            multiplier=2,
            premium_basis="per_contract",
        ),
        _option(100, OptionType.CALL, 10, 12, bid_size=8, ask_size=0),
        _option(110, OptionType.CALL, 2, 4, bid_size=0, ask_size=10),
    )
    (check,) = check_quoted_butterflies(rows, _missing(_row(FUTURE, 99, 101)), 1)
    package = check.best_package
    assert package is not None
    assert package.net_entry_cost == 0
    assert package.displayed_size_cap == 4
    assert package.option_legs[0].price == 16
    body_quote = rows[1].quote
    assert body_quote is not None
    empty = replace(rows[1], quote=replace(body_quote, bid_size=0))
    (unsupported,) = check_quoted_butterflies(
        (rows[0], empty, rows[2]), _row(FUTURE, 99, 101), 1
    )
    assert unsupported.best_package is None
    assert "BTC-100-call:empty_bid_size" in unsupported.flags


def test_small_roundoff_negative_is_not_a_candidate() -> None:
    (check,) = check_quoted_butterflies(
        _chain(10 + 1e-12, 12), _row(FUTURE, 99, 101), 1
    )
    assert check.best_package is not None
    assert check.best_package.net_entry_cost < 0
    assert check.negative_cost_candidate is False


@pytest.mark.parametrize("discount", [0, -1, float("nan"), float("inf"), True])
def test_discount_factor_must_be_finite_positive(discount: float) -> None:
    with pytest.raises(InputValidationError, match="discount_factor"):
        check_quoted_butterflies(_chain(), _row(FUTURE, 99, 101), discount)


def test_duplicate_type_strike_and_incompatible_contracts_rejected() -> None:
    rows = _chain()
    future = _row(FUTURE, 99, 101)
    with pytest.raises(InputValidationError, match="duplicate"):
        check_quoted_butterflies((*rows, rows[0]), future, 1)
    contract = rows[0].instrument
    assert isinstance(contract, OptionContract)
    incompatible = (
        replace(contract, exercise_style="american"),
        replace(contract, settlement_style="physical"),
        replace(contract, quote_currency="USD"),
        replace(contract, settlement_currency="USD"),
        replace(contract, expiry=EXPIRY + timedelta(days=1)),
        replace(contract, underlying=Underlying("ETH", "USDC")),
    )
    for changed in incompatible:
        with pytest.raises(InputValidationError):
            check_quoted_butterflies((_row(changed, 1, 2), *rows[1:]), future, 1)
    with pytest.raises(InputValidationError, match="dated Forward"):
        check_quoted_butterflies(rows, rows[0], 1)
    with pytest.raises(InputValidationError, match="OptionContract"):
        check_quoted_butterflies((future, *rows), future, 1)
    with pytest.raises(InputValidationError, match="underlying/expiry"):
        check_quoted_butterflies(
            rows, _row(replace(FUTURE, expiry=EXPIRY + timedelta(days=1)), 99, 101), 1
        )


def test_fewer_than_three_declared_strikes_have_no_checks() -> None:
    assert check_quoted_butterflies(_chain()[:4], _row(FUTURE, 99, 101), 1) == ()
    assert check_quoted_butterflies((), _row(FUTURE, 99, 101), 1) == ()


def test_opposing_puts_net_before_crossing_futures_spread_and_sizing() -> None:
    rows = (
        _option(90, OptionType.PUT, 1, 2),
        _option(100, OptionType.PUT, 10, 11),
        _option(130, OptionType.CALL, 1, 2),
    )
    future = _row(FUTURE, 119, 121, bid_size=1, ask_size=0)
    (check,) = check_quoted_butterflies(rows, future, 0.9)
    package = check.best_package
    assert package is not None
    # .75 long puts - 1 short put = .25 short future, not two gross hedges.
    hedge = package.futures_hedge
    assert hedge is not None
    assert hedge.side == "sell"
    assert hedge.signed_weight == -0.25
    assert hedge.price == 119
    assert package.option_entry_cost == -8
    assert package.parity_adjustment == pytest.approx(2.475)
    assert package.net_entry_cost == pytest.approx(-5.525)
    assert package.displayed_size_cap == 4


@pytest.mark.parametrize(
    ("bid", "ask", "size"),
    [(3, 2, 10), (float("nan"), 2, 10), (1, 2, float("inf")), (1, 2, -1)],
)
def test_raw_invalid_books_are_excluded_even_without_snapshot_flags(
    bid: float, ask: float, size: float
) -> None:
    rows = tuple(
        _option(strike, OptionType.CALL, bid, ask, ask_size=size)
        for strike in (90, 100, 110)
    )
    (check,) = check_quoted_butterflies(rows, _row(FUTURE, 99, 101), 1)
    assert check.combinations_checked == 0
    assert check.best_package is None
    assert check.negative_cost_candidate is None
    json.dumps(asdict(check), allow_nan=False)


def test_same_package_midpoints_preserve_weighted_hedge_and_normalized_units() -> None:
    rows = (
        _option(
            90, OptionType.CALL, 28, 30, multiplier=2, premium_basis="per_contract"
        ),
        _option(100, OptionType.PUT, 10, 11),
        _option(130, OptionType.CALL, 1, 2),
    )
    future = _row(FUTURE, 119, 121, 3, 8)
    (check,) = check_quoted_butterflies(rows, future, 0.9)
    package = check.best_package
    assert package is not None
    books = {
        "BTC-90-call": ButterflyBook(14, 15),
        "BTC-100-put": ButterflyBook(10, 11),
        "BTC-130-call": ButterflyBook(1, 2),
        "BTC-FUTURE": ButterflyBook(119, 121),
    }
    result = compare_butterfly_costs(package, books, 0.9)
    assert result.quoted_cost == pytest.approx(-15.35)
    assert result.midpoint_cost == pytest.approx(-17.25)
    assert result.crossing_cost == pytest.approx(1.9)
    assert result.flags == ()
    inputs = ButterflyCostInputs(
        package.option_legs, package.futures_hedge, package.net_entry_cost
    )
    assert compare_butterfly_costs(inputs, books, 0.9) == result
    books["BTC-90-call"] = ButterflyBook(14, 15, ("stale", "sequence_gap"))
    blocked = compare_butterfly_costs(package, books, 0.9)
    assert blocked.midpoint_cost is None
    assert blocked.quoted_cost == result.quoted_cost
    assert blocked.flags == ("BTC-90-call:stale", "BTC-90-call:sequence_gap")
    assert compare_butterfly_costs(inputs, books, 0.9) == blocked


@pytest.mark.parametrize(
    "book,flag",
    [
        (None, "missing"),
        (ButterflyBook(20, 10), "crossed"),
        (ButterflyBook(float("nan"), 10), "invalid_price"),
    ],
)
def test_comparison_rejects_missing_invalid_or_crossed_books(
    book: ButterflyBook | None, flag: str
) -> None:
    (check,) = check_quoted_butterflies(_chain(), _row(FUTURE, 99, 101), 1)
    package = check.best_package
    assert package is not None
    books = {
        leg.symbol: ButterflyBook(leg.price, leg.price) for leg in package.option_legs
    }
    symbol = package.option_legs[0].symbol
    if book is None:
        del books[symbol]
    else:
        books[symbol] = book
    result = compare_butterfly_costs(package, books, 1)
    assert result.midpoint_cost is None
    assert result.crossing_cost is None
    assert result.flags == (f"{symbol}:{flag}",)


def test_same_package_comparison_without_hedge_and_overflow_guard() -> None:
    (check,) = check_quoted_butterflies(_chain(), _row(FUTURE, 99, 101), 1)
    package = check.best_package
    assert package is not None
    books = {
        leg.symbol: ButterflyBook(leg.price, leg.price) for leg in package.option_legs
    }
    result = compare_butterfly_costs(package, books, 1)
    assert result.midpoint_cost == result.quoted_cost
    assert result.crossing_cost == 0
    broken = replace(
        package,
        option_legs=tuple(
            replace(leg, signed_weight=1e308) for leg in package.option_legs
        ),
    )
    huge_books = {
        leg.symbol: ButterflyBook(1e308, 1e308) for leg in package.option_legs
    }
    assert compare_butterfly_costs(broken, huge_books, 1).flags == (
        "nonfinite_calculation",
    )
    # Opposing infinities reach fsum's explicit error branch.
    broken = replace(
        broken,
        option_legs=(
            replace(broken.option_legs[0], signed_weight=-1e308),
            *broken.option_legs[1:],
        ),
    )
    assert compare_butterfly_costs(broken, huge_books, 1).midpoint_cost is None
    with pytest.raises(InputValidationError, match="discount_factor"):
        compare_butterfly_costs(package, books, 0)


def test_exact_strike_weights_are_shared_with_quoted_checker() -> None:
    assert tuple(
        str(weight) for weight in butterfly_strike_weights((90, 100, 120))
    ) == (
        "2/3",
        "-1",
        "1/3",
    )
    with pytest.raises(InputValidationError, match="increase strictly"):
        butterfly_strike_weights((90, 90, 120))
