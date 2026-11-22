"""Matched option parity, independent observations, carry and quality cutoffs."""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from quant_pricing.black_scholes import OptionType, price
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
from quant_pricing.option_chain import load_fixture
from quant_pricing.parity import CarryAssumptions, ParityDiagnostic, diagnose_parity

AS_OF = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
EXPIRY = AS_OF + timedelta(days=365)
MAX_AGE = timedelta(seconds=60)
UNDERLYING = Underlying("SYNTH", "USD")
CALL = OptionContract(
    "C100",
    UNDERLYING,
    OptionType.CALL,
    100.0,
    EXPIRY,
    "european",
    "cash",
    100.0,
    "USD",
    "USD",
    "per_unit",
)
PUT = replace(CALL, symbol="P100", option_type=OptionType.PUT)


def _row(instrument: Instrument, bid: float, ask: float) -> SnapshotQuote:
    quote = Quote(instrument, bid, ask, 10, 20, AS_OF, AS_OF, "fixture", 1)
    return SnapshotQuote(instrument, quote, ())


def _diagnose(
    call: SnapshotQuote | None = None,
    put: SnapshotQuote | None = None,
    **kwargs: Any,
) -> ParityDiagnostic:
    return diagnose_parity(
        _row(CALL, 8, 10) if call is None else call,
        _row(PUT, 5, 6) if put is None else put,
        as_of=AS_OF,
        max_age=MAX_AGE,
        discount_factor=0.9,
        **kwargs,
    )


def test_executable_bid_ask_bounds_not_midpoint_observations() -> None:
    result = _diagnose(forward=104)
    assert result.option_flags == result.forward_flags == ()
    assert result.premium_difference == (2, 5)
    assert result.implied_forward == pytest.approx((100 + 2 / 0.9, 100 + 5 / 0.9))
    assert result.parity_residual == pytest.approx((-1.6, 1.4))
    assert result.independent_forward_residual is None
    assert result.implied_dividend_yield is None
    assert result.implied_borrow_yield is None
    assert asdict(result)["premium_difference"] == (2, 5)
    interval = _diagnose(forward=(103, 105))
    assert interval.parity_residual == pytest.approx((-2.5, 2.3))
    disjoint = _diagnose(forward=110)
    assert disjoint.parity_residual == pytest.approx((-7, -4))


def test_each_contract_premium_basis_is_normalized_independently() -> None:
    put = replace(PUT, premium_basis="per_contract", multiplier=25)
    call = replace(CALL, premium_basis="per_contract", multiplier=100)
    result = _diagnose(_row(call, 800, 1000), _row(put, 125, 150))
    assert result.premium_difference == (2, 5)
    mixed = _diagnose(put=_row(put, 125, 150))
    assert mixed.implied_forward == result.implied_forward


def test_independent_forward_is_a_separate_executable_comparison() -> None:
    forward = Forward("F", UNDERLYING, EXPIRY)
    result = _diagnose(forward=104, forward_quote=_row(forward, 109, 111))
    assert result.parity_residual == pytest.approx((-1.6, 1.4))
    assert result.independent_forward_residual == pytest.approx((-7.9, -3.1))
    assert result.forward_flags == ()
    invalid = replace(_row(forward, 109, 111), flags=("stale",))
    rejected = _diagnose(forward_quote=invalid)
    assert rejected.forward_flags == ("forward:stale",)
    assert rejected.implied_forward is not None
    assert rejected.independent_forward_residual is None


def test_bsm_identity_recovers_only_the_conditionally_unknown_carry() -> None:
    s, r, q, b, t = 100.0, 0.05, 0.02, 0.01, 1.0
    c = float(price("call", s, 100, t, r, 0.3, q + b))
    p = float(price("put", s, 100, t, r, 0.3, q + b))
    kwargs = dict(as_of=AS_OF, max_age=MAX_AGE, discount_factor=math.exp(-r * t))
    dividend = diagnose_parity(
        _row(CALL, c, c),
        _row(PUT, p, p),
        carry=CarryAssumptions(s, r, t, borrow_yield=b),
        **kwargs,
    )
    assert dividend.implied_dividend_yield == pytest.approx((q, q), abs=2e-14)
    assert dividend.implied_borrow_yield is None
    borrow = diagnose_parity(
        _row(CALL, c, c),
        _row(PUT, p, p),
        carry=CarryAssumptions(s, r, t, dividend_yield=q),
        **kwargs,
    )
    assert borrow.implied_borrow_yield == pytest.approx((b, b), abs=2e-14)
    assert borrow.implied_dividend_yield is None
    forward = s * math.exp((r - q - b) * t)
    assert borrow.implied_forward == pytest.approx((forward, forward))


def test_carry_interval_reverses_forward_endpoints() -> None:
    assumptions = CarryAssumptions(100, 0.05, 1, borrow_yield=0)
    result = _diagnose(carry=assumptions)
    assert result.implied_forward is not None
    low, high = result.implied_forward
    assert result.implied_dividend_yield == pytest.approx(
        (0.05 - math.log(high / 100), 0.05 - math.log(low / 100))
    )
    impossible = _diagnose(_row(CALL, 0, 0), _row(PUT, 100, 110), carry=assumptions)
    assert impossible.implied_forward is not None
    assert impossible.implied_forward[0] < 0
    assert impossible.implied_dividend_yield is None
    assert impossible.option_flags == ("carry:nonpositive_forward",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"dividend_yield": 0, "borrow_yield": 0},
        {"borrow_yield": float("nan")},
        {"dividend_yield": float("inf")},
    ],
)
def test_carry_requires_exactly_one_finite_known_counterpart(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(InputValidationError):
        CarryAssumptions(100, 0.05, 1, **kwargs)


def test_carry_validates_spot_horizon_and_funding() -> None:
    for spot, rate, t in ((0, 0.05, 1), (100, 0.05, 0), (100, math.nan, 1)):
        with pytest.raises(InputValidationError):
            CarryAssumptions(spot, rate, t, borrow_yield=0)


@pytest.mark.parametrize(
    "changes,flag",
    [
        ({"bid": 12, "ask": 10}, "crossed"),
        ({"bid": float("nan")}, "invalid_price"),
        ({"ask_size": -1}, "invalid_size"),
        ({"bid_size": 0}, "empty_size"),
        ({"quote_time": AS_OF - timedelta(seconds=61)}, "stale"),
        ({"quote_time": AS_OF + timedelta(seconds=1)}, "future"),
        ({"available_at": AS_OF + timedelta(seconds=1)}, "future"),
    ],
)
def test_rechecks_raw_quality_and_both_asof_timestamps(
    changes: dict[str, Any],
    flag: str,
) -> None:
    row = _row(CALL, 8, 10)
    assert row.quote is not None
    invalid = replace(row, quote=replace(row.quote, **changes))
    result = _diagnose(call=invalid)
    assert result.option_flags == (f"call:{flag}",)
    assert result.implied_forward is None
    assert result.premium_difference is None
    assert result.parity_residual is None


def test_existing_flags_missing_quotes_and_boundary_age() -> None:
    missing = SnapshotQuote(PUT, None, ("missing",))
    assert _diagnose(put=missing).option_flags == ("put:missing",)
    flagged = replace(_row(CALL, 8, 10), flags=("stale",))
    assert _diagnose(call=flagged).implied_forward is None
    row = _row(CALL, 8, 10)
    assert row.quote is not None
    boundary = replace(row, quote=replace(row.quote, quote_time=AS_OF - MAX_AGE))
    assert _diagnose(call=boundary).option_flags == ()


def test_sequence_gaps_block_option_and_forward_evidence() -> None:
    row = _row(CALL, 8, 10)
    assert row.quote is not None
    gap = SequenceGap(2, 2, row.quote, replace(row.quote, source_sequence=3))
    assert _diagnose(call=replace(row, sequence_gaps=(gap,))).option_flags == (
        "call:sequence_gap",
    )
    forward = _row(Forward("F", UNDERLYING, EXPIRY), 103, 104)
    assert forward.quote is not None
    forward_gap = SequenceGap(
        2, 2, forward.quote, replace(forward.quote, source_sequence=3)
    )
    result = _diagnose(forward_quote=replace(forward, sequence_gaps=(forward_gap,)))
    assert result.forward_flags == ("forward:sequence_gap",)
    assert result.independent_forward_residual is None
    assert result.implied_forward is not None


@pytest.mark.parametrize(
    "changes",
    [
        {"exercise_style": "american"},
        {"settlement_style": "physical"},
        {"quote_currency": "EUR"},
        {"settlement_currency": "EUR"},
        {"underlying": Underlying("OTHER", "USD")},
        {"expiry": EXPIRY + timedelta(seconds=1)},
        {"strike": 101},
        {"option_type": OptionType.CALL},
    ],
)
def test_rejects_incompatible_option_pairs(changes: dict[str, Any]) -> None:
    with pytest.raises(InputValidationError):
        _diagnose(put=_row(replace(PUT, **changes), 5, 6))


def test_rejects_nonoption_and_mismatched_forward_instruments() -> None:
    with pytest.raises(InputValidationError, match="OptionContract"):
        _diagnose(call=_row(UNDERLYING, 100, 101))
    for instrument in (
        UNDERLYING,
        Forward("F", UNDERLYING, EXPIRY + timedelta(days=1)),
    ):
        with pytest.raises(InputValidationError, match="independent forward"):
            _diagnose(forward_quote=_row(instrument, 100, 101))


@pytest.mark.parametrize("benchmark", [0, float("nan"), (105, 100), (100, 0)])
def test_rejects_invalid_theoretical_benchmark(benchmark: Any) -> None:
    with pytest.raises(InputValidationError):
        _diagnose(forward=benchmark)


def test_rejects_invalid_cutoffs_discounting_and_numeric_overflow() -> None:
    call, put = _row(CALL, 8, 10), _row(PUT, 5, 6)
    for as_of, age, d in (
        (AS_OF.replace(tzinfo=None), MAX_AGE, 0.9),
        (AS_OF, timedelta(seconds=-1), 0.9),
        (AS_OF, MAX_AGE, 0),
        (EXPIRY + timedelta(seconds=1), MAX_AGE, 0.9),
        (AS_OF, MAX_AGE, 1e-320),
    ):
        with pytest.raises(InputValidationError):
            diagnose_parity(call, put, as_of=as_of, max_age=age, discount_factor=d)


def test_frozen_fixture_has_independent_option_sides_and_carry_consistency() -> None:
    path = Path(__file__).resolve().parents[1] / "examples/synthetic_option_chain.json"
    fixture = load_fixture(path)
    rows = {row.instrument.symbol: row for row in fixture.provider.snapshot(AS_OF).rows}
    result = diagnose_parity(
        rows["SYNTH-20260101-C100"],
        rows["SYNTH-20260101-P100"],
        as_of=AS_OF,
        max_age=fixture.provider.max_age,
        discount_factor=math.exp(-fixture.rate),
        forward=100 * math.exp(fixture.rate - fixture.dividend_yield),
        carry=CarryAssumptions(100, fixture.rate, 1, borrow_yield=0),
    )
    assert result.option_flags == ()
    assert result.parity_residual is not None
    assert result.parity_residual[0] < 0 < result.parity_residual[1]
    assert result.implied_dividend_yield is not None
    low, high = result.implied_dividend_yield
    assert low < fixture.dividend_yield < high
    assert result.independent_forward_residual is None
    interim_rows = {
        row.instrument.symbol: row
        for row in fixture.provider.snapshot(AS_OF + timedelta(seconds=30)).rows
    }
    invalid = diagnose_parity(
        interim_rows["SYNTH-20260101-C100"],
        interim_rows["SYNTH-20260101-P100"],
        as_of=AS_OF + timedelta(seconds=30),
        max_age=fixture.provider.max_age,
        discount_factor=math.exp(-fixture.rate),
    )
    assert invalid.option_flags == ("call:crossed",)
    assert invalid.implied_forward is None
