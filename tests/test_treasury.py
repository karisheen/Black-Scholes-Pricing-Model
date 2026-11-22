"""Offline GSW source-format, availability and genuine zero-yield economics."""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from math import exp
from pathlib import Path

import pytest

from quant_pricing.treasury import GSW_SOURCE_URL, TreasuryObservation, read_gsw_csv

FIXTURE = Path(__file__).parent / "fixtures" / "gsw_zero_sample.csv"
OBSERVED = date(1961, 6, 14)
# Test ingestion clock only; not a claimed Fed publication or historical vintage.
CAPTURED = datetime(2026, 11, 22, 12, tzinfo=timezone.utc)


def observation() -> TreasuryObservation:
    return read_gsw_csv(FIXTURE, OBSERVED, captured_at=CAPTURED)


def test_official_zero_columns_are_percent_cc_not_par_yields() -> None:
    record = observation()
    assert record.observation_date == OBSERVED
    assert record.captured_at == record.available_at == CAPTURED
    assert record.source == GSW_SOURCE_URL
    assert record.content_sha256 == sha256(FIXTURE.read_bytes()).hexdigest()
    assert dict(record.zero_rates)[1] == pytest.approx(0.029825)
    assert dict(record.zero_rates)[2] == pytest.approx(0.033771)
    assert dict(record.zero_rates)[7] == pytest.approx(0.037612)
    assert dict(record.zero_rates)[8] is None
    assert dict(record.zero_rates)[30] is None
    curve = record.discount_curve(CAPTURED, (1, 2, 7))
    assert curve.as_of == CAPTURED
    assert curve.discount_factor(CAPTURED) == 1
    assert curve.discount_factor(CAPTURED + timedelta(days=365)) == pytest.approx(
        exp(-0.029825)
    )
    assert curve.discount_factor(CAPTURED + timedelta(days=365 * 7)) == pytest.approx(
        exp(-0.037612 * 7)
    )
    # The first par yield is 3.0025%; using it as a zero is economically different.
    assert curve.zero_rate(CAPTURED + timedelta(days=365)) != pytest.approx(0.030025)
    assert curve.discount_factor(CAPTURED + timedelta(days=365 * 1.5)) == pytest.approx(
        exp((-0.029825 - 0.033771 * 2) / 2)
    )


def test_capture_and_availability_are_distinct_from_observation_date() -> None:
    ready = CAPTURED + timedelta(hours=1)
    record = read_gsw_csv(FIXTURE, OBSERVED, captured_at=CAPTURED, available_at=ready)
    with pytest.raises(ValueError, match="not yet available"):
        record.discount_curve(CAPTURED, (1,))
    with pytest.raises(ValueError, match="not yet available"):
        record.discount_curve(datetime(1961, 6, 14, 23, tzinfo=timezone.utc), (1,))
    assert record.discount_curve(ready, (1,)).as_of == ready
    assert record.discount_curve(ready + timedelta(days=1), (1,)).as_of == (
        ready + timedelta(days=1)
    )
    with pytest.raises(ValueError, match="precede file capture"):
        replace(record, available_at=CAPTURED - timedelta(seconds=1))
    with pytest.raises(ValueError, match="follow capture"):
        replace(record, observation_date=date(2026, 11, 23))


def test_real_holiday_missing_tenors_and_absent_observation_do_not_fallback() -> None:
    holiday = read_gsw_csv(FIXTURE, date(1961, 7, 4), captured_at=CAPTURED)
    assert all(rate is None for _, rate in holiday.zero_rates)
    with pytest.raises(ValueError, match="missing Treasury zero yield SVENY01"):
        holiday.discount_curve(CAPTURED)
    with pytest.raises(ValueError, match="missing Treasury zero yield SVENY03"):
        observation().discount_curve(CAPTURED)
    with pytest.raises(ValueError, match="SVENY08"):
        observation().discount_curve(CAPTURED, (8,))
    with pytest.raises(ValueError, match="absent"):
        read_gsw_csv(FIXTURE, date(1961, 6, 15), captured_at=CAPTURED)


@pytest.mark.parametrize("tenors", [(), (2, 1), (1, 1), (0,), (31,), (True,), (1.5,)])
def test_requested_tenors_are_explicit_unique_and_valid(
    tenors: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        observation().discount_curve(CAPTURED, tenors)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("a file without a header", "no Date header"),
        ("Date,SVENPY01\n1961-06-14,3.0025\n", "not par yields"),
        ("Date,SVENY01,SVENY01\n1961-06-14,3,3\n", "duplicate columns"),
        ("Date,SVENY01\nnot-a-date,3\n", "invalid GSW observation date"),
        ("Date,SVENY01\n1961-06-14,3\n1961-06-14,4\n", "duplicate GSW"),
        ("Date,SVENY01\n1961-06-14\n", "column count"),
        ("Date,SVENY01\n1961-06-14,error\n", "invalid GSW zero yield"),
        ("Date,SVENY01\n1961-06-14,nan\n", "finite or missing"),
        ("Date,SVENY01\n1961-06-14,inf\n", "finite or missing"),
    ],
)
def test_malformed_downloads_fail_explicitly(
    tmp_path: Path, payload: str, message: str
) -> None:
    path = tmp_path / "download.csv"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_gsw_csv(path, OBSERVED, captured_at=CAPTURED)


def test_blank_yields_bom_and_negative_zero_rates(tmp_path: Path) -> None:
    path = tmp_path / "download.csv"
    path.write_text("\ufeffDate,SVENY01,SVENY02\n\n1961-06-14,-1,\n", encoding="utf-8")
    record = read_gsw_csv(str(path), OBSERVED, captured_at=CAPTURED)
    assert dict(record.zero_rates)[1] == -0.01
    assert dict(record.zero_rates)[2] is None
    assert record.discount_curve(CAPTURED, (1,)).discount_factor(
        CAPTURED + timedelta(days=365)
    ) == pytest.approx(exp(0.01))


def test_observation_constructor_preserves_invariants_and_canonical_clocks() -> None:
    record = observation()
    with pytest.raises(ValueError, match="ordered GSW tenors"):
        replace(record, zero_rates=((1, 0.03),))
    eastern = timezone(timedelta(hours=-5))
    converted = replace(
        record,
        captured_at=CAPTURED.astimezone(eastern),
        available_at=CAPTURED.astimezone(eastern),
    )
    assert converted == record
    assert converted.captured_at.tzinfo == timezone.utc
    with pytest.raises(ValueError, match="follow capture"):
        replace(record, observation_date=CAPTURED)
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(record, captured_at=CAPTURED.replace(tzinfo=None))
