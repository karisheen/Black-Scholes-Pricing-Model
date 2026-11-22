"""Frozen chain, pricing units, and as-of availability at the report boundary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from quant_pricing.option_chain import generate_report, main

FIXTURE = Path(__file__).resolve().parents[1] / "examples/synthetic_option_chain.json"
CLEAN = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
CROSSED = "SYNTH-20260101-C100"
LATE = "SYNTH-20260101-P100"


def _report(as_of: datetime, fixture: Path = FIXTURE) -> dict[str, Any]:
    return generate_report(fixture, as_of)


def _options(report: dict[str, Any]) -> dict[str, Any]:
    return {row["symbol"]: row for row in report["options"]}


def test_frozen_chain_recovers_the_known_iv_range_in_both_premium_bases() -> None:
    report = _report(CLEAN)
    assert report["underlying"]["midpoint"] == 100.0
    options = _options(report)
    assert len(options) == 6
    for option in options.values():
        assert not option["quality_flags"]
        assert option["bid"]["iv"] == pytest.approx(0.19, abs=1e-9)
        assert option["ask"]["iv"] == pytest.approx(0.21, abs=1e-9)
        assert option["bid"]["error"] is None
        assert option["ask"]["error"] is None
        assert option["time_to_expiry"] == pytest.approx(1.0)

    put = options[LATE]
    assert put["raw"]["premium_basis"] == "per_contract"
    assert put["raw"]["size_unit"] == "contracts"
    assert put["normalized"]["bid"] == pytest.approx(put["raw"]["bid"] / 100.0)
    assert put["normalized"]["currency"] == put["raw"]["currency"]


def test_crossed_and_late_quotes_are_not_used_at_the_interim_cutoff() -> None:
    interim = _options(_report(CLEAN.replace(second=30)))
    crossed = interim[CROSSED]
    assert crossed["quality_flags"] == ["crossed"]
    assert crossed["provenance"]["source_sequence"] == 2
    assert crossed["raw"]["bid"] > crossed["raw"]["ask"]
    assert crossed["bid"]["iv"] is None and crossed["ask"]["iv"] is None

    initial = _options(_report(CLEAN))[LATE]
    assert interim[LATE]["provenance"] == initial["provenance"]
    assert interim[LATE]["raw"] == initial["raw"]
    later = _options(_report(CLEAN.replace(minute=1)))[LATE]
    assert later["provenance"]["source_sequence"] == 2
    assert later["provenance"]["quote_time"] == "2025-01-01T12:00:20Z"
    assert later["provenance"]["available_at"] == "2025-01-01T12:00:45Z"
    assert later["raw"]["bid"] != initial["raw"]["bid"]
    assert later["bid"]["error"] is None


def test_stale_spot_does_not_discard_fresh_options_or_supply_a_midpoint() -> None:
    report = _report(CLEAN.replace(minute=2, second=1))
    assert report["underlying"]["quality_flags"] == ["stale"]
    assert report["underlying"]["midpoint"] is None
    late = _options(report)[LATE]
    assert late["quality_flags"] == []
    assert late["bid"]["iv"] is None
    assert late["bid"]["error"]["type"] == "UnderlyingUnavailable"


def test_pricing_errors_are_scoped_to_the_affected_side_or_contract(
    tmp_path: Path,
) -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for quote in data["quotes"]:
        if quote["symbol"] == CROSSED and quote["source_sequence"] == 1:
            quote["bid"] = 0.0  # Below the call's positive lower bound.
    for option in data["options"]:
        if option["symbol"] == "SYNTH-20260101-P90":
            option["quote_currency"] = "BTC"
        if option["symbol"] == "SYNTH-20260101-C110":
            option["exercise_style"] = "american"
    fixture = tmp_path / "adjusted.json"
    fixture.write_text(json.dumps(data), encoding="utf-8")

    options = _options(_report(CLEAN, fixture))
    assert options[CROSSED]["bid"]["error"]["type"] == "NoArbitrageError"
    assert options[CROSSED]["bid"]["iv"] is None
    assert options[CROSSED]["ask"]["iv"] == pytest.approx(0.21, abs=1e-9)
    coin = options["SYNTH-20260101-P90"]
    assert coin["normalized"]["currency"] == "BTC"
    assert coin["bid"]["error"]["type"] == "CurrencyMismatch"
    assert options["SYNTH-20260101-C110"]["bid"]["error"]["type"] == "UnsupportedModel"


def test_cli_rejects_an_as_of_without_timezone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([str(FIXTURE), "--as-of", "2025-01-01T12:00:00"]) == 2
    output = capsys.readouterr()
    assert not output.out
    assert "timezone" in output.err


def test_cli_emits_a_stable_as_of_report(capsys: pytest.CaptureFixture[str]) -> None:
    argv = [str(FIXTURE), "--as-of", "2025-01-01T12:00:30Z"]
    assert main(argv) == 0
    first = capsys.readouterr().out
    assert main(argv) == 0
    assert capsys.readouterr().out == first
    assert _options(json.loads(first))[CROSSED]["quality_flags"] == ["crossed"]


def test_no_observations_are_fabricated_before_the_first_quote() -> None:
    report = _report(CLEAN.replace(hour=11))
    assert report["underlying"]["midpoint"] is None
    for row in _options(report).values():
        assert row["quality_flags"] == ["missing"]
        assert row["provenance"] is None
        assert row["raw"] is None
        assert row["bid"]["iv"] is None


def test_bad_numeric_observations_remain_visible_in_strict_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for quote in data["quotes"]:
        if quote["symbol"] == "SYNTH":
            quote.update(bid=0.0, ask=0.0)
        if quote["symbol"] == LATE and quote["source_sequence"] == 1:
            quote.update(bid=float("nan"), ask=float("inf"), bid_size=-float("inf"))
    fixture = tmp_path / "invalid-quotes.json"
    fixture.write_text(json.dumps(data), encoding="utf-8")

    assert main([str(fixture), "--as-of", "2025-01-01T12:00:00Z"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["underlying"]["midpoint"] is None
    put = _options(report)[LATE]
    assert set(put["quality_flags"]) == {"invalid_price", "invalid_size"}
    assert put["raw"]["bid"] == "NaN"
    assert put["raw"]["ask"] == "Infinity"
    assert put["raw"]["bid_size"] == "-Infinity"
    assert put["bid"]["iv"] is None


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        ("schema_version", "1", "must be an integer"),
        ("schema_version", 2, "schema_version 1"),
        ("data_kind", "live", "only frozen_synthetic"),
        ("underlying", [], "must be a JSON object"),
        ("source", "", "must be a nonempty string"),
        ("options", {}, "must be a JSON array"),
        ("rate", True, "must be a number"),
        ("rate", float("inf"), "must be finite"),
    ],
)
def test_malformed_fixture_cannot_emit_pricing_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: object,
    diagnostic: str,
) -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data[field] = value
    fixture = tmp_path / "bad.json"
    fixture.write_text(json.dumps(data), encoding="utf-8")

    assert main([str(fixture), "--as-of", "2025-01-01T12:00:00Z"]) == 2
    output = capsys.readouterr()
    assert not output.out
    assert diagnostic in output.err
