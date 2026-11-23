"""Public replay and local-only report adapters preserve provenance and risk basis."""

from __future__ import annotations

import json
from copy import deepcopy
from math import exp, log
from pathlib import Path
from typing import Any

import pytest

from quant_pricing.black76 import price
from quant_pricing.lab_data import (
    ReplayView,
    ReportLibrary,
    choose_symbol,
    contract_detail,
    load_reports,
    market_context,
    visible_options,
)
from quant_pricing.lab_demo import DEMO_SESSION, DEMO_SOURCE, demo_report
from quant_pricing.synthetic_chain import SSVITruth


def _view(report: dict[str, Any], checkpoint: int = 0, expiry: int = 0) -> ReplayView:
    point = report["checkpoints"][checkpoint]
    return ReplayView("test", report, point, point["expiries"][expiry])


@pytest.fixture
def view() -> ReplayView:
    return _view(demo_report())


def _selected(view: ReplayView, side: str = "call") -> str:
    symbol = choose_symbol(view, side)
    assert symbol is not None
    return symbol


def test_demo_is_isolated_deterministic_and_generated_through_domain() -> None:
    first, second = demo_report(), demo_report()
    assert first == second
    assert first is not second
    assert first["source"] == DEMO_SOURCE
    assert "synthetic" in first["scope"]
    assert "Generated" in first["assumptions"]["availability"]
    assert "spot" not in first["assumptions"]
    assert len(first["checkpoints"]) == 3
    assert all(len(point["expiries"]) == 2 for point in first["checkpoints"])
    for expiry in first["checkpoints"][0]["expiries"]:
        assert expiry["fit"]["converged"]
        assert expiry["parity"]
        assert expiry["quoted_butterflies"]["triplets"]
        assert expiry["coverage"]["usable_books"] == 20
    first["checkpoints"][0]["expiries"][0]["options"].clear()
    assert demo_report() == second
    assert load_reports().sessions == {DEMO_SESSION: second}


def test_demo_smile_has_supported_curvature_and_observation_bands(
    view: ReplayView,
) -> None:
    truth = SSVITruth(max_time=0.2, variance_rate=0.58**2, rho=-0.22, phi=5.0)
    rows = visible_options(view, "otm")
    forward = view.expiry["forward_input"]
    # Quote generation is one second before the first replay checkpoint.
    maturity = market_context(view, None)["dte"] / 365 + 1 / (365 * 86_400)
    midpoint_ivs = []
    for row in rows:
        k = log(row["strike"] / forward)
        sigma = truth.volatility(k, maturity)
        bid = row["sides"]["bid"]["iv"]
        ask = row["sides"]["ask"]["iv"]
        assert bid < sigma < ask
        assert ask - bid == pytest.approx(2 * (0.008 + 0.012 * abs(k)), abs=3e-6)
        midpoint_ivs.append(row["sides"]["midpoint"]["iv"])
    assert max(midpoint_ivs) - min(midpoint_ivs) > 0.08
    assert midpoint_ivs[0] > min(midpoint_ivs) + 0.08
    assert midpoint_ivs[-1] > min(midpoint_ivs) + 0.015


def test_private_loading_is_sorted_exact_and_never_falls_back(
    tmp_path: Path, view: ReplayView
) -> None:
    report = view.report
    report["source"] = "tardis/deribit"
    for name in ("z-report.json", "a-report.json"):
        (tmp_path / name).write_text(json.dumps(report))
    (tmp_path / "ignored.json").write_text(json.dumps(report))
    library = load_reports(tmp_path)
    assert list(library.sessions) == ["a-report", "z-report"]
    assert not library.errors
    selected = library.view("a-report", view.checkpoint["as_of"], view.expiry["expiry"])
    assert selected.report == report
    assert selected.checkpoint is selected.report["checkpoints"][0]
    assert selected.expiry is selected.checkpoint["expiries"][0]
    assert selected.session == "a-report"
    for session, stamp, expiry in (
        ("missing", view.checkpoint["as_of"], view.expiry["expiry"]),
        ("a-report", "2030-01-02T00:00:02Z", view.expiry["expiry"]),
        ("a-report", view.checkpoint["as_of"], "2030-01-26T08:00:00+00:00"),
    ):
        with pytest.raises(KeyError):
            library.view(session, stamp, expiry)
    assert library.view("a-report", view.checkpoint["as_of"], view.expiry["expiry"])


def test_empty_missing_file_and_symlink_directories(tmp_path: Path) -> None:
    assert not load_reports(tmp_path).sessions
    assert load_reports(tmp_path).errors
    assert not load_reports(tmp_path / "absent").sessions
    target = tmp_path / "not-a-directory"
    target.write_text("not a report")
    assert not load_reports(target).sessions
    (tmp_path / "link-report.json").symlink_to(target)
    (tmp_path / "folder-report.json").mkdir()
    result = load_reports(tmp_path)
    assert not result.sessions
    assert sum("not a regular" in error for error in result.errors) == 2


@pytest.mark.parametrize("content", ["{", "[]", "null", "NaN", "1e999", "{}"])
def test_malformed_json_is_visible_without_demo(tmp_path: Path, content: str) -> None:
    (tmp_path / "bad-report.json").write_text(content)
    result = load_reports(tmp_path)
    assert not result.sessions
    assert result.errors[0].startswith("bad-report.json:")


@pytest.mark.parametrize(
    ("location", "field", "bad"),
    [
        ("report", "source", None),
        ("report", "scope", 1),
        ("report", "assumptions", []),
        ("report", "checkpoints", []),
        ("report", "checkpoints", [None]),
        ("checkpoint", "as_of", "not-time"),
        ("checkpoint", "as_of", "2030-01-02T00:00:01"),
        ("checkpoint", "as_of", None),
        ("checkpoint", "expiries", []),
        ("checkpoint", "expiries", [None]),
        ("checkpoint", "native_capture_complete", {}),
        ("checkpoint", "observed_arbitrage", []),
        ("expiry", "expiry", None),
        ("expiry", "fit", []),
        ("expiry", "options", None),
        ("expiry", "options", [None]),
        ("option", "symbol", None),
        ("option", "strike", -1),
        ("option", "strike", True),
        ("option", "strike", "100"),
        ("option", "option_type", "future"),
        ("option", "normalized_quote", []),
        ("option", "sides", {"bid": 1}),
        ("option", "normalized_flags", "stale"),
        ("option", "rejected", [1]),
    ],
)
def test_malformed_report_shape(
    tmp_path: Path, view: ReplayView, location: str, field: str, bad: Any
) -> None:
    target = {
        "report": view.report,
        "checkpoint": view.checkpoint,
        "expiry": view.expiry,
        "option": view.expiry["options"][0],
    }[location]
    target[field] = bad
    (tmp_path / "bad-report.json").write_text(json.dumps(view.report))
    assert not load_reports(tmp_path).sessions


@pytest.mark.parametrize("level", ["checkpoint", "expiry", "option"])
def test_duplicate_recorded_keys_rejected(
    tmp_path: Path, view: ReplayView, level: str
) -> None:
    rows = {
        "checkpoint": view.report["checkpoints"],
        "expiry": view.checkpoint["expiries"],
        "option": view.expiry["options"],
    }[level]
    rows.append(deepcopy(rows[0]))
    (tmp_path / "duplicate-report.json").write_text(json.dumps(view.report))
    assert "duplicate" in load_reports(tmp_path).errors[0]


def test_bad_file_does_not_hide_good_file(tmp_path: Path, view: ReplayView) -> None:
    (tmp_path / "good-report.json").write_text(json.dumps(view.report))
    (tmp_path / "bad-report.json").write_bytes(b"\xff")
    result = load_reports(tmp_path)
    assert list(result.sessions) == ["good-report"]
    assert len(result.errors) == 1


def test_filters_and_selection_use_actual_forward(view: ReplayView) -> None:
    assert len(visible_options(view, "all")) == 20
    assert len(visible_options(view, "call")) == len(visible_options(view, "put")) == 10
    otm = visible_options(view, "otm")
    assert len(otm) == 10
    assert all(
        row["option_type"] == ("put" if row["strike"] < 82_000 else "call")
        for row in otm
    )
    retained = otm[0]["symbol"]
    assert choose_symbol(view, "otm", retained) == retained
    assert choose_symbol(view, "call", retained) == "BTC_USDC-25JAN30-80000-C"
    assert choose_symbol(view, "otm", "removed") == "BTC_USDC-25JAN30-80000-P"
    with pytest.raises(ValueError, match="Unknown chain side"):
        visible_options(view, "calls")
    view.expiry["forward_input"] = 82_500
    assert choose_symbol(view, "call") == "BTC_USDC-25JAN30-80000-C"
    view.expiry["forward_input"] = 80_000
    atm = [row for row in visible_options(view, "otm") if row["strike"] == 80_000]
    assert [row["option_type"] for row in atm] == ["call"]
    view.expiry["forward_input"] = None
    assert not visible_options(view, "otm")
    assert choose_symbol(view, "call") is None
    assert choose_symbol(view, "all", retained) == retained


def test_context_never_invents_spot_and_reports_midpoint_iv(view: ReplayView) -> None:
    context = market_context(view, _selected(view))
    assert context["spot"] is None
    assert context["forward"] == 82_000
    assert context["atm_strike"] == 80_000
    atm = next(row for row in view.expiry["options"] if row["strike"] == 80_000)
    assert context["atm_iv"] == atm["sides"]["midpoint"]["iv"]
    assert 0.57 < context["atm_iv"] < 0.62
    assert "midpoint" in context["atm_iv_basis"]
    assert context["dte"] > 23
    assert context["rate"] == 0.025
    assert market_context(view, "unknown")["selected_symbol"] is None
    view.report["assumptions"].pop("rate")
    view.expiry["forward_input"] = None
    view.expiry["discount_factor"] = None
    absent = market_context(view, None)
    assert all(
        absent[key] is None
        for key in ("spot", "forward", "atm_iv", "rate", "discount_factor")
    )


def test_model_and_forward_greeks_agree_with_black76_partials(view: ReplayView) -> None:
    detail = contract_detail(view, _selected(view))
    assert detail["quality_flags"] == []
    assert detail["age_seconds"] == 1
    assert "not spot" in detail["risk_basis"]
    assert "absolute" in detail["risk_units"]["vega"]
    assert "calendar year" in detail["risk_units"]["theta"]
    assert "not quote midpoint" in detail["model_basis"]
    assert detail["normalized_quote"] is view.expiry["options"][8]["normalized_quote"]
    assert detail["assumptions"] is view.report["assumptions"]
    assert detail["risk_unavailable_reason"] is None
    greeks = detail["greeks"]
    assert set(greeks) == set(detail["risk_units"])
    assert greeks["delta"] > 0 and greeks["gamma"] > 0 and greeks["vega"] > 0
    put = contract_detail(view, _selected(view, "put"))
    assert put["greeks"]["delta"] < 0
    assert put["greeks"]["rho"] < 0 and greeks["rho"] < 0
    forward, strike, sigma = 82_000.0, detail["strike"], detail["model_iv"]
    maturity = market_context(view, None)["dte"] / 365
    rate = view.report["assumptions"]["rate"]

    def value(
        f: float = forward,
        s: float = sigma,
        t: float = maturity,
        r: float = rate,
    ) -> float:
        return float(price("call", f, strike, t, exp(-r * t), s))

    assert detail["model_value"] == pytest.approx(value())
    h = 1.0
    assert greeks["delta"] == pytest.approx(
        (value(forward + h) - value(forward - h)) / (2 * h), rel=1e-6
    )
    assert greeks["gamma"] == pytest.approx(
        (value(forward + h) - 2 * value() + value(forward - h)) / h**2, rel=1e-5
    )
    step = 1e-5
    assert greeks["vega"] == pytest.approx(
        (value(s=sigma + step) - value(s=sigma - step)) / (2 * step), rel=1e-6
    )
    assert greeks["theta"] == pytest.approx(
        (value(t=maturity - step) - value(t=maturity + step)) / (2 * step), rel=1e-6
    )
    assert greeks["rho"] == pytest.approx(
        (value(r=rate + step) - value(r=rate - step)) / (2 * step), rel=1e-6
    )
    json.dumps(detail, allow_nan=False)


def test_gap_failed_iv_and_stale_demo_paths() -> None:
    report = demo_report()
    mixed = _view(report, 1)
    gap = contract_detail(mixed, "BTC_USDC-25JAN30-60000-P")
    assert "sequence_gap" in gap["quality_flags"]
    assert gap["age_seconds"] is None
    assert gap["greeks"] is None and gap["model_value"] is None
    failed = contract_detail(mixed, "BTC_USDC-25JAN30-105000-C")
    assert failed["sides"]["midpoint"]["error"]
    assert failed["greeks"] is None
    assert contract_detail(mixed, _selected(mixed))["greeks"] is not None
    stale = _view(report, 2)
    detail = contract_detail(stale, "BTC_USDC-25JAN30-80000-C")
    assert "stale" in detail["quality_flags"]
    assert detail["age_seconds"] == 37
    assert detail["greeks"] is None and detail["model_iv"] is None
    assert choose_symbol(stale, "otm") is None
    assert market_context(stale, None)["atm_iv"] is None


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("expiry", "fit", None),
        ("expiry", "fit_error", "failed"),
        ("expiry", "forward_input", None),
        ("expiry", "discount_factor", 0),
        ("expiry", "expiry", "2030-01-02T00:00:01+00:00"),
        ("fit", "converged", False),
        ("fit", "parameters", {}),
        ("fit", "domain", [2, 3]),
        ("option", "normalized_quote", None),
        ("option", "sides", {}),
        ("quote", "quote_time", None),
        ("quote", "quote_time", "invalid"),
        ("quote", "quote_time", "2030-01-02T00:00:02Z"),
        ("quote", "quote_time", "2030-01-01T00:00:00Z"),
        ("future", "normalized_quote", None),
        ("future", "normalized_flags", ["stale"]),
    ],
)
def test_missing_or_rejected_model_inputs_withhold_risk(
    view: ReplayView, target: str, field: str, value: Any
) -> None:
    symbol = _selected(view)
    option = next(row for row in view.expiry["options"] if row["symbol"] == symbol)
    targets = {
        "expiry": view.expiry,
        "fit": view.expiry["fit"],
        "option": option,
        "quote": option["normalized_quote"],
        "future": view.expiry["matched_future"],
    }
    targets[target][field] = value
    detail = contract_detail(view, symbol)
    assert detail["model_value"] is None and detail["greeks"] is None
    assert detail["risk_unavailable_reason"]
    assert detail["risk_basis"]


@pytest.mark.parametrize("rate", [None, True, "0.025"])
def test_conditional_rate_risks_are_not_spot_risks(view: ReplayView, rate: Any) -> None:
    view.report["assumptions"]["rate"] = rate
    detail = contract_detail(view, _selected(view))
    assert detail["model_value"] is not None
    assert detail["greeks"]["delta"] is not None
    assert all(
        detail["greeks"][name] is None for name in ("rho", "theta", "charm", "color")
    )


def test_unknown_contract_and_empty_chain(view: ReplayView) -> None:
    with pytest.raises(KeyError, match="Unknown contract"):
        contract_detail(view, "unknown")
    view.expiry["options"] = []
    assert choose_symbol(view, "all") is None
    assert market_context(view, None)["atm_strike"] is None
    assert ReportLibrary({}).errors == ()


def test_report_io_errors_are_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "unreadable-report.json").write_text("{}")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_text", fail)
    assert "unable to read local report file" in load_reports(tmp_path).errors[0]
    monkeypatch.setattr(Path, "glob", fail)
    assert "Unable to read" in load_reports(tmp_path).errors[0]


def test_nonfinite_context_and_unreported_rate_convention(view: ReplayView) -> None:
    symbol = _selected(view)
    view.report["assumptions"]["rate_convention"] = "unspecified"
    assert contract_detail(view, symbol)["greeks"]["rho"] is None
    view.expiry["forward_input"] = float("nan")
    assert market_context(view, None)["forward"] is None
    assert choose_symbol(view, "all") is None


def test_advanced_forward_greeks_match_numerical_partial_changes(
    view: ReplayView,
) -> None:
    symbol = _selected(view)
    detail = contract_detail(view, symbol)
    forward, strike, sigma = 82_000.0, detail["strike"], detail["model_iv"]
    maturity = market_context(view, None)["dte"] / 365
    rate = view.report["assumptions"]["rate"]
    step_f, step_s, step_t = 10.0, 0.0001, 0.0001

    def value(f: float, s: float, t: float) -> float:
        return float(price("call", f, strike, t, exp(-rate * t), s))

    def delta(f: float, s: float, t: float) -> float:
        return (value(f + step_f, s, t) - value(f - step_f, s, t)) / (2 * step_f)

    def gamma(f: float, s: float, t: float) -> float:
        return (
            value(f + step_f, s, t) - 2 * value(f, s, t) + value(f - step_f, s, t)
        ) / step_f**2

    estimates = {
        "vanna": (
            delta(forward, sigma + step_s, maturity)
            - delta(forward, sigma - step_s, maturity)
        )
        / (2 * step_s),
        "volga": (
            value(forward, sigma + step_s, maturity)
            - 2 * value(forward, sigma, maturity)
            + value(forward, sigma - step_s, maturity)
        )
        / step_s**2,
        "charm": (
            delta(forward, sigma, maturity - step_t)
            - delta(forward, sigma, maturity + step_t)
        )
        / (2 * step_t),
        "speed": (
            gamma(forward + step_f, sigma, maturity)
            - gamma(forward - step_f, sigma, maturity)
        )
        / (2 * step_f),
        "color": (
            gamma(forward, sigma, maturity - step_t)
            - gamma(forward, sigma, maturity + step_t)
        )
        / (2 * step_t),
    }
    for greek, estimate in estimates.items():
        assert detail["greeks"][greek] == pytest.approx(estimate, rel=1e-3, abs=1e-10)


def test_unrepresentable_numbers_and_model_overflow_are_unavailable(
    view: ReplayView, monkeypatch: pytest.MonkeyPatch
) -> None:
    symbol = _selected(view)
    monkeypatch.setattr("quant_pricing.lab_data.price", lambda *args: float("inf"))
    detail = contract_detail(view, symbol)
    assert detail["greeks"] is None
    assert "finite numerical range" in detail["risk_unavailable_reason"]
    view.expiry["forward_input"] = 10**1000
    assert market_context(view, None)["forward"] is None
    assert choose_symbol(view, "all") is None


@pytest.mark.parametrize(
    ("path", "bad"),
    [
        (("fit", "parameters"), {}),
        (("fit", "parameters", "sigma"), "not a number"),
        (("fit", "parameters", "rho"), 2),
        (("fit", "domain"), [0]),
        (("fit", "time_to_expiry"), None),
        (("fit", "expiry"), "/private/capture"),
        (("fit", "residuals"), None),
        (("fit", "residuals"), [{}]),
        (("fit", "residuals", 0, "weight"), []),
        (("fit", "arbitrage"), {"passed": True, "findings": [None]}),
        (("parity",), [None]),
        (("parity",), [{"call": [], "put": "missing"}]),
        (("parity", 0, "premium_difference"), [0, "bad"]),
        (("parity", 0, "option_flags"), [{}]),
        (("quoted_butterflies", "triplets"), [None]),
        (("quoted_butterflies", "triplets", 0, "strikes"), None),
        (("quoted_butterflies", "triplets", 0, "strikes"), [1, 2]),
        (("quoted_butterflies", "triplets", 0, "strikes"), [1, 3, 2]),
        (("quoted_butterflies", "triplets", 0, "strikes"), [0, 1, 2]),
        (("quoted_butterflies", "triplets", 0, "strikes"), [1, True, 3]),
        (("quoted_butterflies", "triplets", 0, "strikes"), [1, 1, 2]),
        (("quoted_butterflies", "triplets", 0, "best_package"), {}),
        (("quoted_butterflies", "triplets", 0, "best_package", "option_legs"), [None]),
        (("quoted_butterflies", "triplets", 0, "best_package", "futures_hedge"), {}),
        (("matched_future", "normalized_quote", "bid"), "bad"),
        (("matched_future", "native_book", "flags"), [{}]),
        (("matched_future", "native_book", "last_update"), []),
        (("options", 0, "normalized_quote", "available_at"), []),
        (("options", 0, "native_book", "flags"), None),
        (("options", 0, "sides", "midpoint", "iv"), "bad"),
        (("options", 0, "log_forward_moneyness"), {}),
        (("coverage",), None),
        (("coverage", "iv_successes"), []),
        (("coverage", "native_flag_counts"), {"stale": "many"}),
        (("coverage", "usable_books"), {}),
    ],
)
def test_malformed_consumed_subtrees_reject_only_bad_session(
    tmp_path: Path, view: ReplayView, path: tuple[str | int, ...], bad: Any
) -> None:
    (tmp_path / "good-report.json").write_text(json.dumps(view.report))
    target: Any = view.expiry
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad
    (tmp_path / "bad-report.json").write_text(json.dumps(view.report))
    library = load_reports(tmp_path)
    assert list(library.sessions) == ["good-report"]
    assert len(library.errors) == 1
    assert library.errors[0].startswith("bad-report.json: invalid")
    assert "/private" not in library.errors[0]
    assert str(tmp_path) not in library.errors[0]
    good = library.view("good-report", view.checkpoint["as_of"], view.expiry["expiry"])
    assert contract_detail(good, _selected(good))["greeks"] is not None


@pytest.mark.parametrize(
    "finding",
    [None, {}, {"kind": "calendar", "time_to_expiry": "bad", "log_moneyness": 0}],
)
def test_invalid_midpoint_findings_are_rejected(
    tmp_path: Path, view: ReplayView, finding: Any
) -> None:
    view.checkpoint["observed_arbitrage"]["findings"] = [finding]
    (tmp_path / "bad-report.json").write_text(json.dumps(view.report))
    assert not load_reports(tmp_path).sessions


def test_private_io_error_does_not_expose_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "private-report.json").write_text("{}")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError(f"permission denied: {tmp_path}/private-report.json")

    monkeypatch.setattr(Path, "read_text", fail)
    library = load_reports(tmp_path)
    assert not library.sessions
    assert all(str(tmp_path) not in error for error in library.errors)


@pytest.mark.parametrize(
    ("target", "field", "value", "reason"),
    [
        ("expiry", "forward_input", 83_000, "forward"),
        ("expiry", "discount_factor", 0.5, "discount factor"),
        ("assumptions", "rate", 0.5, "discount factor"),
        ("assumptions", "max_book_age_ms", None, "quality policy"),
        ("fit", "arbitrage", None, "quality"),
        ("fit", "arbitrage", {"passed": False, "findings": []}, "quality"),
        ("fit", "time_to_expiry", 1, "maturity"),
    ],
)
def test_inconsistent_model_basis_withholds_all_model_risk(
    view: ReplayView, target: str, field: str, value: Any, reason: str
) -> None:
    symbol = _selected(view)
    targets = {
        "expiry": view.expiry,
        "assumptions": view.report["assumptions"],
        "fit": view.expiry["fit"],
    }
    targets[target][field] = value
    detail = contract_detail(view, symbol)
    assert detail["model_value"] is None
    assert detail["model_iv"] is None
    assert detail["greeks"] is None
    assert reason in detail["risk_unavailable_reason"]
