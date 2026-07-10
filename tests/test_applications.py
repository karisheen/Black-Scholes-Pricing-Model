"""Integration and compatibility tests for user-facing entry points."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest
from dash import html

from quant_pricing.cli import main as cli_main
from quant_pricing.dashboard import (
    calculate_scenario,
    create_app,
    render_analysis,
)

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_functions_delegate_to_validated_library() -> None:
    legacy = runpy.run_path(str(ROOT / "BlackScholesPricingModel.py"))

    assert legacy["d1"](100.0, 100.0, 1.0, 0.05, 0.2) == pytest.approx(0.35)
    assert legacy["d2"](100.0, 100.0, 1.0, 0.05, 0.2) == pytest.approx(0.15)
    assert legacy["bs_call"](100.0, 100.0, 1.0, 0.05, 0.2) == pytest.approx(
        10.450583572185565
    )
    assert legacy["bs_put"](100.0, 100.0, 1.0, 0.05, 0.2) == pytest.approx(
        5.573526022256971
    )


def test_command_line_entry_point_accepts_reproducible_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli_main(
        [
            "--spot",
            "100",
            "--strike",
            "100",
            "--time",
            "1",
            "--rate",
            "0.05",
            "--volatility",
            "0.2",
            "--type",
            "call",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == "Call value: 10.4505835722\n"


def test_dashboard_scenario_uses_vectorized_core() -> None:
    analysis = calculate_scenario("put", 100.0, 100.0, 365.0, 5.0, 20.0)

    assert analysis.valuation.price == pytest.approx(5.573526022256971)
    assert analysis.time_to_expiry == 1.0
    assert analysis.rate == 0.05
    assert analysis.volatility == 0.2
    assert analysis.spot_grid.shape == (161,)
    assert analysis.volatility_grid.shape == (161,)
    assert np.all(np.diff(analysis.call_by_spot) >= 0.0)
    assert np.all(np.diff(analysis.put_by_spot) <= 0.0)
    assert np.all(np.diff(analysis.value_by_volatility) >= -1e-13)


def test_dashboard_renderer_handles_success_and_validation_errors() -> None:
    success = render_analysis("call", 100.0, 100.0, 365.0, 5.0, 20.0, 0.0)
    missing = render_analysis("call", None, 100.0, 365.0, 5.0, 20.0, 0.0)
    invalid = render_analysis("call", 0.0, 100.0, 365.0, 5.0, 20.0, 0.0)

    assert isinstance(success, html.Div)
    assert "Call value: $10.4506" in success.children[0].children
    assert missing.role == "alert"
    assert "Missing required inputs" in missing.children
    assert invalid.role == "alert"
    assert "spot must be strictly positive" in invalid.children


def test_dash_app_and_compatibility_entry_point_initialize() -> None:
    app = create_app()
    compatibility = runpy.run_path(str(ROOT / "dashboard.py"))

    assert app.title == "Black-Scholes-Merton Analytics"
    assert app.layout is not None
    assert len(app.callback_map) == 1
    assert compatibility["app"].title == app.title
    assert compatibility["server"] is compatibility["app"].server
