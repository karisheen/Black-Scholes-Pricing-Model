"""Integration and compatibility tests for user-facing entry points."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dash import html

import quant_pricing.dashboard as dashboard_module
from quant_pricing import price
from quant_pricing.cli import main as cli_main
from quant_pricing.dashboard import (
    calculate_scenario,
    create_app,
    render_analysis,
    theme_class_name,
)
from quant_pricing.lab_data import ReportLibrary

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
    assert analysis.spot_grid.dtype == np.float64
    assert analysis.volatility_grid.shape == (161,)
    assert analysis.volatility_grid.dtype == np.float64
    assert np.all(np.diff(analysis.call_by_spot) >= 0.0)
    assert np.all(np.diff(analysis.put_by_spot) <= 0.0)
    assert np.all(np.diff(analysis.value_by_volatility) >= -1e-13)


def test_dashboard_renderer_handles_success_and_validation_errors() -> None:
    success = render_analysis("call", 100.0, 100.0, 365.0, 5.0, 20.0, 0.0)
    dark = render_analysis("call", 100.0, 100.0, 365.0, 5.0, 20.0, 0.0, theme="dark")
    missing = render_analysis("call", None, 100.0, 365.0, 5.0, 20.0, 0.0)
    invalid = render_analysis("call", 0.0, 100.0, 365.0, 5.0, 20.0, 0.0)

    assert isinstance(success, html.Div)
    assert "Call value: $10.4506" in success.children[0].children
    light_background = (
        success.children[2].children[0].figure.layout.template.layout.paper_bgcolor
    )
    dark_background = (
        dark.children[2].children[0].figure.layout.template.layout.paper_bgcolor
    )
    assert light_background != dark_background
    assert missing.role == "alert"
    assert "Missing required inputs" in missing.children
    assert invalid.role == "alert"
    assert "spot must be strictly positive" in invalid.children


def test_dashboard_recovers_iv_with_day_and_percent_inputs() -> None:
    premium = price("put", 110.0, 100.0, 0.5, 0.045, 0.27, 0.02)
    assert isinstance(premium, float)
    results = render_analysis(
        "put", 110.0, 100.0, 182.5, 4.5, None, 2.0, mode="iv", premium=premium
    )
    assert results.children[0].children == "Put implied volatility: 27.0000%"


def test_dashboard_iv_distinguishes_zero_premium_from_missing_premium() -> None:
    zero = render_analysis(
        "put", 110.0, 100.0, 365.0, 5.0, None, 0.0, mode="iv", premium=0.0
    )
    missing = render_analysis("put", 110.0, 100.0, 365.0, 5.0, None, 0.0, mode="iv")

    assert zero.children[0].children == "Put implied volatility: 0.0000%"
    assert missing.role == "alert"
    assert "option premium" in missing.children


@pytest.mark.parametrize(
    ("premium", "message"),
    [(100.0, "requires infinite volatility"), (101.0, "outside no-arbitrage bounds")],
)
def test_dashboard_iv_reports_unsolvable_premiums(premium: float, message: str) -> None:
    results = render_analysis(
        "call", 100.0, 100.0, 365.0, 5.0, None, 0.0, mode="iv", premium=premium
    )
    assert results.role == "alert"
    assert message in results.children


def test_theme_switch_preserves_last_calculated_scenario_with_pending_edits() -> None:
    client = create_app().server.test_client()
    callbacks = client.get("/_dash-dependencies").get_json()
    values: dict[str, Any] = {
        "calculate.n_clicks": 1,
        "pricing-mode.value": "price",
        "option-type.value": "call",
        "spot.value": 100.0,
        "strike.value": 100.0,
        "time-days.value": 365.0,
        "rate.value": 5.0,
        "volatility.value": 20.0,
        "dividend-yield.value": 0.0,
        "premium.value": None,
        "theme-mode.value": "light",
    }

    def update(output: str, changed: str) -> Any:
        callback = next(item for item in callbacks if item["output"] == output)

        def arguments(kind: str) -> list[dict[str, Any]]:
            return [
                {**arg, "value": values[f"{arg['id']}.{arg['property']}"]}
                for arg in callback[kind]
            ]

        identifier, prop = output.split(".")
        response = client.post(
            "/_dash-update-component",
            json={
                "output": output,
                "outputs": {"id": identifier, "property": prop},
                "inputs": arguments("inputs"),
                "state": arguments("state"),
                "changedPropIds": [changed],
            },
        )
        assert response.status_code == 200
        return response.get_json()["response"][identifier][prop]

    values["submitted-scenario.data"] = update(
        "submitted-scenario.data", "calculate.n_clicks"
    )
    values["spot.value"] = 110.0
    values["theme-mode.value"] = "dark"
    themed = update("results.children", "theme-mode.value")
    assert themed["props"]["children"][0]["props"]["children"] == "Call value: $10.4506"
    themed_graph = themed["props"]["children"][2]["props"]["children"][0]
    assert themed_graph["props"]["figure"]["layout"]["paper_bgcolor"] == "#48483a"
    values["theme-mode.value"] = "light"
    light = update("results.children", "theme-mode.value")
    assert light["props"]["children"][0]["props"]["children"] == "Call value: $10.4506"
    light_graph = light["props"]["children"][2]["props"]["children"][0]
    assert light_graph["props"]["figure"]["layout"]["paper_bgcolor"] == "#f2f0e5"

    values["calculate.n_clicks"] = 2
    values["submitted-scenario.data"] = update(
        "submitted-scenario.data", "calculate.n_clicks"
    )
    recalculated = update("results.children", "submitted-scenario.data")
    assert (
        recalculated["props"]["children"][0]["props"]["children"]
        == "Call value: $17.6630"
    )


def test_dash_app_and_compatibility_entry_point_initialize() -> None:
    app = create_app()
    compatibility = runpy.run_path(str(ROOT / "dashboard.py"))

    assert app.title == "Options Lab"
    assert app.layout is not None
    assert app.layout.className == "dashboard-shell theme-dark"
    assert theme_class_name("dark") == "dashboard-shell theme-dark"
    assert theme_class_name("unsupported") == "dashboard-shell theme-dark"
    assert theme_class_name(None) == "dashboard-shell theme-dark"
    assert theme_class_name("light") == "dashboard-shell theme-light"
    assert (Path(app.config.assets_folder) / "dashboard.css").is_file()
    stylesheet = app.server.test_client().get("/assets/dashboard.css")
    assert stylesheet.status_code == 200
    assert b".dashboard-shell.theme-dark" in stylesheet.data
    assert compatibility["app"].title == app.title
    assert compatibility["server"] is compatibility["app"].server


def _layout_props(node: Any) -> dict[str, dict[str, Any]]:
    """Index serialized layout components without depending on nesting."""
    result: dict[str, dict[str, Any]] = {}
    if isinstance(node, list):
        for child in node:
            result.update(_layout_props(child))
    elif isinstance(node, dict) and "props" in node:
        props = node["props"]
        identifier = props.get("id")
        if isinstance(identifier, str):
            result[identifier] = props
        result.update(_layout_props(props.get("children")))
    return result


def test_replay_default_mounts_both_views_and_persists_only_theme() -> None:
    client = create_app().server.test_client()
    response = client.get("/_dash-layout")
    assert response.status_code == 200
    props = _layout_props(response.get_json())
    assert props["lab-navigation"]["value"] == "replay"
    assert props["lab-replay-view"]["hidden"] is False
    assert props["lab-calculator-view"]["hidden"] is True
    assert props["theme-mode"]["value"] == "dark"
    assert props["theme-mode"]["persistence"] is True
    assert props["theme-mode"]["persistence_type"] == "local"
    assert props["spot"]["value"] == 100.0
    assert props["premium-field"]["hidden"] is True
    assert "lab-session" in props
    assert "lab-smile" in props


@pytest.mark.parametrize(
    ("view", "replay_hidden", "calculator_hidden"),
    [("calculator", True, False), ("replay", False, True), ("unknown", False, True)],
)
def test_navigation_only_changes_visibility_not_calculator_state(
    view: str, replay_hidden: bool, calculator_hidden: bool
) -> None:
    client = create_app().server.test_client()
    callbacks = client.get("/_dash-dependencies").get_json()
    callback = next(
        item for item in callbacks if "lab-replay-view.hidden" in item["output"]
    )
    assert callback["inputs"] == [{"id": "lab-navigation", "property": "value"}]
    assert callback["state"] == []
    response = client.post(
        "/_dash-update-component",
        json={
            "output": callback["output"],
            "outputs": [
                {"id": "lab-replay-view", "property": "hidden"},
                {"id": "lab-calculator-view", "property": "hidden"},
            ],
            "inputs": [{"id": "lab-navigation", "property": "value", "value": view}],
            "state": [],
            "changedPropIds": ["lab-navigation.value"],
        },
    )
    assert response.status_code == 200
    assert response.get_json()["response"] == {
        "lab-replay-view": {"hidden": replay_hidden},
        "lab-calculator-view": {"hidden": calculator_hidden},
    }


@pytest.mark.parametrize(
    ("mode", "hidden"), [("price", (False, True)), ("iv", (True, False))]
)
def test_calculator_mode_toggles_fields_without_resetting_values(
    mode: str, hidden: tuple[bool, bool]
) -> None:
    client = create_app().server.test_client()
    callback = next(
        item
        for item in client.get("/_dash-dependencies").get_json()
        if "volatility-field.hidden" in item["output"]
    )
    response = client.post(
        "/_dash-update-component",
        json={
            "output": callback["output"],
            "outputs": [
                {"id": "volatility-field", "property": "hidden"},
                {"id": "premium-field", "property": "hidden"},
            ],
            "inputs": [{"id": "pricing-mode", "property": "value", "value": mode}],
            "state": [],
            "changedPropIds": ["pricing-mode.value"],
        },
    )
    assert response.status_code == 200
    assert response.get_json()["response"] == {
        "volatility-field": {"hidden": hidden[0]},
        "premium-field": {"hidden": hidden[1]},
    }


def test_theme_before_first_submission_does_not_create_a_calculation() -> None:
    client = create_app().server.test_client()
    response = client.post(
        "/_dash-update-component",
        json={
            "output": "results.children",
            "outputs": {"id": "results", "property": "children"},
            "inputs": [
                {"id": "submitted-scenario", "property": "data", "value": None},
                {"id": "theme-mode", "property": "value", "value": "dark"},
            ],
            "state": [],
            "changedPropIds": ["theme-mode.value"],
        },
    )
    assert response.status_code == 204


@pytest.mark.parametrize("explicit", [True, False])
def test_report_directory_uses_explicit_argument_before_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    environment_dir = tmp_path / "environment"
    explicit_dir = tmp_path / "explicit"
    monkeypatch.setenv("OPTIONS_LAB_REPORTS", str(environment_dir))
    library = dashboard_module.load_reports(tmp_path)
    loaded: list[str | Path | None] = []

    def capture(directory: str | Path | None = None) -> ReportLibrary:
        loaded.append(directory)
        return library

    monkeypatch.setattr(dashboard_module, "load_reports", capture)
    app = create_app(report_dir=explicit_dir) if explicit else create_app()
    assert loaded == [explicit_dir if explicit else str(environment_dir)]
    assert app.server.test_client().get("/_dash-layout").status_code == 200


@pytest.mark.parametrize("theme", ["dark", "light", None])
def test_shared_theme_callback_updates_shell(theme: str | None) -> None:
    client = create_app().server.test_client()
    response = client.post(
        "/_dash-update-component",
        json={
            "output": "app-shell.className",
            "outputs": {"id": "app-shell", "property": "className"},
            "inputs": [{"id": "theme-mode", "property": "value", "value": theme}],
            "state": [],
            "changedPropIds": ["theme-mode.value"],
        },
    )
    assert response.status_code == 200
    assert response.get_json()["response"]["app-shell"][
        "className"
    ] == theme_class_name(theme)
