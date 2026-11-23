"""Behavioral coverage for source-backed chain and smile presentations."""

from __future__ import annotations

from copy import deepcopy
from math import exp
from typing import Any

import pytest
from dash.development.base_component import Component

from quant_pricing.lab_chain import (
    chain_table,
    context_strip,
    contract_panel,
    fit_panel,
    smile_figure,
)
from quant_pricing.lab_data import (
    ReplayView,
    contract_detail,
    load_reports,
    market_context,
    visible_options,
)
from quant_pricing.surface import FitResidual, SVIParameters


@pytest.fixture
def view() -> ReplayView:
    library = load_reports()
    for session, source in library.sessions.items():
        report = deepcopy(source)
        for checkpoint in report["checkpoints"]:
            for expiry in checkpoint["expiries"]:
                if expiry.get("fit"):
                    return ReplayView(session, report, checkpoint, expiry)
    raise AssertionError("Public synthetic replay must contain a fitted expiry")


def _components(value: Any) -> list[Component]:
    if isinstance(value, Component):
        return [value, *_components(getattr(value, "children", None))]
    if isinstance(value, (list, tuple)):
        return [component for child in value for component in _components(child)]
    return []


def _text(value: Any) -> str:
    if isinstance(value, Component):
        return _text(getattr(value, "children", None))
    if isinstance(value, (list, tuple)):
        return " ".join(_text(child) for child in value)
    return "" if value is None else str(value)


def _selected(view: ReplayView) -> str:
    return str(view.expiry["fit"]["residuals"][0]["contract"])


@pytest.mark.parametrize("mode", ["iv", "quotes", "greeks"])
@pytest.mark.parametrize("side", ["otm", "call", "put", "all"])
def test_chain_modes_and_accessible_selection(
    view: ReplayView, mode: str, side: str
) -> None:
    rows = visible_options(view, side)
    symbol = str(rows[0]["symbol"])
    panel = chain_table(view, mode, side, symbol)
    components = _components(panel)
    buttons = [
        item for item in components if isinstance(getattr(item, "id", None), dict)
    ]
    assert [item.id["symbol"] for item in buttons] == [row["symbol"] for row in rows]
    assert all(item.id["type"] == "lab-contract" for item in buttons)
    assert all(item.n_clicks == 0 for item in buttons)
    assert buttons[0].to_plotly_json()["props"]["aria-pressed"] is True
    assert symbol in buttons[0].to_plotly_json()["props"]["aria-label"]
    assert any(getattr(item, "className", "") == "lab-selected" for item in components)
    text = _text(panel)
    assert ("Bid size" in text) is (mode == "quotes")
    assert ("Ask size" in text) is (mode == "quotes")
    assert "Age" in text
    assert "Quality" in text
    expected = {"iv": "Bid IV", "quotes": "Spread", "greeks": "Rho"}
    assert expected[mode] in text
    headers = [item.children for item in components if item.__class__.__name__ == "Th"]
    if mode == "iv":
        assert headers == [
            "Strike",
            "Side",
            "Bid IV",
            "Mid IV",
            "Ask IV",
            "Age",
            "Quality",
        ]
    assert "USDC/BTC" in text
    scroll = next(
        item
        for item in components
        if getattr(item, "className", "") == "lab-table-wrap"
    )
    assert scroll.tabIndex == 0


def test_chain_otm_is_one_leg_per_strike(view: ReplayView) -> None:
    options = visible_options(view, "otm")
    assert len({row["strike"] for row in options}) == len(options)
    forward = view.expiry["forward_input"]
    assert all(
        row["option_type"] == ("put" if row["strike"] < forward else "call")
        for row in options
    )


def test_chain_invalid_mode_and_empty(view: ReplayView) -> None:
    with pytest.raises(ValueError, match="Unknown chain mode"):
        chain_table(view, "invalid", "all", None)
    view.expiry["options"] = []
    assert "No contracts" in _text(chain_table(view, "iv", "all", None))


def test_excluded_rows_keep_quotes_but_withhold_iv_and_risk(view: ReplayView) -> None:
    symbol = _selected(view)
    row = next(row for row in view.expiry["options"] if row["symbol"] == symbol)
    row["rejected"] = ["stale"]
    row["normalized_flags"] = ["stale"]
    row["sides"] = {}
    for mode in ("iv", "quotes", "greeks"):
        components = _components(chain_table(view, mode, "all", symbol))
        selected = next(
            item
            for item in components
            if getattr(item, "className", "") == "lab-selected"
        )
        assert "stale" in _text(selected)
        if mode != "quotes":
            assert "—" in _text(selected)
    detail = _text(contract_panel(view, symbol))
    assert "Model value · USDC/BTC: —" in detail
    assert "not zero" in detail


def test_side_errors_are_visible_not_usable(view: ReplayView) -> None:
    symbol = _selected(view)
    row = next(row for row in view.expiry["options"] if row["symbol"] == symbol)
    row["rejected"] = []
    row["normalized_flags"] = []
    row["sides"]["bid"]["iv"] = None
    row["sides"]["bid"]["error"] = "below intrinsic bound"
    text = _text(chain_table(view, "iv", "all", symbol))
    assert "bid: below intrinsic bound" in text


def test_context_uses_adapter_and_unknown_is_not_zero(view: ReplayView) -> None:
    symbol = _selected(view)
    text = _text(context_strip(view, symbol))
    assert "Forward · synthetic" in text
    assert "Spot · missing" in text
    assert symbol in text
    assert market_context(view, symbol)["spot"] is None
    assert "Spot · missing —" in text
    assert "Selected contract None" in _text(context_strip(view, None))


def test_contract_exposes_every_risk_and_source_assumption(view: ReplayView) -> None:
    symbol = _selected(view)
    detail = contract_detail(view, symbol)
    panel = contract_panel(view, symbol)
    text = _text(panel)
    assert panel.className == "lab-contract-details"
    assert detail["greeks"] is not None
    for greek in (
        "Delta",
        "Gamma",
        "Vega",
        "Theta",
        "Rho",
        "Vanna",
        "Volga",
        "Charm",
        "Speed",
        "Color",
    ):
        assert greek in text
    assert detail["risk_basis"] in text
    assert "risk_assumptions" in text
    assert "normalized_quote" in text
    assert "source_sequence" in text
    for unit in detail["risk_units"].values():
        assert str(unit) in text
    assert "Select a strike" in _text(contract_panel(view, None))


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("side", ["otm", "call", "put", "all"])
def test_smile_observations_are_selectable_and_fit_is_domain_limited(
    view: ReplayView, theme: str, side: str
) -> None:
    rows = visible_options(view, side)
    symbol = str(rows[0]["symbol"])
    figure = smile_figure(view, side, symbol, theme)
    observations = [trace for trace in figure.data if trace.mode == "markers"]
    assert observations
    assert all(trace.connectgaps is False for trace in observations)
    assert all(
        isinstance(value, str) for trace in observations for value in trace.customdata
    )
    assert all("%{customdata}" in trace.hovertemplate for trace in observations)
    assert any(11 in trace.marker.size for trace in observations)
    assert figure.layout.shapes[0].x0 == rows[0]["strike"]
    fit_trace = next(trace for trace in figure.data if trace.name == "Stored SVI fit")
    assert fit_trace.customdata is None
    fit = view.expiry["fit"]
    forward = view.expiry["forward_input"]
    assert min(fit_trace.x) == pytest.approx(forward * exp(fit["domain"][0]))
    assert max(fit_trace.x) == pytest.approx(forward * exp(fit["domain"][1]))
    parameters = SVIParameters(**fit["parameters"])
    assert fit_trace.y[0] ** 2 * fit["time_to_expiry"] == pytest.approx(
        parameters.total_variance(fit["domain"][0])
    )
    assert fit_trace.line.color == ("#007bb5" if theme == "light" else "#1aabf4")
    assert all(
        trace.marker.color == ("#b86600" if theme == "light" else "#ffa51a")
        for trace in observations
    )
    assert figure.layout.dragmode == "zoom"
    assert not figure.layout.title.text


def test_smile_gaps_never_join_stale_or_failed_observations(view: ReplayView) -> None:
    options = visible_options(view, "all")
    first, second = options[:2]
    first["rejected"] = ["stale"]
    second["sides"] = {"bid": {"iv": None, "error": "solver failed"}}
    figure = smile_figure(view, "all", first["symbol"], "dark")
    for trace in figure.data:
        if trace.mode == "markers":
            for index, name in enumerate(trace.customdata):
                if name in {first["symbol"], second["symbol"]}:
                    assert trace.y[index] is None
            assert trace.mode == "markers"
    assert any(trace.name == "Stored SVI fit" for trace in figure.data)


def test_smile_no_fit_keeps_observations_and_explains_failure(view: ReplayView) -> None:
    view.expiry["fit"] = None
    view.expiry["fit_error"] = "insufficient OTM points"
    figure = smile_figure(view, "all", None, "dark")
    assert all(trace.name != "Stored SVI fit" for trace in figure.data)
    assert any(trace.mode == "markers" for trace in figure.data)
    assert "insufficient OTM points" in figure.layout.annotations[0].text
    assert not figure.layout.shapes
    assert "insufficient OTM points" in _text(fit_panel(view, "dark"))
    view.expiry["fit_error"] = None
    assert "No recorded fit" in _text(fit_panel(view, "light"))


def test_empty_smile_remains_informative(view: ReplayView) -> None:
    view.expiry["options"] = []
    view.expiry["fit"] = None
    figure = smile_figure(view, "all", None, "light")
    assert len(figure.data) == 0
    assert "No usable bid/ask IV observations" in figure.layout.annotations[0].text
    assert figure.layout.xaxis.title.text == "Strike · USDC/BTC"
    assert figure.layout.yaxis.title.text == "Implied volatility"


def test_invalid_stored_fit_is_explicit_not_refitted(view: ReplayView) -> None:
    view.expiry["fit"]["parameters"]["sigma"] = -1
    figure = smile_figure(view, "all", None, "dark")
    assert all(trace.name != "Stored SVI fit" for trace in figure.data)
    assert any(trace.mode == "markers" for trace in figure.data)
    assert "invalid parameters or domain" in figure.layout.annotations[0].text


@pytest.mark.parametrize("converged", [True, False])
@pytest.mark.parametrize("passed", [True, False, None])
def test_fit_residual_evidence_and_independent_statuses(
    view: ReplayView, converged: bool, passed: bool | None
) -> None:
    fit = view.expiry["fit"]
    fit["converged"] = converged
    fit["arbitrage"] = {"passed": passed, "findings": [{"kind": "butterfly"}]}
    panel = fit_panel(view, "dark")
    text = _text(panel)
    assert ("Converged" if converged else "Not converged") in text
    expected = {
        True: "Sampled arbitrage checks passed",
        False: "Sampled arbitrage findings",
        None: "Arbitrage checks unavailable",
    }
    assert expected[passed] in text
    assert "not a continuous arbitrage certificate" in text
    assert "Weighted RMSE (total variance)" in text
    assert "butterfly" in text
    assert "weights" in text
    graph = next(item for item in _components(panel) if hasattr(item, "figure"))
    residuals = fit["residuals"]
    assert list(graph.figure.data[0].y) == [
        FitResidual(**row).residual for row in residuals
    ]
    assert graph.figure.data[0].customdata is None
    assert graph.config["responsive"] is True
    fit["residuals"] = []
    empty = fit_panel(view, "light")
    graph = next(item for item in _components(empty) if hasattr(item, "figure"))
    assert "No recorded residuals" in graph.figure.layout.annotations[0].text


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), "unknown"])
def test_unavailable_quote_values_render_dash(view: ReplayView, value: Any) -> None:
    symbol = _selected(view)
    row = next(row for row in view.expiry["options"] if row["symbol"] == symbol)
    row["normalized_quote"]["bid"] = value
    row["normalized_quote"]["bid_size"] = value
    components = _components(chain_table(view, "quotes", "all", symbol))
    selected = next(
        item for item in components if getattr(item, "className", "") == "lab-selected"
    )
    cells = selected.children
    assert _text(cells[2]) == "—"
    assert _text(cells[3]) == "—"
    assert _text(cells[5]) == "—"
    assert _text(cells[6]) == "—"


def test_stored_fit_remains_visible_without_observations(view: ReplayView) -> None:
    for row in view.expiry["options"]:
        row["sides"] = {}
    figure = smile_figure(view, "all", None, "dark")
    assert any(trace.name == "Stored SVI fit" for trace in figure.data)
    assert "No usable bid/ask IV observations" in figure.layout.annotations[0].text


def _primary_text(value: Any) -> str:
    """Visible evidence must not rely on the optional raw-JSON disclosure."""
    if isinstance(value, Component):
        if value.__class__.__name__ == "Pre":
            return ""
        return _primary_text(getattr(value, "children", None))
    if isinstance(value, (list, tuple)):
        return " ".join(_primary_text(child) for child in value)
    return "" if value is None else str(value)


def test_raw_premium_midpoint_and_spread_survive_blocked_iv(view: ReplayView) -> None:
    symbol = _selected(view)
    row = next(row for row in view.expiry["options"] if row["symbol"] == symbol)
    row["normalized_quote"].update(bid=123.0, ask=137.0, bid_size=2.0, ask_size=3.0)
    row["rejected"] = ["stale"]
    row["sides"] = {"midpoint": {"premium_usdc_per_btc": 999.0, "iv": 0.6}}
    selected = next(
        item
        for item in _components(chain_table(view, "quotes", "all", symbol))
        if getattr(item, "className", "") == "lab-selected"
    )
    assert [_text(cell) for cell in selected.children[2:8]] == [
        "123",
        "130",
        "137",
        "14",
        "2",
        "3",
    ]
    assert "stale" in _text(selected)
    assert "excluded quote" in selected.children[3].title
    primary = _primary_text(contract_panel(view, symbol))
    assert "Mid premium · USDC/BTC 130" in primary
    assert "Spread · USDC/BTC 14" in primary
    assert "Midpoint IV —" in primary


def test_greeks_table_includes_actual_rho(view: ReplayView) -> None:
    symbol = _selected(view)
    detail = contract_detail(view, symbol)
    selected = next(
        item
        for item in _components(chain_table(view, "greeks", "all", symbol))
        if getattr(item, "className", "") == "lab-selected"
    )
    rho = detail["greeks"]["rho"]
    assert _text(selected.children[6]) == f"{rho:,.6g}"


def test_contract_primary_view_contains_quote_evidence(view: ReplayView) -> None:
    symbol = _selected(view)
    detail = contract_detail(view, symbol)
    primary = _primary_text(contract_panel(view, symbol))
    for label in (
        "Side",
        "Strike · USDC/BTC",
        "Exact expiry · UTC",
        "Bid premium · USDC/BTC",
        "Mid premium · USDC/BTC",
        "Ask premium · USDC/BTC",
        "Bid size · contracts",
        "Ask size · contracts",
        "Exchange quote time · UTC",
        "Receipt / available at · UTC",
        "Bid IV",
        "Midpoint IV",
        "Ask IV",
        "Model basis",
        "Risk basis",
    ):
        assert label in primary
    assert detail["expiry"] in primary
    assert detail["normalized_quote"]["quote_time"] in primary
    assert detail["normalized_quote"]["available_at"] in primary
    assert "not the mean of bid/ask IV" in primary
    for assumption in detail["risk_assumptions"]:
        assert assumption in primary


def test_context_labels_quantity_provenance_and_two_decimal_dte(
    view: ReplayView,
) -> None:
    strip = context_strip(view, None)
    text = _primary_text(strip)
    context = market_context(view, None)
    assert "ATM IV · quote-implied" in text
    assert "Forward · synthetic" in text
    assert "Rate · assumed" in text
    assert "Discount factor · modeled" in text
    assert "Spot · missing" in text
    assert f"DTE · derived {context['dte']:.2f}" in text
    metrics = strip.children
    assert len(metrics) == 8
    assert all(len(metric.children) == 2 for metric in metrics)
    assert "USDC/BTC" in metrics[1].title
    assert "synthetic futures proxy" in metrics[1].title
    assert "not observed volatility" in metrics[3].title
    assert "Annual" in metrics[5].title
    assert "not observed volatility" not in text
    assert "not observed volatility" in getattr(metrics[3], "aria-label")
    view.report["source"] = "recorded exchange"
    assert "Forward · observed" in _primary_text(context_strip(view, None))


def test_band_contains_no_excluded_or_failed_contract(view: ReplayView) -> None:
    rows = visible_options(view, "otm")
    excluded, failed = rows[1:3]
    excluded["rejected"] = ["stale"]
    failed["sides"]["ask"].update(error="solver failed", iv=0.99)
    figure = smile_figure(view, "otm", str(rows[0]["symbol"]), "dark")
    bands = [trace for trace in figure.data if trace.fill == "toself"]
    assert bands
    strikes = sorted({row["strike"] for row in rows})
    withheld = {excluded["strike"], failed["strike"]}
    for trace in bands:
        assert trace.connectgaps is False
        assert trace.customdata is None
        assert trace.name == "Bid-ask IV visual envelope"
        assert len(trace.x) % 6 == 0
        for offset in range(0, len(trace.x), 6):
            left, right = trace.x[offset], trace.x[offset + 2]
            assert left not in withheld
            assert right not in withheld
            assert strikes.index(right) == strikes.index(left) + 1
            assert trace.x[offset] == trace.x[offset + 4]
            assert trace.y[offset] == trace.y[offset + 4]
            assert trace.x[offset + 5] is None
            assert trace.y[offset + 5] is None
    midpoint = [trace for trace in figure.data if "midpoint" in trace.name]
    assert midpoint
    for trace in midpoint:
        assert trace.mode == "markers"
        for symbol, value in zip(trace.customdata, trace.y, strict=True):
            detail = contract_detail(view, symbol)
            expected = (
                None
                if symbol == excluded["symbol"]
                else detail["sides"]["midpoint"]["iv"]
            )
            assert value == expected
    assert figure.layout.yaxis.tickformat == ".2%"
    assert figure.layout.legend.orientation == "h"
    assert sum(trace.showlegend is not False for trace in figure.data) <= 4
    assert figure.layout.margin.b == 42
    assert figure.layout.autosize is True
    assert not figure.layout.annotations


def _envelope_options(view: ReplayView, side: str) -> list[dict[str, Any]]:
    """Small deterministic quote evidence with distinct call and put intervals."""
    forward = view.expiry["forward_input"]
    rows: list[dict[str, Any]] = []
    for index in range(5):
        strike = forward + (index - 2) * 1000
        kinds = (
            ("put", "call")
            if side == "all"
            else ("put" if strike < forward else "call",)
        )
        for kind in kinds:
            bid = (0.4 if kind == "put" else 0.5) + index * 0.01
            rows.append(
                {
                    "symbol": f"{kind}-{index}",
                    "strike": strike,
                    "option_type": kind,
                    "sides": {
                        "bid": {"iv": bid},
                        "midpoint": {"iv": bid + 0.01},
                        "ask": {"iv": bid + 0.02},
                    },
                }
            )
    view.expiry["options"] = rows
    return rows


@pytest.mark.parametrize("side", ["otm", "all"])
def test_usable_adjacent_strikes_form_endpoint_only_envelope(
    view: ReplayView, side: str
) -> None:
    rows = _envelope_options(view, side)
    figure = smile_figure(view, side, rows[2]["symbol"], "dark")
    bands = [trace for trace in figure.data if trace.fill == "toself"]
    assert len(bands) == (2 if side == "all" else 1)
    series = (
        [
            [row for row in rows if row["option_type"] == kind]
            for kind in ("put", "call")
        ]
        if side == "all"
        else [rows]
    )
    for trace, observations in zip(bands, series, strict=True):
        assert len(trace.x) == 24
        for index in range(4):
            left, right = observations[index : index + 2]
            offset = index * 6
            assert tuple(trace.x[offset : offset + 6]) == (
                left["strike"],
                left["strike"],
                right["strike"],
                right["strike"],
                left["strike"],
                None,
            )
            assert tuple(trace.y[offset : offset + 6]) == (
                left["sides"]["bid"]["iv"],
                left["sides"]["ask"]["iv"],
                right["sides"]["ask"]["iv"],
                right["sides"]["bid"]["iv"],
                left["sides"]["bid"]["iv"],
                None,
            )
    assert not any(trace.name == "Isolated bid-ask IV range" for trace in figure.data)


@pytest.mark.parametrize("failure", ["excluded", "failed"])
def test_gapped_envelope_keeps_isolated_range_without_bridging(
    view: ReplayView, failure: str
) -> None:
    rows = _envelope_options(view, "otm")
    for row in rows[1:3]:
        if failure == "excluded":
            row["rejected"] = ["stale"]
        else:
            row["sides"]["ask"]["error"] = "solver failed"
    figure = smile_figure(view, "otm", rows[0]["symbol"], "light")
    bands = [trace for trace in figure.data if trace.fill == "toself"]
    assert len(bands) == 1
    assert tuple(bands[0].x) == (
        rows[3]["strike"],
        rows[3]["strike"],
        rows[4]["strike"],
        rows[4]["strike"],
        rows[3]["strike"],
        None,
    )
    isolated = next(
        trace for trace in figure.data if trace.name == "Isolated bid-ask IV range"
    )
    assert tuple(isolated.x) == (rows[0]["strike"], rows[0]["strike"], None)
    assert tuple(isolated.y) == (
        rows[0]["sides"]["bid"]["iv"],
        rows[0]["sides"]["ask"]["iv"],
        None,
    )
    selected = next(trace for trace in figure.data if trace.name == "Put midpoint")
    assert selected.customdata[0] == rows[0]["symbol"]
    assert selected.marker.size[0] == 11
    assert "%{customdata}" in selected.hovertemplate


def test_smile_withholds_computed_expiry_blocks(view: ReplayView) -> None:
    view.expiry["iv_blocked"] = "missing_forward"
    figure = smile_figure(view, "all", None, "dark")
    for trace in figure.data:
        if trace.mode == "markers":
            assert all(value is None for value in trace.y)
        assert trace.fill != "toself"


def test_fit_exposes_counts_warning_and_bounded_graph(view: ReplayView) -> None:
    view.expiry["fit"]["converged"] = False
    view.expiry["options"][0]["rejected"] = ["stale"]
    panel = fit_panel(view, "dark")
    primary = _primary_text(panel)
    used = len({row["contract"] for row in view.expiry["fit"]["residuals"]})
    assert f"Used in fit: {used}" in primary
    assert "Rejected / IV unavailable:" in primary
    assert "Not fitted:" in primary
    assert "Optimizer did not converge" in primary
    graph = next(item for item in _components(panel) if hasattr(item, "figure"))
    assert graph.style["height"] == "190px"
    assert graph.style["width"] == "100%"
    assert graph.figure.layout.width is None
    assert graph.config["doubleClick"] == "reset+autosize"


def test_unavailable_fit_still_exposes_used_and_rejected_counts(
    view: ReplayView,
) -> None:
    view.expiry["fit"] = None
    view.expiry["options"][0]["rejected"] = ["stale"]
    primary = _primary_text(fit_panel(view, "light"))
    assert "Used in fit: 0" in primary
    assert "Rejected / IV unavailable:" in primary
    assert f"Not fitted: {len(view.expiry['options'])}" in primary
