"""Consumer tests for atomic replay controls, linked selections and downloads."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest
from dash import Dash, dcc, html

from quant_pricing.lab_data import ReportLibrary, load_reports, visible_options
from quant_pricing.lab_diagnostics import diagnostic_options, diagnostic_payload
from quant_pricing.lab_export import export_view
from quant_pricing.lab_workspace import (
    clicked_symbol,
    register_replay_callbacks,
    render_replay,
    replay_layout,
    resolve_replay,
)


@pytest.fixture
def library() -> ReportLibrary:
    return load_reports()


def test_recorded_selectors_cascade_without_retaining_invalid_values(
    library: ReportLibrary,
) -> None:
    initial = resolve_replay(library)
    assert initial.view is not None
    assert initial.checkpoint in initial.checkpoints
    assert initial.expiry in initial.expiries
    report = deepcopy(initial.view.report)
    report["checkpoints"] = [deepcopy(initial.view.checkpoint)]
    report["checkpoints"][0]["as_of"] = "2030-01-01T00:00:00Z"
    report["checkpoints"][0]["expiries"] = [deepcopy(initial.view.expiry)]
    report["checkpoints"][0]["expiries"][0]["expiry"] = "2030-06-01T08:00:00+00:00"
    other = ReportLibrary({"original": initial.view.report, "other": report}, ())
    changed = resolve_replay(
        other,
        "other",
        initial.checkpoint,
        initial.expiry,
        selected={"symbol": "NO-LONGER-PRESENT"},
    )
    assert changed.session == "other"
    assert changed.checkpoint == "2030-01-01T00:00:00Z"
    assert changed.expiry == "2030-06-01T08:00:00+00:00"
    assert changed.view is not None
    assert changed.view.checkpoint is report["checkpoints"][0]
    assert changed.symbol in {
        row["symbol"] for row in visible_options(changed.view, changed.side)
    }
    retained = resolve_replay(
        other,
        changed.session,
        changed.checkpoint,
        changed.expiry,
        selected=changed.store(),
    )
    assert retained.store() == changed.store()
    assert resolve_replay(other, "missing").session == "original"


def test_filters_retain_visible_symbols_and_clear_invisible_selection(
    library: ReportLibrary,
) -> None:
    initial = resolve_replay(library, side="call")
    assert initial.view is not None
    assert initial.symbol is not None
    retained = resolve_replay(library, side="all", selected=initial.store())
    assert retained.symbol == initial.symbol
    changed = resolve_replay(library, side="put", selected=initial.store())
    assert changed.symbol != initial.symbol
    assert changed.symbol in {
        row["symbol"] for row in visible_options(initial.view, "put")
    }
    stale = resolve_replay(library, selected={"symbol": "obsolete"})
    assert stale.symbol == resolve_replay(library).symbol
    assert resolve_replay(library, side="invalid").side == "otm"


@pytest.mark.parametrize(
    "trigger",
    ["lab-smile", "lab-contract", "lab-diagnostic-contract"],
)
def test_valid_click_reveals_contract_outside_current_side(
    library: ReportLibrary,
    trigger: str,
) -> None:
    calls = resolve_replay(library, side="call")
    puts = resolve_replay(library, side="put")
    assert puts.symbol is not None
    selected = resolve_replay(
        library,
        side="call",
        selected=calls.store(),
        trigger=trigger,
        clicked=puts.symbol,
    )
    assert selected.side == "all"
    assert selected.symbol == puts.symbol
    repeated = resolve_replay(
        library,
        side=selected.side,
        selected=selected.store(),
        trigger=trigger,
        clicked=puts.symbol,
    )
    assert repeated.store() == selected.store()


def test_stale_plot_click_and_invalid_symbol_cannot_override_new_filter(
    library: ReportLibrary,
) -> None:
    calls = resolve_replay(library, side="call")
    filtered = resolve_replay(
        library,
        side="put",
        selected=calls.store(),
        trigger="lab-side",
        clicked=calls.symbol,
    )
    assert filtered.side == "put"
    assert filtered.symbol != calls.symbol
    invalid = resolve_replay(
        library,
        side="call",
        selected=calls.store(),
        trigger="lab-smile",
        clicked="NOT-IN-REPORT",
    )
    assert invalid.store() == calls.store()
    assert invalid.side == "call"


def test_recorded_stale_checkpoint_clears_otm_contract_and_resets_zoom(
    library: ReportLibrary,
) -> None:
    initial = resolve_replay(library, mode="quotes", tab="diagnostics")
    stale = resolve_replay(
        library,
        initial.session,
        initial.checkpoints[-1],
        initial.expiry,
        selected=initial.store(),
        mode="quotes",
        tab="diagnostics",
        trigger="lab-checkpoint",
    )
    assert stale.symbol is None
    assert stale.view is not None
    assert stale.view.expiry["forward_input"] is None
    initial_render = render_replay(initial, theme="dark")
    stale_render = render_replay(stale, theme="dark")
    assert initial_render["figure"].layout.uirevision != (
        stale_render["figure"].layout.uirevision
    )
    assert stale_render["contract"] is not None
    restored = resolve_replay(
        library,
        initial.session,
        initial.checkpoint,
        initial.expiry,
        selected=stale.store(),
    )
    assert restored.symbol == initial.symbol


@pytest.mark.parametrize(
    "click",
    [
        None,
        [],
        {},
        {"points": []},
        {"points": [None]},
        {"points": [{"customdata": ["not-a-symbol"]}]},
        {"points": [{"curveNumber": 1, "x": 1, "y": 2}]},
    ],
)
def test_fit_and_malformed_clicks_do_not_select(click: Any) -> None:
    assert clicked_symbol(click) is None


def test_observation_click_returns_exact_contract_symbol() -> None:
    assert clicked_symbol({"points": [{"customdata": "BTC_USDC-TEST-C"}]}) == (
        "BTC_USDC-TEST-C"
    )


def test_diagnostic_selector_links_relevant_contract_and_retains_item(
    library: ReportLibrary,
) -> None:
    initial = resolve_replay(library, side="all")
    assert initial.view is not None
    options = diagnostic_options(initial.view, "parity", initial.symbol)
    assert options
    item = options[-1]["value"]
    payload = diagnostic_payload(initial.view, "parity", initial.symbol, item)
    assert payload["symbols"]
    selected = resolve_replay(
        library,
        side="call",
        selected=initial.store(),
        tab="parity",
        item=item,
        trigger="lab-diagnostic-item",
    )
    assert selected.item == item
    assert selected.symbol in payload["symbols"]
    assert selected.view is not None
    assert selected.symbol in {
        row["symbol"] for row in visible_options(selected.view, selected.side)
    }
    invalid = resolve_replay(library, tab="parity", item="invalid-item")
    assert invalid.item in {option["value"] for option in invalid.items}


def test_empty_private_library_is_visible_and_never_replaced_with_demo() -> None:
    library = ReportLibrary({}, ("No readable historical reports",))
    selection = resolve_replay(library, mode="quotes", tab="diagnostics")
    assert selection.view is None
    assert selection.symbol is None
    assert selection.checkpoints == []
    assert selection.expiries == []
    rendered = render_replay(selection, theme="light")
    assert "No report loaded" in rendered["source"]
    assert "no synthetic substitution" in rendered["source"]
    assert rendered["context"].role == "status"
    layout = replay_layout(library)
    alerts = [
        child for child in layout.children if getattr(child, "role", None) == "alert"
    ]
    assert alerts[0].children == "No readable historical reports"
    assert (
        resolve_replay(ReportLibrary({"empty": {"checkpoints": []}}, ())).view is None
    )
    assert (
        resolve_replay(
            ReportLibrary(
                {
                    "empty": {"checkpoints": [{"as_of": "t", "expiries": []}]},
                },
                (),
            )
        ).view
        is None
    )


def test_render_preserves_zoom_for_selection_theme_but_resets_for_dataset(
    library: ReportLibrary,
) -> None:
    first = resolve_replay(library, mode="quotes", tab="diagnostics")
    other = resolve_replay(library, side="put", tab="parity")
    a = render_replay(first, theme="dark")
    b = render_replay(other, theme="light")
    assert a["figure"].layout.uirevision == b["figure"].layout.uirevision
    assert "Public synthetic demo" in a["source"]
    assert first.view is not None
    private = ReportLibrary({"private": deepcopy(first.view.report)}, ())
    private.sessions["private"]["source"] = "historical archive"
    private_render = render_replay(
        resolve_replay(private, mode="greeks", tab="midpoint"),
        theme="dark",
    )
    assert private_render["source"] == "Local report · unverified source · Offline"
    assert private_render["figure"].layout.uirevision != a["figure"].layout.uirevision


def _app(library: ReportLibrary) -> Dash:
    app = Dash(__name__)
    app.layout = html.Div(
        [
            dcc.RadioItems(id="theme-mode", options=["dark", "light"], value="dark"),
            replay_layout(library),
        ]
    )
    register_replay_callbacks(app, library)
    return app


def _values(library: ReportLibrary) -> dict[str, Any]:
    selection = resolve_replay(library)
    return {
        "lab-session.value": selection.session,
        "lab-checkpoint.value": selection.checkpoint,
        "lab-expiry.value": selection.expiry,
        "lab-side.value": selection.side,
        "lab-mode.value": "iv",
        "lab-diagnostic-tab.value": "butterflies",
        "lab-diagnostic-item.value": selection.item,
        "theme-mode.value": "dark",
        "lab-smile.clickData": None,
        '{"symbol":["ALL"],"type":"lab-contract"}.n_clicks': [],
        '{"symbol":["ALL"],"type":"lab-diagnostic-contract"}.n_clicks': [],
        "lab-selected.data": selection.store(),
        "lab-export.n_clicks": 1,
        "lab-export-scope.value": "chain",
        "lab-export-format.value": "json",
    }


def _post(
    app: Dash,
    values: dict[str, Any],
    changed: str,
    *,
    download: bool = False,
) -> Any:
    key = next(
        key for key in app.callback_map if ("lab-download.data" in key) == download
    )
    callback = app.callback_map[key]

    def args(kind: str) -> list[Any]:
        return [
            values[f"{arg['id']}.{arg['property']}"]
            if arg["id"].startswith("{")
            else {**arg, "value": values[f"{arg['id']}.{arg['property']}"]}
            for arg in callback[kind]
        ]

    outputs = callback["output"]
    output_json = (
        [
            {"id": output.component_id, "property": output.component_property}
            for output in outputs
        ]
        if isinstance(outputs, list)
        else {"id": outputs.component_id, "property": outputs.component_property}
    )
    return app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": key,
            "outputs": output_json,
            "inputs": args("inputs"),
            "state": args("state"),
            "changedPropIds": [changed],
        },
    )


def test_callback_has_one_selection_owner_and_no_store_feedback(
    library: ReportLibrary,
) -> None:
    app = _app(library)
    assert len(app.callback_map) == 2
    callback = next(
        value for key, value in app.callback_map.items() if "lab-selected.data" in key
    )
    assert all(arg["id"] != "lab-selected" for arg in callback["inputs"])
    download = app.callback_map["lab-download.data"]
    assert download["inputs"] == [{"id": "lab-export", "property": "n_clicks"}]
    assert download["state"] == [
        {"id": "lab-selected", "property": "data"},
        {"id": "lab-export-scope", "property": "value"},
        {"id": "lab-export-format", "property": "value"},
    ]
    values = _values(library)
    values["lab-side.value"] = "put"
    response = _post(app, values, "lab-side.value")
    assert response.status_code == 200
    result = response.get_json()["response"]
    assert result["lab-side"]["value"] == "put"
    expected = resolve_replay(library, side="put")
    assert result["lab-selected"]["data"] == expected.store()
    assert result["lab-expiry"]["value"] == expected.expiry
    assert result["lab-checkpoint"]["value"] == expected.checkpoint
    assert result["lab-export"]["disabled"] is False
    values["lab-selected.data"] = result["lab-selected"]["data"]
    values["theme-mode.value"] = "light"
    repeated = _post(app, values, "theme-mode.value").get_json()["response"]
    assert repeated["lab-selected"]["data"] == result["lab-selected"]["data"]
    assert (
        repeated["lab-smile"]["figure"]["layout"]["uirevision"]
        == (result["lab-smile"]["figure"]["layout"]["uirevision"])
    )


def test_callbacks_ignore_fit_clicks_and_newly_mounted_buttons(
    library: ReportLibrary,
) -> None:
    app = _app(library)
    values = _values(library)
    assert _post(app, values, "lab-smile.clickData").status_code == 204
    symbol = resolve_replay(library).symbol
    button_id = {"type": "lab-contract", "symbol": symbol}
    changed = json.dumps(button_id, sort_keys=True)
    assert _post(app, values, f"{changed}.n_clicks").status_code == 204
    values['{"symbol":["ALL"],"type":"lab-contract"}.n_clicks'] = [
        {"id": button_id, "property": "n_clicks", "value": 1},
    ]
    response = _post(app, values, f"{changed}.n_clicks")
    assert response.status_code == 200
    assert response.get_json()["response"]["lab-selected"]["data"]["symbol"] == symbol
    values["lab-smile.clickData"] = {"points": [{"customdata": "invalid"}]}
    invalid = _post(app, values, "lab-smile.clickData")
    assert invalid.status_code == 200
    assert (
        invalid.get_json()["response"]["lab-selected"]["data"]
        == (values["lab-selected.data"])
    )


def test_plot_click_atomically_reveals_row_and_next_checkpoint_clears_it(
    library: ReportLibrary,
) -> None:
    app = _app(library)
    values = _values(library)
    calls = resolve_replay(library, side="call")
    puts = resolve_replay(library, side="put")
    values["lab-side.value"] = "call"
    values["lab-selected.data"] = calls.store()
    values["lab-smile.clickData"] = {"points": [{"customdata": puts.symbol}]}
    response = _post(app, values, "lab-smile.clickData")
    assert response.status_code == 200
    result = response.get_json()["response"]
    assert result["lab-side"]["value"] == "all"
    assert result["lab-selected"]["data"]["symbol"] == puts.symbol
    assert puts.symbol in json.dumps(result["lab-chain"])
    values["lab-selected.data"] = result["lab-selected"]["data"]
    values["lab-side.value"] = "otm"
    values["lab-checkpoint.value"] = calls.checkpoints[-1]
    stale = _post(app, values, "lab-checkpoint.value").get_json()["response"]
    assert stale["lab-selected"]["data"]["symbol"] is None
    assert stale["lab-side"]["value"] == "otm"


def test_diagnostic_render_marks_raw_relevant_strikes(library: ReportLibrary) -> None:
    selected = resolve_replay(library, mode="quotes", tab="parity")
    assert selected.view is not None
    assert selected.item is not None
    payload = diagnostic_payload(
        selected.view,
        "parity",
        selected.symbol,
        selected.item,
    )
    rendered = render_replay(selected, theme="dark")
    assert set(payload["strikes"]) <= {
        shape.x0 for shape in rendered["figure"].layout.shapes
    }


def test_download_uses_resolved_selection_and_has_no_initial_download(
    library: ReportLibrary,
) -> None:
    app = _app(library)
    values = _values(library)
    response = _post(app, values, "lab-export.n_clicks", download=True)
    assert response.status_code == 200
    payload = response.get_json()["response"]["lab-download"]["data"]
    assert payload["filename"].endswith(".json")
    assert json.loads(payload["content"])
    for selected in (
        None,
        {},
        {"session": None},
        {
            "session": "missing",
            "checkpoint": "missing",
            "expiry": "missing",
        },
    ):
        values["lab-selected.data"] = selected
        assert (
            _post(app, values, "lab-export.n_clicks", download=True).status_code == 204
        )
    values = _values(library)
    values["lab-export.n_clicks"] = 0
    assert _post(app, values, "lab-export.n_clicks", download=True).status_code == 204


@pytest.mark.parametrize("scope", ["chain", "iv", "fit", "contract", "diagnostic"])
@pytest.mark.parametrize("format", ["json", "csv"])
def test_download_uses_last_rendered_snapshot_until_selection_commits(
    library: ReportLibrary,
    scope: str,
    format: str,
) -> None:
    initial = resolve_replay(library)
    assert initial.view is not None
    pending_report = deepcopy(initial.view.report)
    pending_report["checkpoints"] = [deepcopy(initial.view.checkpoint)]
    pending_checkpoint = pending_report["checkpoints"][0]
    pending_checkpoint["as_of"] = "2030-01-03T00:00:00Z"
    pending_checkpoint["expiries"] = [deepcopy(initial.view.expiry)]
    pending_expiry = pending_checkpoint["expiries"][0]
    pending_expiry["expiry"] = "2030-06-01T08:00:00+00:00"
    library = ReportLibrary(
        {
            "rendered": initial.view.report,
            "pending": pending_report,
        },
        (),
    )
    app = _app(library)
    values = _values(library)
    parity_item = diagnostic_options(initial.view, "parity", initial.symbol)[-1][
        "value"
    ]
    values.update(
        {
            "lab-mode.value": "quotes",
            "lab-side.value": "call",
            "lab-diagnostic-tab.value": "parity",
            "lab-diagnostic-item.value": parity_item,
        }
    )
    response = _post(app, values, "lab-diagnostic-item.value")
    assert response.status_code == 200
    rendered = response.get_json()["response"]
    snapshot = rendered["lab-selected"]["data"]
    assert snapshot["mode"] == "quotes"
    assert snapshot["tab"] == "parity"
    assert snapshot["side"] == rendered["lab-side"]["value"]
    assert snapshot["item"] == rendered["lab-diagnostic-item"]["value"] == parity_item
    assert snapshot["symbol"] in json.dumps(rendered["lab-contract"])
    values["lab-selected.data"] = snapshot
    values.update(
        {
            "lab-session.value": "pending",
            "lab-checkpoint.value": pending_checkpoint["as_of"],
            "lab-expiry.value": pending_expiry["expiry"],
            "lab-mode.value": "greeks",
            "lab-side.value": "put",
            "lab-diagnostic-tab.value": "diagnostics",
            "lab-diagnostic-item.value": "summary",
            "lab-export-scope.value": scope,
            "lab-export-format.value": format,
        }
    )

    def assert_download_matches(committed: dict[str, Any]) -> None:
        response = _post(app, values, "lab-export.n_clicks", download=True)
        assert response.status_code == 200
        view = library.view(
            committed["session"],
            committed["checkpoint"],
            committed["expiry"],
        )
        assert response.get_json()["response"]["lab-download"]["data"] == export_view(
            view,
            scope=scope,
            format=format,
            mode=committed["mode"],
            side=committed["side"],
            symbol=committed["symbol"],
            tab=committed["tab"],
            item=committed["item"],
        )

    # Live controls have changed, but their render transaction has not committed.
    assert_download_matches(snapshot)
    response = _post(app, values, "lab-session.value")
    assert response.status_code == 200
    committed = response.get_json()["response"]["lab-selected"]["data"]
    assert committed["session"] == "pending"
    assert committed["checkpoint"] == pending_checkpoint["as_of"]
    assert committed["expiry"] == pending_expiry["expiry"]
    assert committed["mode"] == "greeks"
    assert committed["side"] == "put"
    assert committed["tab"] == "diagnostics"
    assert committed["item"] == "summary"
    values["lab-selected.data"] = committed
    assert_download_matches(committed)


def test_empty_library_callback_disables_export() -> None:
    library = ReportLibrary({}, ())
    app = _app(library)
    response = _post(app, _values(library), "lab-session.value")
    assert response.status_code == 200
    result = response.get_json()["response"]
    assert result["lab-export"]["disabled"] is True
    assert result["lab-selected"]["data"]["symbol"] is None
    assert result["lab-diagnostic-item"]["disabled"] is True


def _components(component: Any) -> list[Any]:
    if not hasattr(component, "to_plotly_json"):
        return []
    children = getattr(component, "children", [])
    children = children if isinstance(children, list) else [children]
    return [component, *(child for item in children for child in _components(item))]


def test_landing_controls_and_accessible_labels(library: ReportLibrary) -> None:
    layout = replay_layout(library)
    components = _components(layout)
    by_id = {
        component.id: component
        for component in components
        if isinstance(getattr(component, "id", None), str)
    }
    assert any(
        isinstance(component, html.H1)
        and component.children == "BTC/USDC options replay"
        for component in components
    )
    assert by_id["lab-source"].children == "Public synthetic demo · Offline"
    assert by_id["lab-mode"].value == "iv"
    assert [option["label"] for option in by_id["lab-mode"].options] == [
        "IV",
        "Premium",
        "Greeks",
    ]
    assert by_id["lab-diagnostic-tab"].value == "butterflies"
    assert by_id["lab-selected"].data == {
        "session": by_id["lab-session"].value,
        "checkpoint": by_id["lab-checkpoint"].value,
        "expiry": by_id["lab-expiry"].value,
        "symbol": resolve_replay(library).symbol,
        "mode": by_id["lab-mode"].value,
        "side": by_id["lab-side"].value,
        "tab": by_id["lab-diagnostic-tab"].value,
        "item": by_id["lab-diagnostic-item"].value,
    }
    assert [tab.label for tab in by_id["lab-diagnostic-tab"].children] == [
        "Diagnostics",
        "Parity",
        "Midpoint warnings",
        "Quoted butterflies",
    ]
    grid = next(
        component
        for component in components
        if getattr(component, "className", "") == "lab-controls-grid"
    )
    assert len(grid.children) == 3
    assert [field.children[0].htmlFor for field in grid.children] == [
        "lab-session",
        "lab-expiry",
        "lab-checkpoint",
    ]
    for field in grid.children:
        assert field.role == "group"
        assert (
            field.to_plotly_json()["props"]["aria-label"] == field.children[0].children
        )
    assert by_id["lab-quality"].to_plotly_json()["props"]["aria-live"] == "polite"
    assert isinstance(by_id["lab-contract"], html.Div)
    assert isinstance(by_id["lab-contract"].children, html.Details)
    assert all("T" not in tab.label for tab in by_id["lab-checkpoint"].children)
    assert by_id["lab-session"].options[0]["label"] == "Jan 2, 2030"
    assert by_id["lab-expiry"].options[0]["label"] == "Jan 25, 2030"
    assert by_id["lab-smile"].responsive is True
    assert isinstance(by_id["lab-smile-note"], html.P)
    assert isinstance(by_id["lab-smile-note"].children, str)
    assert "not confidence" in by_id["lab-smile-note"].children
    diagnostic_header = next(
        component
        for component in components
        if getattr(component, "className", "") == "lab-diagnostic-header"
    )
    assert isinstance(diagnostic_header.children[0], html.H2)
    assert diagnostic_header.children[0].children == "Diagnostics"
    assert by_id["lab-diagnostic-tab"] in diagnostic_header.children
    assert by_id["lab-diagnostic-item"] in _components(diagnostic_header)
    diagnostic_section = next(
        component
        for component in layout.children
        if getattr(component, "className", "") == "lab-panel lab-diagnostics"
    )
    main_grid = next(
        component
        for component in layout.children
        if getattr(component, "className", "") == "lab-workspace-grid lab-main-grid"
    )
    assert (
        layout.children.index(main_grid)
        < layout.children.index(diagnostic_section)
        < layout.children.index(by_id["lab-contract"])
    )


def test_quality_counts_and_recorded_assumptions(library: ReportLibrary) -> None:
    initial = resolve_replay(library)
    for checkpoint in initial.checkpoints:
        selection = resolve_replay(library, checkpoint=checkpoint)
        assert selection.view is not None
        coverage = selection.view.expiry["coverage"]
        rendered = render_replay(selection, theme="dark")
        requested, usable = coverage["requested_options"], coverage["usable_books"]
        assert rendered["quality"][0].children == (
            f"{requested} requested · {usable} usable · {requested - usable} withheld"
        )
        assert (
            "30 s freshness limit · ACT/365F · D = " in rendered["quality"][1].children
        )


@pytest.mark.parametrize(
    ("source", "badge"),
    [
        ("tardis/deribit", "Deribit via Tardis · Offline"),
        (None, "Local report · unverified source · Offline"),
        (42, "Local report · unverified source · Offline"),
        (
            "synthetic-sounding historical archive",
            "Local report · unverified source · Offline",
        ),
    ],
)
def test_source_badge_requires_exact_provenance(
    library: ReportLibrary,
    source: object,
    badge: str,
) -> None:
    initial = resolve_replay(library)
    assert initial.view is not None
    report = deepcopy(initial.view.report)
    report["source"] = source
    selection = resolve_replay(ReportLibrary({"local": report}, ()))
    assert render_replay(selection, theme="dark")["source"] == badge


def test_checkpoint_dates_disambiguate_days_without_changing_values(
    library: ReportLibrary,
) -> None:
    initial = resolve_replay(library)
    assert initial.view is not None
    report = deepcopy(initial.view.report)
    report["checkpoints"] = [
        deepcopy(initial.view.checkpoint),
        deepcopy(initial.view.checkpoint),
    ]
    values = ["2030-01-02T00:00:01Z", "2030-01-03T00:00:01Z"]
    for checkpoint, value in zip(report["checkpoints"], values, strict=True):
        checkpoint["as_of"] = value
    components = _components(replay_layout(ReportLibrary({"days": report}, ())))
    tabs = next(
        component
        for component in components
        if getattr(component, "id", None) == "lab-checkpoint"
    ).children
    assert [tab.value for tab in tabs] == values
    assert [tab.label for tab in tabs] == [
        "Jan 2, 2030 · 00:00:01",
        "Jan 3, 2030 · 00:00:01",
    ]


def test_default_butterfly_is_atm_related_and_diagnostics_start_with_summary(
    library: ReportLibrary,
) -> None:
    selection = resolve_replay(library)
    assert selection.view is not None
    payload = diagnostic_payload(
        selection.view,
        "butterflies",
        selection.symbol,
        selection.item,
    )
    triplets = selection.view.expiry["quoted_butterflies"]["triplets"]
    related = [
        triplet
        for triplet in triplets
        if any(
            leg["symbol"] == selection.symbol
            for leg in (triplet.get("best_package") or {}).get("option_legs", [])
        )
    ]
    if related:
        assert selection.symbol in payload["symbols"]
    forward = selection.view.expiry["forward_input"]
    nearest = min(
        related or triplets,
        key=lambda triplet: abs(triplet["strikes"][1] - forward),
    )
    assert payload["strikes"] == nearest["strikes"]
    summary = resolve_replay(library, tab="diagnostics", selected=selection.store())
    assert summary.item == "summary"
    assert summary.symbol == selection.symbol


@pytest.mark.parametrize("kind", ["lab-contract", "lab-diagnostic-contract"])
def test_pattern_callback_ignores_other_buttons_click_history(
    library: ReportLibrary,
    kind: str,
) -> None:
    app = _app(library)
    initial = resolve_replay(library)
    assert initial.view is not None
    other_symbol = next(
        row["symbol"]
        for row in visible_options(initial.view, "all")
        if row["symbol"] != initial.symbol
    )
    values = _values(library)
    button_id = {"type": kind, "symbol": other_symbol}
    key = json.dumps({"symbol": ["ALL"], "type": kind}, separators=(",", ":"))
    values[f"{key}.n_clicks"] = [
        {
            "id": {"type": kind, "symbol": initial.symbol},
            "property": "n_clicks",
            "value": 3,
        },
        {"id": button_id, "property": "n_clicks", "value": 0},
    ]
    changed = json.dumps(button_id, sort_keys=True, separators=(",", ":"))
    assert _post(app, values, f"{changed}.n_clicks").status_code == 204
    values[f"{key}.n_clicks"][1]["value"] = 1
    response = _post(app, values, f"{changed}.n_clicks")
    assert response.status_code == 200
    assert (
        response.get_json()["response"]["lab-selected"]["data"]["symbol"]
        == other_symbol
    )
