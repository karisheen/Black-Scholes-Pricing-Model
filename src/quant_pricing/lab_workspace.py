"""Replay workspace with one atomic owner for dependent controls and selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, cast

import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

from quant_pricing.lab_chain import (
    chain_table,
    context_strip,
    contract_panel,
    fit_panel,
    smile_figure,
)
from quant_pricing.lab_data import (
    ReplayView,
    ReportLibrary,
    Row,
    choose_symbol,
    visible_options,
)
from quant_pricing.lab_demo import DEMO_SOURCE
from quant_pricing.lab_diagnostics import (
    diagnostic_options,
    diagnostic_panel,
    diagnostic_payload,
)
from quant_pricing.lab_export import export_view


@dataclass(frozen=True)
class ReplaySelection:
    """One display snapshot for rendering and export, including resolved controls."""

    session: str | None
    checkpoint: str | None
    expiry: str | None
    checkpoints: list[str]
    expiries: list[str]
    side: str
    mode: str
    tab: str
    symbol: str | None
    item: str | None
    items: list[dict[str, str]]
    view: ReplayView | None

    def store(self) -> Row:
        """Serialize the evidence selection committed alongside the rendered view."""
        return {
            "session": self.session,
            "checkpoint": self.checkpoint,
            "expiry": self.expiry,
            "symbol": self.symbol,
            "mode": self.mode,
            "side": self.side,
            "tab": self.tab,
            "item": self.item,
        }


def _retain(value: str | None, available: list[str]) -> str | None:
    return value if value in available else next(iter(available), None)


def clicked_symbol(click: Any) -> str | None:
    """Only observation points carry a selectable symbol, never a fitted line."""
    if not isinstance(click, dict):
        return None
    points = click.get("points")
    if not isinstance(points, list) or not points or not isinstance(points[0], dict):
        return None
    symbol = points[0].get("customdata")
    return symbol if isinstance(symbol, str) else None


def resolve_replay(
    library: ReportLibrary,
    session: str | None = None,
    checkpoint: str | None = None,
    expiry: str | None = None,
    *,
    side: str = "otm",
    mode: str = "iv",
    selected: Row | None = None,
    tab: str = "butterflies",
    item: str | None = None,
    trigger: str | None = None,
    clicked: str | None = None,
) -> ReplaySelection:
    """Cascade exact recorded selectors and validate linked user selections.

    The previous store is state, not an input: writing a selection cannot cause
    another callback. Old Plotly clickData is consulted only on a new plot click.
    """
    session = _retain(session, list(library.sessions))
    report = library.sessions[session] if session is not None else {}
    checkpoints = [str(row["as_of"]) for row in report.get("checkpoints", [])]
    checkpoint = _retain(checkpoint, checkpoints)
    record: Row = next(
        (row for row in report.get("checkpoints", []) if row["as_of"] == checkpoint),
        {},
    )
    expiries = [str(row["expiry"]) for row in record.get("expiries", [])]
    expiry = _retain(expiry, expiries)
    side = side if side in {"otm", "call", "put", "all"} else "otm"
    if session is None or checkpoint is None or expiry is None:
        return ReplaySelection(
            session,
            checkpoint,
            expiry,
            checkpoints,
            expiries,
            side,
            mode,
            tab,
            None,
            None,
            [],
            None,
        )
    view = library.view(session, checkpoint, expiry)
    preferred = (selected or {}).get("symbol")
    symbol = choose_symbol(view, side, preferred)
    all_symbols = {str(row["symbol"]) for row in visible_options(view, "all")}
    if (
        trigger in {"lab-smile", "lab-contract", "lab-diagnostic-contract"}
        and clicked in all_symbols
    ):
        symbol = clicked
        if symbol not in {str(row["symbol"]) for row in visible_options(view, side)}:
            side = "all"
    items = diagnostic_options(view, tab, symbol)
    available_items = [option["value"] for option in items]
    default_item = _default_item(view, tab, symbol, items)
    item = item if item in available_items else default_item
    if item is not None and trigger != "lab-diagnostic-item":
        related = diagnostic_payload(view, tab, symbol, item).get("symbols", [])
        if related and symbol not in related:
            item = default_item
    if trigger == "lab-diagnostic-item" and item is not None:
        related = diagnostic_payload(view, tab, symbol, item).get("symbols", [])
        relevant = [str(value) for value in related if value in all_symbols]
        if relevant:
            symbol = symbol if symbol in relevant else relevant[0]
            if symbol not in {
                str(row["symbol"]) for row in visible_options(view, side)
            }:
                side = "all"
            items = diagnostic_options(view, tab, symbol)
            item = _retain(item, [option["value"] for option in items])
    return ReplaySelection(
        session,
        checkpoint,
        expiry,
        checkpoints,
        expiries,
        side,
        mode,
        tab,
        symbol,
        item,
        items,
        view,
    )


def _default_item(
    view: ReplayView, tab: str, symbol: str | None, items: list[dict[str, str]]
) -> str | None:
    if tab == "diagnostics":
        return "summary" if any(item["value"] == "summary" for item in items) else None
    if tab == "butterflies" and items:
        forward = view.expiry.get("forward_input")
        triplets = (view.expiry.get("quoted_butterflies") or {}).get("triplets", [])
        row: Row = next(
            (row for row in view.expiry.get("options", []) if row["symbol"] == symbol),
            {},
        )
        if isinstance(forward, (int, float)):

            def proximity(item: dict[str, str]) -> tuple[bool, float]:
                triplet = triplets[int(item["value"].split(":")[1])]
                strikes = triplet["strikes"]
                legs = (triplet.get("best_package") or {}).get("option_legs", [])
                related = (
                    any(leg["symbol"] == symbol for leg in legs)
                    if legs
                    else row.get("strike") in strikes
                )
                return not related, abs(strikes[len(strikes) // 2] - forward)

            return min(items, key=proximity)["value"]
    return next((item["value"] for item in items), None)


def _quality(view: ReplayView) -> Any:
    coverage = view.expiry.get("coverage") or {}
    usable = coverage.get("usable_books", "—")
    requested = coverage.get("requested_options", "—")
    withheld = (
        requested - usable if type(requested) is int and type(usable) is int else "—"
    )
    assumptions = view.report.get("assumptions") or {}
    age = assumptions.get("max_book_age_ms")
    freshness = f"{age / 1000:g} s" if isinstance(age, (int, float)) else "—"
    discount = view.expiry.get("discount_factor")
    discount_label = f"{discount:.6g}" if isinstance(discount, (int, float)) else "—"
    return [
        html.Span(
            f"{requested} requested · {usable} usable · {withheld} withheld",
            className="lab-quality-counts",
        ),
        html.Span(
            f"{freshness} freshness limit · {assumptions.get('day_count', '—')} "
            f"· D = {discount_label}",
            className="lab-quality-assumptions",
        ),
    ]


def render_replay(selection: ReplaySelection, *, theme: str) -> Row:
    """Render only resolved state, keeping the graph itself permanently mounted."""
    view = selection.view
    if view is None:
        message = "No recorded expiry is available. Configure a valid report directory."
        figure = go.Figure()
        figure.update_layout(
            template="plotly_dark" if theme == "dark" else "plotly_white",
            annotations=[{"text": "No recorded observations", "showarrow": False}],
            xaxis={"visible": False},
            yaxis={"visible": False},
            uirevision="empty",
        )
        return {
            "source": "No report loaded — no synthetic substitution",
            "quality": "No recorded quality information",
            "context": html.Div(message, className="lab-empty", role="status"),
            "chain": html.Div("No contracts", className="lab-empty"),
            "figure": figure,
            "fit": html.Div("No fitted slice", className="lab-empty"),
            "contract": html.Div("No selected contract", className="lab-empty"),
            "diagnostic": html.Div("No diagnostic items", className="lab-empty"),
        }
    symbol = selection.symbol
    source = view.report.get("source")
    label = {
        DEMO_SOURCE: "Public synthetic demo",
        "tardis/deribit": "Deribit via Tardis",
    }.get(source if isinstance(source, str) else "", "Local report · unverified source")
    figure = smile_figure(view, selection.side, symbol, theme)
    if selection.item is not None and selection.tab != "diagnostics":
        diagnostic = diagnostic_payload(view, selection.tab, symbol, selection.item)
        for strike in dict.fromkeys(diagnostic.get("strikes", [])):
            figure.add_vline(
                x=strike,
                line_width=1,
                line_dash="dot",
                line_color="#1aabf4",
                opacity=0.4,
            )
    figure.update_layout(
        uirevision=f"{selection.session}|{selection.checkpoint}|{selection.expiry}"
    )
    return {
        "source": f"{label} · Offline",
        "quality": _quality(view),
        "context": context_strip(view, symbol),
        "chain": chain_table(view, selection.mode, selection.side, symbol),
        "figure": figure,
        "fit": fit_panel(view, theme),
        "contract": contract_panel(view, symbol),
        "diagnostic": diagnostic_panel(view, selection.tab, symbol, selection.item),
    }


def _options(values: list[str]) -> list[dict[str, str]]:
    return [{"label": value, "value": value} for value in values]


def _timestamp(value: str) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (
        stamp.astimezone(timezone.utc)
        if stamp.tzinfo
        else stamp.replace(tzinfo=timezone.utc)
    )


def _date_label(value: str) -> str:
    stamp = _timestamp(value)
    return f"{stamp:%b} {stamp.day}, {stamp.year}" if stamp else value


def _expiry_options(values: list[str]) -> list[dict[str, str]]:
    labels = [_date_label(value) for value in values]
    return [
        {
            "label": (
                f"{label} · {stamp:%H:%M:%S} UTC"
                if labels.count(label) > 1 and (stamp := _timestamp(value))
                else label
            ),
            "value": value,
        }
        for value, label in zip(values, labels, strict=True)
    ]


def _session_options(library: ReportLibrary) -> list[dict[str, str]]:
    options = []
    for session, report in library.sessions.items():
        checkpoints = report.get("checkpoints") or []
        label = _date_label(str(checkpoints[0]["as_of"])) if checkpoints else session
        options.append({"label": label, "value": session})
    labels = [option["label"] for option in options]
    for option in options:
        if labels.count(option["label"]) > 1:
            option["label"] = f"{option['label']} · {option['value']}"
    return options


def _checkpoint_tabs(values: list[str]) -> list[Any]:
    stamps = [_timestamp(value) for value in values]
    dates = {stamp.date() for stamp in stamps if stamp is not None}
    return [
        dcc.Tab(
            label=(
                f"{_date_label(value)} · {stamp:%H:%M:%S}"
                if stamp and len(dates) > 1
                else f"{stamp:%H:%M:%S}"
                if stamp
                else value
            ),
            value=value,
            className="lab-tab",
            selected_className="lab-tab--selected",
        )
        for value, stamp in zip(values, stamps, strict=True)
    ]


def _field(
    label: str, component: Any, target: str, *, class_name: str = "lab-field"
) -> Any:
    return html.Div(
        [html.Label(label, htmlFor=target, className="lab-control-label"), component],
        className=class_name,
        role="group",
        **cast(Row, {"aria-label": label}),
    )


def replay_layout(library: ReportLibrary) -> Any:
    """Build the replay workspace, excluding the shell's header/navigation."""
    selection = resolve_replay(library)
    rendered = render_replay(selection, theme="dark")
    errors = [
        html.Div(error, role="alert", className="lab-error") for error in library.errors
    ]
    return html.Div(
        [
            dcc.Store(id="lab-selected", data=selection.store()),
            dcc.Download(id="lab-download"),
            *errors,
            html.Div(
                [
                    html.H1("BTC/USDC options replay"),
                    html.Div(
                        rendered["source"],
                        id="lab-source",
                        className="lab-source-badge",
                    ),
                ],
                className="lab-title-row",
            ),
            html.Div(
                [
                    _field(
                        "Session",
                        dcc.Dropdown(
                            id="lab-session",
                            options=_session_options(library),
                            value=selection.session,
                            clearable=False,
                            disabled=not library.sessions,
                        ),
                        "lab-session",
                    ),
                    _field(
                        "Expiry (UTC)",
                        dcc.Dropdown(
                            id="lab-expiry",
                            options=_expiry_options(selection.expiries),
                            value=selection.expiry,
                            clearable=False,
                            disabled=not selection.expiries,
                        ),
                        "lab-expiry",
                    ),
                    _field(
                        "Recorded checkpoint · UTC",
                        dcc.Tabs(
                            id="lab-checkpoint",
                            value=selection.checkpoint,
                            children=_checkpoint_tabs(selection.checkpoints),
                            className="lab-tabs lab-checkpoint-tabs",
                        ),
                        "lab-checkpoint",
                        class_name="lab-field lab-checkpoint-field",
                    ),
                ],
                className="lab-controls-grid",
            ),
            html.Div(
                rendered["quality"],
                id="lab-quality",
                className="lab-status",
                role="status",
                **cast(Row, {"aria-live": "polite"}),
            ),
            html.Div(rendered["context"], id="lab-context", className="lab-context"),
            html.Div(
                [
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.H2("Option chain"),
                                    _field(
                                        "View",
                                        dcc.RadioItems(
                                            id="lab-mode",
                                            options=[
                                                {"label": "IV", "value": "iv"},
                                                {"label": "Premium", "value": "quotes"},
                                                {"label": "Greeks", "value": "greeks"},
                                            ],
                                            value=selection.mode,
                                            inline=True,
                                            className="lab-segments",
                                        ),
                                        "lab-mode",
                                        class_name="lab-toolbar-field",
                                    ),
                                    _field(
                                        "Side",
                                        dcc.RadioItems(
                                            id="lab-side",
                                            options=[
                                                {"label": "OTM", "value": "otm"},
                                                {"label": "Calls", "value": "call"},
                                                {"label": "Puts", "value": "put"},
                                                {"label": "All", "value": "all"},
                                            ],
                                            value=selection.side,
                                            inline=True,
                                            className="lab-segments",
                                        ),
                                        "lab-side",
                                        class_name="lab-toolbar-field",
                                    ),
                                ],
                                className="lab-toolbar",
                            ),
                            html.Div(rendered["chain"], id="lab-chain"),
                        ],
                        className="lab-panel lab-chain-panel",
                    ),
                    html.Section(
                        [
                            html.H2("Volatility smile"),
                            dcc.Loading(
                                dcc.Graph(
                                    id="lab-smile",
                                    figure=rendered["figure"],
                                    config={
                                        "displaylogo": False,
                                        "scrollZoom": False,
                                        "doubleClick": "reset+autosize",
                                    },
                                    responsive=True,
                                ),
                                type="dot",
                                color="#ffa51a",
                            ),
                            html.P(
                                "Amber: bid-ask IV spread, not confidence. "
                                "Markers: midpoint IV; blue: stored fit.",
                                id="lab-smile-note",
                                className="lab-chart-note",
                            ),
                            html.Div(
                                rendered["fit"], id="lab-fit", className="lab-fit-panel"
                            ),
                        ],
                        className="lab-panel lab-smile-panel",
                    ),
                ],
                className="lab-workspace-grid lab-main-grid",
            ),
            html.Div(
                className="lab-panel-handle", **cast(Row, {"aria-hidden": "true"})
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.H2("Diagnostics", className="lab-diagnostics-title"),
                            dcc.Tabs(
                                id="lab-diagnostic-tab",
                                value=selection.tab,
                                children=[
                                    dcc.Tab(
                                        label=label,
                                        value=value,
                                        className="lab-tab",
                                        selected_className="lab-tab--selected",
                                    )
                                    for label, value in (
                                        ("Diagnostics", "diagnostics"),
                                        ("Parity", "parity"),
                                        ("Midpoint warnings", "midpoint"),
                                        ("Quoted butterflies", "butterflies"),
                                    )
                                ],
                                className="lab-tabs",
                                parent_className="lab-diagnostic-tabs",
                            ),
                            _field(
                                "Diagnostic item",
                                dcc.Dropdown(
                                    id="lab-diagnostic-item",
                                    options=selection.items,
                                    value=selection.item,
                                    clearable=False,
                                    disabled=not selection.items,
                                    placeholder="No recorded items",
                                ),
                                "lab-diagnostic-item",
                                class_name="lab-diagnostic-selector",
                            ),
                        ],
                        className="lab-diagnostic-header",
                    ),
                    html.Div(rendered["diagnostic"], id="lab-diagnostic-content"),
                ],
                className="lab-panel lab-diagnostics",
            ),
            html.Div(
                rendered["contract"], id="lab-contract", className="lab-contract-slot"
            ),
            html.Div(
                [
                    _field(
                        "Export scope",
                        dcc.Dropdown(
                            id="lab-export-scope",
                            options=[
                                {"label": label, "value": value}
                                for label, value in (
                                    ("Visible chain", "chain"),
                                    ("IV observations", "iv"),
                                    ("Fitted slice", "fit"),
                                    ("Selected contract", "contract"),
                                    ("Selected diagnostic", "diagnostic"),
                                )
                            ],
                            value="chain",
                            clearable=False,
                        ),
                        "lab-export-scope",
                    ),
                    _field(
                        "Format",
                        dcc.Dropdown(
                            id="lab-export-format",
                            options=_options(["csv", "json"]),
                            value="csv",
                            clearable=False,
                        ),
                        "lab-export-format",
                    ),
                    html.Button(
                        "Export view",
                        id="lab-export",
                        n_clicks=0,
                        disabled=selection.view is None,
                    ),
                ],
                className="lab-export-controls",
            ),
        ],
        className="lab-workspace",
    )


_RENDER_OUTPUTS = (
    ("lab-source", "children", "source"),
    ("lab-quality", "children", "quality"),
    ("lab-context", "children", "context"),
    ("lab-chain", "children", "chain"),
    ("lab-smile", "figure", "figure"),
    ("lab-fit", "children", "fit"),
    ("lab-contract", "children", "contract"),
    ("lab-diagnostic-content", "children", "diagnostic"),
)


def register_replay_callbacks(app: Dash, library: ReportLibrary) -> None:
    """Commit display state atomically; export only that last rendered snapshot."""

    @app.callback(
        Output("lab-expiry", "options"),
        Output("lab-expiry", "value"),
        Output("lab-expiry", "disabled"),
        Output("lab-checkpoint", "children"),
        Output("lab-checkpoint", "value"),
        Output("lab-side", "value"),
        Output("lab-selected", "data"),
        Output("lab-diagnostic-item", "options"),
        Output("lab-diagnostic-item", "value"),
        Output("lab-diagnostic-item", "disabled"),
        Output("lab-export", "disabled"),
        *[Output(component, prop) for component, prop, _ in _RENDER_OUTPUTS],
        Input("lab-session", "value"),
        Input("lab-checkpoint", "value"),
        Input("lab-expiry", "value"),
        Input("lab-side", "value"),
        Input("lab-mode", "value"),
        Input("lab-diagnostic-tab", "value"),
        Input("lab-diagnostic-item", "value"),
        Input("theme-mode", "value"),
        Input("lab-smile", "clickData"),
        Input({"type": "lab-contract", "symbol": ALL}, "n_clicks"),
        Input({"type": "lab-diagnostic-contract", "symbol": ALL}, "n_clicks"),
        State("lab-selected", "data"),
    )
    def update_replay(
        session: str | None,
        checkpoint: str | None,
        expiry: str | None,
        side: str,
        mode: str,
        tab: str,
        item: str | None,
        theme: str,
        click: Any,
        row_clicks: list[int],
        diagnostic_clicks: list[int],
        selected: Row | None,
    ) -> tuple[Any, ...]:
        triggered = ctx.triggered_id
        trigger = triggered.get("type") if isinstance(triggered, dict) else triggered
        clicked = clicked_symbol(click) if trigger == "lab-smile" else None
        if isinstance(triggered, dict):
            # Check the exact button; a different row may retain an earlier click.
            group_index = 9 if trigger == "lab-contract" else 10
            counts = row_clicks if trigger == "lab-contract" else diagnostic_clicks
            inputs = ctx.inputs_list[group_index]
            count = next(
                (
                    count
                    for button, count in zip(inputs, counts, strict=True)
                    if button["id"] == triggered
                ),
                0,
            )
            if not count:
                raise PreventUpdate
            clicked = triggered.get("symbol")
        if trigger == "lab-smile" and clicked is None:
            raise PreventUpdate
        selection = resolve_replay(
            library,
            session,
            checkpoint,
            expiry,
            side=side,
            mode=mode,
            selected=selected,
            tab=tab,
            item=item,
            trigger=trigger,
            clicked=clicked,
        )
        rendered = render_replay(selection, theme=theme)
        return (
            _expiry_options(selection.expiries),
            selection.expiry,
            not selection.expiries,
            _checkpoint_tabs(selection.checkpoints),
            selection.checkpoint,
            selection.side,
            selection.store(),
            selection.items,
            selection.item,
            not selection.items,
            selection.view is None,
            *(rendered[key] for _, _, key in _RENDER_OUTPUTS),
        )

    @app.callback(
        Output("lab-download", "data"),
        Input("lab-export", "n_clicks"),
        State("lab-selected", "data"),
        State("lab-export-scope", "value"),
        State("lab-export-format", "value"),
        prevent_initial_call=True,
    )
    def download_replay(
        clicks: int,
        selected: Row | None,
        scope: str,
        format: str,
    ) -> Row:
        if not clicks or not selected:
            raise PreventUpdate
        session = selected.get("session")
        checkpoint = selected.get("checkpoint")
        expiry = selected.get("expiry")
        if session is None or checkpoint is None or expiry is None:
            raise PreventUpdate
        try:
            view = library.view(session, checkpoint, expiry)
        except KeyError:
            raise PreventUpdate from None
        return export_view(
            view,
            scope=scope,
            mode=selected["mode"],
            side=selected["side"],
            symbol=selected.get("symbol"),
            tab=selected["tab"],
            item=selected.get("item"),
            format=format,
        )
