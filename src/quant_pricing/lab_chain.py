"""Read-only chain, smile and fit views over recorded replay evidence."""

from __future__ import annotations

import json
from datetime import datetime
from itertools import pairwise
from math import exp, isfinite
from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html

from quant_pricing.exceptions import InputValidationError
from quant_pricing.lab_data import (
    ReplayView,
    Row,
    contract_detail,
    market_context,
    visible_options,
)
from quant_pricing.surface import (
    FitResidual,
    SVIParameters,
    SVISlice,
    VolatilitySurface,
)

_GREEKS = (
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "vanna",
    "volga",
    "charm",
    "speed",
    "color",
)
_GRAPH_CONFIG: dcc.Graph.Config = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "doubleClick": "reset+autosize",
}


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if isfinite(value) else None
    return None


def _fmt(value: Any, *, percent: bool = False) -> str:
    number = _number(value)
    if number is None:
        return "—"
    return f"{number:.2%}" if percent else f"{number:,.6g}"


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _flags(row: Row) -> list[str]:
    return list(
        dict.fromkeys(
            str(flag)
            for key in ("quality_flags", "rejected", "normalized_flags")
            for flag in row.get(key, ())
        )
    )


def _quality(row: Row) -> html.Span:
    flags = _flags(row)
    if flags:
        label = ", ".join(flags).replace("_", " ")
        tone = "warn" if any("stale" in flag for flag in flags) else "bad"
    else:
        errors = [
            f"{side}: {data['error']}"
            for side, data in row.get("sides", {}).items()
            if data.get("error")
        ]
        label = "; ".join(errors) if errors else "Usable"
        tone = "warn" if errors else "good"
    return html.Span(label, className=f"lab-quality lab-quality--{tone}")


def _aria(**attributes: str | bool) -> dict[str, Any]:
    """Dash accepts wildcard ARIA props outside its generated typed signature."""
    return {f"aria-{key}": value for key, value in attributes.items()}


def _premiums(quote: Row) -> tuple[float | None, ...]:
    bid, ask = _number(quote.get("bid")), _number(quote.get("ask"))
    midpoint = None if bid is None or ask is None else bid / 2 + ask / 2
    spread = None if bid is None or ask is None else ask - bid
    return bid, midpoint, ask, spread


def _iv(row: Row, side: str) -> float | None:
    evidence = (row.get("sides") or {}).get(side) or {}
    value = _number(evidence.get("iv"))
    if _flags(row) or evidence.get("error") or value is None or value <= 0:
        return None
    return value


def chain_table(view: ReplayView, mode: str, side: str, symbol: str | None) -> html.Div:
    """Compact source quote table; withheld IV/risk never becomes a zero."""
    if mode not in {"iv", "quotes", "greeks"}:
        raise ValueError(f"Unknown chain mode: {mode}")
    options = visible_options(view, side)
    if not options:
        return html.Div("No contracts for this selection.", className="lab-empty")
    columns = {
        "iv": ["Bid IV", "Mid IV", "Ask IV"],
        "quotes": ["Bid", "Mid", "Ask", "Spread", "Bid size", "Ask size"],
        "greeks": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
    }[mode]
    headers = ["Strike", "Side", *columns, "Age", "Quality"]
    units = {
        "iv": "Strike · USDC/BTC · IV · % · Age · seconds",
        "quotes": "Strike and premiums · USDC/BTC · Size · contracts · Age · seconds",
        "greeks": (
            "Strike · USDC/BTC · Age · seconds · Forward partials: "
            "Delta ∂V/∂F · Gamma ∂²V/∂F² · Vega per 1 vol · "
            "Theta per year · Rho per 1 rate"
        ),
    }[mode]
    rows = []
    for option in options:
        name = str(option["symbol"])
        detail = contract_detail(view, name)
        quote = detail.get("normalized_quote") or {}
        flags = _flags(detail)
        if mode == "greeks":
            greeks = detail.get("greeks") or {}
            values = [_fmt(greeks.get(key)) for key in _GREEKS[:5]]
        elif mode == "iv":
            values = [
                _fmt(_iv(detail, key), percent=True)
                for key in ("bid", "midpoint", "ask")
            ]
        else:
            # Source arithmetic remains available even when inversion is blocked.
            values = [_fmt(value) for value in _premiums(quote)]
            values.extend(_fmt(quote.get(key)) for key in ("bid_size", "ask_size"))
        rows.append(
            html.Tr(
                [
                    html.Td(
                        html.Button(
                            _fmt(option.get("strike")),
                            id={"type": "lab-contract", "symbol": name},
                            n_clicks=0,
                            className="lab-contract-button",
                            title=name,
                            **_aria(label=f"Select {name}", pressed=name == symbol),
                        )
                    ),
                    html.Td(str(option.get("option_type", "—")).title()),
                    *[
                        html.Td(
                            value,
                            title="Raw recorded premium; excluded quote"
                            if flags and mode == "quotes"
                            else "",
                        )
                        for value in values
                    ],
                    html.Td(_fmt(detail.get("age_seconds"))),
                    html.Td(_quality(detail), className="lab-quality-cell"),
                ],
                className="lab-selected" if name == symbol else "",
            )
        )
    note = (
        "Forward-coordinate partials; frozen fitted volatility. "
        "Full units and assumptions in selected contract."
        if mode == "greeks"
        else (
            "Quote-implied IV; excluded / failed solves withheld (—). "
            "Select a strike to link views."
            if mode == "iv"
            else "Mid = (bid + ask) / 2; spread = ask - bid. Flagged premiums are raw "
            "recorded evidence, not usable IV inputs or executable fills. "
            "— means unavailable."
        )
    )
    return html.Div(
        [
            html.P(units, className="lab-chain-units lab-muted"),
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            html.Tr([html.Th(label, scope="col") for label in headers])
                        ),
                        html.Tbody(rows),
                    ],
                    className="lab-chain-table",
                ),
                className="lab-table-wrap",
                tabIndex=0,
                **_aria(label="Scrollable option chain"),
            ),
            html.P(note, className="lab-muted"),
        ],
        className=f"lab-chain-content lab-chain-content--{mode}",
    )


def context_strip(view: ReplayView, symbol: str | None) -> html.Div:
    """Distinguish recorded inputs, quote-implied quantities and assumptions."""
    context = market_context(view, symbol)
    synthetic = "synthetic" in str(view.report.get("source", "")).lower()
    fields = (
        ("Spot · missing", "spot", False, "USDC/BTC · no spot source"),
        (
            "Forward · synthetic" if synthetic else "Forward · observed",
            "forward",
            False,
            "USDC/BTC · synthetic futures proxy"
            if synthetic
            else "USDC/BTC · observed futures midpoint proxy",
        ),
        (
            "ATM strike · quoted",
            "atm_strike",
            False,
            "USDC/BTC · nearest usable quoted strike",
        ),
        (
            "ATM IV · quote-implied",
            "atm_iv",
            True,
            "Black-76 midpoint IV, not observed volatility",
        ),
        ("DTE · derived", "dte", False, "Days · exact expiry minus checkpoint"),
        ("Rate · assumed", "rate", True, "Annual · report rate input"),
        ("Discount factor · modeled", "discount_factor", False, "Discount input"),
    )
    metrics = []
    for label, key, percent, basis in fields:
        value = context.get(key)
        number = _number(value)
        display = (
            f"{number:.2f}"
            if key == "dte" and number is not None
            else _fmt(value, percent=percent)
        )
        metrics.append(
            html.Div(
                [
                    html.Span(label, className="lab-muted"),
                    html.Strong(display),
                ],
                className="lab-metric",
                title=basis if number is not None else f"{basis} · unavailable",
                **_aria(
                    label=f"{label}: {display}. {basis}"
                    + ("" if number is not None else " · unavailable")
                ),
            )
        )
    return html.Div(
        [
            *metrics,
            html.Div(
                [
                    html.Span("Selected contract", className="lab-muted"),
                    html.Strong(context.get("selected_symbol") or "None"),
                ],
                className="lab-metric",
            ),
        ],
        className="lab-context-strip",
    )


def contract_panel(view: ReplayView, symbol: str | None) -> html.Details:
    """Expose the selected contract's raw quote and explicit risk conventions."""
    if symbol is None:
        return html.Details(
            [
                html.Summary("Selected contract"),
                html.P(
                    "Select a strike or an observed smile point.", className="lab-empty"
                ),
            ],
            className="lab-contract-details",
        )
    detail = contract_detail(view, symbol)
    quote = detail.get("normalized_quote") or {}
    bid, midpoint, ask, spread = _premiums(quote)
    source_fields = [
        ("Side", str(detail.get("option_type", "Unavailable")).title()),
        ("Strike · USDC/BTC", _fmt(detail.get("strike"))),
        ("Exact expiry · UTC", str(detail.get("expiry", "Unavailable"))),
        ("Bid premium · USDC/BTC", _fmt(bid)),
        ("Mid premium · USDC/BTC", _fmt(midpoint)),
        ("Ask premium · USDC/BTC", _fmt(ask)),
        ("Spread · USDC/BTC", _fmt(spread)),
        ("Bid size · contracts", _fmt(quote.get("bid_size"))),
        ("Ask size · contracts", _fmt(quote.get("ask_size"))),
        ("Exchange quote time · UTC", str(quote.get("quote_time") or "Unavailable")),
        (
            "Receipt / available at · UTC",
            str(quote.get("available_at") or "Unavailable"),
        ),
        ("Quote age · seconds", _fmt(detail.get("age_seconds"))),
        ("Bid IV", _fmt(_iv(detail, "bid"), percent=True)),
        ("Midpoint IV", _fmt(_iv(detail, "midpoint"), percent=True)),
        ("Ask IV", _fmt(_iv(detail, "ask"), percent=True)),
    ]
    greeks = detail.get("greeks") or {}
    units = detail.get("risk_units") or {}
    risk_rows = [
        html.Tr(
            [
                html.Th(key.title(), scope="row"),
                html.Td(_fmt(greeks.get(key))),
                html.Td(str(units.get(key, "Unavailable"))),
            ]
        )
        for key in _GREEKS
    ]
    return html.Details(
        [
            html.Summary(f"Selected contract · {symbol}"),
            _quality(detail),
            html.Div(
                [
                    html.Div(
                        [html.Span(label, className="lab-muted"), html.Strong(value)]
                    )
                    for label, value in source_fields
                ],
                className="lab-detail-grid",
            ),
            html.P(
                "Premiums are normalized recorded BBO in USDC per BTC; sizes are "
                "contracts. Mid premium = (bid + ask) / 2; spread = ask - bid. "
                "Midpoint IV solves the midpoint premium, not the mean of bid/ask IV. "
                "Flagged raw premiums remain visible; IV is withheld for excluded "
                "quotes or failed solves. No executable fill is assumed.",
                className="lab-muted",
            ),
            html.Div(
                [
                    html.P(
                        f"Model value · USDC/BTC: {_fmt(detail.get('model_value'))}"
                    ),
                    html.P(f"Model IV: {_fmt(detail.get('model_iv'), percent=True)}"),
                    html.P(f"Model basis: {detail.get('model_basis', 'Unavailable')}"),
                    html.P(f"Risk basis: {detail.get('risk_basis', 'Unavailable')}"),
                    html.P(
                        "Risk availability: "
                        + str(
                            detail.get("risk_unavailable_reason")
                            or "Available as shown"
                        )
                    ),
                ],
                className="lab-detail-grid",
            ),
            html.Ul(
                [
                    html.Li(str(assumption))
                    for assumption in detail.get("risk_assumptions", [])
                ]
            ),
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [html.Th("Risk"), html.Th("Value"), html.Th("Unit")]
                            )
                        ),
                        html.Tbody(risk_rows),
                    ],
                    className="lab-chain-table",
                ),
                className="lab-table-wrap",
            ),
            html.P("— means withheld/unavailable, not zero.", className="lab-muted"),
            html.Details(
                [
                    html.Summary("Raw JSON · quote, solver evidence and assumptions"),
                    html.Pre(_json(detail)),
                ]
            ),
        ],
        className="lab-contract-details",
    )


def _figure(theme: str, *, residual: bool = False) -> go.Figure:
    light = theme == "light"
    fg, bg, line = (
        ("#292a24", "#f5f3e9", "#d0cebe")
        if light
        else ("#f3efdc", "#48483a", "#666653")
    )
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white" if light else "plotly_dark",
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font={"color": fg, "size": 11},
        height=190 if residual else 350,
        autosize=True,
        margin={
            "l": 62,
            "r": 16,
            "t": 36 if not residual else 20,
            "b": 48 if residual else 42,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font": {"size": 10},
            "tracegroupgap": 0,
        },
        hovermode="closest",
        dragmode="zoom",
    )
    figure.update_xaxes(title_text="Strike · USDC/BTC", gridcolor=line, zeroline=False)
    figure.update_yaxes(
        title_text="Total variance residual" if residual else "Implied volatility",
        gridcolor=line,
        zeroline=residual,
        tickformat=".2g" if residual else ".2%",
        nticks=5,
    )
    return figure


def _stored_surface(fit: Row) -> VolatilitySurface:
    parameters = SVIParameters(**fit["parameters"])
    knot = SVISlice(
        expiry=datetime.fromisoformat(str(fit["expiry"]).replace("Z", "+00:00")),
        time_to_expiry=float(fit["time_to_expiry"]),
        parameters=parameters,
        domain=(float(fit["domain"][0]), float(fit["domain"][1])),
    )
    return VolatilitySurface((knot,))


def smile_figure(
    view: ReplayView, side: str, symbol: str | None, theme: str
) -> go.Figure:
    """Observed endpoints with adjacent-strike envelopes and a stored fit."""
    figure = _figure(theme)
    options = [
        contract_detail(view, str(row["symbol"])) for row in visible_options(view, side)
    ]
    strikes = sorted({float(row["strike"]) for row in options})
    strike_index = {strike: index for index, strike in enumerate(strikes)}
    amber = "#b86600" if theme == "light" else "#ffa51a"
    blue = "#007bb5" if theme == "light" else "#1aabf4"
    has_observation = False
    band_legend = False
    series = (
        [options]
        if side == "otm"
        else [
            [row for row in options if row.get("option_type") == kind]
            for kind in ("put", "call")
        ]
    )
    for rows in series:
        intervals = []
        for row in sorted(rows, key=lambda row: float(row["strike"])):
            bid, midpoint, ask = (_iv(row, key) for key in ("bid", "midpoint", "ask"))
            usable = (
                bid is not None
                and midpoint is not None
                and ask is not None
                and ask >= bid
            )
            intervals.append(
                (float(row["strike"]), str(row["symbol"]), bid, ask, usable)
            )
        band_x: list[float | None] = []
        band_y: list[float | None] = []
        connected: set[str] = set()
        for left, right in pairwise(intervals):
            lx, ls, lb, la, left_usable = left
            rx, rs, rb, ra, right_usable = right
            if not (
                left_usable
                and right_usable
                and strike_index[rx] == strike_index[lx] + 1
            ):
                continue
            # Edges are a visual envelope; only endpoints are observations.
            band_x.extend([lx, lx, rx, rx, lx, None])
            band_y.extend([lb, la, ra, rb, lb, None])
            connected.update((ls, rs))
        if band_x:
            figure.add_trace(
                go.Scatter(
                    x=band_x,
                    y=band_y,
                    name="Bid-ask IV visual envelope",
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(184,102,0,0.24)"
                    if theme == "light"
                    else "rgba(255,165,26,0.24)",
                    line={"color": amber, "width": 0.5},
                    connectgaps=False,
                    legendgroup="band",
                    showlegend=not band_legend,
                    hoverinfo="skip",
                )
            )
            band_legend = True
        range_x: list[float | None] = []
        range_y: list[float | None] = []
        for strike, name, bid, ask, usable in intervals:
            if usable and name not in connected:
                range_x.extend([strike, strike, None])
                range_y.extend([bid, ask, None])
        if range_x:
            figure.add_trace(
                go.Scatter(
                    x=range_x,
                    y=range_y,
                    name="Isolated bid-ask IV range",
                    mode="lines",
                    line={"color": amber, "width": 1},
                    connectgaps=False,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    for kind in ("put", "call"):
        rows = [row for row in options if row.get("option_type") == kind]
        if not rows:
            continue
        for quote_side in ("bid", "ask", "midpoint"):
            values = [_iv(row, quote_side) for row in rows]
            has_observation = has_observation or any(v is not None for v in values)
            midpoint_marker = quote_side == "midpoint"
            figure.add_trace(
                go.Scatter(
                    x=[row["strike"] for row in rows],
                    y=values,
                    customdata=[str(row["symbol"]) for row in rows],
                    name=f"{kind.title()} midpoint"
                    if midpoint_marker
                    else f"{kind.title()} {quote_side}",
                    mode="markers",
                    connectgaps=False,
                    showlegend=midpoint_marker,
                    legendgroup=kind,
                    marker={
                        "color": amber,
                        "symbol": ("circle" if kind == "put" else "triangle-up")
                        if midpoint_marker
                        else "line-ew",
                        "size": [
                            11
                            if row["symbol"] == symbol
                            else (6 if midpoint_marker else 4)
                            for row in rows
                        ],
                        "line": {"width": 1.5, "color": amber},
                    },
                    hovertemplate=(
                        "%{customdata}<br>Strike %{x:,.0f} USDC/BTC"
                        f"<br>{quote_side.title()} IV %{{y:.2%}}<extra></extra>"
                    ),
                )
            )
    fit = view.expiry.get("fit")
    forward = _number(view.expiry.get("forward_input"))
    fit_note = ""
    if fit and forward is not None and forward > 0:
        try:
            surface = _stored_surface(fit)
            grid = np.linspace(*surface.domain, 101)
            maturity = surface.maturity_domain[0]
            figure.add_trace(
                go.Scatter(
                    x=[forward * exp(float(k)) for k in grid],
                    y=[surface.volatility(float(k), maturity) for k in grid],
                    name="Stored SVI fit",
                    mode="lines",
                    line={"color": blue, "width": 2},
                    hovertemplate=(
                        "Stored SVI<br>Strike %{x:,.0f}<br>IV %{y:.2%}<extra></extra>"
                    ),
                )
            )
        except (InputValidationError, KeyError, TypeError, ValueError, OverflowError):
            fit_note = "Stored fit unavailable: invalid parameters or domain."
    else:
        fit_note = f"No stored fit: {view.expiry.get('fit_error') or 'unavailable'}."
    selected = next((row for row in options if row["symbol"] == symbol), None)
    if selected is not None:
        figure.add_vline(
            x=selected["strike"],
            line_color=amber,
            line_dash="dot",
            line_width=1,
        )
    notes = []
    if not has_observation:
        notes.append("No usable bid/ask IV observations for this selection.")
    if fit_note:
        notes.append(fit_note)
    if notes:
        figure.add_annotation(
            text="<br>".join(notes),
            x=0.5,
            y=0.97,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 11},
            bgcolor=figure.layout.paper_bgcolor,
        )
    figure.update_layout(
        uirevision=f"{view.session}:{view.expiry.get('expiry')}:{side}:{theme}",
    )
    return figure


def fit_panel(view: ReplayView, theme: str) -> html.Details:
    """Keep optimizer convergence separate from sampled arbitrage diagnostics."""
    fit = view.expiry.get("fit")
    residuals = (fit or {}).get("residuals", [])
    options = [
        contract_detail(view, str(row["symbol"]))
        for row in view.expiry.get("options", [])
    ]
    used_symbols = {str(row["contract"]) for row in residuals}
    rejected = sum(
        bool(_flags(row))
        or any(data.get("error") for data in (row.get("sides") or {}).values())
        or _iv(row, "midpoint") is None
        for row in options
    )
    unused = sum(str(row["symbol"]) not in used_symbols for row in options)
    coverage = html.P(
        f"Used in fit: {len(used_symbols)} · "
        f"Rejected / IV unavailable: {rejected} · Not fitted: {unused} "
        "(includes usable non-OTM contracts)"
    )
    if not fit:
        return html.Details(
            [
                html.Summary("SVI fit · unavailable"),
                html.P(str(view.expiry.get("fit_error") or "No recorded fit.")),
                coverage,
            ],
            className="lab-fit-details",
        )
    convergence = "Converged" if fit.get("converged") else "Not converged"
    arbitrage = fit.get("arbitrage") or {}
    passed = arbitrage.get("passed")
    arb_label = (
        "Sampled arbitrage checks passed"
        if passed is True
        else "Sampled arbitrage findings"
        if passed is False
        else "Arbitrage checks unavailable"
    )
    strikes = {row["symbol"]: row["strike"] for row in view.expiry.get("options", [])}
    figure = _figure(theme, residual=True)
    figure.add_trace(
        go.Bar(
            x=[strikes.get(row["contract"]) for row in residuals],
            y=[FitResidual(**row).residual for row in residuals],
            text=[row["contract"] for row in residuals],
            marker_color="#007bb5" if theme == "light" else "#1aabf4",
            name="Fitted - observed variance",
            hovertemplate=(
                "%{text}<br>Strike %{x:,.0f}<br>Residual %{y:.6g}<extra></extra>"
            ),
        )
    )
    if not residuals:
        figure.add_annotation(
            text="No recorded residuals.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    warnings = []
    if not fit.get("converged"):
        warnings.append(
            "Optimizer did not converge; the stored curve is not a validated fit."
        )
    if view.expiry.get("fit_error"):
        warnings.append(str(view.expiry["fit_error"]))
    for finding in arbitrage.get("findings", []):
        warnings.append(
            "; ".join(
                f"{str(key).replace('_', ' ')}: {value}"
                for key, value in finding.items()
            )
            if isinstance(finding, dict)
            else str(finding)
        )
    return html.Details(
        [
            html.Summary(f"SVI fit · {convergence} · {arb_label}"),
            coverage,
            html.Ul([html.Li(warning) for warning in warnings])
            if warnings
            else html.P("No recorded fit warnings.", className="lab-muted"),
            html.P(
                f"Weighted RMSE (total variance): {_fmt(fit.get('weighted_rmse'))} · "
                f"Evaluations: {fit.get('evaluations', '—')} · "
                f"Termination: {fit.get('termination', 'Unavailable')}"
            ),
            html.P(
                "Stored fit, never recalibrated here; "
                "no extrapolation beyond its domain. "
                "Sampled checks are not a continuous arbitrage certificate.",
                className="lab-muted",
            ),
            dcc.Graph(
                figure=figure,
                config=_GRAPH_CONFIG,
                className="lab-fit-graph",
                style={"height": "190px", "width": "100%", "minWidth": 0},
            ),
            html.Details(
                [
                    html.Summary(
                        "Parameters, weights, residuals and arbitrage evidence"
                    ),
                    html.Pre(_json(fit)),
                ]
            ),
        ],
        className="lab-fit-details",
    )
