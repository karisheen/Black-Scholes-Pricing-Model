"""Dash user interface built on the deterministic pricing library."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
from numpy.typing import NDArray

from quant_pricing.black_scholes import (
    OptionType,
    Valuation,
    price,
    value_and_greeks,
)
from quant_pricing.exceptions import PricingError
from quant_pricing.implied_volatility import ImpliedVolatilityResult, implied_volatility

FloatArray = NDArray[np.float64]
PricingMode = Literal["price", "iv"]


class _SubmittedScenario(TypedDict):
    mode: PricingMode
    option_type: str
    spot: float | None
    strike: float | None
    time_to_expiry_days: float | None
    rate_percent: float | None
    volatility_percent: float | None
    dividend_yield_percent: float | None
    premium: float | None


@dataclass(frozen=True, slots=True)
class ScenarioAnalysis:
    """A scalar valuation and vectorized sensitivity grids for the dashboard."""

    option_type: OptionType
    spot: float
    strike: float
    time_to_expiry: float
    rate: float
    volatility: float
    dividend_yield: float
    valuation: Valuation
    spot_grid: FloatArray
    call_by_spot: FloatArray
    put_by_spot: FloatArray
    volatility_grid: FloatArray
    value_by_volatility: FloatArray


def _array(value: float | FloatArray) -> FloatArray:
    if not isinstance(value, np.ndarray):  # pragma: no cover - internal guard
        raise TypeError("vectorized dashboard calculation produced a scalar")
    return value


def _scalar(value: float | FloatArray) -> float:
    if not isinstance(value, float):  # pragma: no cover - internal guard
        raise TypeError("scalar dashboard calculation produced an array")
    return value


def calculate_scenario(
    option_type: OptionType | str,
    spot: float,
    strike: float,
    time_to_expiry_days: float,
    rate_percent: float,
    volatility_percent: float,
    dividend_yield_percent: float = 0.0,
) -> ScenarioAnalysis:
    """Calculate a dashboard valuation from human-readable units."""

    kind = OptionType.parse(option_type)
    time_to_expiry = float(time_to_expiry_days) / 365.0
    rate = float(rate_percent) / 100.0
    volatility = float(volatility_percent) / 100.0
    dividend_yield = float(dividend_yield_percent) / 100.0
    valuation = value_and_greeks(
        kind,
        spot,
        strike,
        time_to_expiry,
        rate,
        volatility,
        dividend_yield,
    )

    spot_grid = np.asarray(
        np.linspace(0.5 * float(spot), 1.5 * float(spot), 161),
        dtype=np.float64,
    )
    call_by_spot = _array(
        price(
            OptionType.CALL,
            spot_grid,
            strike,
            time_to_expiry,
            rate,
            volatility,
            dividend_yield,
        )
    )
    put_by_spot = _array(
        price(
            OptionType.PUT,
            spot_grid,
            strike,
            time_to_expiry,
            rate,
            volatility,
            dividend_yield,
        )
    )
    volatility_grid = np.asarray(
        np.linspace(0.0, max(2.0 * volatility, 0.5), 161),
        dtype=np.float64,
    )
    value_by_volatility = _array(
        price(
            kind,
            spot,
            strike,
            time_to_expiry,
            rate,
            volatility_grid,
            dividend_yield,
        )
    )
    return ScenarioAnalysis(
        option_type=kind,
        spot=float(spot),
        strike=float(strike),
        time_to_expiry=time_to_expiry,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        valuation=valuation,
        spot_grid=spot_grid,
        call_by_spot=call_by_spot,
        put_by_spot=put_by_spot,
        volatility_grid=volatility_grid,
        value_by_volatility=value_by_volatility,
    )


_COLORS = {
    "ink": "var(--dashboard-ink)",
    "muted": "var(--dashboard-muted)",
    "panel": "var(--dashboard-panel)",
    "canvas": "var(--dashboard-canvas)",
    "border": "var(--dashboard-border)",
    "input": "var(--dashboard-input)",
    "accent": "var(--dashboard-accent)",
    "danger": "var(--dashboard-danger)",
    "button_text": "var(--dashboard-button-text)",
}
_PLOT_COLORS = {
    "call": "#4c9aff",
    "put": "#ef5350",
    "accent": "#26a69a",
    "marker": "#7b879c",
}


def theme_class_name(theme: str | None) -> str:
    """Return the application class for a supported color theme."""

    normalized = "dark" if theme == "dark" else "light"
    return f"dashboard-shell theme-{normalized}"


def _metric_card(label: str, value: str, note: str) -> html.Div:
    return html.Div(
        [
            html.P(label, style={"margin": "0", "color": _COLORS["muted"]}),
            html.Strong(
                value,
                style={"fontSize": "1.35rem", "color": _COLORS["ink"]},
            ),
            html.P(
                note,
                style={
                    "margin": "0.25rem 0 0",
                    "fontSize": "0.75rem",
                    "color": _COLORS["muted"],
                },
            ),
        ],
        style={
            "padding": "1rem",
            "background": _COLORS["panel"],
            "border": f"1px solid {_COLORS['border']}",
            "borderRadius": "0.5rem",
        },
    )


_METRIC_GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(140px, 1fr))",
    "gap": "0.75rem",
}


def _render_iv(option_type: str, solution: ImpliedVolatilityResult) -> html.Div:
    return html.Div(
        [
            html.H2(
                f"{option_type.capitalize()} implied volatility: "
                f"{solution.volatility:.4%}",
                style={"color": _COLORS["ink"]},
            ),
            html.Div(
                [
                    _metric_card(
                        "Input premium", f"${solution.target_price:.10g}", "per unit"
                    ),
                    _metric_card(
                        "Model premium", f"${solution.model_price:.10g}", "per unit"
                    ),
                    _metric_card(
                        "Residual", f"{solution.residual:+.3e}", "model - input"
                    ),
                    _metric_card(
                        "Iterations",
                        str(solution.iterations),
                        solution.termination_reason.replace("_", " "),
                    ),
                ],
                style=_METRIC_GRID_STYLE,
            ),
            html.P(
                "Annualized volatility implied by the supplied premium under "
                "European exercise and continuous dividend yield.",
                style={"color": _COLORS["muted"], "fontSize": "0.85rem"},
            ),
        ]
    )


def _sensitivity_figures(
    analysis: ScenarioAnalysis, theme: str
) -> tuple[go.Figure, go.Figure]:
    template = "plotly_dark" if theme == "dark" else "plotly_white"
    spot_figure = go.Figure()
    spot_figure.add_scatter(
        x=analysis.spot_grid,
        y=analysis.call_by_spot,
        mode="lines",
        name="Call",
        line={"color": _PLOT_COLORS["call"]},
    )
    spot_figure.add_scatter(
        x=analysis.spot_grid,
        y=analysis.put_by_spot,
        mode="lines",
        name="Put",
        line={"color": _PLOT_COLORS["put"]},
    )
    spot_figure.add_vline(
        x=analysis.spot,
        line_dash="dot",
        line_color=_PLOT_COLORS["marker"],
    )
    spot_figure.update_layout(
        title="Value by underlying price",
        xaxis_title="Underlying price",
        yaxis_title="Option value",
        template=template,
        margin={"l": 50, "r": 20, "t": 55, "b": 45},
        legend={"orientation": "h"},
    )

    volatility_figure = go.Figure()
    volatility_figure.add_scatter(
        x=100.0 * analysis.volatility_grid,
        y=analysis.value_by_volatility,
        mode="lines",
        name=analysis.option_type.value.capitalize(),
        line={"color": _PLOT_COLORS["accent"]},
    )
    volatility_figure.add_vline(
        x=100.0 * analysis.volatility,
        line_dash="dot",
        line_color=_PLOT_COLORS["marker"],
    )
    volatility_figure.update_layout(
        title=f"{analysis.option_type.value.capitalize()} value by volatility",
        xaxis_title="Annualized volatility (%)",
        yaxis_title="Option value",
        template=template,
        margin={"l": 50, "r": 20, "t": 55, "b": 45},
        showlegend=False,
    )
    return spot_figure, volatility_figure


def render_analysis(
    option_type: str,
    spot: float | None,
    strike: float | None,
    time_to_expiry_days: float | None,
    rate_percent: float | None,
    volatility_percent: float | None,
    dividend_yield_percent: float | None,
    theme: str = "light",
    *,
    mode: PricingMode = "price",
    premium: float | None = None,
) -> html.Div:
    """Render validated dashboard output for a callback or integration test."""

    required = {
        "underlying price": spot,
        "strike price": strike,
        "time to expiry": time_to_expiry_days,
        "risk-free rate": rate_percent,
        "option premium" if mode == "iv" else "volatility": (
            premium if mode == "iv" else volatility_percent
        ),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        return html.Div(
            f"Missing required inputs: {', '.join(missing)}.",
            role="alert",
            style={"color": _COLORS["danger"]},
        )
    assert spot is not None
    assert strike is not None
    assert time_to_expiry_days is not None
    assert rate_percent is not None

    try:
        if mode == "iv":
            assert premium is not None
            solution = implied_volatility(
                option_type,
                premium,
                spot,
                strike,
                time_to_expiry_days / 365.0,
                rate_percent / 100.0,
                (dividend_yield_percent or 0.0) / 100.0,
            )
            return _render_iv(option_type, solution)

        assert volatility_percent is not None
        analysis = calculate_scenario(
            option_type,
            spot,
            strike,
            time_to_expiry_days,
            rate_percent,
            volatility_percent,
            dividend_yield_percent or 0.0,
        )
    except (PricingError, TypeError, ValueError) as error:
        return html.Div(
            str(error),
            role="alert",
            style={"color": _COLORS["danger"]},
        )

    risk = analysis.valuation.greeks
    option_value = _scalar(analysis.valuation.price)
    delta = _scalar(risk.delta)
    gamma = _scalar(risk.gamma)
    vega = _scalar(risk.vega)
    theta = _scalar(risk.theta)
    rho = _scalar(risk.rho)
    normalized_theme = "dark" if theme == "dark" else "light"
    spot_figure, volatility_figure = _sensitivity_figures(analysis, normalized_theme)

    return html.Div(
        [
            html.H2(
                f"{analysis.option_type.value.capitalize()} value: "
                f"${option_value:,.4f}",
                style={"color": _COLORS["ink"]},
            ),
            html.Div(
                [
                    _metric_card("Delta", f"{delta:,.6f}", "per $1 spot move"),
                    _metric_card("Gamma", f"{gamma:,.6f}", "Delta change per $1"),
                    _metric_card("Vega", f"{vega / 100.0:,.6f}", "per 1 vol point"),
                    _metric_card(
                        "Theta",
                        f"{theta / 365.0:,.6f}",
                        "per calendar day",
                    ),
                    _metric_card("Rho", f"{rho / 100.0:,.6f}", "per 1 rate point"),
                ],
                style=_METRIC_GRID_STYLE,
            ),
            html.Div(
                [
                    dcc.Graph(figure=spot_figure),
                    dcc.Graph(figure=volatility_figure),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                    "gap": "1rem",
                    "marginTop": "1rem",
                },
            ),
            html.P(
                "European exercise · continuous compounding · continuous dividend "
                "yield · no transaction costs",
                style={"color": _COLORS["muted"], "fontSize": "0.85rem"},
            ),
        ]
    )


def _number_input(
    identifier: str,
    label: str,
    value: float | None,
    *,
    minimum: float | None = None,
    hidden: bool = False,
) -> html.Div:
    return html.Div(
        [
            html.Label(
                label,
                htmlFor=identifier,
                style={"fontWeight": "600", "color": _COLORS["ink"]},
            ),
            dcc.Input(
                id=identifier,
                type="number",
                value=value,
                min=minimum,
                debounce=True,
                style={
                    "width": "100%",
                    "padding": "0.65rem",
                    "boxSizing": "border-box",
                    "border": f"1px solid {_COLORS['border']}",
                    "borderRadius": "0.35rem",
                    "background": _COLORS["input"],
                    "color": _COLORS["ink"],
                },
            ),
        ],
        id=f"{identifier}-field",
        hidden=hidden,
        style={"marginBottom": "0.85rem"},
    )


def create_app() -> Dash:
    """Create the Dash application without starting a server."""

    assets_folder = Path(__file__).with_name("assets")
    app = Dash(__name__, assets_folder=str(assets_folder))
    app.title = "Black-Scholes-Merton Analytics"
    app.layout = html.Div(
        id="app-shell",
        className=theme_class_name("light"),
        children=[
            dcc.Store(id="submitted-scenario"),
            html.Main(
                [
                    html.Header(
                        [
                            html.Div(
                                [
                                    html.H1(
                                        "Black-Scholes-Merton Analytics",
                                        style={"marginBottom": "0.35rem"},
                                    ),
                                    html.P(
                                        "Deterministic European option valuation "
                                        "and analytic risk.",
                                        style={
                                            "marginTop": "0",
                                            "color": _COLORS["muted"],
                                        },
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Appearance",
                                        htmlFor="theme-mode",
                                        style={
                                            "fontWeight": "600",
                                            "fontSize": "0.85rem",
                                        },
                                    ),
                                    dcc.RadioItems(
                                        id="theme-mode",
                                        options=[
                                            {"label": "Light", "value": "light"},
                                            {"label": "Dark", "value": "dark"},
                                        ],
                                        value="light",
                                        inline=True,
                                        persistence=True,
                                        persistence_type="local",
                                        style={"marginTop": "0.4rem"},
                                        labelStyle={"color": _COLORS["ink"]},
                                    ),
                                ],
                                style={"minWidth": "150px"},
                            ),
                        ],
                        className="dashboard-header",
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "space-between",
                            "gap": "1rem",
                        },
                    ),
                    html.Div(
                        [
                            html.Section(
                                [
                                    html.H2("Model inputs"),
                                    html.Label(
                                        "Calculation",
                                        htmlFor="pricing-mode",
                                        style={"fontWeight": "600"},
                                    ),
                                    dcc.RadioItems(
                                        id="pricing-mode",
                                        options=[
                                            {
                                                "label": "Price from volatility",
                                                "value": "price",
                                            },
                                            {
                                                "label": "IV from option premium",
                                                "value": "iv",
                                            },
                                        ],
                                        value="price",
                                        style={"margin": "0.5rem 0 1rem"},
                                        labelStyle={"color": _COLORS["ink"]},
                                    ),
                                    html.Label(
                                        "Option type",
                                        htmlFor="option-type",
                                        style={"fontWeight": "600"},
                                    ),
                                    dcc.RadioItems(
                                        id="option-type",
                                        options=[
                                            {"label": "Call", "value": "call"},
                                            {"label": "Put", "value": "put"},
                                        ],
                                        value="call",
                                        inline=True,
                                        style={"margin": "0.5rem 0 1rem"},
                                        labelStyle={"color": _COLORS["ink"]},
                                    ),
                                    _number_input(
                                        "spot",
                                        "Underlying price",
                                        100.0,
                                        minimum=0.0,
                                    ),
                                    _number_input(
                                        "strike",
                                        "Strike price",
                                        100.0,
                                        minimum=0.0,
                                    ),
                                    _number_input(
                                        "time-days",
                                        "Time to expiry (days)",
                                        365.0,
                                        minimum=0.0,
                                    ),
                                    _number_input("rate", "Risk-free rate (%)", 5.0),
                                    _number_input(
                                        "volatility",
                                        "Annualized volatility (%)",
                                        20.0,
                                        minimum=0.0,
                                    ),
                                    _number_input(
                                        "premium",
                                        "Option premium",
                                        None,
                                        minimum=0.0,
                                        hidden=True,
                                    ),
                                    _number_input(
                                        "dividend-yield",
                                        "Dividend yield (%)",
                                        0.0,
                                    ),
                                    html.Button(
                                        "Calculate",
                                        id="calculate",
                                        n_clicks=0,
                                        style={
                                            "width": "100%",
                                            "padding": "0.75rem",
                                            "border": "none",
                                            "borderRadius": "0.35rem",
                                            "background": _COLORS["accent"],
                                            "color": _COLORS["button_text"],
                                            "fontWeight": "700",
                                            "cursor": "pointer",
                                        },
                                    ),
                                ],
                                style={
                                    "background": _COLORS["panel"],
                                    "padding": "1.25rem",
                                    "border": f"1px solid {_COLORS['border']}",
                                    "borderRadius": "0.65rem",
                                },
                            ),
                            html.Section(
                                id="results",
                                **{"aria-live": "polite"},  # type: ignore[arg-type]
                            ),
                        ],
                        className="dashboard-content-grid",
                        style={
                            "display": "grid",
                            "gridTemplateColumns": (
                                "minmax(240px, 300px) minmax(0, 1fr)"
                            ),
                            "gap": "1.5rem",
                            "alignItems": "start",
                        },
                    ),
                ],
                style={
                    "maxWidth": "1280px",
                    "margin": "0 auto",
                    "padding": "1.5rem",
                    "fontFamily": "Inter, system-ui, sans-serif",
                    "color": _COLORS["ink"],
                },
            ),
        ],
    )

    @app.callback(
        Output("app-shell", "className"),
        Input("theme-mode", "value"),
    )
    def update_theme(theme: str | None) -> str:
        return theme_class_name(theme)

    @app.callback(
        Output("volatility-field", "hidden"),
        Output("premium-field", "hidden"),
        Input("pricing-mode", "value"),
    )
    def update_fields(mode: PricingMode) -> tuple[bool, bool]:
        return mode == "iv", mode != "iv"

    @app.callback(
        Output("submitted-scenario", "data"),
        Input("calculate", "n_clicks"),
        State("pricing-mode", "value"),
        State("option-type", "value"),
        State("spot", "value"),
        State("strike", "value"),
        State("time-days", "value"),
        State("rate", "value"),
        State("volatility", "value"),
        State("dividend-yield", "value"),
        State("premium", "value"),
    )
    def submit_scenario(
        _clicks: int,
        mode: PricingMode,
        option_type: str,
        spot: float | None,
        strike: float | None,
        time_days: float | None,
        rate: float | None,
        volatility: float | None,
        dividend_yield: float | None,
        premium: float | None,
    ) -> _SubmittedScenario:
        return {
            "mode": mode,
            "option_type": option_type,
            "spot": spot,
            "strike": strike,
            "time_to_expiry_days": time_days,
            "rate_percent": rate,
            "volatility_percent": volatility,
            "dividend_yield_percent": dividend_yield,
            "premium": premium,
        }

    @app.callback(
        Output("results", "children"),
        Input("submitted-scenario", "data"),
        Input("theme-mode", "value"),
    )
    def update_results(submitted: _SubmittedScenario | None, theme: str) -> html.Div:
        if submitted is None:
            raise PreventUpdate
        return render_analysis(**submitted, theme=theme)

    return app


def main(**run_options: Any) -> None:
    """Run the local development server."""

    create_app().run(**run_options)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main(debug=True)
