"""Selected, auditable report diagnostics; no ingestion, fitting or trade claims."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from math import isclose
from typing import Any

from dash import html

from quant_pricing.butterfly import (
    ButterflyBook,
    ButterflyCostInputs,
    ButterflyFuturesHedge,
    ButterflyOptionLeg,
    butterfly_strike_weights,
    compare_butterfly_costs,
)
from quant_pricing.conventions import year_fraction
from quant_pricing.lab_data import ReplayView, Row

_LIMITS = {
    "diagnostics": "Stored report quality and coverage, not a live book.",
    "parity": "Intervals cross observed bid/ask sides; consistency is not tradability.",
    "midpoint": (
        "Raw midpoint anomalies are not tradable packages. Repeated strikes may use "
        "worst-case call/put observations; never compare this signal to quoted cost."
    ),
    "butterflies": (
        "Indicative BBO only: no fees, simultaneous fills, size-step, financing or "
        "execution guarantee. Displayed capacity is a minimum across required sides."
    ),
}


def _flags(row: Row) -> list[str]:
    native = row.get("native_book") or {}
    flags = [
        *row.get("normalized_flags", []),
        *row.get("rejected", []),
        *native.get("flags", []),
    ]
    if native.get("sequence_gaps"):
        flags.append("sequence_gap")
    return list(dict.fromkeys(flags))


def _inputs(row: Row) -> Row:
    return {
        "symbol": row.get("symbol"),
        "strike": row.get("strike"),
        "option_type": row.get("option_type"),
        "normalized_quote": row.get("normalized_quote"),
        "quality_flags": _flags(row),
        "native_book": row.get("native_book"),
    }


def _midpoint_findings(view: ReplayView) -> list[tuple[int, Row]]:
    report = view.checkpoint.get("observed_arbitrage") or {}
    start = datetime.fromisoformat(str(view.checkpoint["as_of"]).replace("Z", "+00:00"))
    end = datetime.fromisoformat(str(view.expiry["expiry"]).replace("Z", "+00:00"))
    if end <= start:
        return []
    maturity = year_fraction(start, end)
    return [
        (index, finding)
        for index, finding in enumerate(report.get("findings", []))
        if isclose(finding["time_to_expiry"], maturity, rel_tol=1e-10, abs_tol=1e-12)
    ]


def _finding_rows(view: ReplayView, finding: Row) -> list[Row]:
    rows = [
        row
        for row in view.expiry.get("options", [])
        if row.get("log_forward_moneyness") is not None
        and ((row.get("sides", {}).get("midpoint") or {}).get("iv") or 0) > 0
        and not row.get("rejected")
    ]
    nodes = sorted({round(row["log_forward_moneyness"], 12) for row in rows})
    target = round(finding["log_moneyness"], 12)
    if target not in nodes:
        return []
    index = nodes.index(target)
    kind = finding["kind"]
    relevant = {target}
    if kind == "butterfly" and index:
        relevant.add(nodes[index - 1])
    if kind in ("butterfly", "monotonicity") and index + 1 < len(nodes):
        relevant.add(nodes[index + 1])
    return [row for row in rows if round(row["log_forward_moneyness"], 12) in relevant]


def _items(view: ReplayView, tab: str) -> list[tuple[str, str, Row, list[Row]]]:
    rows = view.expiry.get("options", [])
    by_symbol = {row["symbol"]: row for row in rows}
    if tab == "diagnostics":
        return [("summary", "Coverage & fit warnings", {}, [])] + [
            (f"book:{row['symbol']}", row["symbol"], row, [row]) for row in rows
        ]
    if tab == "parity":
        return [
            (
                f"parity:{index}",
                f"{pair['call']} / {pair['put']}",
                pair,
                [
                    by_symbol[symbol]
                    for symbol in (pair["call"], pair["put"])
                    if symbol in by_symbol
                ],
            )
            for index, pair in enumerate(view.expiry.get("parity", []))
        ]
    if tab == "midpoint":
        return [
            (
                f"midpoint:{index}",
                f"{finding['kind']} · k={finding['log_moneyness']:.6g}",
                finding,
                _finding_rows(view, finding),
            )
            for index, finding in _midpoint_findings(view)
        ]
    if tab == "butterflies":
        result = []
        for index, triplet in enumerate(
            (view.expiry.get("quoted_butterflies") or {}).get("triplets", [])
        ):
            package = triplet.get("best_package") or {}
            symbols = {leg["symbol"] for leg in package.get("option_legs", [])}
            relevant = [row for row in rows if row["symbol"] in symbols]
            if not symbols:
                relevant = [row for row in rows if row["strike"] in triplet["strikes"]]
            label = " / ".join(f"{strike:g}" for strike in triplet["strikes"])
            result.append((f"butterfly:{index}", label, triplet, relevant))
        return result
    raise ValueError(f"Unknown diagnostic tab: {tab}")


def diagnostic_options(
    view: ReplayView, tab: str, symbol: str | None
) -> list[dict[str, str]]:
    """Stable item IDs, with items involving the chosen contract first."""
    items = _items(view, tab)
    items.sort(key=lambda item: not any(row["symbol"] == symbol for row in item[3]))
    return [{"label": label, "value": key} for key, label, _, _ in items]


def _comparison(view: ReplayView, raw: Row, rows: list[Row]) -> Row:
    future = view.expiry.get("matched_future") or {}
    books = {}
    for row in [*rows, future]:
        quote = row.get("normalized_quote")
        if quote:
            books[row["symbol"]] = ButterflyBook(
                quote["bid"], quote["ask"], tuple(_flags(row))
            )
    discount = view.expiry.get("discount_factor")
    if discount is None:
        return {
            "quoted_cost": raw["net_entry_cost"],
            "midpoint_cost": None,
            "crossing_cost": None,
            "flags": ["discount_factor_unavailable"],
        }
    hedge = raw.get("futures_hedge")
    package = ButterflyCostInputs(
        option_legs=tuple(ButterflyOptionLeg(**leg) for leg in raw["option_legs"]),
        futures_hedge=None if hedge is None else ButterflyFuturesHedge(**hedge),
        net_entry_cost=raw["net_entry_cost"],
    )
    return asdict(compare_butterfly_costs(package, books, discount))


def _quoted_cost_assessment(package: Row, candidate: bool | None) -> tuple[str, str]:
    """Keep quoted-cost sign separate from the stored candidate threshold."""
    quoted = package.get("net_entry_cost")
    if quoted is None:
        return (
            "no usable combination",
            "No usable quoted package; midpoint comparison and capacity "
            "are unavailable.",
        )
    if quoted >= 0:
        return (
            "No quoted credit",
            "The stored package requires a positive quoted entry cost; "
            "there is no quoted credit."
            if quoted > 0
            else "The stored package has zero quoted entry cost, not a quoted credit.",
        )
    tolerance = package.get("cost_tolerance")
    if (
        tolerance is None
        or candidate is None
        or package.get("negative_cost_candidate") is None
    ):
        return (
            "stored negative quoted cost · threshold unverified",
            "The stored quoted cost is negative, but candidate threshold "
            "metadata is incomplete. Whether it exceeds the cost tolerance "
            "cannot be established from this record. "
            "Any quoted credit is indicative, not an executable profit.",
        )
    if quoted >= -tolerance:
        return (
            "negative quoted cost within tolerance",
            "The stored quoted cost is slightly negative but within "
            "the cost tolerance "
            f"({tolerance:.6g}); it is not a negative-cost candidate. "
            "This numerical credit is indicative, not an executable profit.",
        )
    return (
        "indicative negative cost",
        "The stored package has an indicative quoted credit, not an executable profit. "
        f"Its negative quoted cost exceeds the cost tolerance ({tolerance:.6g}).",
    )


def _butterfly_data(view: ReplayView, selected: Row, rows: list[Row]) -> Row:
    data = dict(selected)
    report = view.expiry.get("quoted_butterflies") or {}
    low, middle, high = selected["strikes"]
    data["exact_weights"] = [
        str(weight) for weight in butterfly_strike_weights((low, middle, high))
    ]
    data["cost_unit"] = report.get("cost_unit", "USDC per BTC of center option")
    data["capacity_unit"] = "BTC of center option; options in contracts, futures in BTC"
    data["quote_basis"] = report.get("quote_basis")
    raw = selected.get("best_package")
    status, explanation = _quoted_cost_assessment(
        raw or {}, selected.get("negative_cost_candidate")
    )
    data["quoted_cost_status"] = status
    data["quoted_cost_explanation"] = explanation
    if raw is None:
        data["same_package_midpoint"] = None
        data["option_inputs"] = [_inputs(row) for row in rows]
        return data
    by_symbol = {row["symbol"]: row for row in rows}
    data["option_inputs"] = [
        {
            **_inputs(by_symbol.get(leg["symbol"], {"symbol": leg["symbol"]})),
            **leg,
            "crossed_side": "ask" if leg["side"] == "buy" else "bid",
        }
        for leg in raw["option_legs"]
    ]
    hedge = raw.get("futures_hedge")
    data["future_input"] = (
        None
        if hedge is None
        else {
            **_inputs(view.expiry.get("matched_future") or {}),
            **hedge,
            "crossed_side": "ask" if hedge["side"] == "buy" else "bid",
        }
    )
    data["same_package_midpoint"] = _comparison(view, raw, rows)
    return data


def diagnostic_payload(
    view: ReplayView, tab: str, symbol: str | None, item: str | None
) -> Row:
    """Export and display the same selected evidence, never unrelated findings."""
    items = _items(view, tab)
    selected_item = next((entry for entry in items if entry[0] == item), None)
    if selected_item is None and item is None:
        options = diagnostic_options(view, tab, symbol)
        first = options[0]["value"] if options else None
        selected_item = next((entry for entry in items if entry[0] == first), None)
    payload: Row = {
        "tab": tab,
        "item": item,
        "session": view.session,
        "as_of": view.checkpoint.get("as_of"),
        "expiry": view.expiry.get("expiry"),
        "source": view.report.get("source"),
        "symbols": [],
        "strikes": [],
        "limits": [_LIMITS[tab]],
        "status": "unavailable",
        "data": {},
    }
    if selected_item is None:
        payload["message"] = (
            "Selected diagnostic is not present at this expiry and checkpoint."
            if item is not None
            else {
                "parity": "No parity intervals recorded for this expiry/checkpoint.",
                "midpoint": "No recorded raw midpoint anomalies at this expiry/time. "
                "This is not proof of complete coverage or tradability.",
                "butterflies": "No quoted butterfly triplets recorded here; "
                "quotes or sufficient declared strikes may be unavailable.",
            }.get(tab, "No diagnostics recorded.")
        )
        return payload
    key, label, selected, rows = selected_item
    payload.update(
        item=key,
        label=label,
        status="recorded",
        symbols=list(dict.fromkeys(row["symbol"] for row in rows)),
        strikes=sorted({row["strike"] for row in rows}),
    )
    if tab == "diagnostics":
        if key == "summary":
            fit = view.expiry.get("fit") or {}
            butterflies = view.expiry.get("quoted_butterflies") or {}
            payload["data"] = {
                "coverage": view.expiry.get("coverage"),
                "parity_pairs_recorded": len(view.expiry.get("parity", [])),
                "raw_midpoint_findings_at_expiry": len(_midpoint_findings(view)),
                "quoted_butterfly_counts": {
                    "candidate_count": butterflies.get("candidate_count"),
                    "triplets_checked": butterflies.get("triplets_checked"),
                },
                "native_capture_complete": view.checkpoint.get(
                    "native_capture_complete"
                ),
                "iv_blocked": view.expiry.get("iv_blocked"),
                "fit_error": view.expiry.get("fit_error"),
                "fit_arbitrage": fit.get("arbitrage"),
                "joint_fit_error": view.checkpoint.get("joint_fit_error"),
                "joint_fitted_arbitrage": view.checkpoint.get("joint_fitted_arbitrage"),
                "joint_fitted_expiries": view.checkpoint.get("joint_fitted_expiries"),
                "assumptions": view.report.get("assumptions"),
                "matched_future": _inputs(view.expiry.get("matched_future") or {}),
            }
        else:
            payload["data"] = {**selected, "quality_flags": _flags(selected)}
            payload["status"] = "flagged" if _flags(selected) else "usable book"
    elif tab == "parity":
        payload["data"] = {
            **selected,
            "option_inputs": [_inputs(row) for row in rows],
            "future_input": _inputs(view.expiry.get("matched_future") or {}),
            "discount_factor": view.expiry.get("discount_factor"),
            "forward_input": view.expiry.get("forward_input"),
        }
        consistent = selected.get("consistent_with_future")
        payload["status"] = (
            "unavailable"
            if consistent is None
            else "consistent interval"
            if consistent
            else "inconsistent interval"
        )
    elif tab == "midpoint":
        observed = view.checkpoint.get("observed_arbitrage") or {}
        payload["data"] = {
            **selected,
            "claim": observed.get("claim"),
            "tolerance": observed.get("tolerance"),
            "option_inputs": [_inputs(row) for row in rows],
        }
        payload["status"] = "midpoint anomaly · not tradable"
        if selected["kind"] == "calendar":
            payload["limits"].append(
                "Calendar evidence compares shared k with a prior maturity. "
                "Stored findings do not identify that prior expiry or exact legs."
            )
    else:
        payload["data"] = _butterfly_data(view, selected, rows)
        payload["status"] = payload["data"]["quoted_cost_status"]
        comparison = payload["data"]["same_package_midpoint"]
        if comparison and comparison["flags"]:
            payload["status"] = "stored result · quote comparison unavailable"
    return payload


def _number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "Not available"
    return f"{value:,.{digits}f}" if isinstance(value, (int, float)) else str(value)


def _human(value: Any) -> str:
    return str(value).replace("_", " ")


def _interval(value: Any, unit: str) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "Not available"
    return f"{_number(value[0])} to {_number(value[1])} {unit}"


def _metric(label: str, value: str, note: str | None = None) -> Any:
    return html.Div(
        [html.Span(label), html.Strong(value), *([html.Small(note)] if note else [])],
        className="lab-diagnostic-metric",
    )


def _details(label: str, children: list[Any]) -> Any:
    return html.Details(
        [html.Summary(label), *children], className="lab-diagnostic-details"
    )


def _quality_flags(data: Row) -> list[str]:
    flags = [
        *data.get("flags", []),
        *data.get("quality_flags", []),
        *data.get("option_flags", []),
        *data.get("forward_flags", []),
    ]
    for row in [*data.get("option_inputs", []), data.get("future_input") or {}]:
        flags.extend(
            f"{row.get('symbol', 'Input')}: {_human(flag)}"
            for flag in row.get("quality_flags", [])
        )
    comparison = data.get("same_package_midpoint") or {}
    flags.extend(comparison.get("flags", []))
    return list(dict.fromkeys(_human(flag) for flag in flags))


def _flag_panel(flags: list[str]) -> Any:
    return html.Div(
        [
            html.Span("Flags: "),
            *(
                [html.Span(flag, className="lab-warn") for flag in flags]
                if flags
                else [html.Span("None recorded", className="lab-muted")]
            ),
        ],
        className="lab-diagnostic-flags",
    )


def _legs(rows: list[Row]) -> Any:
    body = []
    for row in rows:
        quote = row.get("normalized_quote") or {}
        side = row.get("side")
        crossed = (
            "Buy ask" if side == "buy" else "Sell bid" if side == "sell" else "Observed"
        )
        price_unit = quote.get("price_unit", "USDC/BTC")
        size_unit = quote.get("size_unit", "contracts")
        size = (
            f"{_number(row.get('displayed_size'))} {size_unit}"
            if side
            else (
                f"{_number(quote.get('bid_size'))} / "
                f"{_number(quote.get('ask_size'))} {size_unit}"
            )
        )
        body.append(
            html.Tr(
                [
                    html.Th(str(row.get("symbol", "Not available")), scope="row"),
                    html.Td(_number(row.get("strike"))),
                    html.Td(_number(row.get("signed_weight"), 4) if side else "—"),
                    html.Td(crossed),
                    html.Td(
                        f"{_number(quote.get('bid'))} / "
                        f"{_number(quote.get('ask'))} {price_unit}"
                    ),
                    html.Td(
                        f"{_number(row.get('price'))} {price_unit}"
                        if side
                        else "Not crossed"
                    ),
                    html.Td(size),
                ]
            )
        )
    return html.Div(
        html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th(label)
                            for label in (
                                "Contract",
                                "Strike",
                                "Weight",
                                "Side",
                                "Bid / ask",
                                "Crossed input",
                                "Side size / bid-ask size",
                            )
                        ]
                    )
                ),
                html.Tbody(body),
            ],
            className="lab-table lab-diagnostic-legs",
        ),
        className="lab-table-wrap",
    )


def _provenance(payload: Row) -> Any:
    data = payload["data"]
    rows = data.get("option_inputs", [])
    if data.get("future_input"):
        rows = [*rows, data["future_input"]]
    if data.get("symbol"):
        rows = [data]
    return _details(
        "Provenance & quote timing",
        [
            html.P(
                f"{payload['source']} · checkpoint {payload['as_of']} · "
                f"expiry {payload['expiry']}",
                className="lab-diagnostic-provenance lab-muted",
            ),
            *[
                html.P(
                    f"{row.get('symbol', 'Input')} · quote time "
                    f"{quote.get('quote_time', 'Not available')} · "
                    f"available at {quote.get('available_at', 'Not available')} · "
                    f"sequence {quote.get('source_sequence', 'Not available')}"
                )
                for row in rows
                for quote in [row.get("normalized_quote") or {}]
            ],
        ],
    )


def _butterfly_panel(data: Row) -> list[Any]:
    package = data.get("best_package") or {}
    comparison = data.get("same_package_midpoint") or {}
    midpoint = comparison.get("midpoint_cost")
    quoted = package.get("net_entry_cost")
    unit = data["cost_unit"]
    signal = data["quoted_cost_explanation"]
    if package and midpoint is None:
        signal += (
            " Same-package midpoint comparison unavailable: missing or flagged inputs. "
            "The stored quoted result is retained, not refreshed or validated."
        )
    elif midpoint is not None and midpoint < 0 and quoted is not None and quoted >= 0:
        signal += (
            " The same-package midpoint credit disappears when crossing "
            "the quoted bid/ask sides."
        )
    hedge = data.get("future_input")
    return [
        html.Div(
            [
                _metric("Same-package midpoint cost", _number(midpoint), unit),
                _metric(
                    "Quoted entry cost",
                    f"{quoted:.6g}"
                    if quoted is not None and 0 < abs(quoted) < 0.005
                    else _number(quoted),
                    unit,
                ),
                _metric(
                    "Displayed capacity",
                    _number(package.get("displayed_size_cap")),
                    "BTC of center option · minimum required-side size",
                ),
            ],
            className="lab-diagnostic-metrics",
        ),
        html.P(signal, className="lab-diagnostic-signal"),
        html.P(
            "Same-package midpoint cost uses these exact legs and hedge; "
            "it is not the raw midpoint anomaly. Negative cost means credit. "
            "No fee estimate; no simultaneous fill.",
            className="lab-muted",
        ),
        _flag_panel(_quality_flags(data)),
        _details(
            "Option legs · observed quotes"
            if not package
            else "Option legs · quoted entry",
            [_legs(data.get("option_inputs", []))],
        ),
        _details(
            "Net futures hedge & exact package weights",
            [
                html.P("Exact strike weights: " + " / ".join(data["exact_weights"])),
                *(
                    [_legs([hedge])]
                    if hedge
                    else [
                        html.P(
                            "No net futures hedge required."
                            if package
                            else "Net futures hedge unavailable."
                        )
                    ]
                ),
                html.P(
                    "Option multipliers (BTC / contract): "
                    + "; ".join(
                        f"{leg['symbol']}: {_number(leg.get('multiplier'), 4)}"
                        for leg in package.get("option_legs", [])
                    )
                )
                if package
                else html.P("Option multipliers unavailable."),
                html.P(
                    "Bid/ask crossing increment: "
                    f"{_number(comparison.get('crossing_cost'))} {unit}. "
                    f"Option entry: {_number(package.get('option_entry_cost'))}; "
                    f"parity adjustment: {_number(package.get('parity_adjustment'))}."
                ),
                html.P(
                    "Combinations checked: "
                    f"{_number(data.get('combinations_checked'), 0)} · "
                    f"skipped: {_number(data.get('combinations_skipped'), 0)}. "
                    f"Quote basis: {data.get('quote_basis') or 'Not available'}"
                ),
            ],
        ),
    ]


def _parity_panel(data: Row) -> list[Any]:
    future = data.get("future_input") or {}
    quote = future.get("normalized_quote") or {}
    consistent = data.get("consistent_with_future")
    return [
        html.Div(
            [
                _metric(
                    "Option-implied forward",
                    _interval(data.get("implied_forward"), "USDC/BTC"),
                ),
                _metric(
                    "Independently quoted future",
                    _interval(
                        [quote["bid"], quote["ask"]] if quote else None, "USDC/BTC"
                    ),
                    str(future.get("symbol") or "No matched future"),
                ),
                _metric(
                    "Parity residual",
                    _interval(data.get("independent_forward_residual"), "USDC/BTC"),
                ),
            ],
            className="lab-diagnostic-metrics",
        ),
        html.P(
            "Consistent: residual interval includes zero."
            if consistent is True
            else "Inconsistent: residual interval excludes zero."
            if consistent is False
            else "Consistency unavailable: independent quoted evidence is incomplete.",
            className="lab-diagnostic-signal",
        ),
        _flag_panel(_quality_flags(data)),
        _details(
            "Option legs & parity inputs",
            [
                _legs(data.get("option_inputs", [])),
                html.P(
                    "Call minus put premium: "
                    f"{_interval(data.get('premium_difference'), 'USDC/BTC')}"
                ),
                html.P(
                    f"Discount factor: {_number(data.get('discount_factor'), 6)} · "
                    f"Forward input: {_number(data.get('forward_input'))} USDC/BTC"
                ),
            ],
        ),
    ]


def _midpoint_panel(payload: Row) -> list[Any]:
    data = payload["data"]
    maturity = data.get("time_to_expiry")
    strikes = " / ".join(_number(strike) for strike in payload["strikes"])
    return [
        html.P(
            f"Strikes: {strikes or 'Not identified'} · expiry {payload['expiry']} · "
            f"{_number(maturity * 365 if maturity is not None else None, 1)} "
            "days to maturity",
            className="lab-diagnostic-signal",
        ),
        html.Div(
            [
                _metric(
                    f"{_human(data.get('kind', 'Midpoint')).title()} violation",
                    _number(data.get("value"), 6),
                    "Raw diagnostic value, not a package cost",
                ),
                _metric("Lower bound", _number(data.get("lower_bound"), 6)),
                _metric("Upper bound", _number(data.get("upper_bound"), 6)),
            ],
            className="lab-diagnostic-metrics",
        ),
        _flag_panel(_quality_flags(data)),
        _details(
            "Observed nodes & finding details",
            [
                _legs(data.get("option_inputs", [])),
                html.P(
                    f"Log-forward moneyness: {_number(data.get('log_moneyness'), 6)} · "
                    f"maturity: {_number(maturity, 6)} years"
                ),
                html.P(
                    f"Tolerance: {data.get('tolerance', 'Not available')} · "
                    f"Scope: {data.get('claim') or 'Not available'}"
                ),
            ],
        ),
    ]


def _summary_panel(data: Row) -> list[Any]:
    if data.get("symbol"):
        native = data.get("native_book") or {}
        return [
            _flag_panel(_quality_flags(data)),
            _legs([data]),
            _details(
                "Book quality & IV availability",
                [
                    html.P(
                        "Native sequence gaps: "
                        f"{_number(native.get('sequence_gaps'), 0)}"
                    ),
                    *[
                        html.P(
                            f"{side.title()} IV: {_number((value or {}).get('iv'), 4)}"
                        )
                        for side, value in data.get("sides", {}).items()
                    ],
                ],
            ),
        ]
    coverage = data.get("coverage") or {}
    butterflies = data.get("quoted_butterfly_counts") or {}
    iv_successes = coverage.get("iv_successes") or {}
    median_spread = coverage.get("median_premium_spread_usdc_per_btc")
    flags = [
        f"{_human(key)}: {value}"
        for key, value in (coverage.get("native_flag_counts") or {}).items()
        if value
    ]
    flags.extend(
        f"{label}: {_human(data[key])}"
        for key, label in (
            ("iv_blocked", "IV blocked"),
            ("fit_error", "Fit unavailable"),
            ("joint_fit_error", "Joint fit unavailable"),
        )
        if data.get(key)
    )
    return [
        html.Div(
            [
                _metric(label, _number(value, 0))
                for label, value in (
                    ("Usable books", coverage.get("usable_books")),
                    ("Requested options", coverage.get("requested_options")),
                    ("Parity pairs recorded", data.get("parity_pairs_recorded")),
                    (
                        "Raw midpoint findings",
                        data.get("raw_midpoint_findings_at_expiry"),
                    ),
                    ("Butterfly triplets checked", butterflies.get("triplets_checked")),
                    ("Quoted credit candidates", butterflies.get("candidate_count")),
                )
            ],
            className="lab-diagnostic-metrics",
        ),
        _flag_panel(flags),
        html.P(
            "Native capture: "
            + {True: "complete", False: "incomplete", None: "Not available"}[
                data.get("native_capture_complete")
            ],
            className="lab-diagnostic-signal",
        ),
        _details(
            "Coverage & fit quality details",
            [
                html.P(
                    f"Two-sided IV books: {_number(coverage.get('two_sided_iv'), 0)} · "
                    "Eligible OTM strikes: "
                    f"{_number(coverage.get('eligible_otm_strikes'), 0)}"
                ),
                html.P(
                    "IV successes: "
                    + " · ".join(
                        f"{side}: {_number(iv_successes.get(side), 0)}"
                        for side in ("bid", "midpoint", "ask")
                    )
                ),
                html.P(
                    "Maximum usable book age: "
                    f"{_number(coverage.get('max_usable_book_age_ms'))} ms · "
                    f"Median premium spread: {_number(median_spread)} USDC/BTC"
                ),
                html.P(
                    "Parity pairs consistent: "
                    f"{_number(coverage.get('parity_pairs_consistent'), 0)} / "
                    f"{_number(coverage.get('parity_pairs_compared'), 0)} compared"
                ),
                *[
                    html.P(
                        f"{label}: "
                        + (
                            f"{len(value.get('findings', []))} recorded findings"
                            if isinstance(value, dict)
                            else "Not available"
                        )
                    )
                    for label, value in (
                        (
                            "Per-expiry fitted arbitrage check",
                            data.get("fit_arbitrage"),
                        ),
                        (
                            "Joint fitted arbitrage check",
                            data.get("joint_fitted_arbitrage"),
                        ),
                    )
                ],
            ],
        ),
    ]


def diagnostic_panel(
    view: ReplayView, tab: str, symbol: str | None, item: str | None
) -> Any:
    payload = diagnostic_payload(view, tab, symbol, item)
    status = payload["status"]
    children: list[Any] = [
        html.Div(
            [
                html.H3(payload.get("label", tab.title())),
                html.P(
                    status,
                    className="lab-good"
                    if status
                    in ("usable book", "consistent interval", "No quoted credit")
                    else "lab-warn",
                ),
            ],
            className="lab-diagnostic-head",
        ),
    ]
    if "message" in payload:
        children.append(html.P(payload["message"], className="lab-empty"))
    else:
        data = payload["data"]
        if tab == "butterflies":
            children.extend(_butterfly_panel(data))
        elif tab == "parity":
            children.extend(_parity_panel(data))
        elif tab == "midpoint":
            children.extend(_midpoint_panel(payload))
        else:
            children.extend(_summary_panel(data))
        buttons = []
        for related in payload["symbols"]:
            attrs: dict[str, Any] = {
                "aria-pressed": "true" if related == symbol else "false"
            }
            buttons.append(
                html.Button(
                    related,
                    id={"type": "lab-diagnostic-contract", "symbol": related},
                    n_clicks=0,
                    title=f"Select {related}",
                    **attrs,
                )
            )
        children.append(html.Div(buttons, className="lab-diagnostic-contracts"))
        children.append(_provenance(payload))
    children.extend(html.P(limit, className="lab-muted") for limit in payload["limits"])
    children.append(
        _details(
            "Raw diagnostic JSON",
            [html.Pre(json.dumps(payload, indent=2, ensure_ascii=False, default=str))],
        )
    )
    return html.Section(children, className="lab-panel lab-diagnostic")
