"""Deterministic, in-memory downloads of the selected Options Lab view."""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any

from quant_pricing.lab_data import (
    ReplayView,
    Row,
    contract_detail,
    market_context,
    visible_options,
)
from quant_pricing.lab_demo import DEMO_SOURCE
from quant_pricing.lab_diagnostics import diagnostic_payload

_SCOPES = frozenset({"chain", "iv", "fit", "contract", "diagnostic"})
_SIDES = frozenset({"otm", "call", "put", "all"})
_LIMITS = (
    "Historical research only; not an order, execution, or fill guarantee.",
    "Quoted packages are before fees, slippage, and execution uncertainty.",
    "Midpoint findings are not executable bid/ask opportunities.",
    "Finite-grid fitted checks do not prove global absence of arbitrage.",
)


def _json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _flatten(value: Row, prefix: str = "") -> Row:
    """Use dotted scalar columns; retain ordered structures as canonical JSON."""
    result: Row = {}
    for key, item in sorted(value.items()):
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict) and item:
            result.update(_flatten(item, name))
        elif isinstance(item, (dict, list, tuple)):
            result[name] = _json(item)
        else:
            result[name] = item
    return result


def _rows(
    view: ReplayView,
    scope: str,
    mode: str,
    side: str,
    symbol: str | None,
    tab: str,
    item: str | None,
) -> list[Row]:
    if scope == "contract":
        return [] if symbol is None else [contract_detail(view, symbol)]
    if scope == "diagnostic":
        return [diagnostic_payload(view, tab, symbol, item)]
    if scope == "fit":
        fit = view.expiry.get("fit")
        if fit is None:
            return []
        parameters = {key: value for key, value in fit.items() if key != "residuals"}
        residuals = fit.get("residuals", [])
        return [
            {"fit": parameters, "residual": residual} for residual in residuals
        ] or [{"fit": parameters, "residual": None}]
    options = visible_options(view, side)
    if scope == "chain":
        if mode == "greeks":
            return [contract_detail(view, option["symbol"]) for option in options]
        return [dict(option) for option in options]
    rows = []
    for option in options:
        base = {key: value for key, value in option.items() if key != "sides"}
        sides = option.get("sides", {})
        if not sides:
            rows.append({**base, "quote_side": None, "iv_result": None})
        for quote_side, result in sorted(sides.items()):
            rows.append({**base, "quote_side": quote_side, "iv_result": result})
    return rows


def _filename_token(value: object) -> str:
    # No path components, control characters, or browser-provided paths survive.
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(value)).strip("-")[:80] or "unknown"


def export_view(
    view: ReplayView,
    *,
    scope: str,
    mode: str,
    side: str,
    symbol: str | None,
    tab: str,
    item: str | None,
    format: str,
) -> dict[str, Any]:
    """Return a Dash Download payload without filesystem writes or network I/O.

    JSON preserves source values and nulls. CSV repeats flattened provenance for
    each record, uses empty cells for nulls, and JSON-encodes structured columns.
    Empty scopes have zero JSON records and one CSV metadata-only row.
    """
    if scope not in _SCOPES:
        raise ValueError(f"Unsupported export scope: {scope}")
    if format not in {"csv", "json"}:
        raise ValueError(f"Unsupported export format: {format}")
    if side not in _SIDES:
        raise ValueError(f"Unsupported option side: {side}")
    rows = sorted(_rows(view, scope, mode, side, symbol, tab, item), key=_json)
    report = view.report
    metadata: Row = {
        "export_schema": "options-lab-export/v1",
        "session": view.session,
        "source": report.get("source"),
        "synthetic": report.get("source") == DEMO_SOURCE,
        "as_of": view.checkpoint["as_of"],
        "expiry": view.expiry["expiry"],
        "report_schema_version": report.get("schema_version"),
        "report_scope": report.get("scope"),
        "contract_specification": report.get("contract_specification"),
        "catalog_selection": report.get("catalog_selection"),
        "assumptions": report.get("assumptions"),
        "native_capture_complete": view.checkpoint.get("native_capture_complete"),
        "matched_future": view.expiry.get("matched_future"),
        "market_context": market_context(view, symbol),
        "iv_blocked": view.expiry.get("iv_blocked"),
        "fit_error": view.expiry.get("fit_error"),
        "selection": {
            "scope": scope,
            "mode": mode,
            "side": side,
            "symbol": symbol,
            "tab": tab,
            "item": item,
        },
        "limits": _LIMITS,
        "row_count": len(rows),
        "csv_null_encoding": "empty cell; consult JSON to distinguish null and empty",
    }
    if format == "json":
        content = _json({"metadata": metadata, "data": rows}) + "\n"
        media_type = "application/json"
    else:
        common = _flatten({"metadata": metadata})
        flattened = [{**common, **_flatten({"data": row})} for row in rows] or [common]
        columns = sorted({key for row in flattened for key in row})
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flattened)
        content = output.getvalue()
        media_type = "text/csv"
    tokens = (scope, view.session, view.checkpoint["as_of"], view.expiry["expiry"])
    filename = "options-lab-" + "-".join(map(_filename_token, tokens)) + f".{format}"
    return {"filename": filename, "content": content, "type": media_type}
