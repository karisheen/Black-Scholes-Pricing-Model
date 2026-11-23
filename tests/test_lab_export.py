"""Behavioral tests for local, scoped, reproducible Options Lab downloads."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from quant_pricing.lab_data import (
    ReplayView,
    choose_symbol,
    contract_detail,
    load_reports,
    visible_options,
)
from quant_pricing.lab_diagnostics import diagnostic_options, diagnostic_payload
from quant_pricing.lab_export import export_view


@pytest.fixture
def view() -> ReplayView:
    library = load_reports()
    session = next(iter(library.sessions))
    report = library.sessions[session]
    checkpoint = report["checkpoints"][0]
    return library.view(
        session, checkpoint["as_of"], checkpoint["expiries"][0]["expiry"]
    )


def _download(view: ReplayView, **changes: Any) -> dict[str, Any]:
    arguments = {
        "scope": "chain",
        "mode": "quotes",
        "side": "all",
        "symbol": choose_symbol(view, "all"),
        "tab": "diagnostics",
        "item": None,
        "format": "json",
    }
    arguments.update(changes)
    return export_view(view, **arguments)


@pytest.mark.parametrize("scope", ["chain", "iv", "fit", "contract", "diagnostic"])
@pytest.mark.parametrize("format", ["csv", "json"])
def test_all_download_scopes_are_deterministic_and_local(
    view: ReplayView,
    scope: str,
    format: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    first = _download(view, scope=scope, format=format)
    assert first == _download(view, scope=scope, format=format)
    assert first["filename"].endswith(f".{format}")
    assert first["type"] == ("text/csv" if format == "csv" else "application/json")
    assert list(tmp_path.iterdir()) == []
    if format == "json":
        payload = json.loads(first["content"])
        metadata = payload["metadata"]
        assert metadata["row_count"] == len(payload["data"])
        assert metadata["session"] == view.session
        assert metadata["synthetic"] is True
        assert metadata["source"] == view.report["source"]
        assert metadata["as_of"] == view.checkpoint["as_of"]
        assert metadata["expiry"] == view.expiry["expiry"]
        assert metadata["assumptions"] == view.report["assumptions"]
        assert metadata["matched_future"] == view.expiry["matched_future"]
        assert metadata["selection"]["scope"] == scope
        assert any("fees" in limit for limit in metadata["limits"])
    else:
        reader = csv.DictReader(io.StringIO(first["content"]))
        rows = list(reader)
        assert rows
        assert reader.fieldnames == sorted(reader.fieldnames or [])
        assert rows[0]["metadata.session"] == view.session
        assert rows[0]["metadata.as_of"] == view.checkpoint["as_of"]


@pytest.mark.parametrize("side", ["otm", "call", "put", "all"])
def test_chain_export_honors_side_without_mutating_source(
    view: ReplayView, side: str
) -> None:
    before = json.dumps(view.report, sort_keys=True)
    payload = json.loads(_download(view, side=side)["content"])
    expected = visible_options(view, side)
    assert sorted(row["symbol"] for row in payload["data"]) == sorted(
        row["symbol"] for row in expected
    )
    assert payload["metadata"]["selection"]["side"] == side
    assert json.dumps(view.report, sort_keys=True) == before


def test_chain_csv_has_usable_columns_and_original_flags(view: ReplayView) -> None:
    rows = list(csv.DictReader(io.StringIO(_download(view, format="csv")["content"])))
    source = {row["symbol"]: row for row in visible_options(view, "all")}
    assert len(rows) == len(source)
    for row in rows:
        original = source[row["data.symbol"]]
        assert float(row["data.strike"]) == original["strike"]
        assert row["data.option_type"] == original["option_type"]
        assert json.loads(row["data.normalized_flags"]) == list(
            original["normalized_flags"]
        )
        quote = original["normalized_quote"]
        if quote is not None:
            assert row["data.normalized_quote.price_unit"] == quote["price_unit"]
            assert row["data.normalized_quote.quote_time"] == quote["quote_time"]
        assert "data" not in row


def test_iv_export_keeps_solver_sides_separate(view: ReplayView) -> None:
    payload = json.loads(_download(view, scope="iv", side="put")["content"])
    originals = {row["symbol"]: row for row in visible_options(view, "put")}
    assert len(payload["data"]) == sum(
        max(1, len(row["sides"])) for row in originals.values()
    )
    for row in payload["data"]:
        source = originals[row["symbol"]]
        assert row["option_type"] == "put"
        assert row["iv_result"] == source["sides"].get(row["quote_side"])
        assert row["normalized_flags"] == list(source["normalized_flags"])


def test_missing_iv_is_exported_as_missing_not_zero(view: ReplayView) -> None:
    original = visible_options(view, "all")[0]
    option = {**original, "sides": {}, "rejected": ["stale"]}
    changed = replace(view, expiry={**view.expiry, "options": [option]})
    rows = json.loads(_download(changed, scope="iv")["content"])["data"]
    assert len(rows) == 1
    assert rows[0]["quote_side"] is None
    assert rows[0]["iv_result"] is None
    assert rows[0]["rejected"] == ["stale"]


def test_contract_export_is_exactly_the_selected_contract(view: ReplayView) -> None:
    symbol = visible_options(view, "all")[-1]["symbol"]
    payload = json.loads(_download(view, scope="contract", symbol=symbol)["content"])
    expected = json.loads(json.dumps(contract_detail(view, symbol)))
    assert payload["data"] == [expected]
    assert payload["data"][0]["symbol"] == symbol
    assert payload["metadata"]["selection"]["symbol"] == symbol


@pytest.mark.parametrize("tab", ["diagnostics", "parity", "midpoint", "butterflies"])
def test_diagnostic_export_contains_only_requested_item(
    view: ReplayView, tab: str
) -> None:
    symbol = choose_symbol(view, "all")
    options = diagnostic_options(view, tab, symbol)
    item = options[-1]["value"] if options else None
    payload = json.loads(
        _download(view, scope="diagnostic", tab=tab, item=item)["content"]
    )
    expected = diagnostic_payload(view, tab, symbol, item)
    assert payload["data"] == [json.loads(json.dumps(expected))]
    assert payload["metadata"]["selection"]["item"] == item
    assert payload["metadata"]["selection"]["tab"] == tab


def test_fit_export_retains_parameters_residuals_and_limits(view: ReplayView) -> None:
    fit = {
        "parameters": {"a": 0.01, "b": 0.1, "rho": -0.2, "m": 0.0, "sigma": 0.3},
        "residuals": [
            {"contract": "Z", "residual": 0.002},
            {"contract": "A", "residual": -0.001},
        ],
        "arbitrage": {"claim": "finite grid only", "tolerance": 1e-8},
    }
    changed = replace(view, expiry={**view.expiry, "fit": fit})
    rows = json.loads(_download(changed, scope="fit")["content"])["data"]
    assert [row["residual"]["contract"] for row in rows] == ["A", "Z"]
    assert all(row["fit"]["parameters"] == fit["parameters"] for row in rows)
    assert all(row["fit"]["arbitrage"] == fit["arbitrage"] for row in rows)
    without_residuals = replace(
        changed, expiry={**changed.expiry, "fit": {**fit, "residuals": []}}
    )
    rows = json.loads(_download(without_residuals, scope="fit")["content"])["data"]
    assert len(rows) == 1
    assert rows[0]["residual"] is None


@pytest.mark.parametrize("scope", ["chain", "iv", "fit", "contract"])
def test_empty_scopes_keep_provenance_without_invented_rows(
    view: ReplayView, scope: str
) -> None:
    changed = replace(view, expiry={**view.expiry, "options": [], "fit": None})
    payload = json.loads(_download(changed, scope=scope, symbol=None)["content"])
    assert payload["data"] == []
    assert payload["metadata"]["row_count"] == 0
    csv_rows = list(
        csv.DictReader(
            io.StringIO(
                _download(changed, scope=scope, symbol=None, format="csv")["content"]
            )
        )
    )
    assert len(csv_rows) == 1
    assert csv_rows[0]["metadata.row_count"] == "0"
    assert not any(key.startswith("data.") for key in csv_rows[0])


def test_input_order_does_not_change_chain_download(view: ReplayView) -> None:
    changed = replace(
        view,
        report=dict(reversed(list(view.report.items()))),
        expiry={**view.expiry, "options": list(reversed(view.expiry["options"]))},
    )
    symbol = choose_symbol(view, "all")
    for format in ("csv", "json"):
        assert _download(view, symbol=symbol, format=format) == _download(
            changed, symbol=symbol, format=format
        )


def test_private_provenance_and_safe_download_filename(view: ReplayView) -> None:
    changed = replace(
        view,
        session="../../private\\report\n\x00",
        report={**view.report, "source": "tardis/deribit"},
    )
    download = _download(changed)
    filename = download["filename"]
    assert "/" not in filename and "\\" not in filename
    assert ".." not in filename and "\n" not in filename and "\x00" not in filename
    metadata = json.loads(download["content"])["metadata"]
    assert metadata["synthetic"] is False
    assert metadata["source"] == "tardis/deribit"
    assert metadata["session"] == changed.session


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"scope": "../chain"}, "scope"),
        ({"format": "../../json"}, "format"),
        ({"side": "calls"}, "side"),
    ],
)
def test_invalid_export_selectors_are_rejected(
    view: ReplayView, changes: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _download(view, **changes)


def test_greek_chain_export_matches_adapter_risk_and_units(view: ReplayView) -> None:
    payload = json.loads(_download(view, mode="greeks", side="call")["content"])
    expected = {
        row["symbol"]: json.loads(json.dumps(contract_detail(view, row["symbol"])))
        for row in visible_options(view, "call")
    }
    assert {row["symbol"]: row for row in payload["data"]} == expected
    assert payload["metadata"]["selection"]["mode"] == "greeks"
    csv_rows = list(
        csv.DictReader(
            io.StringIO(
                _download(view, mode="greeks", side="call", format="csv")["content"]
            )
        )
    )
    assert csv_rows
    assert "data.risk_basis" in csv_rows[0]
    assert any(key.startswith("data.risk_units.") for key in csv_rows[0])
