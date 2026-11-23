"""Synthetic-only evidence selection and same-package cost regressions."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Any

import pytest

from quant_pricing.butterfly import (
    ButterflyFuturesHedge,
    ButterflyOptionLeg,
    ButterflyPackage,
)
from quant_pricing.conventions import year_fraction
from quant_pricing.lab_data import ReplayView
from quant_pricing.lab_diagnostics import (
    diagnostic_options,
    diagnostic_panel,
    diagnostic_payload,
)


@pytest.fixture
def view() -> ReplayView:
    options = []
    for symbol, strike, kind, bid, ask, k in (
        ("C90", 90, "call", 14, 15, -0.1),
        ("P100", 100, "put", 10, 11, 0.0),
        ("C130", 130, "call", 1, 2, 0.2),
    ):
        options.append(
            {
                "symbol": symbol,
                "strike": strike,
                "option_type": kind,
                "normalized_quote": {
                    "bid": bid,
                    "ask": ask,
                    "bid_size": 20,
                    "ask_size": 6,
                    "quote_time": "2025-01-01T00:00:00+00:00",
                    "available_at": "2025-01-01T00:00:00+00:00",
                    "source_sequence": 7,
                    "price_unit": "USDC/BTC",
                    "size_unit": "contracts",
                },
                "normalized_flags": [],
                "rejected": [],
                "native_book": {"flags": [], "sequence_gaps": 0},
                "log_forward_moneyness": k,
                "sides": {"midpoint": {"iv": 0.5}},
            }
        )
    package = ButterflyPackage(
        (
            ButterflyOptionLeg("C90", "call", "buy", 0.75, 15, 6, 2),
            ButterflyOptionLeg("P100", "put", "sell", -1, 10, 20, 1),
            ButterflyOptionLeg("C130", "call", "buy", 0.25, 2, 6, 1),
        ),
        ButterflyFuturesHedge("F", "sell", -1, 119, 3),
        1.75,
        -17.1,
        -15.35,
        3,
        True,
        1e-10,
    )
    future = {
        "symbol": "F",
        "normalized_flags": [],
        "native_book": {"flags": []},
        "normalized_quote": {
            "bid": 119,
            "ask": 121,
            "bid_size": 3,
            "ask_size": 8,
            "size_unit": "BTC",
        },
    }
    expiry = {
        "expiry": "2025-04-01T00:00:00+00:00",
        "options": options,
        "discount_factor": 0.9,
        "forward_input": 120,
        "matched_future": future,
        "coverage": {"requested_options": 3, "usable_books": 3},
        "fit": None,
        "fit_error": "insufficient eligible observations",
        "parity": [
            {
                "call": "C90",
                "put": "P100",
                "option_flags": [],
                "forward_flags": [],
                "premium_difference": [3, 5],
                "implied_forward": [100, 120],
                "independent_forward_residual": [-1, 1],
                "consistent_with_future": True,
            }
        ],
        "quoted_butterflies": {
            "cost_unit": "USDC per BTC of center option",
            "quote_basis": "buy asks, sell bids; before fees and not a fill",
            "triplets": [
                {
                    "strikes": [90, 100, 130],
                    "weights": [0.75, -1, 0.25],
                    "combinations_checked": 1,
                    "combinations_skipped": 7,
                    "flags": ["partial_support"],
                    "negative_cost_candidate": True,
                    "best_package": asdict(package),
                }
            ],
        },
    }
    maturity = year_fraction(
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 4, 1, tzinfo=timezone.utc),
    )
    checkpoint = {
        "as_of": "2025-01-01T00:00:00Z",
        "native_capture_complete": True,
        "observed_arbitrage": {
            "claim": "observed nodes only",
            "tolerance": 1e-8,
            "findings": [
                {
                    "kind": "butterfly",
                    "time_to_expiry": maturity,
                    "log_moneyness": 0.0,
                    "value": -99,
                    "lower_bound": 0,
                },
                {
                    "kind": "calendar",
                    "time_to_expiry": maturity + 1,
                    "log_moneyness": 0.0,
                    "value": -100,
                    "lower_bound": 0,
                },
            ],
        },
    }
    return ReplayView(
        "synthetic",
        {"source": "public synthetic", "assumptions": {}},
        checkpoint,
        expiry,
    )


def _rendered(component: Any) -> str:
    def encode(value: Any) -> Any:
        if hasattr(value, "to_plotly_json"):
            return value.to_plotly_json()
        raise TypeError(type(value))

    return json.dumps(component, default=encode)


def test_butterfly_compares_same_weighted_legs_and_hedge(view: ReplayView) -> None:
    result = diagnostic_payload(view, "butterflies", "C90", None)
    assert result["symbols"] == ["C90", "P100", "C130"]
    data = result["data"]
    assert data["weights"] == [0.75, -1, 0.25]
    assert data["exact_weights"] == ["3/4", "-1", "1/4"]
    assert data["same_package_midpoint"]["quoted_cost"] == -15.35
    # Options cost .75*14.5 - 10.5 + .25*1.5 = .75;
    # exactly the stored short-future hedge contributes .9*(100 - 120) = -18.
    assert data["same_package_midpoint"]["midpoint_cost"] == pytest.approx(-17.25)
    assert data["same_package_midpoint"]["crossing_cost"] == pytest.approx(1.9)
    assert data["same_package_midpoint"]["midpoint_cost"] != -99
    assert data["best_package"]["displayed_size_cap"] == 3
    assert [row["crossed_side"] for row in data["option_inputs"]] == [
        "ask",
        "bid",
        "ask",
    ]
    assert data["future_input"]["crossed_side"] == "bid"
    assert data["option_inputs"][0]["normalized_quote"]["source_sequence"] == 7
    assert "no fees" in result["limits"][0]
    rendered = _rendered(diagnostic_panel(view, "butterflies", "C90", None))
    assert "lab-diagnostic-contract" in rendered
    assert "Net futures hedge" in rendered


def test_stale_leg_blocks_new_midpoint_not_stored_evidence(view: ReplayView) -> None:
    view.expiry["options"][0]["normalized_flags"] = ["stale"]
    view.expiry["options"][0]["native_book"]["sequence_gaps"] = 2
    result = diagnostic_payload(view, "butterflies", None, None)
    data = result["data"]
    assert data["best_package"]["net_entry_cost"] == -15.35
    assert data["same_package_midpoint"]["midpoint_cost"] is None
    assert data["same_package_midpoint"]["flags"] == ("C90:stale", "C90:sequence_gap")
    assert data["option_inputs"][0]["quality_flags"] == ["stale", "sequence_gap"]


def test_midpoint_scope_and_actionable_adjacent_strikes(view: ReplayView) -> None:
    options = diagnostic_options(view, "midpoint", "C90")
    assert [item["value"] for item in options] == ["midpoint:0"]
    payload = diagnostic_payload(view, "midpoint", None, "midpoint:0")
    assert payload["strikes"] == [90, 100, 130]
    assert payload["data"]["value"] == -99
    assert "not tradable" in payload["status"]
    assert "worst-case" in payload["limits"][0]
    assert diagnostic_payload(view, "midpoint", None, "midpoint:1")["data"] == {}


def test_calendar_and_unmatched_nodes_do_not_invent_legs(view: ReplayView) -> None:
    finding = view.checkpoint["observed_arbitrage"]["findings"][0]
    finding["kind"] = "calendar"
    result = diagnostic_payload(view, "midpoint", None, None)
    assert result["symbols"] == ["P100"]
    assert "prior expiry" in result["limits"][1]
    finding["log_moneyness"] = 3
    assert diagnostic_payload(view, "midpoint", None, None)["symbols"] == []
    finding["kind"] = "monotonicity"
    finding["log_moneyness"] = -0.1
    assert diagnostic_payload(view, "midpoint", None, None)["symbols"] == [
        "C90",
        "P100",
    ]


@pytest.mark.parametrize(
    "consistent,status",
    [
        (True, "consistent interval"),
        (False, "inconsistent interval"),
        (None, "unavailable"),
    ],
)
def test_parity_intervals_preserve_inputs_and_flags(
    view: ReplayView, consistent: bool | None, status: str
) -> None:
    view.expiry["parity"][0]["consistent_with_future"] = consistent
    view.expiry["parity"][0]["option_flags"] = ["call:stale"]
    payload = diagnostic_payload(view, "parity", None, None)
    assert payload["status"] == status
    assert payload["data"]["independent_forward_residual"] == [-1, 1]
    assert payload["data"]["option_flags"] == ["call:stale"]
    assert payload["data"]["discount_factor"] == 0.9
    assert payload["data"]["future_input"]["symbol"] == "F"
    assert "not tradability" in payload["limits"][0]
    assert "Option legs" in _rendered(diagnostic_panel(view, "parity", None, None))


def test_diagnostic_summary_and_selected_book_quality(view: ReplayView) -> None:
    summary = diagnostic_payload(view, "diagnostics", None, "summary")
    assert summary["data"]["coverage"]["usable_books"] == 3
    assert summary["data"]["fit_error"] == "insufficient eligible observations"
    assert summary["source"] == "public synthetic"
    assert diagnostic_options(view, "diagnostics", "P100")[0]["value"] == "book:P100"
    assert (
        diagnostic_payload(view, "diagnostics", "C90", None)["status"] == "usable book"
    )
    view.expiry["options"][0]["rejected"] = ["stale"]
    assert diagnostic_payload(view, "diagnostics", "C90", None)["status"] == "flagged"
    assert "Not available" in _rendered(
        diagnostic_panel(view, "diagnostics", None, "summary")
    )
    assert "stale" in _rendered(diagnostic_panel(view, "diagnostics", "C90", None))


@pytest.mark.parametrize("tab", ["parity", "midpoint", "butterflies"])
def test_empty_missing_and_expired_states(view: ReplayView, tab: str) -> None:
    empty = replace(view, expiry={"expiry": "2025-01-01T00:00:00Z"})
    assert diagnostic_options(empty, tab, None) == []
    result = diagnostic_payload(empty, tab, None, None)
    assert result["status"] == "unavailable"
    assert result["message"]
    assert "lab-empty" in _rendered(diagnostic_panel(empty, tab, None, None))
    assert "not present" in diagnostic_payload(view, tab, None, "old-item")["message"]


def test_no_candidate_no_hedge_and_absent_discount(view: ReplayView) -> None:
    triplet = view.expiry["quoted_butterflies"]["triplets"][0]
    triplet["negative_cost_candidate"] = False
    triplet["best_package"]["futures_hedge"] = None
    triplet["best_package"]["negative_cost_candidate"] = False
    triplet["best_package"]["net_entry_cost"] = 0.4
    assert diagnostic_payload(view, "butterflies", None, None)["status"] == (
        "No quoted credit"
    )
    view.expiry["discount_factor"] = None
    result = diagnostic_payload(view, "butterflies", None, None)
    assert result["status"] == "stored result · quote comparison unavailable"
    assert result["data"]["future_input"] is None
    assert result["data"]["same_package_midpoint"]["flags"] == [
        "discount_factor_unavailable"
    ]
    triplet["best_package"] = None
    triplet["negative_cost_candidate"] = None
    result = diagnostic_payload(view, "butterflies", None, None)
    assert result["status"] == "no usable combination"
    assert result["symbols"] == ["C90", "P100", "C130"]
    assert result["data"]["same_package_midpoint"] is None


def test_missing_quote_and_unrelated_export_evidence(view: ReplayView) -> None:
    view.expiry["options"][0]["normalized_quote"] = None
    view.expiry["quoted_butterflies"]["triplets"].append(
        deepcopy(view.expiry["quoted_butterflies"]["triplets"][0])
    )
    payload = diagnostic_payload(view, "butterflies", None, "butterfly:0")
    assert "triplets" not in payload["data"]
    assert payload["data"]["same_package_midpoint"]["flags"] == ("C90:missing",)
    with pytest.raises(ValueError, match="Unknown diagnostic tab"):
        diagnostic_options(view, "wrong", None)


def test_view_does_not_mutate_source_report(view: ReplayView) -> None:
    before = deepcopy(view)
    for tab in ("diagnostics", "parity", "midpoint", "butterflies"):
        diagnostic_panel(view, tab, "C90", None)
    assert view == before


def test_midpoint_excludes_zero_iv_rejected_and_unobserved_nodes(
    view: ReplayView,
) -> None:
    rows = view.expiry["options"]
    rows[0]["sides"]["midpoint"]["iv"] = 0
    rows[1]["rejected"] = ["stale"]
    rows[2]["log_forward_moneyness"] = None
    result = diagnostic_payload(view, "midpoint", None, None)
    assert result["symbols"] == []
    assert result["data"]["value"] == -99


def test_midpoint_does_not_reuse_prior_checkpoint_maturity(view: ReplayView) -> None:
    checkpoint = {**view.checkpoint, "as_of": "2025-01-02T00:00:00Z"}
    assert (
        diagnostic_options(replace(view, checkpoint=checkpoint), "midpoint", None) == []
    )


def _visible_text(component: Any, *, expanded: bool = False) -> str:
    """Read user-facing child text, excluding closed disclosure content by default."""
    if component is None:
        return ""
    if isinstance(component, (str, int, float)):
        return str(component)
    if isinstance(component, (list, tuple)):
        return " ".join(_visible_text(child, expanded=expanded) for child in component)
    props = component.to_plotly_json()
    children = props["props"].get("children", [])
    if props["type"] == "Details" and not expanded:
        children = children[:1]
    return _visible_text(children, expanded=expanded)


def test_butterfly_primary_view_explains_cost_capacity_and_crossed_legs(
    view: ReplayView,
) -> None:
    panel = diagnostic_panel(view, "butterflies", None, None)
    text = _visible_text(panel)
    assert "90 / 100 / 130" in text
    assert "Same-package midpoint cost -17.25" in text
    assert "Quoted entry cost -15.35" in text
    assert "USDC per BTC of center option" in text
    assert "Displayed capacity 3.00" in text
    assert "BTC of center option" in text
    assert "indicative quoted credit, not an executable profit" in text
    assert "not the raw midpoint anomaly" in text
    assert "No fee estimate; no simultaneous fill" in text
    assert "Option legs · quoted entry" in text
    assert "Buy ask" not in text and "Sell bid" not in text
    assert "14.00 / 15.00 USDC/BTC 15.00 USDC/BTC 6.00 contracts" not in text
    assert "partial support" in text
    assert "Exact strike weights: 3/4 / -1 / 1/4" not in text
    expanded = _visible_text(panel, expanded=True)
    assert "Buy ask" in expanded and "Sell bid" in expanded
    assert "14.00 / 15.00 USDC/BTC 15.00 USDC/BTC 6.00 contracts" in expanded
    assert "Exact strike weights: 3/4 / -1 / 1/4" in expanded
    assert "119.00 / 121.00 USDC/BTC 119.00 USDC/BTC 3.00 BTC" in expanded
    assert "quote time 2025-01-01T00:00:00+00:00" in expanded


@pytest.mark.parametrize("has_package", [True, False])
def test_butterfly_option_legs_use_one_closed_native_disclosure(
    view: ReplayView, has_package: bool
) -> None:
    if not has_package:
        view.expiry["quoted_butterflies"]["triplets"][0]["best_package"] = None
    panel = diagnostic_panel(view, "butterflies", None, None)
    label = (
        "Option legs · quoted entry" if has_package else "Option legs · observed quotes"
    )
    disclosures = [
        child
        for child in panel.children
        if child.to_plotly_json()["type"] == "Details" and _visible_text(child) == label
    ]
    assert len(disclosures) == 1
    disclosure = disclosures[0]
    assert not disclosure.to_plotly_json()["props"].get("open", False)
    assert disclosure.children[0].to_plotly_json()["type"] == "Summary"
    assert len(disclosure.children) == 2
    table = disclosure.children[1].children
    assert table.to_plotly_json()["type"] == "Table"
    assert _visible_text(panel).count(label) == 1
    assert _visible_text(panel, expanded=True).count(label) == 1
    assert "Side size / bid-ask size" not in _visible_text(panel)
    assert "Side size / bid-ask size" in _visible_text(disclosure, expanded=True)


def test_butterfly_midpoint_credit_disappears_at_quoted_sides(view: ReplayView) -> None:
    triplet = view.expiry["quoted_butterflies"]["triplets"][0]
    triplet["negative_cost_candidate"] = False
    package = triplet["best_package"]
    package["negative_cost_candidate"] = False
    package["net_entry_cost"] = 0.4
    package["option_entry_cost"] = 17.5
    package["option_legs"][0]["price"] = 36
    view.expiry["options"][0]["normalized_quote"].update(bid=35, ask=36)
    text = _visible_text(diagnostic_panel(view, "butterflies", None, None))
    assert "No quoted credit" in text
    assert "Same-package midpoint cost -1.50" in text
    assert "Quoted entry cost 0.40" in text
    assert "midpoint credit disappears" in text


def test_stale_leg_is_readable_without_opening_raw_evidence(view: ReplayView) -> None:
    view.expiry["options"][0]["normalized_flags"] = ["stale"]
    text = _visible_text(diagnostic_panel(view, "butterflies", None, None))
    assert "Same-package midpoint cost Not available" in text
    assert "C90: stale" in text
    assert "stored quoted result is retained" in text
    assert "Quoted entry cost -15.35" in text


@pytest.mark.parametrize("tab", ["diagnostics", "parity", "midpoint", "butterflies"])
def test_primary_diagnostics_hide_nested_evidence_until_expanded(
    view: ReplayView,
    tab: str,
) -> None:
    panel = diagnostic_panel(
        view, tab, None, "summary" if tab == "diagnostics" else None
    )
    text = _visible_text(panel)
    assert "Raw diagnostic JSON" in text
    assert "normalized_quote" not in text
    assert "native_book" not in text
    assert "{" not in text and "}" not in text
    assert '"data": {' in _visible_text(panel, expanded=True)


def test_parity_labels_numeric_forward_residual_and_consistency(
    view: ReplayView,
) -> None:
    text = _visible_text(diagnostic_panel(view, "parity", None, None))
    assert "Option-implied forward 100.00 to 120.00 USDC/BTC" in text
    assert "Independently quoted future 119.00 to 121.00 USDC/BTC F" in text
    assert "Parity residual -1.00 to 1.00 USDC/BTC" in text
    assert "Consistent: residual interval includes zero" in text
    view.expiry["parity"][0]["consistent_with_future"] = None
    view.expiry["parity"][0]["independent_forward_residual"] = None
    view.expiry["matched_future"] = None
    text = _visible_text(diagnostic_panel(view, "parity", None, None))
    assert "Independently quoted future Not available" in text
    assert "Parity residual Not available" in text
    assert "Consistency unavailable" in text


def test_midpoint_names_strikes_maturity_and_non_cost_violation(
    view: ReplayView,
) -> None:
    text = _visible_text(diagnostic_panel(view, "midpoint", None, None))
    assert "Strikes: 90.00 / 100.00 / 130.00" in text
    assert "90.0 days to maturity" in text
    assert "Butterfly violation -99.000000" in text
    assert "Raw diagnostic value, not a package cost" in text
    assert "midpoint anomaly" in text


def test_quality_summary_uses_human_counts_and_flags(view: ReplayView) -> None:
    view.expiry["coverage"]["native_flag_counts"] = {"stale": 2, "sequence_gap": 1}
    view.expiry["fit"] = {"arbitrage": {"grid": [[1, 2], [3, 4]], "findings": []}}
    text = _visible_text(diagnostic_panel(view, "diagnostics", None, "summary"))
    assert "Usable books 3" in text
    assert "Requested options 3" in text
    assert "stale: 2" in text and "sequence gap: 1" in text
    assert "Fit unavailable: insufficient eligible observations" in text
    assert "Native capture: complete" in text
    assert "[1, 2]" not in text


def test_unusable_butterfly_does_not_invent_cost_or_capacity(view: ReplayView) -> None:
    triplet = view.expiry["quoted_butterflies"]["triplets"][0]
    triplet["best_package"] = None
    triplet["negative_cost_candidate"] = None
    text = _visible_text(diagnostic_panel(view, "butterflies", None, None))
    assert "No usable quoted package" in text
    assert "Same-package midpoint cost Not available" in text
    assert "Quoted entry cost Not available" in text
    assert "Displayed capacity Not available" in text


@pytest.mark.parametrize(
    "cost,candidate,status,explanation",
    [
        (
            -5e-11,
            False,
            "negative quoted cost within tolerance",
            "not a negative-cost candidate",
        ),
        (
            -1e-10,
            False,
            "negative quoted cost within tolerance",
            "not a negative-cost candidate",
        ),
        (-2e-10, True, "indicative negative cost", "exceeds the cost tolerance"),
        (0.0, False, "No quoted credit", "zero quoted entry cost"),
        (0.4, False, "No quoted credit", "positive quoted entry cost"),
        (2.4, False, "No quoted credit", "positive quoted entry cost"),
    ],
)
def test_butterfly_cost_sign_and_threshold_match_display_and_export(
    view: ReplayView,
    cost: float,
    candidate: bool,
    status: str,
    explanation: str,
) -> None:
    triplet = view.expiry["quoted_butterflies"]["triplets"][0]
    triplet["negative_cost_candidate"] = candidate
    package = triplet["best_package"]
    package.update(net_entry_cost=cost, negative_cost_candidate=candidate)
    payload = diagnostic_payload(view, "butterflies", None, None)
    data = payload["data"]
    assert payload["status"] == data["quoted_cost_status"] == status
    assert explanation in data["quoted_cost_explanation"]
    assert data["best_package"]["net_entry_cost"] == cost
    assert data["best_package"]["negative_cost_candidate"] is candidate
    assert data["same_package_midpoint"]["midpoint_cost"] == pytest.approx(cost - 1.9)
    assert data["same_package_midpoint"]["crossing_cost"] == pytest.approx(1.9)
    panel = diagnostic_panel(view, "butterflies", None, None)
    text = _visible_text(panel)
    assert status in text
    assert data["quoted_cost_explanation"] in text
    if cost < 0:
        assert f"Quoted entry cost {cost:.6g}" in text
        assert "not an executable profit" in text
        for forbidden in (
            "No quoted credit",
            "no credit remains",
            "no quoted credit remains",
            "midpoint credit disappears",
            "no negative quoted cost",
        ):
            assert forbidden not in _visible_text(panel, expanded=True)
    elif cost < 1.9:
        assert "midpoint credit disappears" in text
    else:
        assert "midpoint credit disappears" not in text


@pytest.mark.parametrize("cost", [-15.35, -5e-11])
@pytest.mark.parametrize("null_metadata", [False, True])
@pytest.mark.parametrize(
    "missing",
    ["cost_tolerance", "package_candidate", "triplet_candidate", "all"],
)
def test_butterfly_missing_threshold_metadata_keeps_bounded_credit_and_midpoint(
    view: ReplayView,
    cost: float,
    missing: str,
    null_metadata: bool,
) -> None:
    triplet = view.expiry["quoted_butterflies"]["triplets"][0]
    package = triplet["best_package"]
    package["net_entry_cost"] = cost
    for name, record, field in (
        ("cost_tolerance", package, "cost_tolerance"),
        ("package_candidate", package, "negative_cost_candidate"),
        ("triplet_candidate", triplet, "negative_cost_candidate"),
    ):
        if missing in (name, "all"):
            if null_metadata:
                record[field] = None
            else:
                del record[field]
    before = deepcopy(view)
    payload = diagnostic_payload(view, "butterflies", None, None)
    assert payload["status"] == "stored negative quoted cost · threshold unverified"
    data = payload["data"]
    assert "metadata is incomplete" in data["quoted_cost_explanation"]
    assert data["same_package_midpoint"]["midpoint_cost"] == pytest.approx(cost - 1.9)
    assert data["same_package_midpoint"]["quoted_cost"] == cost
    assert data["same_package_midpoint"]["crossing_cost"] == pytest.approx(1.9)
    assert data["same_package_midpoint"]["flags"] == ()
    assert data["best_package"] == package
    text = _visible_text(
        diagnostic_panel(view, "butterflies", None, None), expanded=True
    )
    assert "cannot be established from this record" in text
    assert "not an executable profit" in text
    assert "No quoted credit" not in text
    assert "midpoint credit disappears" not in text
    assert view == before


def test_butterfly_within_tolerance_credit_survives_unavailable_comparison(
    view: ReplayView,
) -> None:
    triplet = view.expiry["quoted_butterflies"]["triplets"][0]
    triplet["negative_cost_candidate"] = False
    triplet["best_package"].update(net_entry_cost=-5e-11, negative_cost_candidate=False)
    view.expiry["options"][0]["normalized_flags"] = ["stale"]
    payload = diagnostic_payload(view, "butterflies", None, None)
    assert payload["status"] == "stored result · quote comparison unavailable"
    assert (
        payload["data"]["quoted_cost_status"] == "negative quoted cost within tolerance"
    )
    assert payload["data"]["same_package_midpoint"]["midpoint_cost"] is None
    text = _visible_text(
        diagnostic_panel(view, "butterflies", None, None), expanded=True
    )
    assert "Quoted entry cost -5e-11" in text
    assert "not a negative-cost candidate" in text
    assert "stored quoted result is retained" in text
    assert "No quoted credit" not in text
    assert "no quoted credit remains" not in text
