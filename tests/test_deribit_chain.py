"""Synthetic capture tests for the real-market normalization boundary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from quant_pricing.black76 import price
from quant_pricing.deribit_chain import (
    main,
    plan_chain,
    replay_chain,
    replay_selected_chain,
)

_DAY = "2026-10-01"
_BASE = datetime(2026, 10, 1, tzinfo=timezone.utc)
_EXCHANGE_MS = int(_BASE.timestamp() * 1000)
_EXPIRIES = ("30OCT26", "27NOV26")
_STRIKES = (80, 90, 100, 110, 120, 130)
_TAIL = f'{_DAY}T00:00:10Z {{"id":1,"result":[]}}\n'


def _snapshot(
    symbol: str, bid: float, ask: float, *, receipt: str = "00:00:00.0000001Z"
) -> str:
    message = {
        "method": "subscription",
        "params": {
            "channel": f"book.{symbol}.raw",
            "data": {
                "instrument_name": symbol,
                "type": "snapshot",
                "timestamp": _EXCHANGE_MS,
                "change_id": 42,
                "bids": [["new", bid, 0.25]],
                "asks": [["new", ask, 0.33]],
            },
        },
    }
    return f"{_DAY}T{receipt} {json.dumps(message)}\n"


def _option_tape(
    code: str, strikes: tuple[int, ...], *, forward: float, month: int, day: int
) -> tuple[str, ...]:
    t = (datetime(2026, month, day, 8, tzinfo=timezone.utc) - _BASE).total_seconds()
    t /= 365 * 86400
    tape = []
    for strike in strikes:
        for suffix, kind in (("C", "call"), ("P", "put")):
            premium = float(price(kind, forward, strike, t, 1.0, 0.4))
            bound = max(forward - strike if kind == "call" else strike - forward, 0)
            half = min(0.01 * premium, 0.25 * (premium - bound))
            tape.append(
                _snapshot(
                    f"BTC_USDC-{code}-{strike}-{suffix}", premium - half, premium + half
                )
            )
    return tuple(tape)


def _options(code: str, strikes: tuple[int, ...] = _STRIKES) -> tuple[str, ...]:
    if code == _EXPIRIES[0]:
        return _option_tape(code, strikes, forward=100, month=10, day=30)
    return _option_tape(code, strikes, forward=102, month=11, day=27)


def _reports(
    lines: tuple[str, ...],
    cutoffs: tuple[str, ...],
    *,
    expiries: tuple[str, ...] = _EXPIRIES,
    max_age_ms: int = 30_000,
) -> list[dict[str, Any]]:
    report: dict[str, Any] = replay_chain(
        (*lines, _TAIL), expiries, _STRIKES, cutoffs, rate=0, max_age_ms=max_age_ms
    )
    return report["checkpoints"]


def test_same_expiry_observed_futures_drive_iv_and_parity() -> None:
    tapes = (
        *_options(_EXPIRIES[0]),
        *_options(_EXPIRIES[1]),
        _snapshot("BTC_USDC-30OCT26", 99.99, 100.01),
        _snapshot("BTC_USDC-27NOV26", 101.98, 102.02),
    )
    report = _reports(tapes, (f"{_DAY}T00:00:00.000001Z",))[0]
    assert report["joint_fit_error"] is None
    assert len(report["joint_fitted_expiries"]) == 2
    for expiry, forward in zip(report["expiries"], (100, 102), strict=True):
        assert expiry["forward_input"] == pytest.approx(forward)
        assert expiry["discount_factor"] == 1.0
        assert expiry["coverage"]["iv_successes"] == {
            "bid": 12,
            "midpoint": 12,
            "ask": 12,
        }
        assert expiry["coverage"]["parity_pairs_consistent"] == 6
        assert expiry["coverage"]["eligible_otm_strikes"] == 6
        assert expiry["fit"]["weighted_rmse"] < 1e-7
        assert sum(row["used_in_svi"] for row in expiry["options"]) == 6
        row = expiry["options"][0]
        assert row["sides"]["midpoint"]["iv"] == pytest.approx(0.4, abs=1e-6)
        assert row["bid_ask_iv_width"] > 0
        assert row["normalized_quote"]["bid_size"] == 0.25
        assert row["normalized_quote"]["price_unit"] == "USDC/BTC"
        assert row["normalized_quote"]["size_unit"] == "contracts"
        assert (
            row["normalized_quote"]["available_at"]
            == "2026-10-01T00:00:00.000001+00:00"
        )
        assert row["native_book"]["last_update"]["available_at"].endswith(".0000001Z")
        assert expiry["matched_future"]["normalized_quote"]["size_unit"] == "BTC"
        assert all(pair["implied_forward"] is not None for pair in expiry["parity"])


def test_missing_or_late_futures_do_not_get_marks_or_parity_fallback() -> None:
    tapes = (
        *_options(_EXPIRIES[0]),
        *_options(_EXPIRIES[1]),
        _snapshot("BTC_USDC-30OCT26", 99.9, 100.1, receipt="00:00:02Z"),
    )
    early, late = _reports(tapes, (f"{_DAY}T00:00:01Z", f"{_DAY}T00:00:02Z"))
    for expiry in early["expiries"]:
        assert expiry["iv_blocked"] == "matched_future_unavailable"
        assert expiry["options"][0]["sides"] == {}
        pair = expiry["parity"][0]
        assert pair["implied_forward"] is not None
        assert pair["consistent_with_future"] is None
    assert late["expiries"][0]["fit"] is not None
    assert late["expiries"][1]["forward_input"] is None
    assert late["joint_fitted_arbitrage"] is None


def test_sparse_expiry_and_stale_books_do_not_destroy_good_slice() -> None:
    tapes = (
        *_options(_EXPIRIES[0]),
        *_options(_EXPIRIES[1], _STRIKES[:3]),
        _snapshot("BTC_USDC-30OCT26", 99.9, 100.1),
        _snapshot("BTC_USDC-27NOV26", 101.99, 102.01),
    )
    now, stale = _reports(
        tapes, (f"{_DAY}T00:00:01Z", f"{_DAY}T00:00:10Z"), max_age_ms=1500
    )
    first, second = now["expiries"]
    assert first["fit"] is not None
    assert second["fit"] is None and "five liquid unique strikes" in second["fit_error"]
    assert second["coverage"]["usable_books"] == 6
    assert second["coverage"]["native_flag_counts"] == {"awaiting_snapshot": 6}
    assert stale["expiries"][0]["coverage"]["usable_books"] == 0
    assert stale["expiries"][0]["options"][0]["rejected"] == ("stale",)
    assert stale["expiries"][0]["options"][0]["normalized_quote"] is not None


def test_settlement_window_and_expired_book_are_not_priced() -> None:
    symbol = "BTC_USDC-1OCT26"
    records = tuple(_snapshot(s, 98, 100) for s in (symbol, f"{symbol}-100-C"))
    tail = f'{_DAY}T08:00:01Z {{"id":1,"result":[]}}\n'
    data: dict[str, Any] = replay_chain(
        (*records, tail),
        ["01OCT26"],
        [100],
        [f"{_DAY}T07:30:00Z", f"{_DAY}T08:00:00Z"],
        rate=0,
    )
    during, expired = (s["expiries"][0] for s in data["checkpoints"])
    assert during["iv_blocked"] == "settlement_window" and during["parity"] == []
    assert expired["iv_blocked"] == "expired" and expired["discount_factor"] is None


def test_report_scopes_one_sided_solver_failure(tmp_path: Path, capsys: Any) -> None:
    capture = tmp_path / "test.ndjson"
    capture.write_text(
        _snapshot("BTC_USDC-30OCT26", 99.99, 100.01)
        + _snapshot("BTC_USDC-30OCT26-100-C", 5, 101)
        + _TAIL,
        encoding="utf-8",
    )
    args = [
        str(capture),
        "--expiry",
        "30OCT26",
        "--strike",
        "100",
        "--rate",
        "0",
        "--as-of",
        f"{_DAY}T00:00:01Z",
    ]
    assert main(args) == 0
    result = capsys.readouterr().out
    first = json.loads(result)["checkpoints"][0]["expiries"][0]["options"][0]
    assert first["sides"]["bid"]["iv"] > 0
    assert first["sides"]["ask"]["error"].startswith("NoArbitrageError:")
    assert first["bid_ask_iv_width"] is None
    assert main(args) == 0
    assert capsys.readouterr().out == result


def _catalog_bytes() -> bytes:
    rows = [
        {
            "exchange": "deribit",
            "id": f"BTC_USDC-30OCT26-{strike}-{side}",
            "baseCurrency": "BTC",
            "quoteCurrency": "USDC",
            "type": "option",
            "availableSince": "2026-09-30T00:00:00.000Z",
        }
        for strike in _STRIKES
        for side in "CP"
    ]
    return json.dumps({"symbols": rows, "nextCursor": None}).encode()


def test_catalog_selection_is_frozen_before_option_books_are_seen(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    future = _snapshot("BTC_USDC-30OCT26", 99.9, 100.1)
    capture, catalog, plan = (
        tmp_path / filename
        for filename in ("capture.ndjson", "catalog.json", "plan.json")
    )
    capture.write_text(future + _TAIL, encoding="utf-8")
    catalog.write_bytes(_catalog_bytes())
    selection_args = [
        str(capture),
        "--catalog",
        str(catalog),
        "--expiry",
        "30OCT26",
        "--strikes-per-expiry",
        "3",
        "--as-of",
        f"{_DAY}T00:00:01Z",
    ]
    assert main(selection_args) == 0
    selection_json = capsys.readouterr().out
    plan.write_text(selection_json, encoding="utf-8")
    selection = json.loads(selection_json)
    assert selection["strips"][0]["strikes"] == [90, 100, 110]
    assert selection["tardis_filter"][0]["symbols"] == [
        "BTC_USDC-30OCT26",
        *(
            f"BTC_USDC-30OCT26-{strike}-{side}"
            for strike in (90, 100, 110)
            for side in "CP"
        ),
    ]

    # Both center legs are absent. A valid, unselected strike cannot replace them.
    options = _options("30OCT26", (90, 110, 120))
    capture.write_text(future + "".join(options) + _TAIL, encoding="utf-8")
    report_args = [
        str(capture),
        "--selection",
        str(plan),
        "--rate",
        "0",
        "--as-of",
        f"{_DAY}T00:00:02Z",
    ]
    assert main(report_args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["catalog_selection"] == selection
    assert len(output["checkpoints"]) == 1
    expiry = output["checkpoints"][0]["expiries"][0]
    assert {o["strike"] for o in expiry["options"]} == {90, 100, 110}
    assert expiry["coverage"]["two_sided_iv"] == 4
    assert expiry["quoted_butterflies"]["triplets_checked"] == 0
    assert expiry["quoted_butterflies"]["triplets"][0]["best_package"] is None


def test_catalog_replay_rejects_lookahead_and_a_different_futures_anchor() -> None:
    old = _snapshot("BTC_USDC-30OCT26", 99.9, 100.1)
    plan = plan_chain((old, _TAIL), _catalog_bytes(), ["30OCT26"], f"{_DAY}T00:00:01Z")
    with pytest.raises(ValueError, match="precedes catalog selection"):
        replay_selected_chain((old, _TAIL), plan, [f"{_DAY}T00:00:00Z"], rate=0)
    revised = _snapshot("BTC_USDC-30OCT26", 99.9, 100.2)
    with pytest.raises(ValueError, match="disagrees with selection anchor"):
        replay_selected_chain((revised, _TAIL), plan, [f"{_DAY}T00:00:01Z"], rate=0)


def test_quoted_butterfly_report_flags_a_negative_entry_cost() -> None:
    prefix = "BTC_USDC-30OCT26"
    lines = (
        _snapshot(prefix, 99.9, 100.1),
        _snapshot(f"{prefix}-90-C", 11, 12),
        _snapshot(f"{prefix}-100-C", 25, 26),
        _snapshot(f"{prefix}-110-C", 2, 3),
        _TAIL,
    )
    data: dict[str, Any] = replay_chain(
        lines, ["30OCT26"], [90, 100, 110], [f"{_DAY}T00:00:01Z"], rate=0
    )
    expiry = data["checkpoints"][0]["expiries"][0]
    assert expiry["fit"] is None
    checks = expiry["quoted_butterflies"]
    assert checks["candidate_count"] == 1
    assert checks["triplets"][0]["best_package"]["net_entry_cost"] == -17.5
