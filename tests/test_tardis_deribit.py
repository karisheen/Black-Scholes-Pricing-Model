"""Consumer-visible book reconstruction, availability and data-quality checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from quant_pricing.tardis_deribit import main, replay

A = "SYNTH-TEST-C"
B = "SYNTH-TEST-P"
EPOCH_MS = 1_704_067_200_000


def stamp(ms: int) -> str:
    return f"2024-01-01T00:00:00.{ms:03d}Z"


def frame(
    ms: int,
    change_id: int,
    *,
    prev: int | None = None,
    symbol: str = A,
    exchange_ms: int | None = None,
    bids: list[list[Any]] | None = None,
    asks: list[list[Any]] | None = None,
) -> str:
    data = {
        "instrument_name": symbol,
        "type": "snapshot" if prev is None else "change",
        "change_id": change_id,
        "prev_change_id": prev,
        "timestamp": EPOCH_MS + (ms if exchange_ms is None else exchange_ms),
        "bids": bids if bids is not None else [["new", 10, 1]],
        "asks": asks if asks is not None else [["new", 12, 2]],
    }
    payload = {
        "method": "subscription",
        "params": {
            "channel": f"book.{symbol}.raw",
            "data": data,
        },
    }
    return f"{stamp(ms)} {json.dumps(payload)}\n"


def books(report: dict[str, Any]) -> dict[str, Any]:
    return report["books"]


def test_replay_updates_and_deletes_levels_without_using_later_receipts() -> None:
    tape = (
        frame(100, 10, bids=[["new", 9, 4], ["new", 10, 1]]),
        frame(200, 20, prev=10, bids=[["new", 11, 2]], asks=[]),
        frame(
            300,
            42,
            prev=20,
            exchange_ms=150,
            bids=[
                ["delete", 11, 0],
                ["change", 10, 3],
            ],
            asks=[["change", 12, 5]],
        ),
    )
    first, second, third = replay(tape, [A], [stamp(100), stamp(299), stamp(300)])
    assert books(first)[A]["bid"] == ["10", "1"]
    assert books(second)[A]["bid"] == ["11", "2"]
    assert books(third)[A]["bid"] == ["10", "3"]
    assert books(third)[A]["ask"] == ["12", "5"]
    assert books(third)[A]["last_update"] == {
        "line": 3,
        "available_at": "2024-01-01T00:00:00.3Z",
        "exchange_time": "2024-01-01T00:00:00.15Z",
        "change_id": 42,
    }


def test_missing_update_invalidates_only_its_symbol_until_fresh_snapshot() -> None:
    complete = (
        frame(100, 100),
        frame(110, 5, symbol=B),
        frame(120, 167, prev=100, bids=[["new", 11, 2]], asks=[]),
        frame(130, 250, prev=167, bids=[["change", 11, 3]], asks=[]),
        frame(140, 300, prev=250, bids=[["change", 11, 4]], asks=[]),
        frame(180, 401, bids=[["new", 8, 5]]),
    )
    original = replay(complete, [A, B], [stamp(140)])[0]
    gap, ignored, recovered = replay(
        (*complete[:2], *complete[3:]), [A, B], [stamp(130), stamp(140), stamp(180)]
    )
    assert books(original)[A]["bid"] == ["11", "4"]
    assert not gap["complete"] and not ignored["complete"]
    assert books(gap)[A]["flags"] == ["sequence_gap"]
    assert books(gap)[A]["issue"]["prev_change_id"] == 167
    assert books(gap)[A]["issue"]["last_applied_change_id"] == 100
    assert books(ignored)[A]["bid"] is None
    assert books(ignored)[B]["bid"] == ["10", "1"]
    assert recovered["complete"] and books(recovered)[A]["bid"] == ["8", "5"]
    assert books(recovered)[A]["sequence_gaps"] == 1


def test_disconnect_interval_is_conservative_and_recovers_per_symbol() -> None:
    tape = (
        frame(100, 1),
        frame(110, 2, symbol=B).replace(".110Z", ".1100001Z"),
        "\n",
        frame(150, 20, prev=1),
        frame(160, 22),
        frame(170, 27, symbol=B),
    )
    before, outage, one, both = replay(
        tape,
        [A, B],
        [
            "2024-01-01T00:00:00.1100001Z",
            "2024-01-01T00:00:00.110000101Z",
            stamp(160),
            stamp(170),
        ],
    )
    assert before["complete"]
    assert not outage["complete"]
    assert all(book["flags"] == ["disconnect"] for book in books(outage).values())
    assert books(outage)[A]["last_update"] is None
    assert books(one)[A]["flags"] == [] and books(one)[B]["flags"] == ["disconnect"]
    assert both["complete"]


def test_receipt_ties_use_wire_order_and_future_exchange_clock_blocks_state() -> None:
    tape = (
        frame(100, 9, exchange_ms=180),
        frame(100, 14, prev=9, exchange_ms=90, bids=[["new", 11, 2]], asks=[]),
        frame(200, 16, prev=14, bids=[], asks=[]),
    )
    before, early, known = replay(tape, [A], [stamp(99), stamp(100), stamp(180)])
    assert books(before)[A]["flags"] == ["awaiting_snapshot"]
    assert books(early)[A]["flags"] == ["future_exchange_time"]
    assert books(early)[A]["bid"] is None
    assert books(known)[A]["last_update"]["change_id"] == 14
    assert books(known)[A]["bid"] == ["11", "2"]


def test_empty_crossed_and_stale_books_do_not_count_as_complete() -> None:
    symbol = "UNSEEN"
    initial, crossed, stale = replay(
        (
            frame(100, 1, asks=[]),
            frame(120, 3, prev=1, asks=[["new", 9, 2]]),
            frame(200, 9),
        ),
        [A, symbol],
        [stamp(100), stamp(120), stamp(150)],
        max_age_ms=20,
    )
    assert books(initial)[A]["flags"] == ["one_sided"]
    assert books(initial)[A]["ask"] is None
    assert books(crossed)[A]["flags"] == ["crossed"]
    assert books(stale)[A]["flags"] == ["stale", "crossed"]
    assert books(stale)[symbol]["flags"] == ["awaiting_snapshot"]
    assert not stale["complete"]


@pytest.mark.parametrize("trailer", [(), ("\n",)])
def test_query_past_last_dated_frame_fails(trailer: tuple[str, ...]) -> None:
    tape = (frame(100, 1), *trailer)
    assert replay(tape, [A], [stamp(100)])[0]["complete"]
    with pytest.raises(ValueError, match="capture coverage"):
        replay(tape, [A], [stamp(101)])


def test_cli_reproduces_exact_decimal_strings_and_capture_provenance(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture = tmp_path / "synthetic.ndjson"
    capture.write_text(
        frame(100, 17).replace("10, 1", "0.100000000000000001, 1"), encoding="utf-8"
    )
    args = [str(capture), "--symbol", A, "--as-of", stamp(100)]
    assert main(args) == 0
    raw = capsys.readouterr().out
    report = json.loads(raw)
    assert report["source"] == "tardis/deribit"
    assert books(report["snapshots"][0])[A]["bid"] == ["0.100000000000000001", "1"]
    assert main(args) == 0
    assert capsys.readouterr().out == raw


@pytest.mark.parametrize(
    ("tape", "message"),
    [
        ((frame(150, 1), frame(100, 2)), "out-of-order"),
        (
            (
                frame(100, 1).replace(
                    '"instrument_name": "SYNTH-TEST-C"', '"instrument_name": "wrong"'
                ),
            ),
            "disagree",
        ),
        ((frame(100, 1, bids=[["delete", 10, 0]]),), "snapshot levels"),
        ((frame(100, 1, bids=[["new", 10, 0]]),), "zero amount"),
        ((frame(100, 1, bids=[["new", 10, float("inf")]]),), "invalid book"),
        ((frame(100, 1, bids=[["unknown", 10, 4]]),), "unknown book action"),
        ((frame(100, 1, bids=[["new", "10", 4]]),), "JSON numbers"),
    ],
)
def test_bad_capture_is_rejected_with_exact_line(
    tape: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=f"capture line [12]: .*{message}"):
        replay(tape, [A], [stamp(150)])
