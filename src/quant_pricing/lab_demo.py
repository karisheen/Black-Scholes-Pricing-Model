"""Public books replayed through production IV recovery and SVI calibration.

The BTC-like smile comes from analytically supported SSVI truth. Bid/ask
volatility bands widen away from ATM; the replay still recovers every IV from
generated premiums. Truth guarantees do not certify the observed or fitted
surface, especially at checkpoints with deliberately broken or stale books.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from math import exp, log
from typing import Any

from quant_pricing.black76 import price
from quant_pricing.deribit_chain import replay_chain
from quant_pricing.synthetic_chain import SSVITruth

DEMO_SOURCE = "public synthetic / generated books"
DEMO_SESSION = "Public synthetic demo"
_BASE = datetime(2030, 1, 2, tzinfo=timezone.utc)
_STRIKES = tuple(range(60_000, 105_001, 5_000))
_EXPIRIES = (("25JAN30", 1, 25, 82_000.0), ("22FEB30", 2, 22, 83_500.0))
_RATE = 0.025
# One analytically supported surface for both expiries; BTC-like ATM IV and skew.
_TRUTH = SSVITruth(max_time=0.2, variance_rate=0.58**2, rho=-0.22, phi=5.0)


def _frame(symbol: str, bid: float, ask: float, second: int) -> str:
    stamp = _BASE + timedelta(seconds=second)
    data = {
        "instrument_name": symbol,
        "type": "snapshot",
        "timestamp": int(stamp.timestamp() * 1000),
        "change_id": second + 1,
        "bids": [["new", bid, 2.0]],
        "asks": [["new", ask, 3.0]],
    }
    message = {
        "method": "subscription",
        "params": {"channel": f"book.{symbol}.raw", "data": data},
    }
    return f"{stamp.strftime('%Y-%m-%dT%H:%M:%SZ')} {json.dumps(message)}\n"


def _books(second: int) -> list[str]:
    frames = []
    for index, (code, month, day, initial_forward) in enumerate(_EXPIRIES):
        forward = initial_forward + second * 12.5
        expiry = datetime(2030, month, day, 8, tzinfo=timezone.utc)
        maturity = (expiry - _BASE - timedelta(seconds=second)).total_seconds()
        maturity /= 365 * 86_400
        discount = exp(-_RATE * maturity)
        frames.append(_frame(f"BTC_USDC-{code}", forward - 5, forward + 5, second))
        for strike in _STRIKES:
            moneyness = log(strike / forward)
            sigma = _TRUTH.volatility(moneyness, maturity)
            # Quote in volatility space: a visible, wider-wing observation band,
            # not a constant premium percentage that collapses near intrinsic.
            half_width = 0.008 + 0.012 * abs(moneyness)
            for suffix, kind in (("C", "call"), ("P", "put")):
                symbol = f"BTC_USDC-{code}-{strike}-{suffix}"
                if second and index == 0 and strike == _STRIKES[0] and suffix == "P":
                    # A genuine broken sequence, not an injected report flag.
                    message = json.loads(_frame(symbol, 1, 2, second).split(" ", 1)[1])
                    message["params"]["data"].update(
                        type="change", prev_change_id=999, change_id=1000
                    )
                    stamp = (_BASE + timedelta(seconds=second)).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
                    frames.append(f"{stamp} {json.dumps(message)}\n")
                    continue
                bid = float(
                    price(kind, forward, strike, maturity, discount, sigma - half_width)
                )
                ask = float(
                    price(kind, forward, strike, maturity, discount, sigma + half_width)
                )
                if second and index == 0 and strike == _STRIKES[-1] and suffix == "C":
                    # Two-sided positive book, but outside Black price bounds.
                    bid, ask = forward * 1.1, forward * 1.2
                frames.append(_frame(symbol, bid, ask, second))
    return frames


@lru_cache(maxsize=1)
def _demo_json() -> str:
    lines = [*_books(0), *_books(8)]
    lines.append('2030-01-02T00:00:50Z {"id":1,"result":[]}\n')
    report = replay_chain(
        lines,
        tuple(item[0] for item in _EXPIRIES),
        _STRIKES,
        tuple(f"2030-01-02T00:00:{second:02d}Z" for second in (1, 10, 45)),
        rate=_RATE,
    )
    # Keep precisely the producer schema, replacing historical provenance text.
    report["source"] = DEMO_SOURCE
    report["scope"] = "Public synthetic BTC/USDC scenario; no historical observations"
    assumptions = report["assumptions"]
    assert isinstance(assumptions, dict)
    assumptions["availability"] = "Generated synthetic exchange and receipt timestamps"
    assumptions["forward_input"] = "Generated matched dated futures BBO midpoint"
    return json.dumps(report, allow_nan=False)


def demo_report() -> dict[str, Any]:
    """Return isolated JSON data; cache only immutable generated serialization."""
    report: dict[str, Any] = json.loads(_demo_json())
    return report
