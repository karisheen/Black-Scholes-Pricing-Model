"""Backward-compatible functions and command-line entry point.

New code should import from :mod:`quant_pricing`. The legacy function names are
kept so existing notebooks continue to work.
"""

from __future__ import annotations

from quant_pricing import d1_d2, price
from quant_pricing.cli import main


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Return the legacy scalar ``d1`` term."""

    first, _ = d1_d2(S, K, T, r, sigma)
    if not isinstance(first, float):
        raise TypeError("legacy d1 accepts scalar inputs only")
    return first


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Return the legacy scalar ``d2`` term."""

    _, second = d1_d2(S, K, T, r, sigma)
    if not isinstance(second, float):
        raise TypeError("legacy d2 accepts scalar inputs only")
    return second


def bs_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """Return a scalar European call value."""

    value = price("call", S, K, T, r, sigma, q)
    if not isinstance(value, float):
        raise TypeError("legacy bs_call accepts scalar inputs only")
    return value


def bs_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """Return a scalar European put value."""

    value = price("put", S, K, T, r, sigma, q)
    if not isinstance(value, float):
        raise TypeError("legacy bs_put accepts scalar inputs only")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
