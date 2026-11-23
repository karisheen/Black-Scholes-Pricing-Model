"""Synthetic ex-dividend spot paths, not observed quotes or forward-price paths.

Under the physical measure, dS/S = drift*dt + volatility*dW: ``drift`` is
ex-dividend price appreciation, NOT total return and NOT automatically r-q.
A total-return assumption must first subtract the continuous dividend yield.
Pricing/funding r and dividend yield q remain separate hedging inputs. The
risk-neutral base case sets drift=r-q explicitly; S0=F0 only if T=0 or r=q
under the continuous-yield carry relationship used by ``spot_from_forward``.

Version 1 uses explicit NumPy PCG64 standard normals in time-major C order.
Replay requires the same algorithm version, NumPy version, seed, dimensions,
and parameters; no cross-NumPy/platform bitwise guarantee is made. Regenerate
with the SAME seed and dimensions for shared shocks across volatility stresses.
Do not change the path count to extend a path-column prefix: that changes the
row stride of the random stream. A longer time grid preserves its prefix when
path count and dt are unchanged. All data produced here are synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from quant_pricing.exceptions import InputValidationError

GBM_VERSION = "synthetic-gbm-pcg64-v1"
GBM_SOURCE = "synthetic"
GBM_BIT_GENERATOR = "PCG64"


def _finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise InputValidationError(f"{name} must be a finite real number")


@dataclass(frozen=True, slots=True)
class GBMConfig:
    """Uniform elapsed ACT/365F years; volatility is absolute annualized sigma.

    ``spot`` and ``horizon_years`` are positive, ``volatility`` is nonnegative,
    ``drift`` is any finite physical ex-dividend drift. ``steps`` and ``paths``
    are positive Python integers; ``seed`` is an explicit nonnegative integer.
    """

    spot: float
    drift: float
    volatility: float
    horizon_years: float
    steps: int
    paths: int
    seed: int

    def __post_init__(self) -> None:
        for name in ("spot", "drift", "volatility", "horizon_years"):
            _finite(name, getattr(self, name))
        if self.spot <= 0 or self.horizon_years <= 0:
            raise InputValidationError("spot and horizon_years must be positive")
        if self.volatility < 0:
            raise InputValidationError("volatility must be nonnegative")
        for name in ("steps", "paths", "seed"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise InputValidationError(f"{name} must be an integer >= {minimum}")


def generate_gbm(config: GBMConfig) -> NDArray[np.float64]:
    """Return positive finite spots of shape ``(steps+1, paths)``, row zero S0.

    Exact log transitions use dt=T/steps and independent standard normals:
    log(S[t+dt]/S[t]) = (drift - sigma**2/2)*dt + sigma*sqrt(dt)*Z.
    This is not Euler discretization. The explicit local generator never
    consumes global RNG state. Numeric overflow/underflow is an error, never
    clipped or filled; even finite but unrepresentable configurations can fail.
    """
    try:
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            dt = np.float64(config.horizon_years) / config.steps
            if not np.isfinite(dt) or dt <= 0:
                raise InputValidationError("GBM time step must be positive and finite")
            diffusion = np.float64(config.volatility) * np.sqrt(dt)
            drift_step = np.float64(config.drift) * dt - 0.5 * diffusion * diffusion
            if not np.isfinite(diffusion) or not np.isfinite(drift_step):
                raise InputValidationError("GBM transition must be finite")
            result = np.empty((config.steps + 1, config.paths), dtype=np.float64)
            result[0] = config.spot
            log_prices = result[1:]
            rng = np.random.Generator(np.random.PCG64(config.seed))
            rng.standard_normal(log_prices.shape, dtype=np.float64, out=log_prices)
            log_prices *= diffusion
            log_prices += drift_step
            np.cumsum(log_prices, axis=0, out=log_prices)
            log_prices += log(config.spot)
            np.exp(log_prices, out=log_prices)
    except (FloatingPointError, OverflowError) as exc:
        raise InputValidationError("GBM numeric overflow or underflow") from exc
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise InputValidationError("GBM spots must remain positive and finite")
    return result


def spot_from_forward(
    forward: float, time: float, rate: float, dividend_yield: float
) -> float:
    """Invert F=S*exp((r-q)*T) for deterministic continuous rates/yields.

    ``time`` is nonnegative ACT/365F years. ``forward`` and returned spot are
    positive finite prices in the same units. This is a synthetic carry input,
    not a live quote conversion, futures convexity adjustment, discrete cash
    dividend model, or assumption about the physical GBM drift. The initial
    spot equals the forward at T=0 or r=q; otherwise do not use F as S0.
    """
    for name, value in (
        ("forward", forward),
        ("time", time),
        ("rate", rate),
        ("dividend_yield", dividend_yield),
    ):
        _finite(name, value)
    if forward <= 0 or time < 0:
        raise InputValidationError("forward must be positive and time nonnegative")
    if time == 0 or rate == dividend_yield:
        return float(forward)
    try:
        with np.errstate(over="raise", under="raise", invalid="raise"):
            carry = (np.float64(rate) - np.float64(dividend_yield)) * time
            spot = float(np.exp(log(forward) - carry))
    except (FloatingPointError, OverflowError) as exc:
        raise InputValidationError(
            "spot conversion numeric overflow or underflow"
        ) from exc
    if not isfinite(spot) or spot <= 0:
        raise InputValidationError("converted spot must be positive and finite")
    return spot
