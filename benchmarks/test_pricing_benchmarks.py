"""Reproducible microbenchmarks for scalar and vectorized pricing."""

from __future__ import annotations

import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture

from quant_pricing import price

SIZE = 10_000
SPOTS = np.linspace(60.0, 140.0, SIZE)
STRIKES = np.linspace(80.0, 120.0, SIZE)
TIMES = np.linspace(1.0 / 365.0, 2.0, SIZE)
VOLATILITIES = np.linspace(0.1, 0.6, SIZE)


def _scalar_loop() -> np.ndarray:
    return np.fromiter(
        (
            price("call", spot, strike, time, 0.03, volatility, 0.01)
            for spot, strike, time, volatility in zip(
                SPOTS, STRIKES, TIMES, VOLATILITIES, strict=True
            )
        ),
        dtype=np.float64,
        count=SIZE,
    )


def _vectorized() -> np.ndarray:
    values = price(
        "call",
        SPOTS,
        STRIKES,
        TIMES,
        0.03,
        VOLATILITIES,
        0.01,
    )
    if not isinstance(values, np.ndarray):
        raise TypeError("vectorized benchmark produced a scalar")
    return values


def test_batch_implementations_agree() -> None:
    np.testing.assert_allclose(_vectorized(), _scalar_loop(), rtol=2e-14, atol=2e-14)


def test_scalar_loop_benchmark(benchmark: BenchmarkFixture) -> None:
    benchmark.group = "10k-call-batch"
    result = benchmark.pedantic(_scalar_loop, rounds=5, iterations=1)
    assert result.shape == (SIZE,)


def test_vectorized_benchmark(benchmark: BenchmarkFixture) -> None:
    benchmark.group = "10k-call-batch"
    result = benchmark.pedantic(_vectorized, rounds=20, iterations=1)
    assert result.shape == (SIZE,)
