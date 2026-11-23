from dataclasses import replace
from math import exp, log, sqrt

import numpy as np
import pytest

from quant_pricing.exceptions import InputValidationError
from quant_pricing.hedge_paths import (
    GBM_BIT_GENERATOR,
    GBM_SOURCE,
    GBM_VERSION,
    GBMConfig,
    generate_gbm,
    spot_from_forward,
)

BASE = GBMConfig(
    spot=100.0,
    drift=0.06,
    volatility=0.2,
    horizon_years=0.5,
    steps=8,
    paths=4,
    seed=42,
)


def test_shape_source_and_exact_seed_replay() -> None:
    first = generate_gbm(BASE)
    np.testing.assert_array_equal(first, generate_gbm(BASE))
    assert first.shape == (9, 4)
    assert first.dtype == np.float64
    np.testing.assert_array_equal(first[0], np.full(BASE.paths, BASE.spot))
    assert np.isfinite(first).all()
    assert (first > 0).all()
    assert not np.array_equal(first, generate_gbm(replace(BASE, seed=43)))
    assert GBM_SOURCE == "synthetic"
    assert GBM_VERSION == "synthetic-gbm-pcg64-v1"
    assert GBM_BIT_GENERATOR == "PCG64"


def test_time_prefix_uses_same_dt_and_path_count() -> None:
    longer = replace(BASE, horizon_years=1.0, steps=16)
    np.testing.assert_array_equal(generate_gbm(BASE), generate_gbm(longer)[:9])


def test_pcg64_known_first_step_and_independent_log_transitions() -> None:
    paths = generate_gbm(BASE)
    dt = BASE.horizon_years / BASE.steps
    normal = np.random.Generator(np.random.PCG64(BASE.seed)).standard_normal(
        (BASE.steps, BASE.paths)
    )
    # A fixed, externally inspectable PCG64 prefix also catches RNG substitution.
    np.testing.assert_array_equal(
        normal[0],
        [
            0.30471707975443135,
            -1.0399841062404955,
            0.7504511958064572,
            0.9405647163912139,
        ],
    )
    expected = (BASE.drift - BASE.volatility**2 / 2) * dt
    expected = expected + BASE.volatility * sqrt(dt) * normal
    np.testing.assert_allclose(np.log(paths[1:] / paths[:-1]), expected, atol=2e-15)
    # Each path and each time interval receive their own innovation, not a
    # broadcast one-factor shock or a reused per-path normal.
    assert not np.array_equal(normal[:, 0], normal[:, 1])
    assert not np.array_equal(normal[0], normal[1])


def test_shared_shocks_recovered_across_volatility_stresses() -> None:
    recovered = []
    dt = BASE.horizon_years / BASE.steps
    for sigma in (0.12, 0.20, 0.32):
        paths = generate_gbm(replace(BASE, volatility=sigma))
        recovered.append(
            (np.log(paths[1:] / paths[:-1]) - (BASE.drift - sigma**2 / 2) * dt)
            / (sigma * sqrt(dt))
        )
    for shocks in recovered[1:]:
        np.testing.assert_allclose(shocks, recovered[0], atol=8e-14, rtol=1e-12)


@pytest.mark.parametrize("drift", [0.0, -0.08, 0.12])
def test_zero_volatility_is_deterministic_ex_dividend_growth(drift: float) -> None:
    config = replace(BASE, volatility=0.0, drift=drift)
    paths = generate_gbm(config)
    times = np.arange(config.steps + 1) * config.horizon_years / config.steps
    expected = config.spot * np.exp(drift * times)
    np.testing.assert_allclose(paths, np.broadcast_to(expected[:, None], paths.shape))
    np.testing.assert_array_equal(paths, generate_gbm(replace(config, seed=100)))


def test_terminal_price_moments_match_physical_drift_not_log_drift() -> None:
    config = replace(BASE, steps=2, paths=100_000, horizon_years=1.0)
    terminal = generate_gbm(config)[-1]
    mean = config.spot * exp(config.drift * config.horizon_years)
    variance = mean**2 * (exp(config.volatility**2 * config.horizon_years) - 1)
    assert abs(float(terminal.mean()) - mean) < 6 * sqrt(variance / config.paths)
    assert float(terminal.var(ddof=1)) == pytest.approx(variance, rel=0.035)


def test_innovations_have_independent_time_and_path_moments() -> None:
    config = replace(BASE, steps=4, paths=30_000)
    paths = generate_gbm(config)
    dt = config.horizon_years / config.steps
    shocks = (
        np.log(paths[1:] / paths[:-1]) - (config.drift - config.volatility**2 / 2) * dt
    ) / (config.volatility * sqrt(dt))
    assert abs(float(shocks.mean())) < 0.015
    assert float(shocks.var()) == pytest.approx(1.0, abs=0.025)
    assert np.max(np.abs(np.corrcoef(shocks) - np.eye(config.steps))) < 0.025
    adjacent = np.corrcoef(shocks[:, ::2].ravel(), shocks[:, 1::2].ravel())[0, 1]
    assert abs(float(adjacent)) < 0.025


@pytest.mark.parametrize("name", ["spot", "drift", "volatility", "horizon_years"])
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), True, "1"]
)
def test_invalid_real_config_fields(name: str, value: object) -> None:
    with pytest.raises(InputValidationError, match="finite real"):
        replace(BASE, **{name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("spot", 0.0),
        ("spot", -1.0),
        ("horizon_years", 0.0),
        ("horizon_years", -1.0),
        ("volatility", -0.1),
        ("steps", 0),
        ("steps", -1),
        ("steps", 1.5),
        ("steps", True),
        ("paths", 0),
        ("paths", -1),
        ("paths", 2.0),
        ("paths", False),
        ("seed", -1),
        ("seed", 1.0),
        ("seed", True),
    ],
)
def test_invalid_config_domains(name: str, value: object) -> None:
    with pytest.raises(InputValidationError):
        replace(BASE, **{name: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"drift": 1000.0},
        {"drift": -1000.0},
        {"spot": 1e308, "drift": 10.0},
        {"spot": 1e-300, "drift": -100.0},
        {"volatility": 1e308},
        {"volatility": 1e-300},
        {"horizon_years": 5e-324, "steps": 2},
        {"horizon_years": 1e308, "drift": 1e308},
    ],
)
def test_unrepresentable_gbm_fails_loudly(changes: dict[str, float | int]) -> None:
    inputs = {"horizon_years": 1.0, **changes}
    config = replace(BASE, **inputs)
    with pytest.raises(InputValidationError, match=r"overflow|underflow|finite"):
        generate_gbm(config)


def test_forward_inversion_and_spot_base_conditions() -> None:
    # 102.5315... is the forward corresponding to spot 100, not another spot.
    forward = 100 * exp((0.07 - 0.02) * 0.5)
    spot = spot_from_forward(forward, 0.5, 0.07, 0.02)
    assert spot == pytest.approx(100.0)
    assert spot != forward
    assert spot_from_forward(105.0, 0.0, 0.07, 0.02) == 105.0
    assert spot_from_forward(105.0, 1.0, 0.02, 0.02) == 105.0
    assert spot_from_forward(100.0, 1.0, -0.03, 0.02) == pytest.approx(100 * exp(0.05))
    # Physical drift remains independent of carry/risk-neutral pricing inputs.
    config = replace(BASE, spot=spot, volatility=0.0, drift=-0.04)
    assert generate_gbm(config)[-1, 0] == pytest.approx(100 * exp(-0.04 * 0.5))
    risk_neutral = replace(config, drift=0.07 - 0.02)
    assert generate_gbm(risk_neutral)[-1, 0] == pytest.approx(forward)


def test_forward_conversion_uses_log_domain_before_exponentiating() -> None:
    assert spot_from_forward(1e300, 1.0, 700.0, 0.0) == pytest.approx(
        exp(log(1e300) - 700.0)
    )


@pytest.mark.parametrize("name", ["forward", "time", "rate", "dividend_yield"])
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), True, "1"]
)
def test_invalid_forward_real_fields(name: str, value: object) -> None:
    inputs = {"forward": 100.0, "time": 1.0, "rate": 0.03, "dividend_yield": 0.01}
    inputs[name] = value
    with pytest.raises(InputValidationError, match="finite real"):
        spot_from_forward(**inputs)


@pytest.mark.parametrize(
    "inputs",
    [
        (0.0, 1.0, 0.03, 0.01),
        (-100.0, 1.0, 0.03, 0.01),
        (100.0, -1.0, 0.03, 0.01),
        (100.0, 1.0, 1000.0, 0.0),
        (100.0, 1.0, -1000.0, 0.0),
        (100.0, 1.0, 1e308, -1e308),
        (100.0, 1e308, 1e308, 0.0),
    ],
)
def test_invalid_forward_domains_and_unrepresentable_outputs(
    inputs: tuple[float, float, float, float],
) -> None:
    with pytest.raises(InputValidationError):
        spot_from_forward(*inputs)
