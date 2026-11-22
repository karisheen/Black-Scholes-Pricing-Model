"""Independent lattice economics, policy evidence, jumps and numerical limits."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_pricing.binomial import CashDividend, price
from quant_pricing.black_scholes import price as analytic_price
from quant_pricing.exceptions import InputValidationError


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("rate", [0.05, -0.02])
def test_european_converges_to_independent_spot_bsm(kind: str, rate: float) -> None:
    expected = float(analytic_price(kind, 100, 100, 1, rate, 0.2))
    coarse = price(kind, 100, 100, 1, rate, 0.2, steps=100)
    fine = price(kind, 100, 100, 1, rate, 0.2, steps=800)
    assert abs(fine.price - expected) < abs(coarse.price - expected)
    assert fine.price == pytest.approx(expected, abs=0.004)
    assert fine.early_exercise_probability == 0


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("spot", [70.0, 100.0, 130.0])
@pytest.mark.parametrize("with_dividends", [False, True])
def test_american_bounds(kind: str, spot: float, with_dividends: bool) -> None:
    dividends = (CashDividend(0.5, 6),) if with_dividends else ()
    european = price(kind, spot, 100, 1, 0.04, 0.2, dividends=dividends)
    american = price(
        kind, spot, 100, 1, 0.04, 0.2, style="american", dividends=dividends
    )
    intrinsic = max(spot - 100 if kind == "call" else 100 - spot, 0)
    assert american.price >= european.price - 1e-11
    assert american.price >= intrinsic - 1e-11
    assert american.price <= (spot if kind == "call" else 100) + 1e-10


@pytest.mark.parametrize("rate", [0.0, 0.05])
def test_conventional_non_dividend_call_does_not_exercise_early(rate: float) -> None:
    european = price("call", 110, 90, 1, rate, 0.2)
    american = price("call", 110, 90, 1, rate, 0.2, style="american")
    assert american.price == pytest.approx(european.price)
    assert american.early_exercise_probability == 0
    assert all(boundary.phase == "expiry" for boundary in american.boundaries)


def test_dividend_call_has_pre_ex_early_exercise_and_writer_assignment() -> None:
    dividends = (CashDividend(0.5, 6),)
    european = price("call", 110, 90, 1, 0.04, 0.2, steps=400, dividends=dividends)
    american = price(
        "call", 110, 90, 1, 0.04, 0.2, style="american", steps=400, dividends=dividends
    )
    assert american.price > european.price
    assert american.early_exercise_probability > 0
    pre = [b for b in american.boundaries if b.phase == "pre_dividend"]
    assert len(pre) == 1
    assert pre[0].time == 0.5 and pre[0].step == 200
    assert pre[0].probability > 0
    assert pre[0].minimum_stock > 90
    assert american.dividend_steps == (200,)
    assert american.escrowed_spot == pytest.approx(110 - 6 * math.exp(-0.02))
    short = american.exposure(-2, 100)
    long = american.exposure(2, 100)
    flat = american.exposure(0)
    assert short.action == "writer_assignment"
    assert long.action == "long_exercise"
    assert flat.action == "flat" and flat.total_quantity == 0
    assert short.early_quantity == long.early_quantity
    assert short.early_quantity == pytest.approx(200 * short.early_probability)
    assert short.expiry_quantity == pytest.approx(200 * short.expiry_probability)
    assert short.total_quantity == pytest.approx(
        short.early_quantity + short.expiry_quantity
    )
    assert 0 < short.total_quantity <= 200


def test_policy_probabilities_are_survival_weighted_and_diagnostics_finite() -> None:
    result = price("put", 100, 100, 1, 0.05, 0.3, style="american", steps=80)
    assert 0 < result.early_exercise_probability < 1
    assert result.early_exercise_probability + result.expiry_exercise_probability + (
        result.no_exercise_probability
    ) == pytest.approx(1)
    assert sum(b.probability for b in result.boundaries) == pytest.approx(
        1 - result.no_exercise_probability
    )
    assert result.steps == 80 and result.time_step == 1 / 80
    assert 0 < result.up_probability < 1
    for boundary in result.boundaries:
        assert boundary.exercised_nodes > 0
        assert 0 <= boundary.probability <= 1
        assert 0 < boundary.minimum_stock <= boundary.maximum_stock < 100
        assert all(
            math.isfinite(value)
            for value in (boundary.time, boundary.minimum_stock, boundary.maximum_stock)
        )


def test_cash_dividend_parity_and_duplicate_schedule_aggregation() -> None:
    schedule = (CashDividend(0.75, 2), CashDividend(0.25, 1), CashDividend(0.25, 3))
    call = price("call", 100, 95, 1, 0.03, 0.25, dividends=schedule)
    put = price("put", 100, 95, 1, 0.03, 0.25, dividends=schedule)
    prepaid = 100 - 4 * math.exp(-0.03 * 0.25) - 2 * math.exp(-0.03 * 0.75)
    assert call.price - put.price == pytest.approx(prepaid - 95 * math.exp(-0.03))
    aggregated = price(
        "call",
        100,
        95,
        1,
        0.03,
        0.25,
        dividends=(CashDividend(0.25, 4), CashDividend(0.75, 2)),
    )
    assert aggregated == call


@pytest.mark.parametrize("kind,spot", [("call", 120.0), ("put", 80.0)])
@pytest.mark.parametrize("style", ["european", "american"])
def test_maturity_is_immediate_intrinsic(kind: str, spot: float, style: str) -> None:
    result = price(kind, spot, 100, 0, 0.05, 0.2, style=style)
    assert result.price == pytest.approx(20)
    assert result.steps == 0 and result.time_step == 0
    assert result.early_exercise_probability == 0
    assert result.expiry_exercise_probability == 1
    assert result.no_exercise_probability == 0
    assert len(result.boundaries) == 1 and result.boundaries[0].phase == "expiry"


def test_expiry_tie_and_worthless_option_do_not_exercise() -> None:
    for spot in (50, 100):
        result = price("call", spot, 100, 0, 0, 0)
        assert result.price == 0
        assert result.no_exercise_probability == 1
        assert result.boundaries == ()


@pytest.mark.parametrize("kind", ["call", "put"])
def test_zero_vol_european_matches_deterministic_discounted_payoff(kind: str) -> None:
    result = price(kind, 100, 105, 1, 0.03, 0)
    terminal = 100 * math.exp(0.03)
    payoff = max(terminal - 105 if kind == "call" else 105 - terminal, 0)
    assert result.price == pytest.approx(math.exp(-0.03) * payoff)
    assert result.up_probability == 1


def test_zero_vol_dividend_exercise_occurs_on_both_sides_of_jump() -> None:
    schedule = (CashDividend(0.5, 20),)
    call = price("call", 110, 90, 1, 0.04, 0, style="american", dividends=schedule)
    put = price("put", 100, 100, 1, 0.04, 0, style="american", dividends=schedule)
    assert call.early_exercise_probability == 1
    assert put.early_exercise_probability == 1
    call_boundary = next(b for b in call.boundaries if b.probability == 1)
    put_boundary = next(b for b in put.boundaries if b.probability == 1)
    assert call_boundary.phase == "pre_dividend" and call_boundary.time == 0.5
    assert put_boundary.phase == "post_dividend" and put_boundary.time == 0.5
    assert call.price == pytest.approx(110 - 90 * math.exp(-0.02))
    assert put.price == pytest.approx(120 * math.exp(-0.02) - 100)
    assert put.expiry_exercise_probability == 0


def test_deep_put_exercises_now_and_negative_rate_call_can_exercise() -> None:
    put = price("put", 20, 100, 1, 0.1, 0, style="american")
    call = price("call", 150, 100, 1, -0.1, 0, style="american")
    assert put.price == pytest.approx(80)
    assert call.price == pytest.approx(50)
    assert put.early_exercise_probability == call.early_exercise_probability == 1
    assert put.boundaries[0].time == call.boundaries[0].time == 0


@pytest.mark.parametrize("rate", [-0.2, 0.2])
def test_probability_endpoints_are_valid(rate: float) -> None:
    result = price("call", 100, 100, 1, rate, 0.2, steps=1)
    assert result.up_probability == (0 if rate < 0 else 1)
    assert result.no_exercise_probability + result.expiry_exercise_probability == 1


@pytest.mark.parametrize(
    "arguments",
    [
        (0, 100, 1, 0.05, 0.2),
        (100, -1, 1, 0.05, 0.2),
        (100, 100, -1, 0.05, 0.2),
        (100, 100, 1, 0.05, -0.2),
        (float("nan"), 100, 1, 0.05, 0.2),
        (100, 100, 1, float("inf"), 0.2),
        (True, 100, 1, 0.05, 0.2),
        ("100", 100, 1, 0.05, 0.2),
        (100, 100, 1, 1000, 0),
        (100, 100, 1, -1000, 0),
        (100, 100, 1, 0.9, 0.01),
        (100, 100, 1, 0, 1e308),
        (100, 100, 4, 0, 1e308),
        (100, 100, 0.25, 0, 5e-324),
        (1e308, 100, 1, 0, 100),
        (1e308, 100, 1, 700, 0),
    ],
)
def test_rejects_invalid_inputs_and_unrepresentable_trees(
    arguments: tuple[float, ...],
) -> None:
    with pytest.raises(InputValidationError):
        price("call", *arguments, steps=1)


@pytest.mark.parametrize("steps", [0, -1, 4097, True, 1.5])
def test_rejects_invalid_step_counts(steps: int) -> None:
    with pytest.raises(InputValidationError, match="steps"):
        price("call", 100, 100, 1, 0.05, 0.2, steps=steps)


@pytest.mark.parametrize("kind,style", [("straddle", "european"), ("call", "bermudan")])
def test_rejects_invalid_kind_and_style(kind: str, style: str) -> None:
    with pytest.raises(InputValidationError, match="kind/style"):
        price(kind, 100, 100, 1, 0.05, 0.2, style=style)


@pytest.mark.parametrize(
    "time,amount",
    [(0, 1), (-1, 1), (1, 0), (1, -1), (math.inf, 1), (1, math.nan), (True, 1)],
)
def test_rejects_invalid_cash_dividend(time: float, amount: float) -> None:
    with pytest.raises(InputValidationError):
        CashDividend(time, amount)


@pytest.mark.parametrize("time", [0.51, 1, 1.1, 1e-20])
def test_rejects_unaligned_or_outside_schedule(time: float) -> None:
    with pytest.raises(InputValidationError):
        price(
            "call", 100, 100, 1, 0.05, 0.2, steps=10, dividends=(CashDividend(time, 1),)
        )


def test_rejects_maturity_dividend_and_nonpositive_escrow() -> None:
    with pytest.raises(InputValidationError, match="inside expiry"):
        price("call", 100, 100, 0, 0.05, 0.2, dividends=(CashDividend(0.5, 1),))
    with pytest.raises(InputValidationError, match="PV of cash"):
        price("call", 100, 100, 1, 0, 0.2, dividends=(CashDividend(0.5, 100),))


@pytest.mark.parametrize(
    "contracts,multiplier",
    [
        (math.nan, 100),
        (1, 0),
        (1, -1),
        (1, math.inf),
        (1e308, 1e308),
    ],
)
def test_exposure_rejects_invalid_units(contracts: float, multiplier: float) -> None:
    result = price("call", 100, 100, 1, 0.05, 0.2)
    with pytest.raises(InputValidationError):
        result.exposure(contracts, multiplier)


def test_inputs_and_schedule_are_not_mutated() -> None:
    dividends = (CashDividend(0.5, 2),)
    first = price("put", 100, 100, 1, 0.03, 0.2, dividends=dividends)
    second = price("put", 100, 100, 1, 0.03, 0.2, dividends=dividends)
    assert first == second
    assert dividends == (CashDividend(0.5, 2),)
    assert np.isfinite(first.price)


def test_two_step_probability_matches_enumerated_paths() -> None:
    result = price("call", 100, 90, 1, 0, math.log(2) * math.sqrt(2), steps=2)
    assert result.up_probability == pytest.approx(1 / 3)
    assert result.price == pytest.approx(350 / 9)
    assert result.expiry_exercise_probability == pytest.approx(5 / 9)
    assert result.no_exercise_probability == pytest.approx(4 / 9)


def test_dividend_tree_refinement_preserves_schedule() -> None:
    schedule = (CashDividend(0.5, 6),)
    coarse = price(
        "call",
        110,
        90,
        1,
        0.04,
        0.2,
        style="american",
        steps=400,
        dividends=schedule,
    )
    fine = price(
        "call",
        110,
        90,
        1,
        0.04,
        0.2,
        style="american",
        steps=800,
        dividends=schedule,
    )
    assert abs(fine.price - coarse.price) < 0.03
    assert coarse.dividend_steps == (200,)
    assert fine.dividend_steps == (400,)


def test_rejects_numerical_grid_and_cash_overflow() -> None:
    with pytest.raises(InputValidationError, match="resolution"):
        price("call", 100, 100, 1e-323, 0, 0)
    with pytest.raises(InputValidationError, match="aggregate dividend"):
        price(
            "call",
            100,
            100,
            1,
            0,
            0.2,
            dividends=(CashDividend(0.5, 1e308), CashDividend(0.5, 1e308)),
        )
    with pytest.raises(InputValidationError, match="option values"):
        price("put", 100, 1e308, 1, -700, 0, steps=1)
    with pytest.raises(InputValidationError, match="range"):
        price("put", 100, 100, 2, -700, 0, steps=2)


def test_zero_vol_at_the_money_does_not_create_roundoff_exercise() -> None:
    result = price("call", 100, 100, 1, 0, 0, style="american")
    assert result.price == 0
    assert result.no_exercise_probability == 1


def test_rejects_non_record_cash_schedule() -> None:
    with pytest.raises(InputValidationError, match="CashDividend"):
        price("call", 100, 100, 1, 0, 0.2, dividends=(None,))  # type: ignore[arg-type]


def test_expiry_probability_excludes_numerically_atm_tree_nodes() -> None:
    result = price("call", 100, 100, 1, 0, math.log(2) * math.sqrt(2), steps=2)
    assert result.expiry_exercise_probability == pytest.approx(1 / 9)
    assert result.no_exercise_probability == pytest.approx(8 / 9)
