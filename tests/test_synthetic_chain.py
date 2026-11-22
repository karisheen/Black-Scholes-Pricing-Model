from dataclasses import replace
from datetime import datetime, timezone
from math import sqrt

import pytest

from quant_pricing.exceptions import InputValidationError
from quant_pricing.surface import check_arbitrage, recover_ivs
from quant_pricing.synthetic_chain import ChainFaults, SSVITruth, generate_chain

AS_OF = datetime(2025, 1, 2, 12, tzinfo=timezone.utc)


def test_clean_generation_is_deterministic_proven_and_nonflat() -> None:
    chain = generate_chain(AS_OF)
    assert chain == generate_chain(AS_OF)
    assert chain.provider.snapshot(AS_OF).complete
    assert "proven" in chain.truth.arbitrage_claim
    assert chain.truth.volatility(0, 0.5) == pytest.approx(0.2)
    assert chain.truth.volatility(-0.3, 0.5) > chain.truth.volatility(0.3, 0.5)
    truth = chain.truth
    theta = truth.variance_rate * truth.max_time
    assert theta * truth.phi * (1 + abs(truth.rho)) < 4
    assert theta * truth.phi**2 * (1 + abs(truth.rho)) <= 4
    for t in (0.1, 0.5, 1):
        for k in (-1.0, -0.3, 0.0, 0.3, 1.0):
            expected = (
                truth.variance_rate
                * t
                / 2
                * (
                    1
                    + truth.rho * truth.phi * k
                    + sqrt((truth.phi * k + truth.rho) ** 2 + 1 - truth.rho**2)
                )
            )
            assert truth.total_variance(k, t) == pytest.approx(expected)
    checked = check_arbitrage(truth, (-1.0, -0.5, 0.0, 0.5, 1.0), (0.1, 0.5, 1.0))
    assert checked.passed
    assert "not a global" in checked.claim


def test_faults_are_explicit_repeatable_and_do_not_change_truth() -> None:
    faults = ChainFaults(missing=(0,), stale=(1,), crossed=(2,), noise=0.5)
    faulty = generate_chain(AS_OF, faults=faults)
    assert faulty == generate_chain(AS_OF, faults=faults)
    clean = generate_chain(AS_OF)
    assert faulty.truth == clean.truth
    rows = faulty.provider.snapshot(AS_OF).rows
    assert rows[0].flags == ("missing",)
    assert rows[1].flags == ("stale",)
    assert rows[2].flags == ("crossed",)
    assert rows[3].quote != clean.provider.snapshot(AS_OF).rows[3].quote
    assert len(faulty.provider.instruments) == len(clean.provider.instruments)
    recovered = recover_ivs(
        faulty.provider.snapshot(AS_OF), faulty.discount_curve, faulty.forward_curve
    )
    assert [o.rejected for o in recovered[:3]] == [
        ("missing",),
        ("stale",),
        ("crossed",),
    ]
    assert all(not o.sides for o in recovered[:3])


def test_premium_units_are_normalized_before_side_recovery() -> None:
    unit, contract = generate_chain(AS_OF), generate_chain(AS_OF, per_contract=True)
    a = recover_ivs(
        unit.provider.snapshot(AS_OF), unit.discount_curve, unit.forward_curve
    )
    b = recover_ivs(
        contract.provider.snapshot(AS_OF),
        contract.discount_curve,
        contract.forward_curve,
    )
    for left, right in zip(a, b, strict=True):
        assert left.weight == pytest.approx(right.weight)
        assert [s.premium_per_unit for s in left.sides] == pytest.approx(
            [s.premium_per_unit for s in right.sides]
        )
        assert [s.volatility for s in left.sides] == pytest.approx(
            [s.volatility for s in right.sides]
        )
        assert left.sides[0].volatility < left.sides[1].volatility
        assert left.sides[1].volatility < left.sides[2].volatility
        assert left.sides[1].volatility == pytest.approx(
            unit.truth.volatility(left.log_moneyness, left.time_to_expiry), abs=2e-7
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_time": float("nan")},
        {"max_time": 0},
        {"rho": 1},
        {"phi": 0},
        {"phi": 100},
        {"phi": 10},
        {"phi": 1, "variance_rate": 4},
    ],
)
def test_invalid_truth_conditions(kwargs: dict[str, float]) -> None:
    inputs = {"max_time": 1.0, **kwargs}
    with pytest.raises(InputValidationError):
        SSVITruth(**inputs)


@pytest.mark.parametrize("t", [0.0, -1.0, 2.0, float("nan")])
def test_truth_maturity_domain(t: float) -> None:
    with pytest.raises(InputValidationError, match="maturity"):
        SSVITruth(1).volatility(0, t)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maturity_days": ()},
        {"maturity_days": (0,)},
        {"maturity_days": (True,)},
        {"maturity_days": (180, 90)},
        {"log_moneyness": (0.0,)},
        {"log_moneyness": (-3.0, -2.0, 0.0, 1.0, 2.0)},
        {"log_moneyness": (-0.2, -0.1, 0.0, 0.0, 0.1)},
        {"log_moneyness": (-0.2, -0.1, 0.0, 0.1, float("nan"))},
        {"spot": 0},
        {"rate": float("inf")},
        {"relative_spread": 0},
        {"faults": ChainFaults(missing=(100,))},
    ],
)
def test_invalid_generation_inputs(kwargs: dict[str, object]) -> None:
    with pytest.raises(InputValidationError):
        generate_chain(AS_OF, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"noise": -1},
        {"noise": float("nan")},
        {"stale": (-1,)},
        {"crossed": (True,)},
        {"missing": (1.5,)},
    ],
)
def test_invalid_fault_inputs(kwargs: dict[str, object]) -> None:
    with pytest.raises(InputValidationError):
        ChainFaults(**kwargs)


def test_zero_skew_truth_and_list_faults() -> None:
    truth = replace(SSVITruth(1), rho=0.0)
    assert truth.volatility(-0.2, 0.5) == truth.volatility(0.2, 0.5)
    assert ChainFaults(missing=[0]).missing == (0,)
