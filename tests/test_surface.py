from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from quant_pricing.black76 import price
from quant_pricing.exceptions import InputValidationError
from quant_pricing.market import MarketSnapshot, SnapshotQuote, Underlying
from quant_pricing.surface import (
    IVObservation,
    SurfaceCalibration,
    SVIParameters,
    SVISlice,
    VolatilitySurface,
    calibrate_surface,
    check_arbitrage,
    check_observed_arbitrage,
    recover_ivs,
)
from quant_pricing.synthetic_chain import ChainFaults, SSVITruth, generate_chain

AS_OF = datetime(2025, 1, 2, 12, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def calibrated() -> SurfaceCalibration:
    chain = generate_chain(AS_OF)
    return calibrate_surface(
        chain.provider.snapshot(AS_OF), chain.discount_curve, chain.forward_curve
    )


def test_known_smile_parameters_variance_and_residual_recovery(
    calibrated: SurfaceCalibration,
) -> None:
    truth = SSVITruth(1.0)
    assert len(calibrated.slices) == 3
    assert calibrated.arbitrage.passed
    assert calibrated.observed_arbitrage.passed
    for fitted in calibrated.slices:
        expected = truth.parameters(fitted.time_to_expiry)
        for name in ("a", "b", "rho", "m", "sigma"):
            assert getattr(fitted.parameters, name) == pytest.approx(
                getattr(expected, name), abs=5e-4
            )
        assert fitted.converged and fitted.evaluations > 0 and fitted.termination
        assert fitted.weighted_rmse < 1e-7
        assert sum(r.weight for r in fitted.residuals) == pytest.approx(9)
        assert len({r.contract for r in fitted.residuals}) == 9
        for residual in fitted.residuals:
            assert abs(residual.residual) < 2e-7
        for k in (-0.35, 0.0, 0.35):
            assert calibrated.surface.total_variance(k, fitted.time_to_expiry) == (
                pytest.approx(truth.total_variance(k, fitted.time_to_expiry), abs=2e-7)
            )
    assert calibrated.surface.volatility(0, 180 / 365) == pytest.approx(0.2, abs=1e-6)


def test_total_variance_interpolation_and_derivatives(
    calibrated: SurfaceCalibration,
) -> None:
    surface = calibrated.surface
    low, high = surface.slices[:2]
    t = (low.time_to_expiry + high.time_to_expiry) / 2
    assert surface.total_variance(0.1, t) == pytest.approx(
        (low.parameters.total_variance(0.1) + high.parameters.total_variance(0.1)) / 2
    )
    h = 1e-4
    first, second = surface.variance_derivatives(0.1, t)
    w = surface.total_variance
    assert first == pytest.approx((w(0.1 + h, t) - w(0.1 - h, t)) / (2 * h), rel=1e-6)
    assert second == pytest.approx(
        (w(0.1 + h, t) - 2 * w(0.1, t) + w(0.1 - h, t)) / h**2, rel=1e-5
    )


@pytest.mark.parametrize(
    "k,t",
    [(1.0, 0.5), (float("nan"), 0.5), (0.0, 0.0), (0.0, 2.0), (0.0, float("nan"))],
)
def test_no_wing_or_maturity_extrapolation(
    calibrated: SurfaceCalibration,
    k: float,
    t: float,
) -> None:
    with pytest.raises(InputValidationError):
        calibrated.surface.volatility(k, t)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_forward_delta_unique_bracket_roundtrip(
    calibrated: SurfaceCalibration,
    kind: str,
) -> None:
    surface = calibrated.surface
    for k in (-0.25, 0.0, 0.25):
        delta = surface.forward_delta(k, 0.5, kind)
        assert surface.log_moneyness_from_delta(delta, 0.5, kind) == pytest.approx(
            k, abs=1e-10
        )
    assert surface.forward_delta(0, 0.5, "call") - surface.forward_delta(
        0, 0.5, "put"
    ) == pytest.approx(1.0)
    with pytest.raises(InputValidationError, match="outside"):
        surface.log_moneyness_from_delta(
            1e-12 if kind == "call" else -1 + 1e-12, 0.5, kind
        )


@pytest.mark.parametrize("delta", [0.0, 1.0, -1.0, float("nan")])
def test_invalid_delta(calibrated: SurfaceCalibration, delta: float) -> None:
    with pytest.raises(InputValidationError, match="forward delta"):
        calibrated.surface.log_moneyness_from_delta(delta, 0.5)


def test_delta_refuses_nonunique_smile() -> None:
    surface = VolatilitySurface(
        (SVISlice(AS_OF, 1.0, SVIParameters(0.01, 4.0, 0.9, 0.0, 0.01), (-1.0, 1.0)),)
    )
    with pytest.raises(InputValidationError, match="unique"):
        surface.log_moneyness_from_delta(0.5, 1.0)


def test_noise_missing_stale_crossed_remain_visible() -> None:
    chain = generate_chain(
        AS_OF, faults=ChainFaults(missing=(0,), stale=(1,), crossed=(2,), noise=0.2)
    )
    result = calibrate_surface(
        chain.provider.snapshot(AS_OF), chain.discount_curve, chain.forward_curve
    )
    assert len(result.slices[0].residuals) == 6
    assert [obs.rejected for obs in result.observations[:3]] == [
        ("missing",),
        ("stale",),
        ("crossed",),
    ]
    assert any(abs(r.residual) > 1e-10 for s in result.slices for r in s.residuals)


def test_independent_sides_keep_valid_bid_when_ask_is_out_of_bounds() -> None:
    chain = generate_chain(AS_OF)
    snapshot = chain.provider.snapshot(AS_OF)
    original = snapshot.rows[4]
    quote = original.quote
    assert quote is not None
    row = replace(original, quote=replace(quote, ask=1000.0))
    observations = recover_ivs(
        replace(snapshot, rows=(row,)), chain.discount_curve, chain.forward_curve
    )
    obs = observations[0]
    assert not obs.rejected and obs.row.quote == row.quote
    assert obs.sides[0].volatility is not None
    assert obs.sides[1].volatility is None and "NoArbitrageError" in obs.sides[1].error
    assert obs.sides[2].volatility is None and "NoArbitrageError" in obs.sides[2].error
    assert not check_observed_arbitrage(observations).maturity_grid


def test_quote_scope_clock_zero_liquidity_and_duplicate_diagnostics() -> None:
    chain = generate_chain(AS_OF)
    snapshot = chain.provider.snapshot(AS_OF)
    row = snapshot.rows[4]
    contract, quote = row.instrument, row.quote
    assert quote is not None
    variants = [
        replace(contract, exercise_style="american"),
        replace(contract, settlement_currency="EUR"),
        replace(contract, expiry=AS_OF),
        Underlying("SPOT", "USD"),
    ]
    for variant in variants:
        updated = replace(
            row, instrument=variant, quote=replace(quote, instrument=variant)
        )
        obs = recover_ivs(
            replace(snapshot, rows=(updated,)),
            chain.discount_curve,
            chain.forward_curve,
        )[0]
        assert obs.rejected and not obs.sides
    with pytest.raises(InputValidationError, match="as_of"):
        recover_ivs(
            replace(snapshot, as_of=AS_OF + timedelta(seconds=1)),
            chain.discount_curve,
            chain.forward_curve,
        )
    other_contract = replace(
        contract, symbol="OTHER", underlying=Underlying("X", "USD")
    )
    other = replace(
        row, instrument=other_contract, quote=replace(quote, instrument=other_contract)
    )
    with pytest.raises(InputValidationError, match="one underlying"):
        recover_ivs(
            replace(snapshot, rows=(row, other)),
            chain.discount_curve,
            chain.forward_curve,
        )
    zero = replace(row, quote=replace(quote, bid_size=0, ask_size=0))
    obs = recover_ivs(
        replace(snapshot, rows=(zero,)), chain.discount_curve, chain.forward_curve
    )
    assert obs[0].weight == 0
    assert check_observed_arbitrage((obs[0], obs[0])).passed
    assert not check_observed_arbitrage(obs).evidence
    with pytest.raises(InputValidationError, match="tolerance"):
        check_observed_arbitrage(obs, tolerance=-1)


def test_calibration_failures_are_not_silent_maturity_drops() -> None:
    chain = generate_chain(AS_OF, faults=ChainFaults(missing=tuple(range(5))))
    with pytest.raises(InputValidationError, match="five liquid unique"):
        calibrate_surface(
            chain.provider.snapshot(AS_OF), chain.discount_curve, chain.forward_curve
        )
    clean = generate_chain(AS_OF)
    with pytest.raises(InputValidationError, match="did not converge"):
        calibrate_surface(
            clean.provider.snapshot(AS_OF),
            clean.discount_curve,
            clean.forward_curve,
            max_evaluations=1,
        )
    with pytest.raises(InputValidationError, match="maturity knots"):
        calibrate_surface(
            MarketSnapshot(AS_OF, "empty", ()),
            clean.discount_curve,
            clean.forward_curve,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_evaluations": 0},
        {"max_evaluations": True},
        {"check_points": 2},
        {"check_points": 3.5},
    ],
)
def test_invalid_calibration_controls(kwargs: dict[str, object]) -> None:
    chain = generate_chain(AS_OF)
    with pytest.raises(InputValidationError):
        calibrate_surface(
            chain.provider.snapshot(AS_OF),
            chain.discount_curve,
            chain.forward_curve,
            **kwargs,
        )


@pytest.mark.parametrize(
    "changes", [{"a": float("inf")}, {"b": -1}, {"rho": 1}, {"sigma": 0}, {"a": -1}]
)
def test_invalid_svi_parameters(changes: dict[str, float]) -> None:
    with pytest.raises(InputValidationError):
        replace(SSVITruth(1).parameters(1), **changes)


def test_invalid_log_moneyness_and_surface_domains() -> None:
    p = SSVITruth(1).parameters(1)
    with pytest.raises(InputValidationError, match="finite"):
        p.total_variance(float("nan"))
    knot = SVISlice(AS_OF, 1.0, p, (-1.0, 1.0))
    invalid = [
        (),
        (replace(knot, time_to_expiry=0),),
        (knot, knot),
        (replace(knot, domain=(1.0, 0.0)),),
        (knot, replace(knot, time_to_expiry=2.0, domain=(2.0, 3.0))),
    ]
    for slices in invalid:
        with pytest.raises(InputValidationError):
            VolatilitySurface(slices)


def test_calendar_and_butterfly_defects_have_numeric_evidence() -> None:
    # Both slices individually satisfy raw SVI positivity, but not a calendar.
    p = SVIParameters(0.05, 0.1, -0.2, 0.0, 0.2)
    surface = VolatilitySurface(
        (
            SVISlice(AS_OF, 0.5, p, (-1.0, 1.0)),
            SVISlice(AS_OF, 1.0, replace(p, a=0.01), (-1.0, 1.0)),
        )
    )
    report = check_arbitrage(surface, (-0.5, 0.0, 0.5), (0.5, 1.0))
    assert any(e.kind == "calendar" and e.value < 0 for e in report.evidence)
    # Extreme positive raw-SVI wings are not rendered safe by positivity.
    bad = VolatilitySurface(
        (SVISlice(AS_OF, 1.0, SVIParameters(0.001, 3.0, 0.8, 0.0, 0.01), (-1.0, 1.0)),)
    )
    report = check_arbitrage(bad, tuple(np.linspace(-1.0, 1.0, 101)), (1.0,))
    assert any(e.kind == "butterfly" for e in report.evidence)
    assert any(e.kind == "monotonicity" for e in report.evidence)
    assert report.tolerance == 1e-8
    assert not report.passed


def test_raw_midpoint_defects_are_not_hidden_by_fitting() -> None:
    chain = generate_chain(AS_OF)
    snapshot = chain.provider.snapshot(AS_OF)
    rows = []
    for index, row in enumerate(snapshot.rows):
        contract, quote = row.instrument, row.quote
        assert quote is not None
        t = (contract.expiry - AS_OF).days / 365
        f = chain.forward_curve.forward(contract.expiry)
        d = chain.discount_curve.discount_factor(contract.expiry)
        # Center price spike creates strike defects; second expiry drops variance.
        vol = 1.5 if index == 4 else 0.08 if 9 <= index < 18 else 0.2
        premium = float(price(contract.option_type, f, contract.strike, t, d, vol))
        rows.append(replace(row, quote=replace(quote, bid=premium, ask=premium)))
    observations = recover_ivs(
        replace(snapshot, rows=tuple(rows)), chain.discount_curve, chain.forward_curve
    )
    report = check_observed_arbitrage(observations)
    assert {e.kind for e in report.evidence} == {
        "calendar",
        "butterfly",
        "monotonicity",
    }
    assert "shared k" in report.claim


@pytest.mark.parametrize(
    "ks,ts,tol",
    [
        ((0.0, 1.0), (1.0,), 0.0),
        ((-0.1, 0.0, 0.1), (), 0.0),
        ((-0.1, 0.0, 0.1), (1.0,), -1.0),
        ((-0.1, 0.0, 0.1), (1.0,), float("nan")),
        ((-0.1, 0.0, float("inf")), (1.0,), 0.0),
        ((-0.1, 0.0, 0.0), (1.0,), 0.0),
        ((-0.1, 0.0, 0.1), (0.0,), 0.0),
    ],
)
def test_invalid_arbitrage_grids(
    ks: tuple[float, ...], ts: tuple[float, ...], tol: float
) -> None:
    with pytest.raises(InputValidationError):
        check_arbitrage(SSVITruth(1), ks, ts, tolerance=tol)


def test_nonpositive_variance_is_a_check_failure() -> None:
    class InvalidVariance:
        def total_variance(self, log_moneyness: float, time_to_expiry: float) -> float:
            return 0.0

    with pytest.raises(InputValidationError, match="positive finite"):
        check_arbitrage(InvalidVariance(), (-0.1, 0.0, 0.1), (1.0,))


def test_observations_with_no_iv_are_explicitly_excluded() -> None:
    chain = generate_chain(AS_OF)
    snapshot = chain.provider.snapshot(AS_OF)
    row = snapshot.rows[4]
    assert row.quote is not None
    zero = replace(row, quote=replace(row.quote, bid=0.0, ask=0.0))
    obs = recover_ivs(
        replace(snapshot, rows=(zero,)), chain.discount_curve, chain.forward_curve
    )
    assert obs[0].sides[1].volatility == 0
    assert not check_observed_arbitrage(obs).maturity_grid
    missing = IVObservation(SnapshotQuote(row.instrument, None, ("missing",)), (), ())
    assert not check_observed_arbitrage((missing,)).maturity_grid


def test_sequence_gaps_exclude_quotes_without_quality_fallback() -> None:
    from quant_pricing.market import SequenceGap

    chain = generate_chain(AS_OF)
    snapshot = chain.provider.snapshot(AS_OF)
    row = snapshot.rows[4]
    assert row.quote is not None
    after = replace(row.quote, source_sequence=3)
    gap = SequenceGap(2, 2, row.quote, after)
    changed = replace(row, quote=after, sequence_gaps=(gap,))
    recovered = recover_ivs(
        replace(snapshot, rows=(changed,)), chain.discount_curve, chain.forward_curve
    )
    assert recovered[0].rejected == ("sequence_gap",)
    assert not recovered[0].sides


def test_zero_liquidity_and_zero_iv_cannot_supply_a_fit() -> None:
    chain = generate_chain(AS_OF)
    snapshot = chain.provider.snapshot(AS_OF)
    rows = []
    for index, row in enumerate(snapshot.rows[:9]):
        assert row.quote is not None
        if index % 2:
            quote = replace(row.quote, bid_size=0, ask_size=0)
        else:
            quote = replace(row.quote, bid=0, ask=0)
        rows.append(replace(row, quote=quote))
    with pytest.raises(InputValidationError, match="five liquid unique"):
        calibrate_surface(
            replace(snapshot, rows=tuple(rows)),
            chain.discount_curve,
            chain.forward_curve,
        )
