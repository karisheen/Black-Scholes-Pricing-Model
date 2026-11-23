"""Auditable Black-76 IV recovery and liquidity-weighted raw-SVI calibration.

Coordinates are k=log(K/F) and ACT/365F T. Only total variance is interpolated
(linearly in T); strike wings and maturity extrapolation always raise. Positive
raw-SVI variance is a fit constraint, NOT a no-arbitrage guarantee. Diagnostics
check a declared finite grid and never certify global absence of arbitrage.
"""

from __future__ import annotations

import itertools
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from math import exp, isfinite, log, sqrt
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.optimize import brentq
from scipy.special import ndtr

from quant_pricing.black76 import implied_volatility, price
from quant_pricing.black_scholes import OptionType
from quant_pricing.conventions import year_fraction
from quant_pricing.curves import DiscountCurve, ForwardCurve
from quant_pricing.exceptions import (
    ImpliedVolatilityError,
    InputValidationError,
    NoArbitrageError,
)
from quant_pricing.implied_volatility import ImpliedVolatilityResult
from quant_pricing.market import MarketSnapshot, OptionContract, SnapshotQuote


@dataclass(frozen=True, slots=True)
class SVIParameters:
    """w(k)=a+b[rho(k-m)+sqrt((k-m)^2+sigma^2)]."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if not all(isfinite(x) for x in (self.a, self.b, self.rho, self.m, self.sigma)):
            raise InputValidationError("SVI parameters must be finite")
        if self.b < 0 or abs(self.rho) >= 1 or self.sigma <= 0:
            raise InputValidationError("SVI requires b>=0, |rho|<1, sigma>0")
        if self.a + self.b * self.sigma * sqrt(1 - self.rho**2) <= 0:
            raise InputValidationError("SVI minimum total variance must be positive")

    def total_variance(self, log_moneyness: float) -> float:
        if not isfinite(log_moneyness):
            raise InputValidationError("log moneyness must be finite")
        x = log_moneyness - self.m
        return self.a + self.b * (self.rho * x + sqrt(x * x + self.sigma**2))

    def derivatives(self, k: float) -> tuple[float, float]:
        x = k - self.m
        r = sqrt(x * x + self.sigma**2)
        return self.b * (self.rho + x / r), self.b * self.sigma**2 / r**3


@dataclass(frozen=True, slots=True)
class IVSide:
    side: Literal["bid", "midpoint", "ask"]
    premium_per_unit: float
    result: ImpliedVolatilityResult | None
    error: str | None

    @property
    def volatility(self) -> float | None:
        return None if self.result is None else self.result.volatility


@dataclass(frozen=True, slots=True)
class IVObservation:
    """Original snapshot provenance and independent side successes/failures."""

    row: SnapshotQuote
    rejected: tuple[str, ...]
    sides: tuple[IVSide, ...]
    time_to_expiry: float | None = None
    log_moneyness: float | None = None
    weight: float = 0.0

    @property
    def fit_eligible(self) -> bool:
        """A liquid, positive midpoint IV can enter the SVI objective."""
        if self.rejected or self.weight <= 0.0:
            return False
        volatility = self.sides[1].volatility
        return volatility is not None and volatility > 0.0


@dataclass(frozen=True, slots=True)
class FitResidual:
    contract: str
    log_moneyness: float
    observed_variance: float
    fitted_variance: float
    weight: float

    @property
    def residual(self) -> float:
        return self.fitted_variance - self.observed_variance


@dataclass(frozen=True, slots=True)
class SVISlice:
    """One fit knot with normalized weights and unweighted variance residuals.

    evaluations and termination describe the selected deterministic start;
    weighted_rmse=sqrt(sum(weight*residual**2)/number_of_observations).
    """

    expiry: datetime
    time_to_expiry: float
    parameters: SVIParameters
    domain: tuple[float, float]
    residuals: tuple[FitResidual, ...] = ()
    converged: bool = True
    evaluations: int = 0
    termination: str = "supplied parameters"
    weighted_rmse: float = 0.0


@dataclass(frozen=True, slots=True)
class VolatilitySurface:
    """Raw SVI knots; linear total-variance interpolation, no extrapolation."""

    slices: tuple[SVISlice, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slices", tuple(self.slices))
        times = tuple(s.time_to_expiry for s in self.slices)
        if not times or any(not isfinite(t) or t <= 0 for t in times):
            raise InputValidationError("surface needs positive finite maturity knots")
        if any(a >= b for a, b in itertools.pairwise(times)):
            raise InputValidationError("surface maturities must strictly increase")
        if any(
            not all(isfinite(k) for k in s.domain) or s.domain[0] >= s.domain[1]
            for s in self.slices
        ):
            raise InputValidationError("each strike domain must be finite and ordered")
        if self.domain[0] >= self.domain[1]:
            raise InputValidationError("slice strike domains must overlap")

    @property
    def domain(self) -> tuple[float, float]:
        return max(s.domain[0] for s in self.slices), min(
            s.domain[1] for s in self.slices
        )

    @property
    def maturity_domain(self) -> tuple[float, float]:
        return self.slices[0].time_to_expiry, self.slices[-1].time_to_expiry

    def _knots(self, k: float, t: float) -> tuple[SVISlice, SVISlice, float]:
        if not isfinite(k) or not self.domain[0] <= k <= self.domain[1]:
            raise InputValidationError("strike outside common observed domain")
        if (
            not isfinite(t)
            or not self.maturity_domain[0] <= t <= self.maturity_domain[1]
        ):
            raise InputValidationError("maturity extrapolation is not supported")
        index = bisect_left(tuple(s.time_to_expiry for s in self.slices), t)
        right = self.slices[index]
        if right.time_to_expiry == t:
            return right, right, 0.0
        left = self.slices[index - 1]
        alpha = (t - left.time_to_expiry) / (right.time_to_expiry - left.time_to_expiry)
        return left, right, alpha

    def total_variance(self, log_moneyness: float, time_to_expiry: float) -> float:
        left, right, alpha = self._knots(log_moneyness, time_to_expiry)
        return (1 - alpha) * left.parameters.total_variance(
            log_moneyness
        ) + alpha * right.parameters.total_variance(log_moneyness)

    def volatility(self, log_moneyness: float, time_to_expiry: float) -> float:
        return sqrt(self.total_variance(log_moneyness, time_to_expiry) / time_to_expiry)

    def variance_derivatives(self, k: float, t: float) -> tuple[float, float]:
        left, right, alpha = self._knots(k, t)
        l1, l2 = left.parameters.derivatives(k)
        r1, r2 = right.parameters.derivatives(k)
        return (1 - alpha) * l1 + alpha * r1, (1 - alpha) * l2 + alpha * r2

    def forward_delta(
        self, k: float, t: float, kind: OptionType | str = "call"
    ) -> float:
        """Undiscounted, unadjusted Black delta N(d1), put N(d1)-1.

        Volatility is held fixed in the delta definition, not differentiated
        through the smile. No premium adjustment or spot-discount convention.
        """
        option = OptionType.parse(kind)
        w = self.total_variance(k, t)
        return float(ndtr((-k + w / 2) / sqrt(w))) - (option == OptionType.PUT)

    def log_moneyness_from_delta(
        self, delta: float, t: float, kind: OptionType | str = "call"
    ) -> float:
        """Invert only on the common observed domain with proven monotone d1.

        Each subinterval uses conservative analytic bounds on convex SVI w and
        increasing w'. If those cannot prove d1'<0, fail rather than select an
        arbitrary smile root. This is a sufficient, not necessary, uniqueness test.
        """
        option = OptionType.parse(kind)
        target = delta + (option == OptionType.PUT)
        if not isfinite(delta) or not 0 < target < 1:
            raise InputValidationError("forward delta must be in (0,1) or (-1,0)")
        grid = np.linspace(*self.domain, 129)
        for grid_lo, grid_hi in itertools.pairwise(grid):
            lo, hi = float(grid_lo), float(grid_hi)
            wlo, whi = self.total_variance(lo, t), self.total_variance(hi, t)
            dlo, _ = self.variance_derivatives(lo, t)
            dhi, _ = self.variance_derivatives(hi, t)
            # Convexity gives w >= both endpoint tangent lines on this interval.
            wmin = max(wlo + min(0.0, dlo * (hi - lo)), whi - max(0.0, dhi * (hi - lo)))
            wmax = max(wlo, whi)
            products = [
                x * y for x in (lo + wmin / 2, hi + wmax / 2) for y in (dlo, dhi)
            ]
            if wmin <= 0 or max(products) >= 2 * wmin:
                raise InputValidationError("cannot certify a unique delta bracket")
        low, high = self.domain
        if (
            not self.forward_delta(high, t, option)
            <= delta
            <= self.forward_delta(low, t, option)
        ):
            raise InputValidationError("delta lies outside the observed strike bracket")
        return float(
            brentq(
                lambda k: self.forward_delta(k, t, option) - delta,
                low,
                high,
                xtol=1e-13,
            )
        )


class VarianceModel(Protocol):
    def total_variance(self, log_moneyness: float, time_to_expiry: float) -> float: ...


@dataclass(frozen=True, slots=True)
class ArbitrageEvidence:
    kind: Literal["monotonicity", "butterfly", "calendar"]
    log_moneyness: float
    time_to_expiry: float
    value: float
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True, slots=True)
class ArbitrageReport:
    log_moneyness_grid: tuple[float, ...]
    maturity_grid: tuple[float, ...]
    tolerance: float
    evidence: tuple[ArbitrageEvidence, ...]
    claim: str = "finite grid checked; not a global no-arbitrage proof"

    @property
    def passed(self) -> bool:
        return not self.evidence


def check_arbitrage(
    surface: VarianceModel,
    log_moneyness_grid: tuple[float, ...],
    maturity_grid: tuple[float, ...],
    *,
    tolerance: float = 1e-8,
) -> ArbitrageReport:
    """Check normalized calls C/(DF) vs K/F, and w(k,T) calendars.

    Call secants must lie in [-1,0] and be nondecreasing (butterflies).
    Normalization avoids interpreting carry-driven calendar price differences
    as arbitrage. Tolerance is absolute in secant slopes / total variance.
    """
    ks, ts = tuple(log_moneyness_grid), tuple(maturity_grid)
    if len(ks) < 3 or not ts or not isfinite(tolerance) or tolerance < 0:
        raise InputValidationError("need >=3 strikes, >=1 maturity and tolerance>=0")
    for grid in (ks, ts):
        if not all(isfinite(x) for x in grid) or any(
            a >= b for a, b in itertools.pairwise(grid)
        ):
            raise InputValidationError("arbitrage grids must be finite and increasing")
    if ts[0] <= 0:
        raise InputValidationError("arbitrage maturity grid must be positive")
    evidence: list[ArbitrageEvidence] = []
    previous: list[float] | None = None
    strikes = [exp(k) for k in ks]
    for t in ts:
        ws = [surface.total_variance(k, t) for k in ks]
        if any(not isfinite(w) or w <= 0 for w in ws):
            raise InputValidationError(
                "arbitrage check requires positive finite variance"
            )
        calls = [
            float(price("call", 1, strike, 1, 1, sqrt(w)))
            for strike, w in zip(strikes, ws, strict=True)
        ]
        evidence.extend(_strike_evidence(ks, calls, calls, t, tolerance))
        if previous is not None:
            for k, w, before in zip(ks, ws, previous, strict=True):
                if w - before < -tolerance:
                    evidence.append(
                        ArbitrageEvidence("calendar", k, t, w - before, 0, float("inf"))
                    )
        previous = ws
    return ArbitrageReport(ks, ts, tolerance, tuple(evidence))


def recover_ivs(
    snapshot: MarketSnapshot, discount_curve: DiscountCurve, forward_curve: ForwardCurve
) -> tuple[IVObservation, ...]:
    """Recover each usable side; no stale/crossed/missing-quote fallback.

    Liquidity weight = harmonic displayed size / squared relative premium spread.
    Relative spread uses midpoint floored at 1e-6*D*F and is floored at 1e-4.
    Zero liquidity has zero weight; fit weights normalize to mean one per expiry.
    """
    if snapshot.as_of != discount_curve.as_of or snapshot.as_of != forward_curve.as_of:
        raise InputValidationError("snapshot and curves must have identical as_of")
    observations: list[IVObservation] = []
    underlyings = {
        r.instrument.underlying
        for r in snapshot.rows
        if isinstance(r.instrument, OptionContract)
    }
    if len(underlyings) > 1:
        raise InputValidationError("surface requires one underlying")
    for row in snapshot.rows:
        contract, quote = row.instrument, row.quote
        reasons = list(row.flags)
        rejected: list[str] = list(reasons)
        if row.sequence_gaps:
            rejected.append("sequence_gap")
        if not isinstance(contract, OptionContract):
            rejected.append("not_option")
        elif (
            contract.exercise_style != "european"
            or contract.quote_currency != contract.underlying.currency
            or contract.settlement_currency != contract.underlying.currency
        ):
            rejected.append("unsupported_contract")
        elif contract.expiry <= snapshot.as_of:
            rejected.append("expired")
        if rejected:
            observations.append(IVObservation(row, tuple(rejected), ()))
            continue
        assert isinstance(contract, OptionContract) and quote is not None
        t = year_fraction(snapshot.as_of, contract.expiry)
        f, d = (
            forward_curve.forward(contract.expiry),
            discount_curve.discount_factor(contract.expiry),
        )
        bid, ask = (
            contract.premium_per_unit(quote.bid),
            contract.premium_per_unit(quote.ask),
        )
        sides: list[IVSide] = []
        premiums: tuple[tuple[Literal["bid", "midpoint", "ask"], float], ...] = (
            ("bid", bid),
            ("midpoint", bid / 2 + ask / 2),
            ("ask", ask),
        )
        for side, premium in premiums:
            result, error = None, None
            try:
                result = implied_volatility(
                    contract.option_type, premium, f, contract.strike, t, d
                )
            except (
                InputValidationError,
                NoArbitrageError,
                ImpliedVolatilityError,
            ) as exc:
                error = f"{type(exc).__name__}: {exc}"
            sides.append(IVSide(side, premium, result, error))
        size = (
            2 * quote.bid_size * quote.ask_size / (quote.bid_size + quote.ask_size)
            if quote.bid_size + quote.ask_size > 0
            else 0.0
        )
        relative_spread = (ask - bid) / max(bid / 2 + ask / 2, 1e-6 * d * f)
        weight = size / max(relative_spread, 1e-4) ** 2
        observations.append(
            IVObservation(row, (), tuple(sides), t, log(contract.strike / f), weight)
        )
    return tuple(observations)


def _parameters(x: NDArray[np.float64]) -> SVIParameters:
    c, b, rho, m, sigma = (float(v) for v in x)
    return SVIParameters(c - b * sigma * sqrt(1 - rho * rho), b, rho, m, sigma)


class _LeastSquaresResult(Protocol):
    x: NDArray[np.float64]
    success: bool
    cost: float
    status: int
    message: str
    nfev: int


class _LeastSquares(Protocol):
    def __call__(
        self,
        fun: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        x0: list[float],
        *,
        bounds: tuple[list[float], list[float]],
        ftol: float,
        xtol: float,
        gtol: float,
        max_nfev: int,
        x_scale: Literal["jac"],
    ) -> _LeastSquaresResult: ...


# SciPy 1.15 stubs resolve the public export to its containing module.
_least_squares = cast(_LeastSquares, optimize.least_squares)


def _fit_slice(
    expiry: datetime, observations: list[IVObservation], max_evaluations: int
) -> SVISlice:
    ks = np.array([o.log_moneyness for o in observations], dtype=float)
    if len(set(ks)) < 5:
        raise InputValidationError(
            f"{expiry.isoformat()}: need five liquid unique strikes"
        )
    t = observations[0].time_to_expiry
    assert t is not None
    values = []
    for obs in observations:
        vol = obs.sides[1].volatility
        assert vol is not None
        values.append(vol * vol * t)
    ws = np.array(values)
    weights = np.array([o.weight for o in observations])
    weights /= np.mean(weights)
    roots = np.sqrt(weights)

    def residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
        p = _parameters(x)
        z = ks - p.m
        return np.asarray(
            roots * (p.a + p.b * (p.rho * z + np.sqrt(z * z + p.sigma**2)) - ws),
            dtype=np.float64,
        )

    best = None
    # Deterministic starts; transformed a enforces strictly positive global min w.
    for rho in (-0.6, 0.0, 0.6):
        start = [min(float(np.min(ws)), 1.0), 0.05, rho, 0.0, 0.3]
        fit = _least_squares(
            residual,
            start,
            bounds=([1e-12, 1e-10, -0.995, -3, 0.001], [4, 2, 0.995, 3, 3]),
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            max_nfev=max_evaluations,
            x_scale="jac",
        )
        if fit.success and (best is None or fit.cost < best.cost):
            best = fit
    if best is None:
        raise InputValidationError(
            f"{expiry.isoformat()}: raw SVI did not converge; "
            f"max_evaluations={max_evaluations}; last_status={fit.status}; "
            f"{fit.message}"
        )
    parameters = _parameters(best.x)
    residuals = tuple(
        FitResidual(
            obs.row.instrument.symbol,
            float(k),
            float(w),
            parameters.total_variance(float(k)),
            float(weight),
        )
        for obs, k, w, weight in zip(observations, ks, ws, weights, strict=True)
    )
    return SVISlice(
        expiry,
        t,
        parameters,
        (float(min(ks)), float(max(ks))),
        residuals,
        True,
        int(best.nfev),
        str(best.message),
        sqrt(2 * float(best.cost) / len(ks)),
    )


def fit_svi_slice(
    expiry: datetime, observations: tuple[IVObservation, ...]
) -> SVISlice:
    """Fit one expiry from recovered IV, with the standard SVI objective."""
    if any(
        not isinstance(obs.row.instrument, OptionContract)
        or obs.row.instrument.expiry != expiry
        for obs in observations
    ):
        raise InputValidationError("SVI observations must share the requested expiry")
    return _fit_slice(expiry, [obs for obs in observations if obs.fit_eligible], 3000)


def check_fitted_surface(
    surface: VolatilitySurface, *, check_points: int = 101, tolerance: float = 1e-8
) -> ArbitrageReport:
    """Check the common strike grid at expiry knots and maturity midpoints."""
    times = tuple(s.time_to_expiry for s in surface.slices)
    check_times = tuple(
        sorted((*times, *((a + b) / 2 for a, b in itertools.pairwise(times))))
    )
    return check_arbitrage(
        surface,
        tuple(float(k) for k in np.linspace(*surface.domain, check_points)),
        check_times,
        tolerance=tolerance,
    )


@dataclass(frozen=True, slots=True)
class SurfaceCalibration:
    surface: VolatilitySurface
    observations: tuple[IVObservation, ...]
    arbitrage: ArbitrageReport
    observed_arbitrage: ArbitrageReport

    @property
    def slices(self) -> tuple[SVISlice, ...]:
        return self.surface.slices


def calibrate_surface(
    snapshot: MarketSnapshot,
    discount_curve: DiscountCurve,
    forward_curve: ForwardCurve,
    *,
    max_evaluations: int = 3000,
    check_points: int = 101,
    arbitrage_tolerance: float = 1e-8,
) -> SurfaceCalibration:
    """Fit midpoint total variance; preserve bid/ask diagnostics and all rejects.

    All declared unexpired European expiries must have >=5 positive-IV liquid
    unique strikes. Failures raise, never silently drop a failed maturity.
    Arbitrage findings are returned as explicit flags, never smoothed away.

    Minimize sum(weight*(w_model-w_midpoint)**2) with SciPy bounded
    trust-region least squares, three deterministic starts, selecting the lowest
    converged cost. Bounds are b in [1e-10,2], rho in [-.995,.995],
    m in [-3,3], sigma in [.001,3], and global min(w) in [1e-12,4].
    a is recovered from min(w)-b*sigma*sqrt(1-rho**2), not clipped afterward.
    These bounds enforce positive variance, not butterfly/calendar freedom.
    ftol=xtol=gtol=1e-13; max_evaluations applies independently to each start.
    """
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, int)
        or max_evaluations < 1
        or isinstance(check_points, bool)
        or not isinstance(check_points, int)
        or check_points < 3
    ):
        raise InputValidationError("positive integer evaluations and >=3 check points")
    observations = recover_ivs(snapshot, discount_curve, forward_curve)
    groups: dict[datetime, list[IVObservation]] = {}
    for obs in observations:
        contract = obs.row.instrument
        if (
            isinstance(contract, OptionContract)
            and contract.expiry > snapshot.as_of
            and contract.exercise_style == "european"
        ):
            group = groups.setdefault(contract.expiry, [])
            if obs.fit_eligible:
                group.append(obs)
    surface = VolatilitySurface(
        tuple(
            _fit_slice(expiry, group, max_evaluations)
            for expiry, group in sorted(groups.items())
        )
    )
    report = check_fitted_surface(
        surface, check_points=check_points, tolerance=arbitrage_tolerance
    )
    observed = check_observed_arbitrage(observations, tolerance=arbitrage_tolerance)
    return SurfaceCalibration(surface, observations, report, observed)


def check_observed_arbitrage(
    observations: tuple[IVObservation, ...], *, tolerance: float = 1e-8
) -> ArbitrageReport:
    """Check recovered midpoint variances before fitting can obscure defects.

    Strike checks use each maturity's available nodes. Calendars compare only
    common k (rounded to 12 decimals to absorb log/exp roundoff), with no missing
    node interpolation. The report grids are unions, not a rectangular coverage
    claim. Repeated strikes (e.g. calls and puts) use worst-case combinations of
    their recovered midpoint prices, never averaging conflicting observations.
    """
    if not isfinite(tolerance) or tolerance < 0:
        raise InputValidationError("arbitrage tolerance must be finite and nonnegative")
    groups: dict[float, dict[float, list[float]]] = {}
    for obs in observations:
        if obs.rejected or obs.time_to_expiry is None or obs.log_moneyness is None:
            continue
        vol = obs.sides[1].volatility
        if vol is None or vol <= 0:
            continue
        group = groups.setdefault(obs.time_to_expiry, {})
        k = round(obs.log_moneyness, 12)
        group.setdefault(k, []).append(vol**2 * obs.time_to_expiry)
    evidence: list[ArbitrageEvidence] = []
    prior: dict[float, float] = {}
    for t, nodes in sorted(groups.items()):
        ks = tuple(sorted(nodes))
        lower = [float(price("call", 1, exp(k), 1, 1, sqrt(min(nodes[k])))) for k in ks]
        upper = [float(price("call", 1, exp(k), 1, 1, sqrt(max(nodes[k])))) for k in ks]
        evidence.extend(_strike_evidence(ks, lower, upper, t, tolerance))
        for k, values in nodes.items():
            w = min(values)
            if k in prior and w - prior[k] < -tolerance:
                evidence.append(
                    ArbitrageEvidence("calendar", k, t, w - prior[k], 0, float("inf"))
                )
            prior[k] = max(values)
    return ArbitrageReport(
        tuple(sorted({k for nodes in groups.values() for k in nodes})),
        tuple(sorted(groups)),
        tolerance,
        tuple(evidence),
        "observed nodes checked; calendars only at shared k; not a global proof",
    )


def _strike_evidence(
    ks: tuple[float, ...],
    lower: list[float],
    upper: list[float],
    t: float,
    tolerance: float,
) -> list[ArbitrageEvidence]:
    """Worst-case secants and butterflies, also supporting repeated strikes."""
    evidence: list[ArbitrageEvidence] = []
    strikes = [exp(k) for k in ks]
    for i in range(len(ks) - 1):
        width = strikes[i + 1] - strikes[i]
        lowest = (lower[i + 1] - upper[i]) / width
        highest = (upper[i + 1] - lower[i]) / width
        if lowest < -1 - tolerance:
            evidence.append(ArbitrageEvidence("monotonicity", ks[i], t, lowest, -1, 0))
        if highest > tolerance:
            evidence.append(ArbitrageEvidence("monotonicity", ks[i], t, highest, -1, 0))
    for i in range(1, len(ks) - 1):
        difference = (lower[i + 1] - upper[i]) / (strikes[i + 1] - strikes[i]) - (
            upper[i] - lower[i - 1]
        ) / (strikes[i] - strikes[i - 1])
        if difference < -tolerance:
            evidence.append(
                ArbitrageEvidence("butterfly", ks[i], t, difference, 0, float("inf"))
            )
    return evidence
