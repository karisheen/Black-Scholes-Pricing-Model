"""Independent escrowed-cash-dividend CRR reference, not an analytic BSM wrapper.

Prices are cash per underlying unit, volatility is annualized, and the flat rate
is continuously compounded per ACT/365F year. Times are elapsed ACT/365F years;
callers convert aware dates before calling. Cash dividends are paid at ex-time
(no delayed payment). They must lie strictly inside expiry and on the uniform
grid within 1e-10 of a step. Misalignment is rejected, never silently rounded.
Refinement must preserve that alignment; even/odd CRR convergence can oscillate.

The recombining state is X = spot minus PV of remaining cash dividends. X follows
CRR with the supplied volatility; spot adds back the deterministic escrow.
Spot therefore jumps down by the cash amount at each ex-time. This is an
approximation: volatility belongs to X, not total spot, and escrow must remain
positive. It is not an exact local-volatility cash-dividend model. American
exercise is checked both immediately before and after each dividend, and at
other grid times (a Bermudan approximation to continuous American exercise).

Work and policy storage are O(steps**2), bounded to 4096 steps. Zero volatility
uses one deterministic path; zero maturity settles intrinsic immediately.
Exercise ties continue; expiry exercises only positive intrinsic. Numerical ties
use relative tolerance 1e-12 (option values early, stock/strike at expiry).
Boundaries describe exercised nodes, not certified continuous boundaries.
Probabilities use
risk-neutral surviving path mass under this policy, NOT real-world forecasts or
broker assignment probabilities. No taxes, fees, borrow, settlement lags,
physical delivery conventions, holder heterogeneity, FX or inverse payoffs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, expm1, isfinite, log, sqrt
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from quant_pricing.exceptions import InputValidationError

FloatArray = NDArray[np.float64]
ActionArray = NDArray[np.uint8]
Phase = Literal["regular", "pre_dividend", "post_dividend", "expiry"]


def _finite(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise InputValidationError(f"{name} must be a finite real number")


def _exp(value: float) -> float:
    try:
        result = exp(value)
    except OverflowError as exc:
        raise InputValidationError("lattice exceeds finite numerical range") from exc
    if not isfinite(result) or result == 0:
        raise InputValidationError("lattice exceeds finite numerical range")
    return result


@dataclass(frozen=True, slots=True)
class CashDividend:
    """Positive cash per unit at an exact elapsed ex/payment time in years."""

    time: float
    amount: float

    def __post_init__(self) -> None:
        _finite(self.time, "dividend time")
        _finite(self.amount, "dividend amount")
        if self.time <= 0 or self.amount <= 0:
            raise InputValidationError("dividend time and amount must be positive")


@dataclass(frozen=True, slots=True)
class ExerciseBoundary:
    """Finite exercised-node range, including unreachable nodes, and reached mass.

    For calls the lowest exercised stock is the grid threshold; for puts it is
    the highest. Mass is unconditional, after earlier exercise has been removed.
    Distinct pre/post dividend rows share a time but different stock levels.
    """

    step: int
    time: float
    phase: Phase
    minimum_stock: float
    maximum_stock: float
    exercised_nodes: int
    probability: float


@dataclass(frozen=True, slots=True)
class ExerciseExposure:
    """Risk-neutral modeled exercise/assignment, not a forecast.

    Quantities are nonnegative underlying units (contracts times multiplier),
    not signed delta or a settlement cash flow. A short writer's assignment is
    modeled by the representative long holder's optimal exercise policy.
    """

    action: Literal["long_exercise", "writer_assignment", "flat"]
    early_probability: float
    expiry_probability: float
    early_quantity: float
    expiry_quantity: float
    total_quantity: float


@dataclass(frozen=True, slots=True)
class BinomialResult:
    """Model price, exercise evidence and explicit grid/numerical diagnostics."""

    price: float
    early_exercise_probability: float
    expiry_exercise_probability: float
    no_exercise_probability: float
    boundaries: tuple[ExerciseBoundary, ...]
    steps: int
    time_step: float
    up_probability: float
    escrowed_spot: float
    dividend_steps: tuple[int, ...]

    def exposure(self, contracts: float, multiplier: float = 1.0) -> ExerciseExposure:
        """Scale exercise probabilities to modeled unsigned underlying quantity."""
        _finite(contracts, "contracts")
        _finite(multiplier, "multiplier")
        if multiplier <= 0:
            raise InputValidationError("multiplier must be positive")
        units = abs(contracts) * multiplier
        _finite(units, "exposure units")
        action: Literal["long_exercise", "writer_assignment", "flat"] = "flat"
        if contracts > 0:
            action = "long_exercise"
        elif contracts < 0:
            action = "writer_assignment"
        early = units * self.early_exercise_probability
        expiry = units * self.expiry_exercise_probability
        return ExerciseExposure(
            action,
            self.early_exercise_probability,
            self.expiry_exercise_probability,
            early,
            expiry,
            early + expiry,
        )


def _schedule(
    dividends: tuple[CashDividend, ...], time: float, steps: int, dt: float
) -> FloatArray:
    cash = np.zeros(steps + 1)
    for dividend in dividends:
        if not isinstance(dividend, CashDividend):
            raise InputValidationError("dividends must contain CashDividend records")
        if dividend.time >= time:
            raise InputValidationError("dividends must be strictly inside expiry")
        index = dividend.time / dt
        if abs(index - round(index)) > 1e-10:
            raise InputValidationError("dividend time must align exactly with the grid")
        step = round(index)
        if not 0 < step < steps:
            raise InputValidationError("dividend time must be an interior grid point")
        cash[step] = float(cash[step]) + dividend.amount
    if not np.all(np.isfinite(cash)):
        raise InputValidationError("aggregate dividend amount must be finite")
    return cash


def _stocks(
    step: int,
    spot: float,
    h: float,
    growth: float,
    cash: float,
    deterministic: bool,
) -> FloatArray:
    if step == 0:
        nodes = np.array([spot + cash])
    elif deterministic:
        nodes = np.array([spot * _exp(growth * step) + cash])
    else:
        with np.errstate(over="ignore", under="ignore"):
            nodes = np.exp(log(spot) + (2 * np.arange(step + 1) - step) * h) + cash
    if not np.all(np.isfinite(nodes)) or np.any(nodes <= 0):
        raise InputValidationError("stock nodes exceed positive finite numerical range")
    return nodes


def _payoff(kind: str, stock: FloatArray, strike: float) -> FloatArray:
    return np.maximum(stock - strike if kind == "call" else strike - stock, 0.0)


def _exercise(intrinsic: FloatArray, continuation: FloatArray) -> NDArray[np.bool_]:
    # Suppress numerical ties, notably a zero-rate non-dividend call.
    tolerance = 1e-12 * np.maximum(intrinsic, continuation)
    return (intrinsic > 0) & (intrinsic - continuation > tolerance)


def price(
    kind: str,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    *,
    style: str = "european",
    steps: int = 200,
    dividends: tuple[CashDividend, ...] = (),
) -> BinomialResult:
    """Value a linear single-currency call/put and its representative policy.

    European exercise occurs only at expiry; American exercise can occur at
    every grid time including time zero. The CRR probability must lie in [0, 1];
    coarse grids violating this condition are rejected (increase steps). No
    convergence claim is implied by one run: compare successively aligned
    refinements. Positive rate with no dividends implies no early call exercise.
    Negative rates need not satisfy that conventional result.
    """
    for name, value in (
        ("spot", spot),
        ("strike", strike),
        ("time_to_expiry", time_to_expiry),
        ("rate", rate),
        ("volatility", volatility),
    ):
        _finite(value, name)
    if spot <= 0 or strike <= 0 or time_to_expiry < 0 or volatility < 0:
        raise InputValidationError(
            "positive spot/strike and nonnegative time/vol required"
        )
    if kind not in ("call", "put") or style not in ("european", "american"):
        raise InputValidationError("kind/style must be call/put and european/american")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 4096:
        raise InputValidationError("steps must be an integer in [1, 4096]")
    n = steps if time_to_expiry > 0 else 0
    dt = time_to_expiry / steps
    if n > 0 and dt == 0:
        raise InputValidationError("time step is below finite numerical resolution")
    cash = _schedule(dividends, time_to_expiry, n, dt)
    discount = _exp(-rate * dt)
    growth = rate * dt
    deterministic = volatility == 0 or n == 0
    h = volatility * sqrt(dt)
    if deterministic:
        probability = 1.0
    else:
        if not isfinite(h) or h == 0 or abs(growth) > h or h > 350:
            raise InputValidationError("invalid CRR probability/range; refine the grid")
        probability = expm1(growth + h) / expm1(2 * h)
    remaining = np.zeros(n + 1)
    with np.errstate(over="ignore", invalid="ignore"):
        for step in range(n - 1, -1, -1):
            remaining[step] = (remaining[step + 1] + cash[step + 1]) * discount
    escrow = spot - float(remaining[0])
    if not isfinite(escrow) or escrow <= 0:
        raise InputValidationError("spot minus PV of cash dividends must be positive")
    stocks = _stocks(n, escrow, h, growth, 0.0, deterministic)
    values = _payoff(kind, stocks, strike)
    policies: list[ActionArray] = [np.empty(0, dtype=np.uint8) for _ in range(n + 1)]
    expiry_tolerance = 1e-12 * np.maximum(stocks, strike)
    policies[n] = np.where(values > expiry_tolerance, 3, 0).astype(np.uint8)
    # Each stored range is associated with an action code in the policy array.
    ranges: dict[tuple[int, int], tuple[Phase, float, float, int]] = {}

    def record(step: int, code: int, phase: Phase, levels: FloatArray) -> None:
        exercised = levels[policies[step] == code]
        if exercised.size:
            ranges[step, code] = (
                phase,
                float(exercised.min()),
                float(exercised.max()),
                exercised.size,
            )

    record(n, 3, "expiry", stocks)
    for step in range(n - 1, -1, -1):
        with np.errstate(over="ignore", invalid="ignore"):
            values = discount * (
                values
                if deterministic
                else (1 - probability) * values[:-1] + probability * values[1:]
            )
        if not np.all(np.isfinite(values)):
            raise InputValidationError("option values exceed finite numerical range")
        policies[step] = np.zeros(values.size, dtype=np.uint8)
        if style == "american":
            stocks = _stocks(
                step, escrow, h, growth, float(remaining[step]), deterministic
            )
            intrinsic = _payoff(kind, stocks, strike)
            post = _exercise(intrinsic, values)
            values = np.where(post, intrinsic, values)
            policies[step][post] = 2
            if cash[step] > 0:
                pre_stock = stocks + cash[step]
                if not np.all(np.isfinite(pre_stock)):
                    raise InputValidationError("pre-dividend stocks must be finite")
                intrinsic = _payoff(kind, pre_stock, strike)
                pre = _exercise(intrinsic, values)
                values = np.where(pre, intrinsic, values)
                policies[step][pre] = 1
                record(step, 1, "pre_dividend", pre_stock)
            record(step, 2, "post_dividend" if cash[step] > 0 else "regular", stocks)
    model_price = float(values[0])
    mass = np.ones(1)
    boundaries: list[ExerciseBoundary] = []
    early, expiry = 0.0, 0.0
    for step, policy in enumerate(policies):
        for code in (1, 2, 3):
            metadata = ranges.get((step, code))
            if metadata is not None:
                phase, low, high, count = metadata
                reached = float(mass[policy == code].sum())
                boundaries.append(
                    ExerciseBoundary(step, step * dt, phase, low, high, count, reached)
                )
                if code == 3:
                    expiry += reached
                else:
                    early += reached
        mass[policy != 0] = 0
        if step < n and not deterministic:
            next_mass = np.zeros(step + 2)
            next_mass[:-1] += mass * (1 - probability)
            next_mass[1:] += mass * probability
            mass = next_mass
    # Accumulation can overshoot unity by roundoff only; normalize all outcomes.
    unexercised = float(mass.sum())
    total = early + expiry + unexercised
    return BinomialResult(
        model_price,
        early / total,
        expiry / total,
        unexercised / total,
        tuple(boundaries),
        n,
        dt,
        probability,
        escrow,
        tuple(int(step) for step in np.flatnonzero(cash)),
    )
