"""Paid-upfront Black-76 for single-currency linear European options.

``price(kind, F, K, T, D, sigma)`` and ``price_bounds(kind, F, K, T, D)``
broadcast NumPy inputs. Prices and IV monetary diagnostics are present values per
underlying unit. D is supplied directly, including at T=0: the deterministic
limit is D times intrinsic, not a rate reconstructed from -log(D)/T.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from quant_pricing.black_scholes import (
    FloatArray,
    NumericResult,
    OptionType,
    _as_float_array,
    _lognormal_price,
    _positive_difference,
    _to_public,
)
from quant_pricing.exceptions import (
    ImpliedVolatilityError,
    InputValidationError,
    NoArbitrageError,
)
from quant_pricing.implied_volatility import ImpliedVolatilityResult, _require_finite


def _inputs(
    forward: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    discount_factor: ArrayLike,
    volatility: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    names = ("forward", "strike", "time_to_expiry", "discount_factor", "volatility")
    arrays = tuple(
        _as_float_array(value, name)
        for value, name in zip(
            (forward, strike, time_to_expiry, discount_factor, volatility),
            names,
            strict=True,
        )
    )
    try:
        f, k, t, d, sigma = np.broadcast_arrays(*arrays)
    except ValueError as error:
        raise InputValidationError("inputs cannot be broadcast together") from error
    for name, value in (("forward", f), ("strike", k), ("discount_factor", d)):
        if np.any(value <= 0.0):
            raise InputValidationError(f"{name} must be strictly positive")
    if np.any(t < 0.0) or np.any(sigma < 0.0):
        raise InputValidationError("time_to_expiry and volatility cannot be negative")
    log_f, log_k = np.log(f), np.log(k)
    log_df, log_dk = np.log(d) + log_f, np.log(d) + log_k
    if np.any(np.maximum(log_df, log_dk) > np.log(np.finfo(np.float64).max)):
        raise InputValidationError("discounted forward or strike exceeds float range")
    with np.errstate(over="ignore", under="ignore"):
        total_vol = sigma * np.sqrt(t)
    return log_df, log_dk, log_f - log_k, total_vol


def price_bounds(
    option_type: OptionType | str,
    forward: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    discount_factor: ArrayLike,
) -> tuple[NumericResult, NumericResult]:
    """Return discounted intrinsic and the infinite-volatility price bound."""
    kind = OptionType.parse(option_type)
    log_f, log_k, _, _ = _inputs(forward, strike, time_to_expiry, discount_factor, 0.0)
    first, second = (log_f, log_k) if kind is OptionType.CALL else (log_k, log_f)
    return _to_public(_positive_difference(first, second)), _to_public(np.exp(first))


def price(
    option_type: OptionType | str,
    forward: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    discount_factor: ArrayLike,
    volatility: ArrayLike,
) -> NumericResult:
    """Price using stable discounted log-assets, with NumPy broadcasting.

    Zero time or volatility returns discounted intrinsic. A factor above one
    is valid (negative rates); neither FX conversion nor inverse payoffs apply.
    """
    kind = OptionType.parse(option_type)
    return _to_public(
        _lognormal_price(
            kind, *_inputs(forward, strike, time_to_expiry, discount_factor, volatility)
        )
    )


def implied_volatility(
    option_type: OptionType | str,
    target_price: float,
    forward: float,
    strike: float,
    time_to_expiry: float,
    discount_factor: float,
    *,
    max_volatility: float = 10.0,
    relative_price_tolerance: float = 1e-12,
    volatility_tolerance: float = 1e-10,
    max_iterations: int = 120,
) -> ImpliedVolatilityResult:
    """Invert a scalar PV price by bracketed bisection, never by an implied rate.

    All price diagnostics, including scale and residual, retain the original PV
    units. At expiry IV is undefined. At the lower bound IV is zero; the upper
    bound requires infinite volatility. Controls and errors follow the spot IV
    convention; ``method`` explicitly identifies the bisection solver.
    """
    kind = OptionType.parse(option_type)
    target = _require_finite(target_price, "target_price")
    f = _require_finite(forward, "forward")
    k = _require_finite(strike, "strike")
    t = _require_finite(time_to_expiry, "time_to_expiry")
    d = _require_finite(discount_factor, "discount_factor")
    maximum = _require_finite(max_volatility, "max_volatility")
    relative_tol = _require_finite(relative_price_tolerance, "relative_price_tolerance")
    vol_tol = _require_finite(volatility_tolerance, "volatility_tolerance")
    if t <= 0.0:
        raise InputValidationError(
            "time_to_expiry must be positive for implied volatility"
        )
    if maximum <= 0.0 or relative_tol <= 0.0 or vol_tol <= 0.0:
        raise InputValidationError(
            "solver tolerances and max_volatility must be positive"
        )
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise InputValidationError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise InputValidationError("max_iterations must be positive")
    lower_value, upper_value = price_bounds(kind, f, k, t, d)
    lower, upper = float(lower_value), float(upper_value)
    scale = max(upper, abs(target), math.ulp(0.0))
    tolerance = relative_tol * scale
    bound_tolerance = 32.0 * math.ulp(scale)
    if target < lower - bound_tolerance or target > upper + bound_tolerance:
        raise NoArbitrageError(
            f"price {target:g} is outside bounds [{lower:g}, {upper:g}]"
        )

    def result(
        vol: float, value: float, iterations: int, low: float, high: float
    ) -> ImpliedVolatilityResult:
        return ImpliedVolatilityResult(
            volatility=vol,
            converged=True,
            iterations=iterations,
            model_price=value,
            target_price=target,
            residual=value - target,
            bracket=(low, high),
            price_scale=scale,
            termination_reason=(
                "lower_bound"
                if iterations == 0
                else "price_residual"
                if abs(value - target) <= tolerance
                else "volatility_bracket"
            ),
            method="bisection",
        )

    if target <= lower + max(tolerance, bound_tolerance):
        return result(0.0, lower, 0, 0.0, 0.0)
    if target >= upper - bound_tolerance:
        raise ImpliedVolatilityError(
            "price at upper bound requires infinite volatility"
        )
    low, high = 0.0, min(0.2, maximum)
    high_price = float(price(kind, f, k, t, d, high))
    while high_price < target and high < maximum:
        high = min(2.0 * high, maximum)
        high_price = float(price(kind, f, k, t, d, high))
    if high_price < target - tolerance:
        raise ImpliedVolatilityError("could not bracket IV below max_volatility")
    if abs(high_price - target) <= tolerance:
        return result(high, high_price, 1, low, high)
    for iteration in range(1, max_iterations + 1):
        midpoint = 0.5 * (low + high)
        value = float(price(kind, f, k, t, d, midpoint))
        if abs(value - target) <= tolerance or high - low <= vol_tol:
            return result(midpoint, value, iteration, low, high)
        if value < target:
            low = midpoint
        else:
            high = midpoint
    raise ImpliedVolatilityError(
        f"implied volatility did not converge in {max_iterations} iterations; "
        f"PV residual={value - target:g}, bracket=[{low:g}, {high:g}]"
    )
