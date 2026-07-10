"""Safeguarded implied-volatility inversion for Black-Scholes-Merton."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

from quant_pricing.black_scholes import (
    OptionType,
    greeks,
    price,
    price_bounds,
)
from quant_pricing.exceptions import (
    ImpliedVolatilityError,
    InputValidationError,
    NoArbitrageError,
)

TerminationReason: TypeAlias = Literal[
    "lower_bound",
    "price_residual",
    "volatility_bracket",
]


@dataclass(frozen=True, slots=True)
class ImpliedVolatilityResult:
    """Diagnostics for a converged implied-volatility solve."""

    volatility: float
    converged: bool
    iterations: int
    model_price: float
    target_price: float
    residual: float
    bracket: tuple[float, float]
    price_scale: float
    termination_reason: TerminationReason
    method: str = "safeguarded-newton"

    @property
    def normalized_residual(self) -> float:
        """Return the signed pricing error divided by the monetary scale."""

        return self.residual / self.price_scale


def _require_finite(value: float, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise InputValidationError(f"{name} must be a real scalar") from error
    if not math.isfinite(converted):
        raise InputValidationError(f"{name} must be finite")
    return converted


def _scalar_price(
    option_type: OptionType,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
) -> float:
    value = price(
        option_type,
        spot,
        strike,
        time_to_expiry,
        rate,
        volatility,
        dividend_yield,
    )
    if not isinstance(value, float):  # pragma: no cover - internal contract guard
        raise TypeError("scalar implied-volatility inputs produced an array price")
    return value


def _result(
    *,
    volatility: float,
    iterations: int,
    model_price: float,
    target_price: float,
    bracket: tuple[float, float],
    price_scale: float,
    termination_reason: TerminationReason,
) -> ImpliedVolatilityResult:
    return ImpliedVolatilityResult(
        volatility=volatility,
        converged=True,
        iterations=iterations,
        model_price=model_price,
        target_price=target_price,
        residual=model_price - target_price,
        bracket=bracket,
        price_scale=price_scale,
        termination_reason=termination_reason,
    )


def implied_volatility(
    option_type: OptionType | str,
    target_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float = 0.0,
    *,
    initial_guess: float = 0.2,
    max_volatility: float = 10.0,
    price_tolerance: float = 0.0,
    relative_price_tolerance: float = 1e-12,
    volatility_tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> ImpliedVolatilityResult:
    """Recover volatility from an observed European option price.

    A Newton step is accepted only when it remains safely inside a valid
    volatility bracket; otherwise the method bisects the bracket. This retains
    Newton's local speed while preserving bisection's global convergence.
    Price convergence uses ``price_tolerance + relative_price_tolerance * scale``,
    where ``scale`` is derived from the option's no-arbitrage bounds. This makes
    the solve invariant to a change in monetary units.

    Raises:
        NoArbitrageError: If ``target_price`` violates model-independent bounds.
        ImpliedVolatilityError: If no finite root can be bracketed or convergence
            is not reached.
        InputValidationError: If inputs or solver controls are invalid.
    """

    kind = OptionType.parse(option_type)
    target = _require_finite(target_price, "target_price")
    spot_value = _require_finite(spot, "spot")
    strike_value = _require_finite(strike, "strike")
    time_value = _require_finite(time_to_expiry, "time_to_expiry")
    rate_value = _require_finite(rate, "rate")
    yield_value = _require_finite(dividend_yield, "dividend_yield")
    guess = _require_finite(initial_guess, "initial_guess")
    maximum = _require_finite(max_volatility, "max_volatility")
    price_tol = _require_finite(price_tolerance, "price_tolerance")
    relative_price_tol = _require_finite(
        relative_price_tolerance, "relative_price_tolerance"
    )
    volatility_tol = _require_finite(volatility_tolerance, "volatility_tolerance")

    if time_value <= 0.0:
        raise InputValidationError(
            "time_to_expiry must be positive because volatility is undefined at expiry"
        )
    if guess <= 0.0:
        raise InputValidationError("initial_guess must be strictly positive")
    if maximum <= 0.0:
        raise InputValidationError("max_volatility must be strictly positive")
    if price_tol < 0.0:
        raise InputValidationError("price_tolerance cannot be negative")
    if relative_price_tol < 0.0:
        raise InputValidationError("relative_price_tolerance cannot be negative")
    if price_tol == 0.0 and relative_price_tol == 0.0:
        raise InputValidationError(
            "at least one price convergence tolerance must be positive"
        )
    if volatility_tol <= 0.0:
        raise InputValidationError("volatility_tolerance must be strictly positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise InputValidationError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise InputValidationError("max_iterations must be strictly positive")

    lower_result, upper_result = price_bounds(
        kind,
        spot_value,
        strike_value,
        time_value,
        rate_value,
        yield_value,
    )
    if not isinstance(lower_result, float) or not isinstance(upper_result, float):
        raise TypeError("scalar implied-volatility inputs produced array bounds")
    lower_bound = lower_result
    upper_bound = upper_result
    price_scale = max(
        abs(lower_bound),
        abs(upper_bound),
        abs(target),
        math.ulp(0.0),
    )
    effective_price_tolerance = price_tol + relative_price_tol * price_scale
    bound_tolerance = 32.0 * math.ulp(price_scale)

    if target < lower_bound - bound_tolerance or target > upper_bound + bound_tolerance:
        raise NoArbitrageError(
            f"{kind.value} price {target:.16g} is outside no-arbitrage bounds "
            f"[{lower_bound:.16g}, {upper_bound:.16g}]"
        )
    if target <= lower_bound + max(bound_tolerance, effective_price_tolerance):
        return _result(
            volatility=0.0,
            iterations=0,
            model_price=lower_bound,
            target_price=target,
            bracket=(0.0, 0.0),
            price_scale=price_scale,
            termination_reason="lower_bound",
        )
    if target >= upper_bound - bound_tolerance:
        raise ImpliedVolatilityError(
            f"{kind.value} price at its upper bound requires infinite volatility"
        )

    low = 0.0
    high = min(max(guess, 0.2), maximum)
    high_price = _scalar_price(
        kind,
        spot_value,
        strike_value,
        time_value,
        rate_value,
        high,
        yield_value,
    )
    while high_price < target and high < maximum:
        high = min(2.0 * high, maximum)
        high_price = _scalar_price(
            kind,
            spot_value,
            strike_value,
            time_value,
            rate_value,
            high,
            yield_value,
        )

    if high_price < target - effective_price_tolerance:
        raise ImpliedVolatilityError(
            f"could not bracket implied volatility below max_volatility={maximum:g}; "
            f"price residual is {high_price - target:.6g}"
        )

    volatility = min(max(guess, low), high)
    if volatility in (low, high):
        volatility = 0.5 * (low + high)
    model_price = _scalar_price(
        kind,
        spot_value,
        strike_value,
        time_value,
        rate_value,
        volatility,
        yield_value,
    )

    for iteration in range(1, max_iterations + 1):
        residual = model_price - target
        if abs(residual) <= effective_price_tolerance:
            return _result(
                volatility=volatility,
                iterations=iteration,
                model_price=model_price,
                target_price=target,
                bracket=(low, high),
                price_scale=price_scale,
                termination_reason="price_residual",
            )

        if residual < 0.0:
            low = volatility
        else:
            high = volatility

        if high - low <= volatility_tol:
            midpoint = 0.5 * (low + high)
            midpoint_price = _scalar_price(
                kind,
                spot_value,
                strike_value,
                time_value,
                rate_value,
                midpoint,
                yield_value,
            )
            return _result(
                volatility=midpoint,
                iterations=iteration,
                model_price=midpoint_price,
                target_price=target,
                bracket=(low, high),
                price_scale=price_scale,
                termination_reason="volatility_bracket",
            )

        risk = greeks(
            kind,
            spot_value,
            strike_value,
            time_value,
            rate_value,
            volatility,
            yield_value,
        )
        if not isinstance(risk.vega, float):
            raise TypeError("scalar implied-volatility inputs produced array vega")

        width = high - low
        candidate = volatility - residual / risk.vega if risk.vega > 1e-14 else math.nan
        safe_low = low + 0.1 * width
        safe_high = high - 0.1 * width
        volatility = (
            candidate
            if math.isfinite(candidate) and safe_low < candidate < safe_high
            else 0.5 * (low + high)
        )
        model_price = _scalar_price(
            kind,
            spot_value,
            strike_value,
            time_value,
            rate_value,
            volatility,
            yield_value,
        )

    final_residual = model_price - target
    if final_residual < 0.0:
        low = max(low, volatility)
    else:
        high = min(high, volatility)
    if abs(final_residual) <= effective_price_tolerance:
        return _result(
            volatility=volatility,
            iterations=max_iterations,
            model_price=model_price,
            target_price=target,
            bracket=(low, high),
            price_scale=price_scale,
            termination_reason="price_residual",
        )
    if high - low <= volatility_tol:
        return _result(
            volatility=volatility,
            iterations=max_iterations,
            model_price=model_price,
            target_price=target,
            bracket=(low, high),
            price_scale=price_scale,
            termination_reason="volatility_bracket",
        )
    raise ImpliedVolatilityError(
        f"implied volatility did not converge in {max_iterations} iterations; "
        f"last volatility={volatility:.16g}, price residual={final_residual:.6g}, "
        f"normalized residual={final_residual / price_scale:.6g}, "
        f"bracket=[{low:.16g}, {high:.16g}]"
    )
