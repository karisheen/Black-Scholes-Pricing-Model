"""Black-Scholes-Merton pricing and analytic risk measures.

The implementation uses continuous compounding, a continuous dividend yield,
and calendar theta (the change in value as calendar time advances). Inputs may
be scalars or NumPy-broadcastable arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import log_ndtr, ndtr

from quant_pricing.exceptions import InputValidationError

FloatArray: TypeAlias = NDArray[np.float64]
NumericResult: TypeAlias = float | FloatArray
_SQRT_TWO_PI = np.sqrt(2.0 * np.pi)
_LOG_FLOAT_MAX = np.log(np.finfo(np.float64).max)


class OptionType(str, Enum):
    """Supported European option payoff types."""

    CALL = "call"
    PUT = "put"

    @classmethod
    def parse(cls, value: OptionType | str) -> OptionType:
        """Normalize a string or enum value to an :class:`OptionType`."""

        if isinstance(value, cls):
            return value
        try:
            return cls(value.lower())
        except (AttributeError, ValueError) as error:
            choices = ", ".join(member.value for member in cls)
            raise InputValidationError(
                f"option_type must be one of: {choices}; received {value!r}"
            ) from error


@dataclass(frozen=True, slots=True)
class Greeks:
    """Analytic Black-Scholes-Merton risk measures.

    Vega and rho are derivatives for a unit change in volatility and rate,
    respectively. Divide them by 100 for a one-percentage-point sensitivity.
    Theta is calendar theta per year.
    """

    delta: NumericResult
    gamma: NumericResult
    vega: NumericResult
    theta: NumericResult
    rho: NumericResult


@dataclass(frozen=True, slots=True)
class Valuation:
    """Option value and its analytic Greeks."""

    price: NumericResult
    greeks: Greeks


@dataclass(frozen=True, slots=True)
class _Inputs:
    spot: FloatArray
    strike: FloatArray
    time_to_expiry: FloatArray
    rate: FloatArray
    volatility: FloatArray
    dividend_yield: FloatArray


def _as_float_array(value: ArrayLike, name: str) -> FloatArray:
    if np.iscomplexobj(value):
        raise InputValidationError(f"{name} must contain real numbers")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InputValidationError(f"{name} must contain real numbers") from error
    if not np.all(np.isfinite(array)):
        raise InputValidationError(f"{name} must contain only finite values")
    return array


def _validate_and_broadcast(
    spot: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    dividend_yield: ArrayLike,
) -> _Inputs:
    names = (
        "spot",
        "strike",
        "time_to_expiry",
        "rate",
        "volatility",
        "dividend_yield",
    )
    arrays = tuple(
        _as_float_array(value, name)
        for value, name in zip(
            (spot, strike, time_to_expiry, rate, volatility, dividend_yield),
            names,
            strict=True,
        )
    )
    try:
        broadcast = np.broadcast_arrays(*arrays)
    except ValueError as error:
        shapes = ", ".join(
            f"{name}={array.shape}" for name, array in zip(names, arrays, strict=True)
        )
        raise InputValidationError(
            f"inputs cannot be broadcast together: {shapes}"
        ) from error

    result = _Inputs(*(np.asarray(item, dtype=np.float64) for item in broadcast))
    if np.any(result.spot <= 0.0):
        raise InputValidationError("spot must be strictly positive")
    if np.any(result.strike <= 0.0):
        raise InputValidationError("strike must be strictly positive")
    if np.any(result.time_to_expiry < 0.0):
        raise InputValidationError("time_to_expiry cannot be negative")
    if np.any(result.volatility < 0.0):
        raise InputValidationError("volatility cannot be negative")

    with np.errstate(invalid="ignore", over="ignore"):
        rate_time = result.rate * result.time_to_expiry
        yield_time = result.dividend_yield * result.time_to_expiry
    if not np.all(np.isfinite(rate_time)) or not np.all(np.isfinite(yield_time)):
        raise InputValidationError(
            "rate or dividend discount exponent exceeds float range"
        )
    if np.any(rate_time < -np.log(np.finfo(np.float64).max)):
        raise InputValidationError("rate produces a non-finite discount factor")
    if np.any(yield_time < -np.log(np.finfo(np.float64).max)):
        raise InputValidationError(
            "dividend_yield produces a non-finite discount factor"
        )
    return result


def _to_public(array: FloatArray) -> NumericResult:
    return float(array) if array.ndim == 0 else array


def _positive_difference(log_first: FloatArray, log_second: FloatArray) -> FloatArray:
    """Return ``max(exp(log_first) - exp(log_second), 0)`` stably."""

    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        log_ratio = np.minimum(log_second - log_first, 0.0)
        difference = np.exp(log_first) * -np.expm1(log_ratio)
    return np.where(np.isneginf(log_first), 0.0, difference)


def _log_discounted_assets(inputs: _Inputs) -> tuple[FloatArray, FloatArray]:
    log_spot = np.log(inputs.spot) - inputs.dividend_yield * inputs.time_to_expiry
    log_strike = np.log(inputs.strike) - inputs.rate * inputs.time_to_expiry
    if np.any(log_spot > _LOG_FLOAT_MAX):
        raise InputValidationError("discounted spot exceeds float range")
    if np.any(log_strike > _LOG_FLOAT_MAX):
        raise InputValidationError("discounted strike exceeds float range")
    return log_spot, log_strike


def _total_volatility(inputs: _Inputs) -> FloatArray:
    with np.errstate(over="ignore", under="ignore"):
        return inputs.volatility * np.sqrt(inputs.time_to_expiry)


def _standardized_terms(inputs: _Inputs) -> tuple[FloatArray, FloatArray]:
    sigma_root_time = _total_volatility(inputs)
    log_forward_moneyness = (
        np.log(inputs.spot)
        - np.log(inputs.strike)
        + inputs.rate * inputs.time_to_expiry
        - inputs.dividend_yield * inputs.time_to_expiry
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scaled_moneyness = log_forward_moneyness / sigma_root_time
    zero_scale_limit = np.where(
        log_forward_moneyness > 0.0,
        np.inf,
        np.where(log_forward_moneyness < 0.0, -np.inf, 0.0),
    )
    scaled_moneyness = np.where(
        sigma_root_time == 0.0, zero_scale_limit, scaled_moneyness
    )
    with np.errstate(invalid="ignore"):
        d1_value = scaled_moneyness + 0.5 * sigma_root_time
        d2_value = scaled_moneyness - 0.5 * sigma_root_time
    return d1_value, d2_value


def d1_d2(
    spot: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
) -> tuple[NumericResult, NumericResult]:
    """Return the standardized Black-Scholes-Merton terms ``d1`` and ``d2``.

    These terms are undefined at expiry or zero volatility; use :func:`price`
    and :func:`greeks` for those explicitly supported boundary cases.
    """

    inputs = _validate_and_broadcast(
        spot, strike, time_to_expiry, rate, volatility, dividend_yield
    )
    if np.any(inputs.time_to_expiry == 0.0):
        raise InputValidationError("d1 and d2 are undefined at expiry")
    if np.any(inputs.volatility == 0.0):
        raise InputValidationError("d1 and d2 are undefined at zero volatility")
    first, second = _standardized_terms(inputs)
    return _to_public(first), _to_public(second)


def price_bounds(
    option_type: OptionType | str,
    spot: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    rate: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
) -> tuple[NumericResult, NumericResult]:
    """Return model-independent lower and upper European option-price bounds."""

    kind = OptionType.parse(option_type)
    inputs = _validate_and_broadcast(
        spot, strike, time_to_expiry, rate, 0.0, dividend_yield
    )
    log_spot, log_strike = _log_discounted_assets(inputs)
    discounted_spot = np.exp(log_spot)
    discounted_strike = np.exp(log_strike)
    if kind is OptionType.CALL:
        lower = _positive_difference(log_spot, log_strike)
        upper = discounted_spot
    else:
        lower = _positive_difference(log_strike, log_spot)
        upper = discounted_strike
    return _to_public(lower), _to_public(upper)


def price(
    option_type: OptionType | str,
    spot: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
) -> NumericResult:
    """Price a European call or put under Black-Scholes-Merton.

    At expiry the exact intrinsic payoff is returned. At zero volatility the
    discounted deterministic payoff is returned. All other inputs use stable
    log-CDF tail calculations to reduce cancellation error.
    """

    kind = OptionType.parse(option_type)
    inputs = _validate_and_broadcast(
        spot, strike, time_to_expiry, rate, volatility, dividend_yield
    )
    log_spot, log_strike = _log_discounted_assets(inputs)

    if kind is OptionType.CALL:
        deterministic = _positive_difference(log_spot, log_strike)
        expiry_payoff = np.maximum(inputs.spot - inputs.strike, 0.0)
    else:
        deterministic = _positive_difference(log_strike, log_spot)
        expiry_payoff = np.maximum(inputs.strike - inputs.spot, 0.0)

    regular = (inputs.time_to_expiry > 0.0) & (_total_volatility(inputs) > 0.0)
    if np.any(regular):
        first, second = _standardized_terms(inputs)
        if kind is OptionType.CALL:
            log_first = log_spot + log_ndtr(first)
            log_second = log_strike + log_ndtr(second)
        else:
            log_first = log_strike + log_ndtr(-second)
            log_second = log_spot + log_ndtr(-first)
        stochastic = _positive_difference(log_first, log_second)
        result = np.where(regular, stochastic, deterministic)
    else:
        result = deterministic

    result = np.where(inputs.time_to_expiry == 0.0, expiry_payoff, result)
    return _to_public(np.maximum(result, 0.0))


def greeks(
    option_type: OptionType | str,
    spot: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
) -> Greeks:
    """Calculate Delta, Gamma, Vega, calendar Theta, and Rho analytically.

    At expiry this function uses an operational payoff convention: midpoint
    Delta at the kink and zero Gamma, Vega, Theta, and Rho. At positive maturity
    and zero effective volatility, first-order risk uses one-sided limits and
    midpoint values at the deterministic exercise boundary. Gamma and Vega are
    zero because their distributional limits are not finite point values.
    """

    kind = OptionType.parse(option_type)
    inputs = _validate_and_broadcast(
        spot, strike, time_to_expiry, rate, volatility, dividend_yield
    )
    log_spot, log_strike = _log_discounted_assets(inputs)
    discounted_spot = np.exp(log_spot)
    discounted_strike = np.exp(log_strike)
    discount_yield = np.exp(-inputs.dividend_yield * inputs.time_to_expiry)

    expiry = inputs.time_to_expiry == 0.0
    total_volatility = _total_volatility(inputs)
    deterministic = (~expiry) & (total_volatility == 0.0)
    regular = (~expiry) & (total_volatility > 0.0)

    spot_greater = inputs.spot > inputs.strike
    spot_equal = inputs.spot == inputs.strike
    equality_tolerance = (
        8.0
        * np.finfo(np.float64).eps
        * np.maximum(np.maximum(np.abs(log_spot), np.abs(log_strike)), 1.0)
    )
    forward_equal = np.abs(log_spot - log_strike) <= equality_tolerance
    forward_greater = (log_spot > log_strike) & ~forward_equal

    if kind is OptionType.CALL:
        expiry_weight = np.where(spot_greater, 1.0, np.where(spot_equal, 0.5, 0.0))
        deterministic_weight = np.where(
            forward_greater, 1.0, np.where(forward_equal, 0.5, 0.0)
        )
        delta_value = expiry_weight
        rho_value = deterministic_weight * (
            inputs.strike
            * inputs.time_to_expiry
            * np.exp(-inputs.rate * inputs.time_to_expiry)
        )
        theta_value = deterministic_weight * (
            inputs.dividend_yield * discounted_spot - inputs.rate * discounted_strike
        )
    else:
        expiry_weight = np.where(spot_greater, 0.0, np.where(spot_equal, -0.5, -1.0))
        deterministic_weight = np.where(
            forward_greater, 0.0, np.where(forward_equal, -0.5, -1.0)
        )
        delta_value = expiry_weight
        rho_value = deterministic_weight * (
            inputs.strike
            * inputs.time_to_expiry
            * np.exp(-inputs.rate * inputs.time_to_expiry)
        )
        theta_value = deterministic_weight * (
            inputs.dividend_yield * discounted_spot - inputs.rate * discounted_strike
        )

    delta_value = np.where(
        deterministic, discount_yield * deterministic_weight, delta_value
    )
    rho_value = np.where(expiry, 0.0, rho_value)
    theta_value = np.where(expiry, 0.0, theta_value)
    gamma_value = np.zeros_like(inputs.spot)
    vega_value = np.zeros_like(inputs.spot)

    if np.any(regular):
        first, second = _standardized_terms(inputs)
        first = np.where(regular, first, 0.0)
        second = np.where(regular, second, 0.0)
        safe_time = np.where(regular, inputs.time_to_expiry, 1.0)
        safe_volatility = np.where(regular, inputs.volatility, 1.0)
        root_time = np.sqrt(safe_time)
        with np.errstate(over="ignore", under="ignore"):
            density = np.exp(-0.5 * first**2) / _SQRT_TWO_PI
            common_theta = (
                -discounted_spot * density * safe_volatility / (2.0 * root_time)
            )
            regular_gamma = (
                discount_yield * density / (inputs.spot * safe_volatility * root_time)
            )
            regular_vega = discounted_spot * density * root_time
        if kind is OptionType.CALL:
            regular_delta = discount_yield * ndtr(first)
            regular_rho = (
                inputs.strike
                * inputs.time_to_expiry
                * np.exp(-inputs.rate * inputs.time_to_expiry)
                * ndtr(second)
            )
            regular_theta = (
                common_theta
                - inputs.rate * discounted_strike * ndtr(second)
                + inputs.dividend_yield * discounted_spot * ndtr(first)
            )
        else:
            regular_delta = discount_yield * (ndtr(first) - 1.0)
            regular_rho = (
                -inputs.strike
                * inputs.time_to_expiry
                * np.exp(-inputs.rate * inputs.time_to_expiry)
                * ndtr(-second)
            )
            regular_theta = (
                common_theta
                + inputs.rate * discounted_strike * ndtr(-second)
                - inputs.dividend_yield * discounted_spot * ndtr(-first)
            )

        delta_value = np.where(regular, regular_delta, delta_value)
        gamma_value = np.where(regular, regular_gamma, gamma_value)
        vega_value = np.where(regular, regular_vega, vega_value)
        theta_value = np.where(regular, regular_theta, theta_value)
        rho_value = np.where(regular, regular_rho, rho_value)

    return Greeks(
        delta=_to_public(delta_value),
        gamma=_to_public(gamma_value),
        vega=_to_public(vega_value),
        theta=_to_public(theta_value),
        rho=_to_public(rho_value),
    )


def value_and_greeks(
    option_type: OptionType | str,
    spot: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
) -> Valuation:
    """Return an option price and all supported analytic Greeks."""

    return Valuation(
        price=price(
            option_type,
            spot,
            strike,
            time_to_expiry,
            rate,
            volatility,
            dividend_yield,
        ),
        greeks=greeks(
            option_type,
            spot,
            strike,
            time_to_expiry,
            rate,
            volatility,
            dividend_yield,
        ),
    )
