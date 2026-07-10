"""Quantitative option-pricing models and numerical methods."""

from quant_pricing.black_scholes import (
    Greeks,
    OptionType,
    Valuation,
    d1_d2,
    greeks,
    price,
    price_bounds,
    value_and_greeks,
)
from quant_pricing.exceptions import (
    ImpliedVolatilityError,
    InputValidationError,
    NoArbitrageError,
    PricingError,
)

__version__ = "0.1.0"
__all__ = [
    "Greeks",
    "ImpliedVolatilityError",
    "InputValidationError",
    "NoArbitrageError",
    "OptionType",
    "PricingError",
    "Valuation",
    "d1_d2",
    "greeks",
    "price",
    "price_bounds",
    "value_and_greeks",
]
