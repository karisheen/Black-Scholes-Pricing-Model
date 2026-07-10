"""Domain-specific exceptions raised by the pricing library."""


class PricingError(Exception):
    """Base class for pricing and numerical-method errors."""


class InputValidationError(PricingError, ValueError):
    """Raised when model inputs are non-finite or outside their valid domain."""


class NoArbitrageError(PricingError, ValueError):
    """Raised when an observed option price violates no-arbitrage bounds."""


class ImpliedVolatilityError(PricingError):
    """Raised when no finite implied volatility can be found."""
