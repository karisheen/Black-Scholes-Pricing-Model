"""Deterministic nonflat SSVI truth and deliberately imperfect observed chains.

Truth uses Gatheral-Jacquier SSVI, w=theta/2*(1+rho*phi*k+
 sqrt((phi*k+rho)^2+1-rho^2)), theta=variance_rate*T, constant phi.
For 0<T<=max_time, the sufficient global-strike butterfly conditions are
 theta*phi*(1+|rho|)<4 and theta*phi^2*(1+|rho|)<=4. Calendar conditions are
 increasing theta and 0<=d(theta*phi)/dtheta<=phi*(1+sqrt(1-rho^2))/rho^2
 (upper bound infinite at rho=0). Constant phi satisfies this condition.
These analytic conditions prove the TRUTH family on its stated maturity domain,
not the finite observed bid/ask chain, noisy premiums, or independently fitted SVI.
Reference: Gatheral & Jacquier, Arbitrage-free SVI volatility surfaces (2014),
Theorems 4.1 and 4.2, https://arxiv.org/abs/1204.0646 .
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import exp, isfinite, sin, sqrt

from quant_pricing.black76 import price, price_bounds
from quant_pricing.black_scholes import OptionType
from quant_pricing.conventions import utc_timestamp, year_fraction
from quant_pricing.curves import DiscountCurve, ForwardCurve
from quant_pricing.exceptions import InputValidationError
from quant_pricing.market import FrozenQuoteProvider, OptionContract, Quote, Underlying
from quant_pricing.surface import SVIParameters


@dataclass(frozen=True, slots=True)
class SSVITruth:
    """Analytically sufficient no-arbitrage conditions, bounded maturity domain."""

    max_time: float
    variance_rate: float = 0.04
    rho: float = -0.35
    phi: float = 2.0

    def __post_init__(self) -> None:
        values = (self.max_time, self.variance_rate, self.rho, self.phi)
        if not all(isfinite(x) for x in values):
            raise InputValidationError("SSVI inputs must be finite")
        if min(self.max_time, self.variance_rate, self.phi) <= 0 or abs(self.rho) >= 1:
            raise InputValidationError("SSVI requires positive scales and |rho|<1")
        theta = self.variance_rate * self.max_time
        if (
            theta * self.phi * (1 + abs(self.rho)) >= 4
            or theta * self.phi**2 * (1 + abs(self.rho)) > 4
        ):
            raise InputValidationError("SSVI sufficient butterfly conditions fail")

    @property
    def arbitrage_claim(self) -> str:
        return "SSVI sufficient conditions proven for all k and 0<T<=max_time"

    def parameters(self, time_to_expiry: float) -> SVIParameters:
        if not isfinite(time_to_expiry) or not 0 < time_to_expiry <= self.max_time:
            raise InputValidationError("truth maturity outside (0,max_time]")
        theta = self.variance_rate * time_to_expiry
        return SVIParameters(
            theta * (1 - self.rho**2) / 2,
            theta * self.phi / 2,
            self.rho,
            -self.rho / self.phi,
            sqrt(1 - self.rho**2) / self.phi,
        )

    def total_variance(self, log_moneyness: float, time_to_expiry: float) -> float:
        return self.parameters(time_to_expiry).total_variance(log_moneyness)

    def volatility(self, log_moneyness: float, time_to_expiry: float) -> float:
        return sqrt(self.total_variance(log_moneyness, time_to_expiry) / time_to_expiry)


@dataclass(frozen=True, slots=True)
class ChainFaults:
    """Zero-based expiry-major/strike-minor contract indices; faults may overlap.

    Missing means a declared instrument without an observed quote, not removal
    from the universe. Noise is a deterministic sin(index+1) displacement in
    units of the unperturbed half-spread, preserving raw prices for diagnostics.
    """

    missing: tuple[int, ...] = ()
    stale: tuple[int, ...] = ()
    crossed: tuple[int, ...] = ()
    noise: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.noise) or self.noise < 0:
            raise InputValidationError("noise must be finite and nonnegative")
        for name in ("missing", "stale", "crossed"):
            indices = tuple(getattr(self, name))
            if any(
                isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices
            ):
                raise InputValidationError("fault indices must be nonnegative integers")
            object.__setattr__(self, name, indices)


@dataclass(frozen=True, slots=True)
class SyntheticChain:
    as_of: datetime
    provider: FrozenQuoteProvider
    discount_curve: DiscountCurve
    forward_curve: ForwardCurve
    truth: SSVITruth
    faults: ChainFaults


_CLEAN_FAULTS = ChainFaults()


def generate_chain(
    as_of: datetime,
    *,
    maturity_days: tuple[int, ...] = (90, 180, 365),
    log_moneyness: tuple[float, ...] = (
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
    ),
    faults: ChainFaults = _CLEAN_FAULTS,
    spot: float = 100.0,
    rate: float = 0.03,
    dividend_yield: float = 0.01,
    relative_spread: float = 0.01,
    per_contract: bool = False,
) -> SyntheticChain:
    """Generate clean OTM calls/puts unless explicitly injecting quote faults.

    Uses curve F and D, European linear USD contracts, multiplier 100. The
    spread is symmetric about truth, capped inside Black bounds. Displayed
    sizes vary deterministically to make calibration weights explicit.
    """
    anchor = utc_timestamp(as_of)
    if (
        not maturity_days
        or any(
            isinstance(d, bool) or not isinstance(d, int) or d <= 0
            for d in maturity_days
        )
        or any(a >= b for a, b in itertools.pairwise(maturity_days))
    ):
        raise InputValidationError("maturity_days must be positive increasing integers")
    if (
        len(log_moneyness) < 5
        or not all(isfinite(k) and abs(k) <= 2 for k in log_moneyness)
        or any(a >= b for a, b in itertools.pairwise(log_moneyness))
    ):
        raise InputValidationError("need >=5 increasing finite strikes in [-2,2]")
    if (
        not all(isfinite(x) for x in (spot, rate, dividend_yield, relative_spread))
        or spot <= 0
        or not 0 < relative_spread < 1
    ):
        raise InputValidationError(
            "positive spot, finite carry, spread in (0,1) required"
        )
    count = len(maturity_days) * len(log_moneyness)
    if any(i >= count for i in (*faults.missing, *faults.stale, *faults.crossed)):
        raise InputValidationError("fault index outside generated universe")
    expiries = tuple(anchor + timedelta(days=d) for d in maturity_days)
    times = tuple(year_fraction(anchor, expiry) for expiry in expiries)
    truth = SSVITruth(times[-1])
    discount = DiscountCurve(
        anchor,
        tuple(
            (expiry, exp(-rate * t)) for expiry, t in zip(expiries, times, strict=True)
        ),
    )
    forward = ForwardCurve(
        anchor,
        (
            (anchor, spot),
            *tuple(
                (expiry, spot * exp((rate - dividend_yield) * t))
                for expiry, t in zip(expiries, times, strict=True)
            ),
        ),
    )
    underlying = Underlying("SYNTH", "USD")
    contracts: list[OptionContract] = []
    quotes: list[Quote] = []
    for expiry, t in zip(expiries, times, strict=True):
        f, d = forward.forward(expiry), discount.discount_factor(expiry)
        for k in log_moneyness:
            index = len(contracts)
            kind = OptionType.PUT if k < 0 else OptionType.CALL
            contract = OptionContract(
                f"SYNTH-{index:04d}",
                underlying,
                kind,
                f * exp(k),
                expiry,
                "european",
                "cash",
                100.0,
                "USD",
                "USD",
                "per_contract" if per_contract else "per_unit",
            )
            contracts.append(contract)
            if index in faults.missing:
                continue
            mid = float(price(kind, f, contract.strike, t, d, truth.volatility(k, t)))
            lower, upper = price_bounds(kind, f, contract.strike, t, d)
            half = min(
                relative_spread * mid / 2,
                (mid - float(lower)) / 4,
                (float(upper) - mid) / 4,
            )
            noisy_mid = mid + faults.noise * sin(index + 1) * half
            bid, ask = noisy_mid - half, noisy_mid + half
            if index in faults.crossed:
                bid, ask = ask, bid
            scale = contract.multiplier if per_contract else 1.0
            age = (
                timedelta(minutes=10) if index in faults.stale else timedelta(seconds=1)
            )
            quotes.append(
                Quote(
                    contract,
                    bid * scale,
                    ask * scale,
                    float(10 + index % 7),
                    float(12 + index % 5),
                    anchor - age,
                    anchor - age,
                    "synthetic-ssvi",
                    1,
                )
            )
    provider = FrozenQuoteProvider(
        tuple(contracts), tuple(quotes), "synthetic-ssvi", timedelta(minutes=1)
    )
    return SyntheticChain(anchor, provider, discount, forward, truth, faults)
