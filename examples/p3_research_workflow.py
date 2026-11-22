"""Reproduce synthetic surface calibration, portfolio stress, and exercise risk."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from quant_pricing.binomial import CashDividend
from quant_pricing.binomial import price as tree_price
from quant_pricing.black_scholes import OptionType
from quant_pricing.conventions import year_fraction
from quant_pricing.market import OptionContract, Portfolio, Position
from quant_pricing.risk import (
    RiskInputs,
    RiskLimits,
    RiskScenario,
    Trade,
    attribute_pnl,
    check_limits,
    scenario_portfolio,
    value_portfolio,
)
from quant_pricing.surface import calibrate_surface
from quant_pricing.synthetic_chain import ChainFaults, generate_chain

AS_OF = datetime(2025, 1, 2, 12, tzinfo=timezone.utc)


def research_report() -> dict[str, object]:
    clean = generate_chain(AS_OF)
    snapshot = clean.provider.snapshot(AS_OF)
    fitted = calibrate_surface(snapshot, clean.discount_curve, clean.forward_curve)
    damaged = generate_chain(
        AS_OF, faults=ChainFaults(missing=(0,), stale=(1,), crossed=(2,), noise=0.3)
    )
    checked = calibrate_surface(
        damaged.provider.snapshot(AS_OF), damaged.discount_curve, damaged.forward_curve
    )

    expiry = AS_OF + timedelta(days=180)
    t = year_fraction(AS_OF, expiry)
    forward = clean.forward_curve.forward(expiry)
    contracts = [
        row.instrument
        for row in snapshot.rows
        if isinstance(row.instrument, OptionContract)
        and row.instrument.expiry == expiry
    ]
    call = min(
        (c for c in contracts if c.option_type is OptionType.CALL),
        key=lambda c: abs(math.log(c.strike / forward)),
    )
    put = max(
        (c for c in contracts if c.option_type is OptionType.PUT),
        key=lambda c: c.strike,
    )
    spot = clean.forward_curve.forward(AS_OF)
    rate = clean.discount_curve.zero_rate(expiry)
    dividend_yield = rate - math.log(forward / spot) / t
    inputs = {
        call.underlying: RiskInputs(
            spot=spot,
            volatility=fitted.surface,
            rate=rate,
            dividend_yield=dividend_yield,
        )
    }
    portfolio = Portfolio((Position(call, 2.0), Position(put, -1.0)))
    base = value_portfolio(portfolio, AS_OF, inputs, vol_buckets=(0.15, 0.3))
    stress = scenario_portfolio(
        portfolio,
        AS_OF,
        inputs,
        RiskScenario(
            spot_return=-0.02,
            volatility_shift=0.015,
            elapsed_years=1 / 365,
            rate_shift=0.001,
            skew_shift=0.01,
            curvature_shift=0.02,
        ),
    )
    currency = call.quote_currency
    limits = check_limits(base, RiskLimits(max_position=1.5))
    closing_inputs = {
        call.underlying: RiskInputs(
            spot=spot * 0.98,
            volatility=fitted.surface,
            rate=rate + 0.001,
            dividend_yield=dividend_yield,
        )
    }
    premium = base.positions[0].greeks.price
    if call.premium_basis == "per_contract":
        premium *= call.multiplier
    pnl = attribute_pnl(
        portfolio,
        AS_OF,
        AS_OF + timedelta(days=1),
        inputs,
        closing_inputs,
        trades=(Trade(call, 0.5, premium, fees=1.0),),
        carry_cash={currency: -0.5},
    )[currency]

    dividends = (CashDividend(0.5, 6.0),)
    european = tree_price(
        "call", 110.0, 90.0, 1.0, 0.04, 0.2, steps=400, dividends=dividends
    )
    american = tree_price(
        "call",
        110.0,
        90.0,
        1.0,
        0.04,
        0.2,
        style="american",
        steps=400,
        dividends=dividends,
    )
    assignment = american.exposure(contracts=-2.0, multiplier=100.0)
    return {
        "as_of": AS_OF.isoformat(),
        "evidence": "synthetic, not market data or an execution backtest",
        "surface": {
            "fitted_expiries": len(fitted.slices),
            "atm_iv_180d": fitted.surface.volatility(0.0, t),
            "max_weighted_slice_rmse": max(s.weighted_rmse for s in fitted.slices),
            "clean_grid_findings": len(fitted.arbitrage.evidence),
            "dirty_rejected_quotes": sum(
                bool(row.rejected) for row in checked.observations
            ),
            "dirty_grid_findings": len(checked.arbitrage.evidence),
            "dirty_observed_grid_findings": len(checked.observed_arbitrage.evidence),
        },
        "portfolio": {
            "currency": currency,
            "base_pv": base.totals[currency].value,
            "stress_pv": stress.totals[currency].value,
            "limit_breaches": [breach.message for breach in limits],
            "pnl_attribution": asdict(pnl),
        },
        "exercise_tree": {
            "european_pv_per_unit": european.price,
            "american_pv_per_unit": american.price,
            "early_assignment_probability": assignment.early_probability,
            "expected_early_assigned_units": assignment.early_quantity,
        },
    }


if __name__ == "__main__":
    print(json.dumps(research_report(), indent=2, sort_keys=True, allow_nan=False))
