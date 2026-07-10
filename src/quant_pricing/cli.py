"""Deterministic command-line interface for option valuation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from quant_pricing.black_scholes import OptionType, price


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Price a European option with Black-Scholes-Merton."
    )
    parser.add_argument("--spot", type=float)
    parser.add_argument("--strike", type=float)
    parser.add_argument("--time", type=float, help="time to expiry in years")
    parser.add_argument(
        "--rate", type=float, help="continuously compounded decimal rate"
    )
    parser.add_argument(
        "--volatility", type=float, help="annualized decimal volatility"
    )
    parser.add_argument(
        "--dividend-yield",
        type=float,
        default=0.0,
        help="continuous decimal dividend yield",
    )
    parser.add_argument(
        "--type",
        choices=[member.value for member in OptionType],
        dest="option_type",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI, prompting for any omitted arguments."""

    arguments = _parser().parse_args(argv)
    spot = arguments.spot
    strike = arguments.strike
    time = arguments.time
    rate = arguments.rate
    volatility = arguments.volatility
    option_type = arguments.option_type

    if spot is None:
        spot = float(input("Underlying price: "))
    if strike is None:
        strike = float(input("Strike price: "))
    if time is None:
        time = float(input("Time to expiry in years: "))
    if rate is None:
        rate = float(input("Continuously compounded rate (decimal): "))
    if volatility is None:
        volatility = float(input("Annualized volatility (decimal): "))
    if option_type is None:
        option_type = input("Option type (call/put): ").strip().lower()

    value = price(
        option_type,
        spot,
        strike,
        time,
        rate,
        volatility,
        arguments.dividend_yield,
    )
    if not isinstance(value, float):  # pragma: no cover - internal contract guard
        raise TypeError("scalar CLI inputs produced an array")
    print(f"{option_type.capitalize()} value: {value:.10f}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through console script
    raise SystemExit(main())
