"""Normalize historical linear BTC/USDC books and report quote and IV quality."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from math import isfinite
from pathlib import Path
from statistics import median
from typing import TypedDict, cast

from quant_pricing.black_scholes import OptionType
from quant_pricing.butterfly import check_quoted_butterflies
from quant_pricing.curves import DiscountCurve, ForwardCurve
from quant_pricing.deribit_catalog import choose_catalog_strikes
from quant_pricing.exceptions import InputValidationError
from quant_pricing.market import (
    Forward,
    MarketSnapshot,
    OptionContract,
    Quote,
    QuoteFlag,
    SnapshotQuote,
    Underlying,
)
from quant_pricing.parity import diagnose_parity
from quant_pricing.surface import (
    ArbitrageReport,
    IVObservation,
    SVISlice,
    VolatilitySurface,
    check_fitted_surface,
    check_observed_arbitrage,
    fit_svi_slice,
    recover_ivs,
)
from quant_pricing.tardis_deribit import BookCheckpoint, BookState, replay

_SOURCE = "tardis/deribit"
_UNDERLYING = Underlying("BTC_USDC", "USDC")
_SPEC = (
    "https://support.deribit.com/hc/en-us/articles/31424932728093-Linear-USDC-Options"
)
_SETTLEMENT_WINDOW = timedelta(minutes=30)
_DEFAULT_MAX_AGE_MS = 30_000
_SELECTION_SCHEMA = "btc-usdc-catalog-selection/v1"
_MONTHS = {
    name: i
    for i, name in enumerate(
        (
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
        ),
        1,
    )
}
_EXPIRY = re.compile(r"([1-9]|[0-2]\d|3[01])([A-Z]{3})(\d\d)")


@dataclass(frozen=True, slots=True)
class _ExpiryStrip:
    future: Forward
    options: tuple[OptionContract, ...]

    @property
    def expiry(self) -> datetime:
        return self.future.expiry


class _FutureAnchor(TypedDict):
    bid: list[str]
    ask: list[str]
    change_id: int
    available_at: str
    exchange_time: str


class _SelectedStrip(TypedDict):
    expiry_code: str
    future_symbol: str
    anchor: _FutureAnchor
    strikes: list[int]
    paired_strikes_available: int
    excluded_same_or_future_day: int


class CatalogSelection(TypedDict):
    selection_schema: str
    as_of: str
    max_age_ms: int
    catalog_sha256: str
    selection_rule: str
    strikes_per_expiry: int
    strips: list[_SelectedStrip]
    tardis_filter: list[dict[str, object]]


def _strip(code: str, strikes: tuple[int, ...]) -> _ExpiryStrip:
    match = _EXPIRY.fullmatch(code)
    if match is None or match[2] not in _MONTHS:
        raise InputValidationError(f"invalid Deribit expiry {code!r}; expected DDMMMYY")
    expiry = datetime(
        2000 + int(match[3]), _MONTHS[match[2]], int(match[1]), 8, tzinfo=timezone.utc
    )
    # Deribit native symbols do not pad the first nine days of the month.
    prefix = f"BTC_USDC-{expiry.day}{match[2]}{match[3]}"
    future = Forward(prefix, _UNDERLYING, expiry)
    return _ExpiryStrip(
        future,
        tuple(
            OptionContract(
                f"{prefix}-{strike}-{suffix}",
                _UNDERLYING,
                kind,
                float(strike),
                expiry,
                "european",
                "cash",
                1.0,
                "USDC",
                "USDC",
                "per_unit",
            )
            for strike in strikes
            for suffix, kind in (("C", OptionType.CALL), ("P", OptionType.PUT))
        ),
    )


def _utc_microseconds(stamp: str) -> datetime:
    seconds, _, fraction = stamp.removesuffix("Z").partition(".")
    # Round availability upward for the microsecond-resolution market domain.
    nanos = int(fraction.ljust(9, "0"))
    return datetime.fromisoformat(seconds + "+00:00") + timedelta(
        microseconds=(nanos + 999) // 1000
    )


def _normalize(instrument: OptionContract | Forward, book: BookState) -> SnapshotQuote:
    bid, ask, last = book["bid"], book["ask"], book["last_update"]
    if bid is None or ask is None or last is None:
        return SnapshotQuote(instrument, None, ("missing",))
    flags: list[QuoteFlag] = []
    for flag in book["flags"]:
        if flag == "stale":
            flags.append("stale")
        elif flag == "crossed":
            flags.append("crossed")
        else:
            return SnapshotQuote(instrument, None, ("missing",))
    bid_price, bid_amount = map(float, bid)
    ask_price, ask_amount = map(float, ask)
    if not all(
        isfinite(x) and x > 0 for x in (bid_price, bid_amount, ask_price, ask_amount)
    ):
        raise InputValidationError(
            f"{instrument.symbol}: book values exceed float domain"
        )
    # Premiums are already USDC/BTC. Native book amounts are BTC; option
    # quantity is amount / contract size (1 BTC here). Forward size stays BTC.
    multiplier = (
        instrument.multiplier if isinstance(instrument, OptionContract) else 1.0
    )
    quote = Quote(
        instrument,
        bid_price,
        ask_price,
        bid_amount / multiplier,
        ask_amount / multiplier,
        _utc_microseconds(last["exchange_time"]),
        _utc_microseconds(last["available_at"]),
        _SOURCE,
        last["change_id"],
    )
    return SnapshotQuote(instrument, quote, tuple(flags))


def _book_report(row: SnapshotQuote, native: BookState) -> dict[str, object]:
    quote = row.quote
    return {
        "symbol": row.instrument.symbol,
        "native_book": native,
        "normalized_flags": row.flags,
        "normalized_quote": None
        if quote is None
        else {
            "bid": quote.bid,
            "ask": quote.ask,
            "bid_size": quote.bid_size,
            "ask_size": quote.ask_size,
            "quote_time": quote.quote_time.isoformat(),
            "available_at": quote.available_at.isoformat(),
            "source_sequence": quote.source_sequence,
            "price_unit": "USDC/BTC",
            "size_unit": "contracts"
            if isinstance(row.instrument, OptionContract)
            else "BTC",
        },
    }


def _arb(report: ArbitrageReport) -> dict[str, object]:
    return {
        "claim": report.claim,
        "passed": report.passed,
        "log_moneyness_grid": report.log_moneyness_grid,
        "maturity_grid": report.maturity_grid,
        "tolerance": report.tolerance,
        "findings": [
            {
                **asdict(finding),
                "upper_bound": finding.upper_bound
                if isfinite(finding.upper_bound)
                else None,
            }
            for finding in report.evidence
        ],
    }


def _option_report(
    contract: OptionContract,
    row: SnapshotQuote,
    state: BookState,
    observation: IVObservation | None,
    blocked: str | None,
    fitted_symbols: set[str],
) -> dict[str, object]:
    sides = () if observation is None else observation.sides
    vols = {side.side: side.volatility for side in sides}
    bid_iv, ask_iv = vols.get("bid"), vols.get("ask")
    return {
        **_book_report(row, state),
        "strike": contract.strike,
        "option_type": contract.option_type.value,
        "rejected": tuple(state["flags"]) or ((blocked,) if blocked else ()),
        "log_forward_moneyness": None
        if observation is None
        else observation.log_moneyness,
        "sides": {
            side.side: {
                "premium_usdc_per_btc": side.premium_per_unit,
                "iv": side.volatility,
                "error": side.error,
                "iterations": None if side.result is None else side.result.iterations,
                "residual_usdc_per_btc": None
                if side.result is None
                else side.result.residual,
                "termination": None
                if side.result is None
                else side.result.termination_reason,
            }
            for side in sides
        },
        "bid_ask_iv_width": None
        if bid_iv is None or ask_iv is None
        else ask_iv - bid_iv,
        "used_in_svi": contract.symbol in fitted_symbols,
    }


def _analyze_expiry(
    strip: _ExpiryStrip,
    checkpoint: BookCheckpoint,
    as_of: datetime,
    rate: float,
    max_age: timedelta,
) -> tuple[dict[str, object], tuple[IVObservation, ...], SVISlice | None]:
    native = checkpoint["books"]
    future = _normalize(strip.future, native[strip.future.symbol])
    rows = tuple(_normalize(c, native[c.symbol]) for c in strip.options)
    chain = MarketSnapshot(as_of, _SOURCE, rows)
    fwd_quote = future.quote
    forward = (
        fwd_quote.bid / 2 + fwd_quote.ask / 2
        if fwd_quote and not future.flags
        else None
    )
    blocked = None
    if strip.expiry <= as_of:
        blocked = "expired"
    elif strip.expiry - as_of <= _SETTLEMENT_WINDOW:
        blocked = "settlement_window"
    elif forward is None:
        blocked = "matched_future_unavailable"
    discounts = None
    discount = None
    parity = []
    if strip.expiry - as_of > _SETTLEMENT_WINDOW:
        discounts = DiscountCurve.from_zero_rates(as_of, ((strip.expiry, rate),))
        discount = discounts.discount_factor(strip.expiry)
        for call, put in zip(rows[::2], rows[1::2], strict=True):
            match = diagnose_parity(
                call,
                put,
                as_of=as_of,
                max_age=max_age,
                discount_factor=discount,
                forward_quote=future,
            )
            residual = match.independent_forward_residual
            parity.append(
                {
                    "call": call.instrument.symbol,
                    "put": put.instrument.symbol,
                    **asdict(match),
                    "consistent_with_future": (
                        None if residual is None else residual[0] <= 0 <= residual[1]
                    ),
                }
            )
    butterflies = None
    if discount is not None:
        checks = check_quoted_butterflies(rows, future, discount)
        butterflies = {
            "quote_basis": "buy asks, sell bids; before fees and not a fill",
            "cost_unit": "USDC per BTC of center option",
            "candidate_count": sum(
                check.negative_cost_candidate is True for check in checks
            ),
            "triplets_checked": sum(check.combinations_checked > 0 for check in checks),
            "triplets": [asdict(check) for check in checks],
        }
    observations: tuple[IVObservation, ...] = ()
    eligible: tuple[IVObservation, ...] = ()
    fitted = None
    fit_error = blocked
    if blocked is None:
        assert forward is not None and discounts is not None
        curve = ForwardCurve(as_of, ((strip.expiry, forward),))
        observations = recover_ivs(chain, discounts, curve)
        eligible = tuple(
            obs
            for contract, obs in zip(strip.options, observations, strict=True)
            if obs.fit_eligible
            and contract.option_type
            is (OptionType.PUT if contract.strike < forward else OptionType.CALL)
        )
        try:
            fitted = fit_svi_slice(strip.expiry, eligible)
        except InputValidationError as exc:
            fit_error = str(exc)
    used = {r.contract for r in fitted.residuals} if fitted else set()
    iv_by_symbol = {obs.row.instrument.symbol: obs for obs in observations}
    options = [
        _option_report(
            contract,
            row,
            native[contract.symbol],
            iv_by_symbol.get(contract.symbol),
            blocked,
            used,
        )
        for contract, row in zip(strip.options, rows, strict=True)
    ]
    widths = [
        float(opt["bid_ask_iv_width"])
        for opt in options
        if isinstance(opt["bid_ask_iv_width"], float)
    ]
    by_side = Counter(
        side.side
        for obs in observations
        for side in obs.sides
        if side.volatility is not None
    )
    usable_quotes = [row.quote for row in rows if not row.flags and row.quote]
    spreads = [quote.ask - quote.bid for quote in usable_quotes]
    ages = [
        (as_of - quote.quote_time).total_seconds() * 1000.0 for quote in usable_quotes
    ]
    return (
        {
            "expiry": strip.expiry.isoformat(),
            "discount_factor": discount,
            "forward_input": forward,
            "matched_future": _book_report(future, native[strip.future.symbol]),
            "iv_blocked": blocked,
            "coverage": {
                "requested_options": len(options),
                "usable_books": len(usable_quotes),
                "max_usable_book_age_ms": max(ages, default=None),
                "median_premium_spread_usdc_per_btc": median(spreads)
                if spreads
                else None,
                "native_flag_counts": dict(
                    Counter(
                        flag
                        for c in strip.options
                        for flag in native[c.symbol]["flags"]
                    )
                ),
                "iv_successes": {
                    side: by_side[side] for side in ("bid", "midpoint", "ask")
                },
                "two_sided_iv": len(widths),
                "median_bid_ask_iv_width": median(widths) if widths else None,
                "eligible_otm_strikes": len(eligible),
                "parity_pairs_compared": sum(
                    m["consistent_with_future"] is not None for m in parity
                ),
                "parity_pairs_consistent": sum(
                    m["consistent_with_future"] is True for m in parity
                ),
            },
            "options": options,
            "parity": parity,
            "quoted_butterflies": butterflies,
            "fit": None
            if fitted is None
            else {
                **asdict(fitted),
                "expiry": strip.expiry.isoformat(),
                "arbitrage": _arb(check_fitted_surface(VolatilitySurface((fitted,)))),
            },
            "fit_error": fit_error,
        },
        observations,
        fitted,
    )


def _analyze(
    checkpoint: BookCheckpoint,
    strips: tuple[_ExpiryStrip, ...],
    rate: float,
    max_age: timedelta,
) -> dict[str, object]:
    as_of = _utc_microseconds(checkpoint["as_of"])
    reports: list[dict[str, object]] = []
    observations: list[IVObservation] = []
    fits: list[SVISlice] = []
    for strip in strips:
        report, recovered, fit = _analyze_expiry(
            strip, checkpoint, as_of, rate, max_age
        )
        reports.append(report)
        observations.extend(recovered)
        if fit is not None:
            fits.append(fit)
    joint = None
    joint_error = None
    if len(fits) < 2:
        joint_error = "need at least two fitted expiries"
    else:
        try:
            joint = _arb(check_fitted_surface(VolatilitySurface(tuple(fits))))
        except InputValidationError as exc:
            joint_error = str(exc)
    return {
        "as_of": checkpoint["as_of"],
        "native_capture_complete": checkpoint["complete"],
        "observed_arbitrage": _arb(check_observed_arbitrage(tuple(observations)))
        if any(observation.fit_eligible for observation in observations)
        else None,
        "expiries": reports,
        "joint_fitted_arbitrage": joint,
        "joint_fitted_expiries": [fit.expiry.isoformat() for fit in fits],
        "joint_fit_error": joint_error,
    }


def _ordered(strips: Iterable[_ExpiryStrip]) -> tuple[_ExpiryStrip, ...]:
    result = tuple(sorted(strips, key=lambda strip: strip.expiry))
    if not result or len({strip.expiry for strip in result}) != len(result):
        raise InputValidationError("specify unique expiries")
    return result


def _strikes(strikes: Sequence[int], *, allow_empty: bool = False) -> tuple[int, ...]:
    if (
        (not strikes and not allow_empty)
        or len(set(strikes)) != len(strikes)
        or any(type(k) is not int or k <= 0 for k in strikes)
    ):
        raise InputValidationError("specify positive, unique integer strikes")
    return tuple(sorted(strikes))


def _check_cutoffs(cutoffs: Sequence[str]) -> None:
    if not cutoffs or any(
        not t.endswith("Z") or len(t.partition(".")[2].removesuffix("Z")) > 6
        for t in cutoffs
    ):
        raise InputValidationError(
            "supply UTC cutoffs with at most six fractional digits"
        )


def _future_anchor(book: BookState, symbol: str) -> _FutureAnchor:
    bid, ask, last = book["bid"], book["ask"], book["last_update"]
    if book["flags"] or bid is None or ask is None or last is None:
        raise InputValidationError(
            f"{symbol}: no usable futures book at selection cutoff"
        )
    return {
        "bid": bid,
        "ask": ask,
        "change_id": last["change_id"],
        "available_at": last["available_at"],
        "exchange_time": last["exchange_time"],
    }


def plan_chain(
    lines: Iterable[str],
    catalog_bytes: bytes,
    expiries: Sequence[str],
    cutoff: str,
    *,
    strikes_per_expiry: int = 9,
    max_age_ms: int = _DEFAULT_MAX_AGE_MS,
) -> CatalogSelection:
    """Freeze symbol choices using only a prior-day archive catalog and futures."""
    _check_cutoffs((cutoff,))
    catalog: object = json.loads(catalog_bytes)
    strips = _ordered(_strip(code, ()) for code in expiries)
    futures = [strip.future.symbol for strip in strips]
    checkpoint = replay(lines, futures, (cutoff,), max_age_ms=max_age_ms)[0]
    as_of = _utc_microseconds(checkpoint["as_of"])
    selected: list[_SelectedStrip] = []
    symbols = list(futures)
    for strip in strips:
        if strip.expiry - as_of <= _SETTLEMENT_WINDOW:
            raise InputValidationError("selection expiry is inside or after settlement")
        symbol = strip.future.symbol
        anchor = _future_anchor(checkpoint["books"][symbol], symbol)
        forward = float(anchor["bid"][0]) / 2 + float(anchor["ask"][0]) / 2
        choice = choose_catalog_strikes(
            catalog, symbol, as_of, forward, count=strikes_per_expiry
        )
        selected.append(
            {
                "expiry_code": symbol.split("-", 1)[1],
                "future_symbol": symbol,
                "anchor": anchor,
                "strikes": list(choice.strikes),
                "paired_strikes_available": choice.paired_strikes_available,
                "excluded_same_or_future_day": choice.excluded_same_or_future_day,
            }
        )
        symbols.extend(f"{symbol}-{k}-{cp}" for k in choice.strikes for cp in "CP")
    if len(symbols) > 50:
        raise InputValidationError("a Tardis filter can include at most 50 symbols")
    return {
        "selection_schema": _SELECTION_SCHEMA,
        "as_of": checkpoint["as_of"],
        "max_age_ms": max_age_ms,
        "catalog_sha256": sha256(catalog_bytes).hexdigest(),
        "selection_rule": "prior UTC day metadata; paired strikes nearest per future",
        "strikes_per_expiry": strikes_per_expiry,
        "strips": selected,
        "tardis_filter": [{"channel": "book", "symbols": symbols}],
    }


def replay_chain(
    lines: Iterable[str],
    expiries: Sequence[str],
    strikes: Sequence[int],
    cutoffs: Sequence[str],
    *,
    rate: float,
    max_age_ms: int = _DEFAULT_MAX_AGE_MS,
) -> dict[str, object]:
    """Produce the normalized research report for an explicit historical universe."""
    grid = _strikes(strikes)
    return _replay_strips(
        lines,
        _ordered(_strip(code, grid) for code in expiries),
        cutoffs,
        rate,
        max_age_ms,
    )


def replay_selected_chain(
    lines: Iterable[str],
    selection: CatalogSelection,
    cutoffs: Sequence[str],
    *,
    rate: float,
) -> dict[str, object]:
    """Replay a frozen catalog selection and verify its futures against the tape."""
    if selection["selection_schema"] != _SELECTION_SCHEMA:
        raise InputValidationError("unsupported catalog selection schema")
    strips = _ordered(
        _strip(item["expiry_code"], _strikes(item["strikes"], allow_empty=True))
        for item in selection["strips"]
    )
    return _replay_strips(
        lines, strips, cutoffs, rate, selection["max_age_ms"], selection=selection
    )


def _replay_strips(
    lines: Iterable[str],
    strips: tuple[_ExpiryStrip, ...],
    cutoffs: Sequence[str],
    rate: float,
    max_age_ms: int,
    *,
    selection: CatalogSelection | None = None,
) -> dict[str, object]:
    if not isfinite(rate):
        raise InputValidationError("rate must be a finite annual decimal")
    _check_cutoffs(cutoffs)
    symbols = [strip.future.symbol for strip in strips]
    symbols.extend(contract.symbol for strip in strips for contract in strip.options)
    skip_anchor = False
    if selection is not None:
        first = _utc_microseconds(cutoffs[0])
        anchor = _utc_microseconds(selection["as_of"])
        if first < anchor:
            raise InputValidationError("report cutoff precedes catalog selection")
        skip_anchor = first > anchor
        if skip_anchor:
            cutoffs = (selection["as_of"], *cutoffs)
    checkpoints = replay(lines, symbols, cutoffs, max_age_ms=max_age_ms)
    if selection is not None:
        captured = checkpoints[0]["books"]
        for item in selection["strips"]:
            symbol = item["future_symbol"]
            if _future_anchor(captured[symbol], symbol) != item["anchor"]:
                raise InputValidationError(
                    f"{symbol}: capture disagrees with selection anchor"
                )
        if skip_anchor:
            checkpoints.pop(0)
    max_age = timedelta(milliseconds=max_age_ms)
    return {
        "schema_version": 1,
        "source": _SOURCE,
        "scope": "historical linear BTC_USDC options and same-expiry futures only",
        "contract_specification": _SPEC,
        "catalog_selection": selection,
        "assumptions": {
            "option_contract_size_btc": 1.0,
            "premium_basis": "USDC per BTC",
            "settlement_currency": "USDC",
            "exercise": "European",
            "settlement": "via dated futures immediately cash-settled in USDC",
            "volatility_units": "annualized decimal",
            "rate": rate,
            "rate_convention": "explicit flat continuously compounded USDC assumption",
            "day_count": "ACT/365F",
            "mark_or_parity_forward_fallback": False,
            "forward_input": "matched dated futures BBO midpoint, not venue mark",
            "max_book_age_ms": max_age_ms,
            "excluded_settlement_window_minutes": 30,
            "availability": "Tardis receipt rounded up to microseconds",
            "svi_selection": "liquid positive midpoint IV; put below F, call otherwise",
            "observed_findings_use_price_midpoints": True,
        },
        "checkpoints": [_analyze(c, strips, rate, max_age) for c in checkpoints],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="decoded raw Tardis HTTP response")
    parser.add_argument("--expiry", action="append", help="Deribit DDMMMYY")
    parser.add_argument("--strike", action="append", type=int)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--catalog", type=Path, help="plan from a local Tardis catalog")
    modes.add_argument("--selection", type=Path, help="replay a frozen catalog plan")
    parser.add_argument("--strikes-per-expiry", type=int, help="catalog plan size (9)")
    parser.add_argument(
        "--as-of", action="append", required=True, help="UTC cutoff ending in Z"
    )
    parser.add_argument("--rate", type=float, help="annual USDC decimal for reports")
    parser.add_argument("--max-age-ms", type=int)
    args = parser.parse_args(argv)
    try:
        with args.capture.open(encoding="utf-8") as src:
            if args.catalog is not None:
                if not args.expiry or args.strike or len(args.as_of) != 1:
                    raise InputValidationError(
                        "planning requires --expiry and one --as-of, not --strike"
                    )
                if args.rate is not None:
                    raise InputValidationError(
                        "set --rate when replaying the selection"
                    )
                output: CatalogSelection | dict[str, object] = plan_chain(
                    src,
                    args.catalog.read_bytes(),
                    args.expiry,
                    args.as_of[0],
                    strikes_per_expiry=(
                        9
                        if args.strikes_per_expiry is None
                        else args.strikes_per_expiry
                    ),
                    max_age_ms=(
                        _DEFAULT_MAX_AGE_MS
                        if args.max_age_ms is None
                        else args.max_age_ms
                    ),
                )
            else:
                if args.rate is None or args.strikes_per_expiry is not None:
                    raise InputValidationError(
                        "replay requires --rate; --strikes-per-expiry is for planning"
                    )
                if args.selection is not None:
                    if args.expiry or args.strike or args.max_age_ms is not None:
                        raise InputValidationError(
                            "selection already defines expiries, strikes and book age"
                        )
                    raw: object = json.loads(args.selection.read_bytes())
                    if not isinstance(raw, dict):
                        raise InputValidationError(
                            "expected a catalog selection object"
                        )
                    selection = cast(CatalogSelection, raw)
                    output = replay_selected_chain(
                        src, selection, args.as_of, rate=args.rate
                    )
                else:
                    if not args.expiry or not args.strike:
                        raise InputValidationError("provide --expiry and --strike")
                    output = replay_chain(
                        src,
                        args.expiry,
                        args.strike,
                        args.as_of,
                        rate=args.rate,
                        max_age_ms=(
                            _DEFAULT_MAX_AGE_MS
                            if args.max_age_ms is None
                            else args.max_age_ms
                        ),
                    )
        rendered = json.dumps(output, sort_keys=True, indent=2, allow_nan=False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"deribit-chain: {exc}", file=sys.stderr)
        return 2
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover - local CLI entry point
    raise SystemExit(main())
