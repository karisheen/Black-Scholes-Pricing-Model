"""Exact replay selection and JSON adapters, never historical-data fallbacks.

Reports retain the deribit_chain schema: checkpoints/as_of, expiries/expiry,
options/symbol. Detail adds model_value/model_iv (nullable floats), model_basis,
risk_basis (text), risk_units (Greek-to-unit mapping), risk_assumptions (text
list), greeks (mapping or None), quality_flags and age_seconds. Source quote,
sides, flags and assumptions are preserved, including their original units.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from math import exp, isclose, isfinite, log, sqrt
from pathlib import Path
from typing import Any

from quant_pricing.black76 import price
from quant_pricing.conventions import utc_timestamp, year_fraction
from quant_pricing.lab_demo import DEMO_SESSION, demo_report
from quant_pricing.risk import advanced_greeks
from quant_pricing.surface import (
    FitResidual,
    SVIParameters,
    SVISlice,
    VolatilitySurface,
)

Row = dict[str, Any]

_RISK_BASIS = "Forward-coordinate partials; frozen fitted volatility; not spot Greeks"
_RISK_UNITS = {
    "delta": "(USDC/BTC) / (USDC/BTC forward)",
    "gamma": "(USDC/BTC) / (USDC/BTC forward)^2",
    "vega": "USDC/BTC per absolute annualized decimal sigma",
    "theta": "USDC/BTC per calendar year",
    "rho": "USDC/BTC per absolute continuously compounded annual rate",
    "vanna": "(USDC/BTC) / (USDC/BTC forward) / absolute sigma",
    "volga": "USDC/BTC per absolute sigma squared",
    "charm": "forward delta per calendar year",
    "speed": "(USDC/BTC) / (USDC/BTC forward)^3",
    "color": "forward gamma per calendar year",
}
_RISK_ASSUMPTIONS = [
    "Black-76 paid-upfront PV per BTC; no position or contract-size scaling.",
    "F is the matched dated future, not spot. Strike and fitted sigma held fixed.",
    "No smile recalibration, sticky-delta adjustment or forward-curve response.",
    "Vega/vanna/volga use absolute sigma, not one volatility percentage point.",
    "Rho varies only D=exp(-rT), holding F and sigma fixed; never BSM spot rho.",
    "Theta/charm/color advance calendar time with F, sigma and flat r fixed.",
    "Time/rate risks require an explicit flat rate consistent with reported D.",
    "Reported F must match the usable future; "
    "reported D must match any flat-rate basis.",
    "Missing or failed sampled fit checks withhold risk; "
    "passing is not a global proof.",
]


@dataclass(frozen=True)
class ReplayView:
    session: str
    report: Row
    checkpoint: Row
    expiry: Row


@dataclass(frozen=True)
class ReportLibrary:
    sessions: dict[str, Row]
    errors: tuple[str, ...] = ()

    def view(self, session: str, as_of: str, expiry: str) -> ReplayView:
        """Select exact recorded keys; never choose a nearest or default row."""
        report = self.sessions[session]
        for checkpoint in report["checkpoints"]:
            if checkpoint["as_of"] == as_of:
                for row in checkpoint["expiries"]:
                    if row["expiry"] == expiry:
                        return ReplayView(session, report, checkpoint, row)
                raise KeyError(f"Unknown expiry: {expiry}")
        raise KeyError(f"Unknown checkpoint: {as_of}")


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if isfinite(value) else None
    except OverflowError:
        return None


def _stamp(value: str) -> datetime:
    return utc_timestamp(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _json_number(value: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError("nonfinite JSON numbers are not supported")
    return result


def _mapping(value: Any, label: str) -> Row:
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: expected an object")
    return value


def _records(value: Any, label: str) -> list[Row]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"invalid {label}: expected object entries")
    return value


def _strings(value: Any, label: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"invalid {label}: expected text entries")


def _numeric_fields(row: Row, fields: tuple[str, ...], label: str) -> None:
    if any(_number(row.get(field)) is None for field in fields):
        raise ValueError(f"invalid {label}: missing or nonfinite numeric fields")


def _timestamp(value: Any, label: str) -> datetime:
    try:
        if not isinstance(value, str):
            raise ValueError
        return _stamp(value)
    except (ValueError, OverflowError):
        # Do not include arbitrary input or private paths in UI errors.
        raise ValueError(
            f"invalid {label}: expected timezone-aware timestamp"
        ) from None


def _interval(value: Any, label: str, *, length: int = 2) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(_number(item) is None for item in value)
        or any(left > right for left, right in pairwise(value))
    ):
        raise ValueError(f"invalid {label}: expected ordered finite interval")
    return value


def _optional_bool(value: Any, label: str) -> None:
    if value is not None and type(value) is not bool:
        raise ValueError(f"invalid {label}: expected boolean or null")


def _validate_book(row: Row) -> None:
    for field in ("normalized_flags", "rejected"):
        _strings(row.get(field, []), f"book {field}")
    quote = row.get("normalized_quote")
    if quote is not None:
        quote = _mapping(quote, "normalized quote")
        _numeric_fields(quote, ("bid", "ask", "bid_size", "ask_size"), "quote")
        for field in ("quote_time", "available_at"):
            _timestamp(quote.get(field), f"quote {field}")
        for field in ("price_unit", "size_unit"):
            if quote.get(field) is not None and not isinstance(quote[field], str):
                raise ValueError(f"invalid quote {field}")
    native = row.get("native_book")
    if native is not None:
        native = _mapping(native, "native book")
        _strings(native.get("flags", []), "native book flags")
        for field in ("snapshot", "last_update"):
            evidence = native.get(field)
            if evidence is not None:
                evidence = _mapping(evidence, f"native {field}")
                for stamp in ("available_at", "exchange_time"):
                    _timestamp(evidence.get(stamp), f"native {field} {stamp}")
        if native.get("issue") is not None:
            _mapping(native["issue"], "native issue")


def _validate_arbitrage(value: Any) -> None:
    if value is None:
        return
    evidence = _mapping(value, "arbitrage evidence")
    if type(evidence.get("passed")) is not bool:
        raise ValueError("invalid arbitrage status")
    for finding in _records(evidence.get("findings"), "arbitrage findings"):
        if finding.get("kind") not in ("monotonicity", "butterfly", "calendar"):
            raise ValueError("invalid arbitrage finding kind")
        _numeric_fields(
            finding,
            ("log_moneyness", "time_to_expiry", "value", "lower_bound"),
            "arbitrage finding",
        )
        if finding.get("upper_bound") is not None:
            _numeric_fields(finding, ("upper_bound",), "arbitrage finding")


def _validate_fit(value: Any) -> None:
    if value is None:
        return
    fit = _mapping(value, "fit")
    parameters = _mapping(fit.get("parameters"), "fit parameters")
    _numeric_fields(parameters, ("a", "b", "rho", "m", "sigma"), "fit parameters")
    _numeric_fields(fit, ("time_to_expiry", "weighted_rmse", "evaluations"), "fit")
    _interval(fit.get("domain"), "fit domain")
    if type(fit.get("converged")) is not bool:
        raise ValueError("invalid fit convergence status")
    expiry = _timestamp(fit.get("expiry"), "fit expiry")
    try:
        # Reuse domain constraints rather than implement another SVI schema/model.
        VolatilitySurface(
            (
                SVISlice(
                    expiry=expiry,
                    time_to_expiry=fit["time_to_expiry"],
                    parameters=SVIParameters(**parameters),
                    domain=(fit["domain"][0], fit["domain"][1]),
                ),
            )
        )
        for row in _records(fit.get("residuals"), "fit residuals"):
            if not isinstance(row.get("contract"), str):
                raise ValueError
            _numeric_fields(
                row,
                ("log_moneyness", "observed_variance", "fitted_variance", "weight"),
                "fit residual",
            )
            FitResidual(**row)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("invalid fit parameters, domain or residuals") from None
    _validate_arbitrage(fit.get("arbitrage"))


def _validate_diagnostics(expiry: Row) -> None:
    for pair in _records(expiry.get("parity", []), "parity"):
        _optional_bool(pair.get("consistent_with_future"), "parity consistency")
        if any(not isinstance(pair.get(key), str) for key in ("call", "put")):
            raise ValueError("invalid parity contract identities")
        for field in ("option_flags", "forward_flags"):
            _strings(pair.get(field, []), f"parity {field}")
        for field in (
            "premium_difference",
            "implied_forward",
            "parity_residual",
            "independent_forward_residual",
            "implied_dividend_yield",
            "implied_borrow_yield",
        ):
            if pair.get(field) is not None:
                _interval(pair[field], f"parity {field}")
    butterflies = expiry.get("quoted_butterflies")
    if butterflies is None:
        return
    butterflies = _mapping(butterflies, "quoted butterflies")
    for triplet in _records(butterflies.get("triplets", []), "butterfly triplets"):
        _optional_bool(triplet.get("negative_cost_candidate"), "butterfly candidate")
        strikes = _interval(triplet.get("strikes"), "butterfly strikes", length=3)
        if not 0 < strikes[0] < strikes[1] < strikes[2]:
            raise ValueError("invalid butterfly strikes")
        _strings(triplet.get("flags", []), "butterfly flags")
        package = triplet.get("best_package")
        if package is None:
            continue
        package = _mapping(package, "butterfly package")
        _numeric_fields(
            package,
            (
                "option_entry_cost",
                "parity_adjustment",
                "net_entry_cost",
                "displayed_size_cap",
                "cost_tolerance",
            ),
            "butterfly package",
        )
        if type(package.get("negative_cost_candidate")) is not bool:
            raise ValueError("invalid butterfly candidate status")
        legs = _records(package.get("option_legs"), "butterfly legs")
        for leg in legs:
            _validate_leg(leg, option=True)
        if package.get("futures_hedge") is not None:
            _validate_leg(
                _mapping(package["futures_hedge"], "futures hedge"), option=False
            )


def _validate_leg(leg: Row, *, option: bool) -> None:
    numeric: tuple[str, ...] = ("signed_weight", "price", "displayed_size")
    if option:
        numeric += ("multiplier",)
    expected = {"symbol", "side", *numeric}
    if option:
        expected.add("option_type")
        if leg.get("option_type") not in ("call", "put"):
            raise ValueError("invalid butterfly option type")
    if (
        set(leg) != expected
        or not isinstance(leg.get("symbol"), str)
        or leg.get("side") not in ("buy", "sell")
    ):
        raise ValueError("invalid butterfly leg identity")
    _numeric_fields(leg, numeric, "butterfly leg")


def _validate_report(report: Any) -> Row:
    if (
        not isinstance(report, dict)
        or type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
    ):
        raise ValueError("expected deribit_chain schema_version 1")
    for field in ("source", "scope"):
        if not isinstance(report.get(field), str):
            raise ValueError(f"missing report {field}")
    if not isinstance(report.get("assumptions"), dict):
        raise ValueError("missing report assumptions")
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("report has no checkpoints")
    seen_times: set[str] = set()
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            raise ValueError("invalid checkpoint")
        stamp = checkpoint.get("as_of")
        if not isinstance(stamp, str) or stamp in seen_times:
            raise ValueError("invalid or duplicate checkpoint time")
        _timestamp(stamp, "checkpoint time")
        _optional_bool(checkpoint.get("native_capture_complete"), "capture status")
        for field in ("observed_arbitrage", "joint_fitted_arbitrage"):
            _validate_arbitrage(checkpoint.get(field))
        seen_times.add(stamp)
        expiries = checkpoint.get("expiries")
        if not isinstance(expiries, list) or not expiries:
            raise ValueError("checkpoint has no expiries")
        seen_expiries: set[str] = set()
        for expiry in expiries:
            if not isinstance(expiry, dict):
                raise ValueError("invalid expiry")
            key = expiry.get("expiry")
            if not isinstance(key, str) or key in seen_expiries:
                raise ValueError("invalid or duplicate expiry")
            _timestamp(key, "expiry")
            seen_expiries.add(key)
            _validate_fit(expiry.get("fit"))
            _validate_diagnostics(expiry)
            if "coverage" in expiry:
                coverage = _mapping(expiry["coverage"], "coverage")
                for field in ("native_flag_counts", "iv_successes"):
                    if coverage.get(field) is not None:
                        counts = _mapping(coverage[field], f"coverage {field}")
                        _numeric_fields(counts, tuple(counts), f"coverage {field}")
                for field in (
                    "requested_options",
                    "usable_books",
                    "two_sided_iv",
                    "eligible_otm_strikes",
                    "max_usable_book_age_ms",
                    "median_premium_spread_usdc_per_btc",
                    "median_bid_ask_iv_width",
                    "parity_pairs_consistent",
                    "parity_pairs_compared",
                ):
                    if coverage.get(field) is not None:
                        _numeric_fields(coverage, (field,), f"coverage {field}")
            future = expiry.get("matched_future")
            if future is not None:
                future = _mapping(future, "matched future")
                if not isinstance(future.get("symbol"), str):
                    raise ValueError("invalid matched future identity")
                _validate_book(future)
            for field in ("forward_input", "discount_factor"):
                if expiry.get(field) is not None:
                    _numeric_fields(expiry, (field,), f"expiry {field}")
            discount = expiry.get("discount_factor")
            if discount is not None and discount <= 0:
                raise ValueError("invalid expiry discount factor")
            options = expiry.get("options")
            if not isinstance(options, list):
                raise ValueError("invalid options")
            seen_symbols: set[str] = set()
            for option in options:
                if not isinstance(option, dict):
                    raise ValueError("invalid option")
                symbol, strike = option.get("symbol"), _number(option.get("strike"))
                if (
                    not isinstance(symbol, str)
                    or not symbol
                    or symbol in seen_symbols
                    or strike is None
                    or strike <= 0
                    or option.get("option_type") not in ("call", "put")
                ):
                    raise ValueError("invalid or duplicate option identity")
                seen_symbols.add(symbol)
                _validate_book(option)
                sides = _mapping(option.get("sides", {}), "IV sides")
                for side in sides.values():
                    side = _mapping(side, "IV side")
                    for field in (
                        "iv",
                        "premium_usdc_per_btc",
                        "iterations",
                        "residual_usdc_per_btc",
                    ):
                        if side.get(field) is not None:
                            _numeric_fields(side, (field,), f"IV side {field}")
                if option.get("log_forward_moneyness") is not None:
                    _numeric_fields(option, ("log_forward_moneyness",), "option")
    return report


def load_reports(directory: str | Path | None = None) -> ReportLibrary:
    """Read only direct *-report.json files in a configured local directory.

    No supplied directory means the public demo. A supplied directory never
    silently switches to demo; symlinks and directories are not report files.
    The UI never supplies a pathname and this adapter does not serve files.
    """
    if directory is None:
        return ReportLibrary({DEMO_SESSION: demo_report()})
    root = Path(directory).expanduser()
    sessions: dict[str, Row] = {}
    errors: list[str] = []
    try:
        if not root.is_dir():
            return ReportLibrary({}, ("Configured report directory does not exist.",))
        paths = sorted(root.glob("*-report.json"), key=lambda path: path.name)
        for path in paths:
            if path.is_symlink() or not path.is_file():
                errors.append(f"{path.name}: not a regular local report file")
                continue
            try:
                report = json.loads(
                    path.read_text(encoding="utf-8"),
                    parse_float=_json_number,
                    parse_constant=_json_number,
                )
                sessions[path.stem] = _validate_report(report)
            except OSError:
                errors.append(f"{path.name}: unable to read local report file")
            except (
                UnicodeError,
                json.JSONDecodeError,
                TypeError,
                OverflowError,
                RecursionError,
            ):
                errors.append(
                    f"{path.name}: invalid report data or unsupported structure"
                )
            except ValueError as exc:
                errors.append(f"{path.name}: {exc}")
    except OSError:
        errors.append("Unable to read configured report directory.")
    if not sessions:
        errors.append("No valid *-report.json sessions in the configured directory.")
    return ReportLibrary(sessions, tuple(errors))


def visible_options(view: ReplayView, side: str) -> list[Row]:
    if side not in ("all", "otm", "call", "put"):
        raise ValueError(f"Unknown chain side: {side}")
    options: list[Row] = view.expiry["options"]
    if side == "all":
        return list(options)
    if side in ("call", "put"):
        return [row for row in options if row["option_type"] == side]
    forward = _number(view.expiry.get("forward_input"))
    if forward is None:
        return []
    return [
        row
        for row in options
        if row["option_type"] == ("put" if row["strike"] < forward else "call")
    ]


def choose_symbol(
    view: ReplayView, side: str, preferred: str | None = None
) -> str | None:
    options = visible_options(view, side)
    for row in options:
        if row["symbol"] == preferred:
            return preferred
    forward = _number(view.expiry.get("forward_input"))
    if not options or forward is None:
        return None
    selected = min(
        options, key=lambda row: (abs(row["strike"] - forward), row["symbol"])
    )
    symbol: str = selected["symbol"]
    return symbol


def _age(view: ReplayView, row: Row) -> float | None:
    quote = row.get("normalized_quote") or {}
    stamp = quote.get("quote_time")
    if not isinstance(stamp, str):
        return None
    try:
        return (_stamp(view.checkpoint["as_of"]) - _stamp(stamp)).total_seconds()
    except ValueError:
        return None


def _flags(view: ReplayView, row: Row) -> list[str]:
    flags = [*row.get("normalized_flags", []), *row.get("rejected", [])]
    flags.extend((row.get("native_book") or {}).get("flags", []))
    blocked = view.expiry.get("iv_blocked")
    if blocked:
        flags.append(str(blocked))
    age = _age(view, row)
    max_age = _number(view.report["assumptions"].get("max_book_age_ms"))
    if age is not None and max_age is not None and age * 1000 > max_age:
        flags.append("stale")
    if age is not None and age < 0:
        flags.append("future_quote_time")
    return list(dict.fromkeys(flags))


def _midpoint_iv(row: Row) -> float | None:
    side = (row.get("sides") or {}).get("midpoint", {})
    value = _number(side.get("iv"))
    return value if value is not None and value > 0 and not side.get("error") else None


def market_context(view: ReplayView, symbol: str | None) -> Row:
    """Observed F is never substituted for absent spot; ATM IV is midpoint IV."""
    forward = _number(view.expiry.get("forward_input"))
    candidates = [
        row
        for row in view.expiry["options"]
        if not _flags(view, row) and _midpoint_iv(row) is not None
    ]
    atm = (
        min(candidates, key=lambda row: (abs(row["strike"] - forward), row["symbol"]))
        if candidates and forward is not None
        else None
    )
    return {
        "spot": None,
        "forward": forward,
        "atm_strike": None if atm is None else atm["strike"],
        "atm_iv": None if atm is None else _midpoint_iv(atm),
        "atm_iv_basis": "observed quote midpoint IV at nearest usable strike",
        "dte": (
            _stamp(view.expiry["expiry"]) - _stamp(view.checkpoint["as_of"])
        ).total_seconds()
        / 86_400,
        "selected_symbol": symbol
        if any(row["symbol"] == symbol for row in view.expiry["options"])
        else None,
        "rate": _number(view.report["assumptions"].get("rate")),
        "discount_factor": _number(view.expiry.get("discount_factor")),
    }


def _model_inputs(view: ReplayView, row: Row, flags: list[str]) -> tuple[float, ...]:
    if flags or not row.get("normalized_quote") or _midpoint_iv(row) is None:
        raise ValueError("quote or midpoint IV is unavailable/rejected")
    fit = view.expiry.get("fit")
    if not fit or fit.get("converged") is not True or view.expiry.get("fit_error"):
        raise ValueError("no converged reported SVI fit")
    arbitrage = fit.get("arbitrage") or {}
    if (
        arbitrage.get("passed") is not True
        or not isinstance(arbitrage.get("findings"), list)
        or arbitrage["findings"]
        or not fit.get("residuals")
    ):
        raise ValueError("fitted smile quality is unsupported by recorded checks")
    future = view.expiry.get("matched_future") or {}
    if _flags(view, future) or not future.get("normalized_quote"):
        raise ValueError("matched future unavailable/rejected")
    forward = _number(view.expiry.get("forward_input"))
    discount = _number(view.expiry.get("discount_factor"))
    if forward is None or forward <= 0 or discount is None or discount <= 0:
        raise ValueError("missing positive forward or discount factor")
    future_quote = future["normalized_quote"]
    future_bid, future_ask = (_number(future_quote.get(key)) for key in ("bid", "ask"))
    if (
        future_bid is None
        or future_ask is None
        or not 0 < future_bid <= future_ask
        or not isclose(forward, future_bid / 2 + future_ask / 2, rel_tol=1e-9)
    ):
        raise ValueError("reported forward does not match the usable dated future")
    max_age = _number(view.report["assumptions"].get("max_book_age_ms"))
    if max_age is None or max_age < 0:
        raise ValueError("quote-age quality policy is unavailable")
    for book in (row, future):
        quote = book["normalized_quote"]
        bid, ask = (_number(quote.get(key)) for key in ("bid", "ask"))
        if bid is None or ask is None or not 0 <= bid <= ask:
            raise ValueError("quote quality is unsupported")
        sizes = (_number(quote.get(key)) for key in ("bid_size", "ask_size"))
        if quote.get("price_unit") != "USDC/BTC":
            raise ValueError("normalized quote price units do not support model risk")
        if any(size is None or size <= 0 for size in sizes):
            raise ValueError("displayed quote sizes do not support model risk")
        available = quote.get("available_at")
        if not isinstance(available, str) or _stamp(available) > _stamp(
            view.checkpoint["as_of"]
        ):
            raise ValueError("quote was not available at this checkpoint")
    maturity = year_fraction(
        _stamp(view.checkpoint["as_of"]), _stamp(view.expiry["expiry"])
    )
    if maturity <= 0 or _age(view, row) is None or _age(view, future) is None:
        raise ValueError("missing live maturity or quote timestamps")
    if _stamp(fit["expiry"]) != _stamp(view.expiry["expiry"]) or not isclose(
        fit["time_to_expiry"], maturity, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError("fitted maturity does not match this checkpoint and expiry")
    rate = _number(view.report["assumptions"].get("rate"))
    convention = view.report["assumptions"].get("rate_convention", "")
    if (
        rate is not None
        and isinstance(convention, str)
        and "flat continuously compounded" in convention
        and not isclose(discount, exp(-rate * maturity), rel_tol=1e-9, abs_tol=1e-12)
    ):
        raise ValueError("reported discount factor does not match the flat-rate basis")
    strike = float(row["strike"])
    moneyness = log(strike / forward)
    domain = fit["domain"]
    if not domain[0] - 1e-12 <= moneyness <= domain[1] + 1e-12:
        raise ValueError("strike outside fitted domain; no extrapolation")
    parameters = SVIParameters(**fit["parameters"])
    volatility = sqrt(parameters.total_variance(moneyness) / maturity)
    return forward, strike, maturity, discount, volatility


def contract_detail(view: ReplayView, symbol: str) -> Row:
    """Adapt a real row, withholding unsubstantiated model and risk values."""
    row = next((row for row in view.expiry["options"] if row["symbol"] == symbol), None)
    if row is None:
        raise KeyError(f"Unknown contract: {symbol}")
    flags = _flags(view, row)
    detail: Row = {
        **row,
        "expiry": view.expiry["expiry"],
        "quality_flags": flags,
        "age_seconds": _age(view, row),
        "assumptions": view.report["assumptions"],
        "model_value": None,
        "model_iv": None,
        "model_basis": (
            "Black-76 at reported SVI fitted IV; not quote midpoint price/IV"
        ),
        "greeks": None,
        "risk_basis": _RISK_BASIS,
        "risk_units": dict(_RISK_UNITS),
        "risk_assumptions": list(_RISK_ASSUMPTIONS),
        "risk_unavailable_reason": None,
    }
    try:
        forward, strike, maturity, discount, sigma = _model_inputs(view, row, flags)
        value = float(
            price(row["option_type"], forward, strike, maturity, discount, sigma)
        )
        base = advanced_greeks(
            row["option_type"], forward, strike, maturity, 0, sigma, 0
        )
        greeks: dict[str, float | None] = {
            name: discount * float(getattr(base, name))
            for name in ("delta", "gamma", "vega", "vanna", "volga", "speed")
        }
        greeks.update(theta=None, rho=None, charm=None, color=None)
        rate = _number(view.report["assumptions"].get("rate"))
        convention = view.report["assumptions"].get("rate_convention", "")
        if (
            rate is not None
            and isinstance(convention, str)
            and "flat continuously compounded" in convention
            and isclose(discount, exp(-rate * maturity), rel_tol=1e-9, abs_tol=1e-12)
        ):
            # Coordinate/discount conversion, not a duplicate pricing model.
            greeks.update(
                rho=-maturity * value,
                theta=discount * base.theta + rate * value,
                charm=discount * (base.charm + rate * base.delta),
                color=discount * (base.color + rate * base.gamma),
            )
        else:
            detail["risk_unavailable_reason"] = (
                "Theta, rho, charm and color require a recorded flat, continuously "
                "compounded rate. Other forward sensitivities remain available."
            )
        if not isfinite(value) or any(
            risk is not None and not isfinite(risk) for risk in greeks.values()
        ):
            raise ValueError("model or risk exceeds finite numerical range")
        detail.update(model_value=value, model_iv=sigma, greeks=greeks)
    except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError) as exc:
        detail["risk_unavailable_reason"] = str(exc)
    return detail
