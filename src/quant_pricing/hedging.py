"""Synthetic, single-currency European option hedging on an elapsed ACT/365F grid.

Every row is available at its own elapsed time; decisions use that row only.
Option marks use the fixed model volatility, never an inferred path volatility.
Cash interest compounds; signed dividend carry is left-point ``h*S*q*dt``.
Stock average cost uses mid prices, with execution costs realized separately.
This is cash-settled research accounting, not an executable market simulator.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from math import exp, expm1, isfinite
from numbers import Real
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quant_pricing.black_scholes import OptionType, value_and_greeks
from quant_pricing.exceptions import InputValidationError

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


def _finite(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise InputValidationError(f"{name} must be a finite real number")


def _array(value: ArrayLike, name: str) -> FloatArray:
    if isinstance(value, np.ma.MaskedArray) and np.any(value.mask):
        raise InputValidationError(f"{name} must not contain missing observations")
    if np.iscomplexobj(value):
        raise InputValidationError(f"{name} must be real")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise InputValidationError(f"{name} must be numeric") from error
    if not np.all(np.isfinite(result)):
        raise InputValidationError(f"{name} must be finite")
    return result


def _frozen(value: FloatArray) -> FloatArray:
    # Bytes backing prevents callers from re-enabling WRITEABLE on owned arrays.
    return np.frombuffer(value.tobytes(), dtype=np.float64).reshape(value.shape)


@dataclass(frozen=True, slots=True)
class HedgeContract:
    """One new option position; quantity is signed, premium is per option unit.

    Multiplier converts contracts to underlying units. Pre-entry wealth is cash
    plus initial stock at S0; that stock's initial average cost is S0.
    """

    kind: str
    strike: float
    maturity: float
    model_volatility: float
    rate: float = 0.0
    dividend_yield: float = 0.0
    quantity: float = -1.0
    multiplier: float = 1.0
    premium: float | None = None
    initial_cash: float = 0.0
    initial_inventory: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", OptionType.parse(self.kind).value)
        for field in fields(self):
            if field.name not in {"kind", "premium"}:
                _finite(getattr(self, field.name), field.name)
        for name in ("strike", "maturity", "multiplier"):
            if getattr(self, name) <= 0:
                raise InputValidationError(f"{name} must be strictly positive")
        if self.model_volatility < 0:
            raise InputValidationError("model_volatility must be nonnegative")
        if self.premium is not None:
            _finite(self.premium, "premium")
            if self.premium < 0:
                raise InputValidationError("premium must be nonnegative")
        _finite(self.quantity * self.multiplier, "scaled option quantity")

    def _pricing_key(self) -> tuple[str, float, float, float, float, float]:
        return (
            self.kind,
            self.strike,
            self.maturity,
            self.model_volatility,
            self.rate,
            self.dividend_yield,
        )


@dataclass(frozen=True, slots=True)
class HedgePolicy:
    """Periodic rebalances at t0; threshold is a strict underlying-unit bound."""

    kind: Literal["none", "periodic", "threshold"]
    every: int = 1
    threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"none", "periodic", "threshold"}:
            raise InputValidationError("unsupported hedge policy")
        if (
            isinstance(self.every, bool)
            or not isinstance(self.every, (int, np.integer))
            or self.every <= 0
        ):
            raise InputValidationError("every must be a positive integer")
        _finite(self.threshold, "threshold")
        if self.threshold < 0:
            raise InputValidationError("threshold must be nonnegative")


@dataclass(frozen=True, slots=True)
class HedgeCosts:
    """Symmetric stock side-price costs; fees use absolute executed notional."""

    half_spread_bps: float = 0.0
    slippage_bps: float = 0.0
    fee_bps: float = 0.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            _finite(value, field.name)
            if value < 0:
                raise InputValidationError(f"{field.name} must be nonnegative")
        if self.half_spread_bps + self.slippage_bps >= 10_000:
            raise InputValidationError(
                "combined side-price costs must be below 10000 bps"
            )


@dataclass(frozen=True, slots=True)
class PreparedMarket:
    """Owned immutable spots and BSM marks, with exact pricing-input identity.

    ``times`` are elapsed ACT/365F years and also observation availability times.
    Rows span zero through contract maturity on a uniform deterministic grid.
    Use :func:`prepare_market` to produce internally consistent BSM valuations.
    """

    contract: HedgeContract
    spots: FloatArray
    marks: FloatArray = dataclasses.field(init=False)
    deltas: FloatArray = dataclasses.field(init=False)
    times: FloatArray = dataclasses.field(init=False)

    def __init__(self, contract: HedgeContract, spots: ArrayLike) -> None:
        prices = _frozen(_array(spots, "spots"))
        if prices.ndim != 2 or prices.shape[0] < 2 or prices.shape[1] < 1:
            raise InputValidationError(
                "spots must have shape (steps+1, paths), both nonempty"
            )
        if np.any(prices <= 0):
            raise InputValidationError("spots must be strictly positive")
        times = np.linspace(0.0, contract.maturity, prices.shape[0], dtype=np.float64)
        if np.any(np.diff(times) <= 0):
            raise InputValidationError("elapsed time grid must be strictly increasing")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                valuation = value_and_greeks(
                    contract.kind,
                    prices,
                    contract.strike,
                    (contract.maturity - times)[:, None],
                    contract.rate,
                    contract.model_volatility,
                    contract.dividend_yield,
                )
        except (FloatingPointError, OverflowError) as error:
            raise InputValidationError(
                "market valuation exceeds numerical range"
            ) from error
        marks = _array(valuation.price, "marks")
        if np.any(marks < 0):
            raise InputValidationError("option marks must be nonnegative")
        object.__setattr__(self, "contract", contract)
        object.__setattr__(self, "spots", prices)
        object.__setattr__(self, "times", _frozen(times))
        object.__setattr__(self, "marks", _frozen(marks))
        deltas = _array(valuation.greeks.delta, "deltas")
        object.__setattr__(self, "deltas", _frozen(deltas))


@dataclass(frozen=True, slots=True)
class HedgeLedgerRow:
    """One post-event observation, including opening and terminal events.

    Cashflows, hedge_trade, option_trade, turnover and cost/carry fields are for
    this event only. Inventory, cash, equity, average_cost and P&L are cumulative
    state. realized_pnl + unrealized_pnl = equity - pre-entry wealth. Costs and
    carry are immediately realized; option P&L becomes realized at settlement.
    Stock P&L uses signed average-mid-cost, resetting on a cross through zero.
    execution_price equals spot on no-trade rows. Terminal option_mark and delta
    retain the per-unit payoff valuation, but option_units and residual are zero.
    """

    step: int
    time: float
    available_at: float
    spot: float
    option_units: float
    option_mark: float
    option_delta: float
    option_trade: float
    option_cashflow: float
    hedge_inventory: float
    hedge_trade: float
    execution_price: float
    hedge_cashflow: float
    fees: float
    spread_cost: float
    slippage_cost: float
    cash_interest: float
    dividend_credit: float
    turnover: float
    cash: float
    equity: float
    residual_delta: float
    average_cost: float
    stock_realized_pnl: float
    stock_unrealized_pnl: float
    realized_pnl: float
    unrealized_pnl: float


@dataclass(frozen=True, slots=True)
class HedgeResult:
    """Per-path terminal USD attribution; turnover is absolute executed notional.

    hedge_error is discounted terminal equity minus pre-entry equity (net of all
    costs), not a centered statistic. Drawdown is absolute peak-to-trough equity
    including pre-entry wealth; residual delta is interval-weighted post-trade
    absolute exposure, excluding expiry. trade_count counts nonzero stock trades
    including opening and terminal liquidation, not option transactions.
    """

    final_pnl: FloatArray
    hedge_error: FloatArray
    turnover: FloatArray
    fees: FloatArray
    spread_cost: FloatArray
    slippage_cost: FloatArray
    financing: FloatArray
    dividends: FloatArray
    option_pnl: FloatArray
    hedge_pnl: FloatArray
    mean_abs_residual_delta: FloatArray
    max_drawdown: FloatArray
    trade_count: IntArray
    ledger: tuple[HedgeLedgerRow, ...]

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name not in {"ledger", "trade_count"}:
                value = _array(getattr(self, field.name), field.name)
                object.__setattr__(self, field.name, _frozen(value))
        object.__setattr__(
            self,
            "trade_count",
            np.frombuffer(self.trade_count.tobytes(), dtype=np.int64),
        )


def prepare_market(contract: HedgeContract, spots: ArrayLike) -> PreparedMarket:
    """Price a spot matrix once for reuse across policy and transaction-cost runs."""
    return PreparedMarket(contract, spots)


def _stock_basis(
    inventory: float, trade: float, spot: float, basis: float, realized: float
) -> tuple[float, float]:
    """Update signed average-mid-cost; execution costs are a separate channel."""
    remaining = inventory + trade
    opposite = (inventory > 0 and trade < 0) or (inventory < 0 and trade > 0)
    if opposite:
        closed = min(abs(inventory), abs(trade))
        realized += closed * (1.0 if inventory > 0 else -1.0) * (spot - basis)
    if remaining == 0:
        basis = 0.0
    elif inventory == 0 or (inventory > 0) != (remaining > 0):
        basis = spot
    elif trade != 0 and not opposite:
        # Convex weights avoid overflowing a large stock notional unnecessarily.
        basis += (spot - basis) * (abs(trade) / abs(remaining))
    return basis, realized


def simulate_hedge(
    contract: HedgeContract,
    market: PreparedMarket,
    policy: HedgePolicy,
    costs: HedgeCosts,
    *,
    trace_path: int | None = None,
) -> HedgeResult:
    """Step through known-at-time observations, vectorizing across all paths.

    Expiry always cash-settles the option and liquidates stock, regardless of
    policy. A cached market may vary position/premium/initial funding but must
    match every BSM pricing field exactly. Numeric range failures are errors.
    """
    if contract._pricing_key() != market.contract._pricing_key():
        raise InputValidationError(
            "contract pricing fields do not match prepared market"
        )
    if trace_path is not None and (
        isinstance(trace_path, bool)
        or not isinstance(trace_path, (int, np.integer))
        or not 0 <= trace_path < market.spots.shape[1]
    ):
        raise InputValidationError("trace_path must be a valid integer path index")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _simulate(contract, market, policy, costs, trace_path)
    except (FloatingPointError, OverflowError) as error:
        raise InputValidationError(
            "hedge accounting exceeds numerical range"
        ) from error


def _simulate(
    contract: HedgeContract,
    market: PreparedMarket,
    policy: HedgePolicy,
    costs: HedgeCosts,
    trace_path: int | None,
) -> HedgeResult:
    spots, marks, deltas, times = (
        market.spots,
        market.marks,
        market.deltas,
        market.times,
    )
    rows, paths = spots.shape
    units = contract.quantity * contract.multiplier
    premium = marks[0] if contract.premium is None else np.full(paths, contract.premium)
    initial = contract.initial_cash + contract.initial_inventory * spots[0]
    cash = np.full(paths, contract.initial_cash, dtype=np.float64)
    inventory: FloatArray = np.full(paths, contract.initial_inventory, dtype=np.float64)
    financing = np.zeros(paths)
    dividends = np.zeros(paths)
    hedge_pnl = np.zeros(paths)
    fees = np.zeros(paths)
    spread = np.zeros(paths)
    slippage = np.zeros(paths)
    turnover = np.zeros(paths)
    residual_sum = np.zeros(paths)
    drawdown = np.zeros(paths)
    peak = initial.copy()
    trade_count = np.zeros(paths, dtype=np.int64)
    ledger: list[HedgeLedgerRow] = []
    basis = float(spots[0, trace_path]) if trace_path is not None else 0.0
    stock_realized = 0.0
    spread_rate = costs.half_spread_bps * 1e-4
    slippage_rate = costs.slippage_bps * 1e-4
    fee_rate = costs.fee_bps * 1e-4
    zero: FloatArray = np.zeros(paths)
    trade: FloatArray
    for step in range(rows):
        spot = spots[step]
        terminal = step == rows - 1
        interest, dividend = zero, zero
        option_cashflow = zero
        option_trade = 0.0
        if step == 0:
            option_cashflow = -units * premium
            option_trade = units
        else:
            dt = float(times[step] - times[step - 1])
            interest = cash * expm1(contract.rate * dt)
            dividend = inventory * spots[step - 1] * contract.dividend_yield * dt
            financing += interest
            dividends += dividend
            hedge_pnl += inventory * (spot - spots[step - 1])
        if terminal:
            option_cashflow = units * marks[step]
            option_trade = -units
            trade = -inventory
        else:
            target = -units * deltas[step]
            if policy.kind == "periodic" and step % policy.every == 0:
                trade = target - inventory
            elif policy.kind == "threshold":
                difference = target - inventory
                trade = np.where(np.abs(difference) > policy.threshold, difference, 0.0)
            else:
                trade = zero
        execution_price = spot * (1.0 + np.sign(trade) * (spread_rate + slippage_rate))
        event_turnover = np.abs(trade) * execution_price
        event_fees = event_turnover * fee_rate
        mid_notional = np.abs(trade) * spot
        event_spread = mid_notional * spread_rate
        event_slippage = mid_notional * slippage_rate
        hedge_cashflow = -trade * execution_price
        cash += interest + dividend + option_cashflow + hedge_cashflow - event_fees
        if trace_path is not None:
            basis, stock_realized = _stock_basis(
                float(inventory[trace_path]),
                float(trade[trace_path]),
                float(spot[trace_path]),
                basis,
                stock_realized,
            )
        inventory = inventory + trade
        fees += event_fees
        spread += event_spread
        slippage += event_slippage
        turnover += event_turnover
        trade_count += trade != 0
        option_units = 0.0 if terminal else units
        equity = cash + option_units * marks[step] + inventory * spot
        residual = option_units * deltas[step] + inventory
        if not terminal:
            weight = float((times[step + 1] - times[step]) / contract.maturity)
            residual_sum += np.abs(residual) * weight
        peak = np.maximum(peak, equity)
        drawdown = np.maximum(drawdown, peak - equity)
        for value in (cash, inventory, equity, residual, hedge_pnl):
            if not np.all(np.isfinite(value)):
                raise InputValidationError("hedge accounting exceeds numerical range")
        if trace_path is not None:
            p = trace_path
            stock_unrealized = float(inventory[p]) * (float(spot[p]) - basis)
            option_unrealized = option_units * float(marks[step, p] - premium[p])
            option_realized = (
                units * float(marks[step, p] - premium[p]) if terminal else 0.0
            )
            realized = (
                stock_realized
                + option_realized
                + float(financing[p] + dividends[p])
                - float(fees[p] + spread[p] + slippage[p])
            )
            row = HedgeLedgerRow(
                step=step,
                time=float(times[step]),
                available_at=float(times[step]),
                spot=float(spot[p]),
                option_units=option_units,
                option_mark=float(marks[step, p]),
                option_delta=float(deltas[step, p]),
                option_trade=option_trade,
                option_cashflow=float(option_cashflow[p]),
                hedge_inventory=float(inventory[p]),
                hedge_trade=float(trade[p]),
                execution_price=float(execution_price[p]),
                hedge_cashflow=float(hedge_cashflow[p]),
                fees=float(event_fees[p]),
                spread_cost=float(event_spread[p]),
                slippage_cost=float(event_slippage[p]),
                cash_interest=float(interest[p]),
                dividend_credit=float(dividend[p]),
                turnover=float(event_turnover[p]),
                cash=float(cash[p]),
                equity=float(equity[p]),
                residual_delta=float(residual[p]),
                average_cost=basis,
                stock_realized_pnl=stock_realized,
                stock_unrealized_pnl=stock_unrealized,
                realized_pnl=realized,
                unrealized_pnl=stock_unrealized + option_unrealized,
            )
            for field in fields(row):
                _finite(getattr(row, field.name), field.name)
            ledger.append(row)
    option_pnl = units * (marks[-1] - premium)
    final_pnl = cash - initial
    attributed = (
        option_pnl + hedge_pnl + financing + dividends - fees - spread - slippage
    )
    scale = (
        1.0
        + np.abs(initial)
        + np.abs(cash)
        + np.abs(option_pnl)
        + np.abs(hedge_pnl)
        + np.abs(financing)
        + np.abs(dividends)
        + fees
        + spread
        + slippage
    )
    if not np.all(np.abs(final_pnl - attributed) <= 1e-10 * scale):
        raise InputValidationError("terminal P&L attribution failed to reconcile")
    discount = exp(-contract.rate * contract.maturity)
    if discount == 0 or not isfinite(discount):
        raise InputValidationError("terminal discount exceeds numerical range")
    return HedgeResult(
        final_pnl=final_pnl,
        hedge_error=discount * cash - initial,
        turnover=turnover,
        fees=fees,
        spread_cost=spread,
        slippage_cost=slippage,
        financing=financing,
        dividends=dividends,
        option_pnl=option_pnl,
        hedge_pnl=hedge_pnl,
        mean_abs_residual_delta=residual_sum,
        max_drawdown=drawdown,
        trade_count=trade_count,
        ledger=tuple(ledger),
    )
