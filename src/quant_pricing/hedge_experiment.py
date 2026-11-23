"""Bounded synthetic hedging research with deterministic, reanalysable artifacts.

Run ``python -m quant_pricing.hedge_experiment CONFIG --output-dir DIR``.
All monetary quantities are USD; volatility inputs are absolute annual decimals.
The experiment is not an optimizer, a backtest, or evidence of trading profits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import scipy
from numpy.typing import NDArray

from quant_pricing import __version__
from quant_pricing.hedge_paths import (
    GBM_BIT_GENERATOR,
    GBM_VERSION,
    GBMConfig,
    generate_gbm,
)
from quant_pricing.hedging import (
    HedgeContract,
    HedgeCosts,
    HedgePolicy,
    HedgeResult,
    prepare_market,
    simulate_hedge,
)

SCHEMA_VERSION = "p5.19a.v1"
FloatArray: TypeAlias = NDArray[np.float64]
Row: TypeAlias = dict[str, Any]
_RESULT_FIELDS = (
    "final_pnl",
    "hedge_error",
    "turnover",
    "fees",
    "spread_cost",
    "slippage_cost",
    "financing",
    "dividends",
    "option_pnl",
    "hedge_pnl",
    "mean_abs_residual_delta",
    "max_drawdown",
    "trade_count",
)
_ARTIFACTS = (
    "results.json",
    "summary.csv",
    "per_seed_summary.csv",
    "paired_comparisons.csv",
    "path_results.csv",
    "example_ledger.csv",
    "hedging_tradeoff.svg",
)


@dataclass(frozen=True)
class MarketSpec:
    """Physical ex-dividend price process; horizon comes from the contract."""

    spot: float
    drift: float
    steps: int
    paths: int
    seeds: tuple[int, ...]
    path_volatilities: tuple[float, ...]


@dataclass(frozen=True)
class PolicySpec:
    id: str
    policy: HedgePolicy


@dataclass(frozen=True)
class CostSpec:
    id: str
    costs: HedgeCosts


@dataclass(frozen=True)
class TraceSpec:
    seed: int
    path_volatility: float
    policy: str
    cost: str
    path: int


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated configuration returned by :func:`parse_config`."""

    schema_version: str
    experiment_id: str
    contract: HedgeContract
    market: MarketSpec
    policies: tuple[PolicySpec, ...]
    cost_schedules: tuple[CostSpec, ...]
    zero_cost_reference: str
    example: TraceSpec

    def to_dict(self) -> Row:
        """Return the canonical, public JSON configuration shape."""
        result = asdict(self)
        result["policies"] = [
            {"id": item.id, **asdict(item.policy)} for item in self.policies
        ]
        result["cost_schedules"] = [
            {"id": item.id, **asdict(item.costs)} for item in self.cost_schedules
        ]
        return result


def _object(value: object, label: str, required: set[str], optional: set[str]) -> Row:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{label} must be a JSON object")
    missing, unknown = required - value.keys(), value.keys() - required - optional
    if missing or unknown:
        raise ValueError(
            f"{label}: missing fields {sorted(missing)}, "
            f"unknown fields {sorted(unknown)}"
        )
    return cast(Row, value)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _name(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not all(
            char.isascii() and (char.isalnum() or char in "_-. ") for char in value
        )
    ):
        raise ValueError(f"{label} must be a nonempty ASCII name (letters/digits/_-. )")
    return value


def _items(value: object, label: str) -> list[Any]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a nonempty array")
    return list(value)


def _unique(values: Sequence[object], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def parse_config(value: object) -> ExperimentConfig:
    """Validate a JSON-shaped config, including an explicit zero-cost reference.

    Small grids are accepted for reproducibility tests. The committed example is
    the predeclared multi-seed, three-volatility, three-cost research experiment.
    Unknown fields are rejected, so misspellings never silently change a study.
    """
    root = _object(
        value,
        "config",
        {
            "schema_version",
            "experiment_id",
            "contract",
            "market",
            "policies",
            "cost_schedules",
            "zero_cost_reference",
            "example",
        },
        set(),
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    contract_data = _object(
        root["contract"],
        "contract",
        {
            "kind",
            "strike",
            "maturity",
            "model_volatility",
        },
        {
            "rate",
            "dividend_yield",
            "quantity",
            "multiplier",
            "premium",
            "initial_cash",
            "initial_inventory",
        },
    )
    kind = contract_data["kind"]
    if kind not in ("call", "put"):
        raise ValueError("contract.kind must be call or put")
    numbers = {
        key: _number(item, f"contract.{key}")
        for key, item in contract_data.items()
        if key not in ("kind", "premium")
    }
    premium = contract_data.get("premium")
    contract = HedgeContract(
        kind=kind,
        strike=numbers["strike"],
        maturity=numbers["maturity"],
        model_volatility=numbers["model_volatility"],
        rate=numbers.get("rate", 0.0),
        dividend_yield=numbers.get("dividend_yield", 0.0),
        quantity=numbers.get("quantity", -1.0),
        multiplier=numbers.get("multiplier", 1.0),
        premium=None if premium is None else _number(premium, "contract.premium"),
        initial_cash=numbers.get("initial_cash", 0.0),
        initial_inventory=numbers.get("initial_inventory", 0.0),
    )
    data = _object(
        root["market"],
        "market",
        {
            "spot",
            "drift",
            "steps",
            "paths",
            "seeds",
            "path_volatilities",
        },
        set(),
    )
    seeds = tuple(_integer(item, "seed") for item in _items(data["seeds"], "seeds"))
    vols = tuple(
        _number(item, "path_volatility")
        for item in _items(data["path_volatilities"], "path_volatilities")
    )
    _unique(seeds, "seeds")
    _unique(vols, "path_volatilities")
    market = MarketSpec(
        _number(data["spot"], "spot"),
        _number(data["drift"], "drift"),
        _integer(data["steps"], "steps", 1),
        _integer(data["paths"], "paths", 2),
        seeds,
        vols,
    )
    # Reuse the process validator without drawing samples.
    for volatility in vols:
        GBMConfig(
            market.spot,
            market.drift,
            volatility,
            contract.maturity,
            market.steps,
            market.paths,
            seeds[0],
        )
    policies: list[PolicySpec] = []
    for item in _items(root["policies"], "policies"):
        data = _object(item, "policy", {"id", "kind"}, {"every", "threshold"})
        policy_kind = data["kind"]
        if policy_kind not in ("none", "periodic", "threshold"):
            raise ValueError("policy.kind must be none, periodic or threshold")
        policies.append(
            PolicySpec(
                _name(data["id"], "policy.id"),
                HedgePolicy(
                    kind=policy_kind,
                    every=_integer(data.get("every", 1), "policy.every", 1),
                    threshold=_number(data.get("threshold", 0.0), "policy.threshold"),
                ),
            )
        )
    costs: list[CostSpec] = []
    for item in _items(root["cost_schedules"], "cost_schedules"):
        data = _object(
            item,
            "cost schedule",
            {
                "id",
                "half_spread_bps",
                "slippage_bps",
                "fee_bps",
            },
            set(),
        )
        costs.append(
            CostSpec(
                _name(data["id"], "cost.id"),
                HedgeCosts(
                    **{
                        key: _number(data[key], f"cost.{key}")
                        for key in ("half_spread_bps", "slippage_bps", "fee_bps")
                    }
                ),
            )
        )
    policy_ids = [item.id for item in policies]
    cost_ids = [item.id for item in costs]
    _unique(policy_ids, "policy ids")
    _unique(cost_ids, "cost ids")
    reference = _name(root["zero_cost_reference"], "zero_cost_reference")
    if reference not in cost_ids:
        raise ValueError("zero_cost_reference must name a configured cost schedule")
    reference_cost = costs[cost_ids.index(reference)].costs
    if any(value != 0.0 for value in asdict(reference_cost).values()):
        raise ValueError("zero_cost_reference must have all three costs exactly zero")
    data = _object(
        root["example"],
        "example",
        {
            "seed",
            "path_volatility",
            "policy",
            "cost",
            "path",
        },
        set(),
    )
    trace = TraceSpec(
        _integer(data["seed"], "example.seed"),
        _number(data["path_volatility"], "example.path_volatility"),
        _name(data["policy"], "example.policy"),
        _name(data["cost"], "example.cost"),
        _integer(data["path"], "example.path"),
    )
    if (
        trace.seed not in seeds
        or trace.path_volatility not in vols
        or trace.policy not in policy_ids
        or trace.cost not in cost_ids
        or trace.path >= market.paths
    ):
        raise ValueError("example must select a configured scenario and valid path")
    return ExperimentConfig(
        SCHEMA_VERSION,
        _name(root["experiment_id"], "experiment_id"),
        contract,
        market,
        tuple(policies),
        tuple(costs),
        reference,
        trace,
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Read and validate the versioned experiment JSON."""
    return parse_config(json.loads(Path(path).read_text(encoding="utf-8")))


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _source_fingerprint() -> Row:
    """Hash shipped Python sources, independent of installation path or git."""
    root = Path(__file__).parent
    hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*.py")
    }
    return {"sha256": _digest(hashes), "module_sha256": hashes}


def _provenance(config: ExperimentConfig) -> Row:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "quant_pricing": __version__,
    }
    source = _source_fingerprint()
    data_generation = {
        "source": "synthetic",
        "algorithm": GBM_VERSION,
        "rng": GBM_BIT_GENERATOR,
        "normal_draw_order": "time-major C order",
        "market": asdict(config.market),
        "horizon_years": config.contract.maturity,
        "numpy_version": np.__version__,
        "common_random_numbers": (
            "Identical seed and (steps,paths) regenerate the same standard-normal "
            "matrix across generating volatilities; all policies/costs share spots."
        ),
    }
    identity = {
        "config_sha256": _digest(config.to_dict()),
        "source_sha256": source["sha256"],
        "versions": versions,
        "data_generation_sha256": _digest(data_generation),
    }
    forward = config.market.spot * math.exp(
        (config.contract.rate - config.contract.dividend_yield)
        * config.contract.maturity
    )
    if not math.isfinite(forward) or forward <= 0:
        raise ValueError("initial flat-curve forward is not finite and positive")
    return {
        **identity,
        "experiment_sha256": _digest(identity),
        "source": source,
        "data_generation": data_generation,
        "initial_forward": {
            "value": forward,
            "unit": "USD per underlying unit",
            "formula": "S0*exp((r-q)*T)",
            "label": "flat-curve implied forward",
        },
        "conventions": {
            "time": "elapsed uniform ACT/365F years; synthetic valuation origin 0",
            "drift": "physical ex-dividend price drift, separately specified from r,q",
            "volatility": (
                "absolute annual decimals; model vol fixed for hedge decisions; "
                "path vol is generating diffusion, not sample realized volatility"
            ),
            "currency": "USD",
            "quantity": "signed option contracts times multiplier",
            "threshold": "absolute post-position residual delta in underlying units",
            "turnover": "absolute hedge notional at execution, USD; fee basis",
            "exposure": "time-weighted absolute post-trade residual delta, stock units",
            "drawdown": "peak-to-trough equity decline, USD, including initial costs",
            "error": "exp(-r*T)*terminal_equity - pre_entry_equity, USD",
            "nominal_pnl": "terminal_equity - pre_entry_equity, USD",
            "cost_drag": "paired zero-cost minus charged nominal P&L, with funding",
            "sd": "sample standard deviation (ddof=1)",
            "quantiles": "NumPy linear sample quantiles; no tail extrapolation",
            "uncertainty": (
                "iid path standard errors within fixed seed/dimension generation; "
                "per-seed summaries and paired differences, not profitability intervals"
            ),
            "cash_dividends": "left-point h*S*q*dt cash credit",
            "ledger": (
                "one row per epoch; fees/spread/slippage/cash_interest/dividend_credit/"
                "turnover/option_cashflow/hedge_cashflow are flows; cash, equity, "
                "realized_pnl/unrealized_pnl are post-event cumulative levels; "
                "stock basis uses average mid cost; costs/carry realized immediately"
            ),
        },
    }


def _arrays(result: HedgeResult, zero: HedgeResult) -> dict[str, FloatArray]:
    values = {
        name: np.asarray(getattr(result, name), dtype=np.float64)
        for name in _RESULT_FIELDS
    }
    values["paired_cost_drag"] = zero.final_pnl - result.final_pnl
    for name, array in values.items():
        if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
            raise ValueError(f"invalid/nonfinite result array: {name}")
    return values


def _summarize(values: Mapping[str, FloatArray]) -> Row:
    nominal, error = values["final_pnl"], values["hedge_error"]
    count = nominal.size
    result: Row = {"paths": int(count)}
    for prefix, array in (("nominal_pnl", nominal), ("discounted_error", error)):
        sd = float(np.std(array, ddof=1))
        p05, median, p95 = np.quantile(array, [0.05, 0.5, 0.95])
        result.update(
            {
                f"{prefix}_mean": float(np.mean(array)),
                f"{prefix}_sample_sd": sd,
                f"{prefix}_mean_se": sd / math.sqrt(count),
                f"{prefix}_p05": float(p05),
                f"{prefix}_median": float(median),
                f"{prefix}_p95": float(p95),
            }
        )
    result["discounted_error_bias"] = result["discounted_error_mean"]
    result["discounted_error_rmse"] = float(np.sqrt(np.mean(np.square(error))))
    for name in (*_RESULT_FIELDS[2:], "paired_cost_drag"):
        result[f"{name}_mean"] = float(np.mean(values[name]))
    drag_sd = float(np.std(values["paired_cost_drag"], ddof=1))
    result["paired_cost_drag_sample_sd"] = drag_sd
    result["paired_cost_drag_mean_se"] = drag_sd / math.sqrt(count)
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("nonfinite summary statistic; refusing to emit filled results")
    return result


def _write_csv(path: Path, rows: Sequence[Row]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _paired_row(label: Row, values: FloatArray) -> Row:
    sd = float(np.std(values, ddof=1))
    return {
        **label,
        "paths": int(values.size),
        "paired_difference_mean": float(values.mean()),
        "paired_difference_sample_sd": sd,
        "paired_difference_mean_se": sd / math.sqrt(values.size),
    }


def _comparisons(
    config: ExperimentConfig,
    pooled: Mapping[tuple[float, str, str], dict[str, FloatArray]],
) -> list[Row]:
    """Paired P&L and squared-error changes, never independent-sample SEs."""
    reference_policy = min(
        config.policies,
        key=lambda item: (
            item.policy.every if item.policy.kind == "periodic" else math.inf,
            item.id,
        ),
    ).id
    reference_vol = min(
        config.market.path_volatilities,
        key=lambda value: (abs(value - config.contract.model_volatility), value),
    )
    rows: list[Row] = []
    for (volatility, policy, cost), values in pooled.items():
        for axis, reference_key in (
            ("policy", (volatility, reference_policy, cost)),
            ("path_volatility", (reference_vol, policy, cost)),
        ):
            reference = pooled[reference_key]
            for metric in ("final_pnl", "hedge_error", "squared_discounted_error"):
                if metric == "squared_discounted_error":
                    difference = np.square(values["hedge_error"]) - np.square(
                        reference["hedge_error"]
                    )
                else:
                    difference = values[metric] - reference[metric]
                rows.append(
                    _paired_row(
                        {
                            "axis": axis,
                            "path_volatility": volatility,
                            "policy": policy,
                            "cost": cost,
                            "reference_path_volatility": reference_key[0],
                            "reference_policy": reference_key[1],
                            "metric": metric,
                            "difference_convention": "scenario minus paired reference",
                        },
                        difference,
                    )
                )
    return rows


def _chart(summary: Sequence[Row], config: ExperimentConfig) -> str:
    """Standalone SVG drawn only from measured summary values."""
    width, height = 1200, 610
    panel_width = 1100 / len(config.cost_schedules)
    xmax = max(float(row["turnover_mean"]) for row in summary) or 1.0
    ymax = max(float(row["discounted_error_sample_sd"]) for row in summary) or 1.0
    colors = ("#2563eb", "#aa5808", "#047857", "#9333ea", "#dc2626")
    policy_numbers = {item.id: index for index, item in enumerate(config.policies, 1)}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Synthetic hedge error dispersion versus turnover</title>',
        '<desc id="desc">Common-shock synthetic scenarios, grouped by execution costs. '
        "Points show measured pooled sample standard deviations, not forecasts.</desc>",
        '<rect width="1200" height="610" fill="white"/>',
        '<g font-family="sans-serif" fill="#172033">',
        '<text x="55" y="35" font-size="22">Synthetic hedging: '
        "dispersion / turnover trade-off</text>",
        '<text x="55" y="60" font-size="13">Discounted net error sample SD (USD); '
        "turnover is mean hedge notional (USD). Axes shared across costs.</text>",
    ]
    for index, cost in enumerate(config.cost_schedules):
        left, top, bottom = 65 + index * panel_width, 125.0, 455.0
        right = left + panel_width - 40
        svg.append(
            f'<text x="{left:.2f}" y="100" font-size="16">{escape(cost.id)}</text>'
        )
        svg.append(
            f'<text x="{left:.2f}" y="117" font-size="10">half-spread/slippage/fee: '
            f"{cost.costs.half_spread_bps:g}/{cost.costs.slippage_bps:g}/"
            f"{cost.costs.fee_bps:g} bps</text>"
        )
        for tick in range(5):
            fraction = tick / 4
            x = left + fraction * (right - left)
            y = bottom - fraction * (bottom - top)
            svg.extend(
                [
                    f'<path d="M {left:.2f} {y:.2f} H {right:.2f}" stroke="#e2e8f0"/>',
                    f'<text x="{left - 5:.2f}" y="{y + 4:.2f}" text-anchor="end" '
                    f'font-size="10">{fraction * ymax:.2f}</text>',
                    f'<text x="{x:.2f}" y="475" text-anchor="middle" '
                    f'font-size="10">{fraction * xmax:.0f}</text>',
                ]
            )
        svg.append(
            f'<path d="M {left:.2f} {top:.2f} V {bottom:.2f} H {right:.2f}" '
            'fill="none" stroke="#64748b"/>'
        )
        for vol_index, volatility in enumerate(config.market.path_volatilities):
            color = colors[vol_index % len(colors)]
            rows = sorted(
                (
                    row
                    for row in summary
                    if row["cost"] == cost.id and row["path_volatility"] == volatility
                ),
                key=lambda row: row["turnover_mean"],
            )
            points: list[str] = []
            for row in rows:
                x = left + float(row["turnover_mean"]) / xmax * (right - left)
                y = bottom - float(row["discounted_error_sample_sd"]) / ymax * (
                    bottom - top
                )
                points.append(f"{x:.2f},{y:.2f}")
                tooltip = escape(
                    f"{row['policy']}; generating vol={volatility:g}; "
                    f"SD={row['discounted_error_sample_sd']:.6g} USD; "
                    f"turnover={row['turnover_mean']:.6g} USD; "
                    f"paired drag={row['paired_cost_drag_mean']:.6g} USD"
                )
                svg.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}">'
                    f"<title>{tooltip}</title></circle>"
                )
                if volatility == config.contract.model_volatility:
                    svg.append(
                        f'<text x="{x + 5:.2f}" y="{y - 6:.2f}" font-size="9">'
                        f"{policy_numbers[row['policy']]}</text>"
                    )
            svg.append(
                f'<polyline points="{" ".join(points)}" fill="none" '
                f'stroke="{color}" stroke-opacity="0.45"/>'
            )
    for index, volatility in enumerate(config.market.path_volatilities):
        x = 65 + 200 * index
        svg.append(
            f'<text x="{x}" y="515" font-size="14" '
            f'fill="{colors[index % len(colors)]}">path vol = {volatility:g}</text>'
        )
    svg.extend(
        [
            '<text x="65" y="545" font-size="12">Matched-volatility labels: '
            + " · ".join(f"{i}: {escape(name)}" for name, i in policy_numbers.items())
            + "</text>",
            '<text x="65" y="567" font-size="12">SD is not RMSE or expected profit. '
            "Common shocks pair policy, cost and volatility stress comparisons.</text>",
            f'<text x="65" y="589" font-size="11">{escape(config.experiment_id)}; '
            f"{len(config.market.seeds)} seeds x {config.market.paths} paths; "
            f"{config.market.steps} steps; "
            f"model vol={config.contract.model_volatility:g}</text>",
            "</g></svg>\n",
        ]
    )
    return "\n".join(svg)


def run_experiment(config: ExperimentConfig, output_dir: str | Path) -> Row:
    """Run all declared scenarios and write seven deterministic artifacts.

    A seed/path index is the pairing key across all policies, costs and generating
    volatilities. CSV path rows retain all final components; the selected trace
    contains every epoch, including entry and terminal liquidation. Re-running
    overwrites these seven names only. Exceptions are not converted into results.
    """
    config = parse_config(config.to_dict())
    provenance = _provenance(config)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    per_seed: list[Row] = []
    chunks: dict[tuple[float, str, str], dict[str, list[FloatArray]]] = {}
    ledger: list[Row] = []
    data_hashes: dict[str, str] = {}
    zero_spec = next(
        item for item in config.cost_schedules if item.id == config.zero_cost_reference
    )
    costs = sorted(config.cost_schedules, key=lambda item: item != zero_spec)
    trace = config.example
    with (directory / "path_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = ["seed", "path_volatility", "policy", "cost", "path"]
        writer = csv.DictWriter(
            stream, fieldnames=fields + list(_RESULT_FIELDS) + ["paired_cost_drag"]
        )
        writer.writeheader()
        for seed in config.market.seeds:
            for volatility in config.market.path_volatilities:
                spots = generate_gbm(
                    GBMConfig(
                        config.market.spot,
                        config.market.drift,
                        volatility,
                        config.contract.maturity,
                        config.market.steps,
                        config.market.paths,
                        seed,
                    )
                )
                data_hashes[f"seed={seed};vol={volatility!r}"] = hashlib.sha256(
                    spots.astype("<f8", copy=False).tobytes(order="C")
                ).hexdigest()
                market = prepare_market(config.contract, spots)
                for policy in config.policies:
                    zero: HedgeResult | None = None
                    for cost in costs:
                        selected = (
                            seed == trace.seed
                            and volatility == trace.path_volatility
                            and policy.id == trace.policy
                            and cost.id == trace.cost
                        )
                        result = simulate_hedge(
                            config.contract,
                            market,
                            policy.policy,
                            cost.costs,
                            trace_path=trace.path if selected else None,
                        )
                        if zero is None:
                            zero = result
                        arrays = _arrays(result, zero)
                        label = {
                            "seed": seed,
                            "path_volatility": volatility,
                            "policy": policy.id,
                            "cost": cost.id,
                        }
                        per_seed.append({**label, **_summarize(arrays)})
                        key = (volatility, policy.id, cost.id)
                        destination = chunks.setdefault(
                            key, {name: [] for name in arrays}
                        )
                        for name, array in arrays.items():
                            destination[name].append(array)
                        writer.writerows(
                            {
                                **label,
                                "path": index,
                                **{
                                    name: float(array[index])
                                    for name, array in arrays.items()
                                },
                            }
                            for index in range(config.market.paths)
                        )
                        if selected:
                            ledger = [
                                {**label, "path": trace.path, **asdict(row)}
                                for row in result.ledger
                            ]
    pooled = {
        key: {name: np.concatenate(arrays) for name, arrays in values.items()}
        for key, values in chunks.items()
    }
    summary = [
        {
            "path_volatility": volatility,
            "policy": policy,
            "cost": cost,
            "seeds": len(config.market.seeds),
            **_summarize(values),
        }
        for (volatility, policy, cost), values in pooled.items()
    ]
    paired = _comparisons(config, pooled)
    if not ledger:
        raise ValueError("selected example produced no ledger rows")
    provenance["path_data_sha256"] = data_hashes
    payload: Row = {
        "schema_version": SCHEMA_VERSION,
        "config": config.to_dict(),
        "provenance": provenance,
        "summary": summary,
        "per_seed_summary": per_seed,
        "paired_comparisons": paired,
        "example_ledger_columns": list(ledger[0]),
        "artifacts": list(_ARTIFACTS),
    }
    # Validate the entire JSON payload before writing summaries; no NaN/Inf fills.
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_csv(directory / "summary.csv", summary)
    _write_csv(directory / "per_seed_summary.csv", per_seed)
    _write_csv(directory / "paired_comparisons.csv", paired)
    _write_csv(directory / "example_ledger.csv", ledger)
    (directory / "hedging_tradeoff.svg").write_text(
        _chart(summary, config), encoding="utf-8"
    )
    (directory / "results.json").write_text(encoded, encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; invalid configs and numeric failures exit nonzero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="versioned experiment JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        run_experiment(config, args.output_dir)
    except (ValueError, OSError, ArithmeticError) as error:
        parser.error(str(error))
    print(f"Wrote {len(_ARTIFACTS)} artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
