# Quantitative Pricing Toolkit

A typed, vectorized, and numerically defensive implementation of the
Black–Scholes–Merton model for European options. The project emphasizes the work
that matters in a quantitative production library: explicit assumptions,
stable boundary behavior, analytic risk, robust inversion, invariant-based
testing, and measured performance.

## Implemented scope

- European call and put pricing with continuous dividends
- Analytic Delta, Gamma, Vega, calendar Theta, and Rho
- Safeguarded implied volatility with no-arbitrage validation and diagnostics
- NumPy broadcasting for heterogeneous batch pricing
- Explicit expiry and zero-volatility conventions
- Stable log-CDF tail evaluation for extreme moneyness
- Typed command-line and Dash interfaces that delegate to the pricing package
- Immutable market-data contracts and an offline, as-of bid/ask IV example
- Dated curves, Black-76 pricing and IV, and bid/ask parity diagnostics
- Optional partitioned Parquet ingestion and causal snapshot replay
- Liquidity-weighted SVI calibration, bid/mid/ask IV, and surface diagnostics
- Portfolio Greeks, scenario shocks, P&L attribution, and risk limits
- Independent binomial reference for European and American exercise with
  discrete cash dividends
- Regression, finite-difference, parity, monotonicity, randomized invariant,
  edge-case, and integration tests
- Ruff formatting/linting, strict mypy checks, locked dependencies, and CI

See [ROADMAP.md](ROADMAP.md) for remaining work and
[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for design decisions and
reproducible workflows.

## Model and conventions

For spot \(S\), strike \(K\), time to expiry \(T\), continuously compounded
risk-free rate \(r\), continuous dividend yield \(q\), and annualized volatility
\(\sigma\):

\[
d_1 =
\frac{\ln(S/K) + (r-q+\tfrac{1}{2}\sigma^2)T}
     {\sigma\sqrt{T}},
\qquad
d_2 = d_1-\sigma\sqrt{T}.
\]

\[
C = Se^{-qT}N(d_1)-Ke^{-rT}N(d_2),
\qquad
P = Ke^{-rT}N(-d_2)-Se^{-qT}N(-d_1).
\]

The implementation assumes European exercise, frictionless markets, continuous
trading, constant rates and volatility, lognormal diffusion, and no discrete
cash dividends. Rates, yields, and volatility are decimal annualized inputs.

Greek conventions are explicit:

- Vega and Rho are derivatives for a unit change in volatility and rate. Divide
  by 100 for a one-percentage-point sensitivity.
- Theta is calendar theta per year, so it is the negative derivative with
  respect to remaining maturity.
- At expiry, price is exact intrinsic value. Delta uses 0.5 at the payoff kink;
  Gamma, Vega, Theta, and Rho are reported as zero because their distributional
  limits are not finite point sensitivities.
- At zero volatility, price is the discounted deterministic payoff. First-order
  sensitivities use one-sided limits and midpoint values at the deterministic
  exercise boundary.

## Numerical design

Pricing validates finite, broadcast-compatible inputs and rejects non-positive
spot/strike, negative maturity, negative volatility, and discount factors that
overflow IEEE-754 double precision. OTM tail terms are evaluated in log space
with `log_ndtr` and `expm1`, reducing cancellation between nearly equal terms.

Implied volatility is solved only after checking the European no-arbitrage
bounds:

\[
\max(Se^{-qT}-Ke^{-rT},0) \le C < Se^{-qT},
\]

with the analogous put bounds. The solver expands a volatility bracket, accepts
Newton updates only inside a safeguarded interior, and otherwise bisects.
Success returns volatility, iteration count, residual, normalized residual,
termination reason, final bracket, and the repriced value. Bound violations,
infinite-volatility limits, bracket failures, and iteration failures are
distinct exceptions.

Price convergence combines an absolute tolerance with a relative tolerance
times a no-arbitrage-bound-derived monetary scale. The default absolute
tolerance is zero, making identical economics produce the same implied
volatility whether prices are quoted in dollars, cents, or normalized notional
units. A narrow volatility bracket is a separate, explicitly reported stopping
condition. This distinction matters for low-Vega options: a tiny normalized
price residual does not imply many meaningful decimal places of volatility.

## Installation

The supported Python versions are 3.10 and newer. `uv.lock` records the resolved
environment used by CI and the reference benchmark.

```bash
git clone <repository-url>
cd "Black Scholes Pricing Model"
uv sync --extra dashboard --extra replay --group dev --group benchmark
```

For library-only use with `pip`:

```bash
python -m pip install .
```

Install the optional dashboard dependencies with:

```bash
python -m pip install ".[dashboard]"
```

Install `".[replay]"` for partitioned Parquet ingestion and replay. The pricing
engine does not require PyArrow.

## Python API

Scalar pricing and risk:

```python
from quant_pricing import value_and_greeks

result = value_and_greeks(
    "call",
    spot=100.0,
    strike=100.0,
    time_to_expiry=1.0,
    rate=0.05,
    volatility=0.20,
    dividend_yield=0.0,
)

print(result.price)         # 10.450583572...
print(result.greeks.delta)  # 0.636830651...
print(result.greeks.vega)   # 37.524034692 per unit volatility
```

Broadcast batch pricing:

```python
import numpy as np
from quant_pricing import price

spots = np.array([[80.0], [100.0], [120.0]])
strikes = np.array([90.0, 100.0, 110.0])
values = price("put", spots, strikes, 0.5, 0.03, 0.25)
assert values.shape == (3, 3)
```

Implied volatility with diagnostics:

```python
from quant_pricing import implied_volatility

solution = implied_volatility(
    "call",
    target_price=10.450583572,
    spot=100.0,
    strike=100.0,
    time_to_expiry=1.0,
    rate=0.05,
)

print(solution.volatility)  # approximately 0.20
print(solution.normalized_residual, solution.termination_reason)
print(solution.iterations, solution.bracket)
```

## Applications

Run a reproducible CLI valuation:

```bash
uv run black-scholes \
  --spot 100 --strike 100 --time 1 \
  --rate 0.05 --volatility 0.20 --type call
```

Run the local dashboard:

```bash
uv run black-scholes-dashboard
```

Choose **Price from volatility** to see the theoretical price, Greeks, and
spot and volatility sensitivity charts. Choose **IV from option premium** to
solve for annualized implied volatility from a supplied premium per unit (not
the total contract cost). The result includes the model premium, signed
repricing residual, iteration count, and stopping reason.

The app opens with a default valuation. Click **Calculate** to apply changes
to the mode or inputs. The persistent light/dark setting restyles the last
calculated result without applying pending edits.

The dashboard does not fetch market data or estimate volatility from stock
returns. Supply your own inputs; timestamped quote acquisition belongs in a
separate data pipeline.

The original `bs_call`, `bs_put`, `d1`, and `d2` names remain available through
`BlackScholesPricingModel.py` for existing notebooks.

### Offline option chain

Run the [frozen synthetic chain](examples/synthetic_option_chain.json) from
the repository root:

```bash
uv run python -m quant_pricing.option_chain \
  examples/synthetic_option_chain.json --as-of 2025-01-01T12:00:00Z
```

- **12:00:00 UTC:** The clean, synthetic chain recovers the known 19% bid and
  21% ask IV at each strike.
- **12:00:30 UTC:** `C100` is crossed; the report shows the raw quote but no IV.
  `P100` still uses sequence 1 because its revision, stamped 12:00:20, does
  not arrive until 12:00:45.
- **12:01:00 UTC:** `P100` uses sequence 2, the now-available revision.

The instrument universe is fixed in the fixture. The JSON report retains
quote and availability times, source sequence, raw prices and sizes, premium
basis, and quality flags. Both clocks must be at or before the cutoff.
Selection is deterministic within the declared source: latest quote time,
then sequence, then availability time. Freshness is measured from quote time.
Missing quotes are not filled; bad or stale quotes are not replaced by older
ones.

Use `quant_pricing.market.SnapshotProvider` at the application boundary;
`FrozenQuoteProvider` implements it over an immutable quote tape. A snapshot's
rows distinguish a missing observation from an observed but unusable quote.

The workflow uses the eligible spot midpoint, fixed decimal rate and dividend
yield, and elapsed UTC seconds on an ACT/365F basis. It solves each side
separately for single-currency European cash-settled options; forward quotes
have a domain type but are not used here. Per-contract premiums are divided
by the multiplier without changing the denomination. Option quote sizes and
positions are in contracts; underlying and forward sizes are in underlying
units. Coin premiums can be represented and normalized in the domain model,
but the IV workflow does not convert currencies or model inverse payoffs.

For persistent replay, ingest into a fresh writable archive and rerun the same
as-of query without `--ingest`:

```bash
uv run --extra replay python -m quant_pricing.option_chain \
  examples/synthetic_option_chain.json --as-of 2025-01-01T12:00:30Z \
  --archive /tmp/quant-chain-example --ingest
uv run --extra replay python -m quant_pricing.option_chain \
  examples/synthetic_option_chain.json --as-of 2025-01-01T12:00:30Z \
  --archive /tmp/quant-chain-example
```

The report includes bid/ask-implied forward intervals and carries a separate
spot-derived forward benchmark. It withholds IV or parity calculations that
depend on an unavailable, invalid, stale, crossed, or sequence-gapped quote.

### Synthetic surface, portfolio, and exercise workflow

```bash
uv run python examples/p3_research_workflow.py
```

Fit a clean and deliberately damaged synthetic chain, reprice a portfolio
against the fitted surface, and compare European and American exercise with
a discrete cash dividend. This is a verification workflow, not a backtest.

## Validation and quality checks

```bash
uv run pytest --cov --cov-report=term-missing
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv build
```

The test suite validates:

- canonical published prices and Greeks for the \(S=K=100\) one-year case;
- analytic Greeks against central finite differences;
- dividend-adjusted put–call parity and call/put Greek identities;
- no-arbitrage bounds and expected monotonicity;
- deterministic randomized parity across broad parameter ranges;
- extreme moneyness, near expiry, zero volatility, negative rates, malformed
  inputs, and non-broadcastable arrays;
- implied-volatility recovery and every documented failure regime;
- monetary-scale invariance and final-iteration convergence in the solver; and
- CLI, compatibility wrapper, dashboard calculation, layout, and callback
  initialization.

## Performance

The reference microbenchmark prices 10,000 heterogeneous calls. On the
development machine, the scalar loop averaged 2.072 seconds and the vectorized
path averaged 1.644 milliseconds—about a 1,261× speedup. This measures the
intended amortization of Python conversion, validation, and call overhead; it is
not treated as a cross-machine performance guarantee.

See [BENCHMARKS.md](BENCHMARKS.md) for methodology, caveats, and reproduction
commands.

## Architecture

```text
src/quant_pricing/
├── black_scholes.py         # deterministic pricing, bounds, and Greeks
├── black76.py               # forward pricing, bounds, and IV
├── conventions.py           # day-count and exchange calendar rules
├── curves.py                # discount, forward, and cash-dividend curves
├── implied_volatility.py    # safeguarded scalar root solve
├── market.py               # instruments, quotes, and as-of snapshot contract
├── replay.py                # synthetic provider and optional Parquet archive
├── parity.py                # parity and conditional carry diagnostics
├── treasury.py              # Federal Reserve GSW zero-yield adapter
├── synthetic_chain.py       # deterministic SSVI chains and injected quote faults
├── surface.py               # IV observations, SVI fits, and arbitrage checks
├── risk.py                  # portfolio Greeks, scenarios, attribution, and limits
├── binomial.py              # independent listed-option exercise reference
├── option_chain.py          # frozen-chain CLI and per-side IV report
├── cli.py                   # command-line adapter
├── dashboard.py             # Dash adapter and vectorized sensitivity views
├── exceptions.py            # domain-specific failures
└── py.typed                 # PEP 561 typing marker

tests/                       # numerical, invariant, and integration tests
benchmarks/                  # correctness-checked performance experiments
.github/workflows/ci.yml     # multi-version quality gate
```

Pricing and numerical methods do not depend on Dash or external market data.
The UI layer is optional and one-way: applications depend on the library, never
the reverse.

## Limitations

Black–Scholes–Merton is not an empirical claim that volatility is constant or
returns are exactly lognormal. It is a transparent baseline and quoting
framework. Fitted surfaces are numerically checked on a declared domain;
the checks are not a global no-arbitrage guarantee. The binomial reference
uses escrowed, grid-aligned cash dividends. The toolkit does not model inverse
settlement, stochastic volatility, transaction costs, or execution.

Use of this software is educational; it is not investment advice or a
production trading system.

## License

MIT. See [LICENSE](LICENSE).
