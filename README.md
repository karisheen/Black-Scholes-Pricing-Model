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
- Regression, finite-difference, parity, monotonicity, randomized invariant,
  edge-case, and integration tests
- Ruff formatting/linting, strict mypy checks, locked dependencies, and CI

The deliberate focus is depth, not model count. American exercise, stochastic
volatility, Monte Carlo, trees, and volatility surfaces are not presented as
implemented features. See [ROADMAP.md](ROADMAP.md) for the ordered research
backlog.

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
uv sync --extra dashboard --group dev --group benchmark
```

For library-only use with `pip`:

```bash
python -m pip install .
```

Install the optional dashboard dependencies with:

```bash
python -m pip install ".[dashboard]"
```

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

The dashboard reports price and Greeks, plots vectorized spot and volatility
sensitivities, and provides persistent light and dark appearance modes. It
intentionally does not fetch equity prices or estimate volatility. The previous
market-data path combined historical volatility with a hard-coded rate and
labeled the output as real-time pricing; removing that non-reproducible workflow
makes the model assumptions auditable. Market-data estimation belongs in a
separate, timestamped data pipeline.

The original `bs_call`, `bs_put`, `d1`, and `d2` names remain available through
`BlackScholesPricingModel.py` for existing notebooks.

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
├── implied_volatility.py    # safeguarded scalar root solve
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
framework. The implementation does not currently support American exercise,
discrete dividends, term structures, early-exercise boundaries, transaction
costs, jumps, stochastic volatility, or arbitrage-checked volatility surfaces.

Use of this software is educational; it is not investment advice or a
production trading system.

## License

MIT. See [LICENSE](LICENSE).
