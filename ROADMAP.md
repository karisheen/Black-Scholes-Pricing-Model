# Quantitative Pricing Toolkit Roadmap

Scores run from 1 (low) to 5 (high). Effort measures relative implementation cost;
the other scores measure expected value. Work is ordered by correctness and depth,
not by feature count.

## P0 — Core quantitative rigor

1. **Package architecture and reproducible tooling** — Portfolio 5, Quant 3,
   Engineering 5, Effort 3.
   - [x] Create a typed `src/` package with explicit public interfaces.
   - [x] Declare runtime and development dependencies in `pyproject.toml`.
   - [ ] Add formatting, linting, static typing, tests, and CI.
   - [x] Stop tracking generated and machine-local artifacts.

2. **Numerically robust Black–Scholes–Merton pricing** — Portfolio 5, Quant 5,
   Engineering 5, Effort 4.
   - [x] Support calls and puts, continuous dividend yield, and scalar inputs.
   - [x] Define and test behavior at expiry and at zero volatility.
   - [x] Enforce finite, economically valid inputs.
   - [x] Use stable tail calculations and no-arbitrage bounds.

3. **Analytic first-order and second-order risk** — Portfolio 5, Quant 5,
   Engineering 4, Effort 3.
   - [x] Implement Delta, Gamma, Vega, calendar Theta, and Rho.
   - [x] Validate Greeks against central finite differences.
   - [x] Document units, sign conventions, and singular boundary conventions.

4. **Safeguarded implied volatility** — Portfolio 5, Quant 5, Engineering 5,
   Effort 4.
   - [x] Validate model-independent price bounds before iteration.
   - [x] Combine Newton steps with a guaranteed bisection bracket.
   - [x] Report convergence, iterations, residual, and failure causes.
   - [x] Test volatility recovery across moneyness and maturity regimes.

5. **Numerical validation suite** — Portfolio 5, Quant 5, Engineering 5,
   Effort 4.
   - [x] Add published reference-value regression tests.
   - [x] Test put–call parity, bounds, monotonicity, and Greek identities.
   - [x] Cover invalid, extreme, near-expiry, and degenerate inputs.
   - [x] Add application-level smoke and compatibility tests.

## P1 — Performance and product quality

6. **Broadcasting batch API** — Portfolio 4, Quant 4, Engineering 5, Effort 3.
   - [x] Broadcast heterogeneous NumPy inputs without Python loops.
   - [x] Preserve a convenient scalar return type.
   - [x] Verify batch results against scalar calculations.

7. **Performance evidence** — Portfolio 4, Quant 3, Engineering 4, Effort 2.
   - [ ] Add reproducible scalar and vectorized benchmarks.
   - [ ] Quantify vectorization speedup without enforcing fragile timing gates.

8. **Thin, deterministic user interfaces** — Portfolio 4, Quant 3,
   Engineering 4, Effort 3.
   - [x] Remove duplicated pricing formulas from the dashboard.
   - [x] Separate market-data estimation from deterministic pricing.
   - [x] Present prices, Greeks, assumptions, and sensitivity charts.

9. **Portfolio-quality documentation** — Portfolio 5, Quant 4, Engineering 4,
   Effort 3.
   - [ ] Explain formulas, assumptions, architecture, conventions, and limits.
   - [ ] Document installation, API, dashboard, validation, and benchmarks.
   - [ ] Distinguish implemented functionality from research extensions.

## P2 — Deferred research extensions

These are intentionally deferred until every P0 and P1 acceptance criterion is
met. Adding models before validating the core would reduce project credibility.

- [ ] Discrete dividends and term structures — Portfolio 4, Quant 5,
  Engineering 3, Effort 4.
- [ ] Binomial validation model and American exercise — Portfolio 4, Quant 5,
  Engineering 4, Effort 5.
- [ ] Monte Carlo estimator with confidence intervals and variance reduction —
  Portfolio 4, Quant 5, Engineering 4, Effort 5.
- [ ] Volatility-smile and surface analysis with arbitrage diagnostics —
  Portfolio 5, Quant 5, Engineering 4, Effort 5.
