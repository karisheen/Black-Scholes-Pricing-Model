# Quantitative Pricing Toolkit Roadmap

Scores run from 1 (low) to 5 (high). Effort measures relative implementation cost;
the other scores measure expected value. Work is ordered by correctness and depth,
not by feature count.

The target is a provider-neutral options research and risk platform for
systematic trading and market-making research. The pricing library remains the
source of truth; dashboards, data feeds, storage, and strategy simulation are
adapters around it.

Delivery principles:

- Use deterministic fixtures and synthetic markets before depending on an
  external feed.
- Preserve raw bid/ask observations and provenance; exchange marks, implied
  volatilities, and Greeks are comparison data rather than ground truth.
- Keep market-data providers replaceable so free crypto feeds and future
  licensed U.S. options feeds share one normalized model.
- Require as-of timestamps, gap detection, and reproducible replay before
  making strategy-performance claims.
- Keep live order entry out of scope until replay, accounting, risk limits, and
  failure controls are validated.

## P0 — Core quantitative rigor

1. **Package architecture and reproducible tooling** — Portfolio 5, Quant 3,
   Engineering 5, Effort 3.
   - [x] Create a typed `src/` package with explicit public interfaces.
   - [x] Declare runtime and development dependencies in `pyproject.toml`.
   - [x] Add formatting, linting, static typing, tests, and CI.
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
   - [x] Add reproducible scalar and vectorized benchmarks.
   - [x] Quantify vectorization speedup without enforcing fragile timing gates.

8. **Thin, deterministic user interfaces** — Portfolio 4, Quant 3,
   Engineering 4, Effort 3.
   - [x] Remove duplicated pricing formulas from the dashboard.
   - [x] Separate market-data estimation from deterministic pricing.
   - [x] Present prices, Greeks, assumptions, and sensitivity charts.

9. **Portfolio-quality documentation** — Portfolio 5, Quant 4, Engineering 4,
   Effort 3.
   - [x] Explain formulas, assumptions, architecture, conventions, and limits.
   - [x] Document installation, API, dashboard, validation, and benchmarks.
   - [x] Distinguish implemented functionality from research extensions.

## P2 — Market and data foundations

P2 has no paid-data dependency. It establishes the contracts that every
synthetic, free, or licensed data source must satisfy.

10. **Instrument, quote, and snapshot domain model** — Portfolio 5, Quant 4,
    Engineering 5, Effort 3.
    - [x] Model option contracts with underlying, strike, exact expiry,
      exercise style, settlement style, multiplier, and quote/settlement
      currencies.
    - [x] Model timestamped underlying, option, and forward quotes with bid/ask
      prices, sizes, source, and source sequence.
    - [x] Define immutable `MarketSnapshot`, `Position`, `Portfolio`, and
      provider interfaces without vendor-specific fields.
    - [x] Normalize premium basis and signed contract quantities explicitly.
      Coin-denominated premiums retain their currency; no implicit FX.

    Offline acceptance: the frozen chain recovers bid and ask implied
    volatility at an explicit as-of time, reports crossed quotes without
    fallback, and only admits a late revision after it becomes available.

11. **Curves, forwards, and market conventions** — Portfolio 5, Quant 5,
    Engineering 4, Effort 4.
    - [ ] Add zero-rate, discount-factor, forward, and discrete-dividend curves.
    - [ ] Add trading calendars, day-count rules, valuation timestamps, and
      exact expiration times.
    - [ ] Implement Black-76/forward-based pricing for futures and crypto
      options while retaining spot Black–Scholes–Merton pricing.
    - [ ] Report put–call parity, implied-forward, implied-dividend, and
      implied-borrow diagnostics.
    - [ ] Add a U.S. Treasury curve adapter while keeping fixed fixtures for
      deterministic tests.

12. **Normalized storage, quality controls, and replay boundary** — Portfolio
    5, Quant 4, Engineering 5, Effort 4.
    - [ ] Store versioned raw and normalized events in partitioned Parquet with
      UTC timestamps and source metadata.
    - [ ] Make ingestion idempotent and detect duplicates, crossed markets,
      stale quotes, sequence gaps, and invalid prices or sizes.
    - [ ] Implement `SyntheticProvider` and `ParquetReplayProvider` behind the
      same interface used by live feeds.
    - [ ] Reconstruct snapshots deterministically and report incomplete data
      rather than silently filling gaps.

## P3 — Volatility and portfolio analytics

P3 turns the single-contract model into a useful options research and inventory
risk engine. All acceptance tests can run on generated chains.

13. **Implied-volatility chain and surface engine** — Portfolio 5, Quant 5,
    Engineering 5, Effort 5.
    - [ ] Generate arbitrage-free synthetic chains from known smiles and inject
      controlled noise, missing strikes, stale quotes, and crossed markets.
    - [ ] Solve bid, midpoint, and ask implied volatility across complete
      chains with per-contract diagnostics.
    - [ ] Fit liquidity-weighted SVI smiles by expiry and interpolate total
      variance across maturities.
    - [ ] Detect strike, butterfly, and calendar arbitrage and expose rejected
      quotes and fit residuals.
    - [ ] Support forward moneyness, delta coordinates, and explicit
      interpolation and extrapolation policies.
    - [ ] Test parameter recovery and no-arbitrage invariants against generated
      surfaces.

14. **Portfolio, inventory, and scenario risk** — Portfolio 5, Quant 5,
    Engineering 4, Effort 4.
    - [ ] Aggregate price and Greeks by portfolio, underlying, expiry, strike,
      and volatility bucket.
    - [ ] Add Vanna, Volga/Vomma, Charm, Speed, and Color with documented units
      and finite-difference validation.
    - [ ] Run spot, volatility, time, rate, skew, and curvature shocks against
      positions and surfaces.
    - [ ] Explain P&L through Delta, Gamma, Vega, Theta, carry, trades, and an
      explicit residual.
    - [ ] Enforce configurable position, notional, Delta, Gamma, and Vega limits.

15. **Listed-option exercise and dividend models** — Portfolio 4, Quant 5,
    Engineering 4, Effort 5.
    - [ ] Implement a binomial reference model for European and American
      exercise with discrete dividends.
    - [ ] Report early-exercise boundaries and exercise/assignment exposure.
    - [ ] Verify European tree convergence to analytic prices and test American
      no-arbitrage inequalities.
    - [ ] Keep the tree as an independent validation model rather than mixing
      it into the analytic Black–Scholes implementation.

## P4 — Free live crypto-options data

P4 uses public exchange data without making the analytics depend on an exchange
schema. Historical order books are collected prospectively because a complete
free archive is not assumed.

16. **Deribit market-data adapter and recorder** — Portfolio 5, Quant 5,
    Engineering 5, Effort 4.
    - [ ] Discover BTC option instruments and track listing, state, and expiry
      changes without repeatedly polling the full catalog.
    - [ ] Ingest index and forward prices, chain summaries, bid/ask quotes,
      grouped order books, trades, open interest, and settlement prices.
    - [ ] Maintain books from snapshots and incremental updates with sequence
      and reconnect-gap validation.
    - [ ] Persist raw and normalized events continuously, then add ETH after the
      BTC pipeline passes replay and recovery checks.
    - [ ] Compare independent prices, implied volatilities, and Greeks with
      Deribit marks without using exchange analytics as model inputs.
    - [ ] Document rate limits, authentication boundaries, units, retention,
      and data-use terms.

17. **Bybit cross-venue adapter** — Portfolio 4, Quant 4, Engineering 4,
    Effort 3.
    - [ ] Normalize Bybit instruments, ticker IV/Greeks, trades, and option
      order books into the same domain model.
    - [ ] Compare contemporaneous BTC and ETH forwards, smiles, spreads, and
      liquidity while preserving venue provenance.
    - [ ] Use the second venue to test adapter isolation and data-quality
      diagnostics, not to merge incompatible marks into one synthetic quote.

18. **Research and risk dashboard** — Portfolio 5, Quant 4, Engineering 4,
    Effort 4.
    - [ ] Add option-chain tables with bid/ask IV, liquidity, quality flags, and
      calculation diagnostics.
    - [ ] Add smile, term-structure, surface, fit-residual, and arbitrage views.
    - [ ] Add portfolio positions, Greek buckets, limits, scenario heatmaps,
      and P&L attribution.
    - [ ] Add collector health, freshness, sequence-gap, and replay status.
    - [ ] Keep calculations in the package and leave the dashboard as a thin
      presentation adapter.

## P5 — Systematic trading and market-making research

P5 is simulation and replay only. Synthetic results establish correctness and
stress behavior; they do not establish real-world strategy profitability.

19. **Synthetic market and hedging simulator** — Portfolio 5, Quant 5,
    Engineering 4, Effort 4.
    - [ ] Generate deterministic spot, forward, volatility, spread, jump, and
      stale-quote scenarios.
    - [ ] Simulate periodic and threshold-based Delta hedging with fees,
      slippage, financing, and explicit cash accounting.
    - [ ] Attribute option, hedge, carry, transaction-cost, and residual P&L.
    - [ ] Test expiration, settlement, exercise, missing-data, and volatility
      shock behavior.

20. **As-of historical replay and strategy framework** — Portfolio 5, Quant 5,
    Engineering 5, Effort 5.
    - [ ] Drive strategies from a deterministic event clock with no access to
      future snapshots or revised data.
    - [ ] Model bid/ask execution, configurable latency, partial fills,
      transaction fees, and rejected orders.
    - [ ] Record an immutable trade ledger, positions, cash, marks, and risk for
      every run.
    - [ ] Version data, code, parameters, and random seeds so results are
      reproducible.
    - [ ] Report turnover, drawdown, risk-adjusted returns, hedge error,
      capacity assumptions, and transaction-cost sensitivity.

21. **Market-making simulator** — Portfolio 5, Quant 5, Engineering 5,
    Effort 5.
    - [ ] Generate theoretical bid/ask quotes from the independent surface,
      model uncertainty, fees, and minimum required edge.
    - [ ] Skew quotes using Delta, Gamma, Vega, expiry, and concentration
      inventory.
    - [ ] Simulate quote age, cancel/replace latency, queue assumptions, partial
      fills, and adverse selection.
    - [ ] Measure fill probability, spread capture, inventory P&L, hedge cost,
      and post-fill markouts.
    - [ ] Add stale-market protection, per-contract and portfolio limits, and
      deterministic kill-switch scenarios.
    - [ ] Do not add live order submission in this phase.

## P6 — Hardening, licensed U.S. data, and advanced research

22. **Operational and model-risk hardening** — Portfolio 5, Quant 5,
    Engineering 5, Effort 4.
    - [ ] Add golden market snapshots, property tests, and differential tests
      against independent implementations.
    - [ ] Benchmark full-chain ingestion, surface fitting, portfolio scenarios,
      storage, and replay without fragile timing assertions.
    - [ ] Add structured calculation provenance, metrics, logs, health checks,
      restart recovery, and data-retention controls.
    - [ ] Separate research, dashboard, collection, and optional service
      processes so failure in one cannot corrupt stored market data.

23. **Licensed U.S. options provider adapter** — Portfolio 5, Quant 5,
    Engineering 4, Effort 4.
    - [ ] Validate parsers against legally distributable samples before
      selecting a paid provider.
    - [ ] Add OPRA contract symbology, multipliers, American exercise,
      corporate actions, and exchange-calendar conventions.
    - [ ] Keep entitlements, retention, non-display use, and redistribution
      restrictions explicit in configuration and documentation.
    - [ ] Require paid historical bid/ask data before presenting U.S.
      systematic backtests as realistic.
    - [ ] Reuse the normalized provider interface; do not fork pricing,
      surfaces, portfolio risk, or replay for U.S. data.

24. **Advanced estimators and calibrated models** — Portfolio 4, Quant 5,
    Engineering 4, Effort 5.
    - [ ] Add Monte Carlo estimators with confidence intervals, antithetic
      sampling, control variates, and reproducible random streams.
    - [ ] Add local-volatility or stochastic-volatility calibration only after
      the vanilla surface and calibration diagnostics are reliable.
    - [ ] Compare model prices, hedges, parameter stability, and out-of-sample
      errors rather than ranking models by in-sample fit alone.

## Delivery order and release gates

1. **v0.2 — Market foundations:** items 10–12 complete with deterministic
   storage and replay.
2. **v0.3 — Surface and portfolio risk:** items 13–15 complete using generated
   chains and independent validation.
3. **v0.4 — Live crypto research:** items 16–18 complete with recoverable
   Deribit collection, then Bybit comparison.
4. **v0.5 — Strategy laboratory:** items 19–21 complete with immutable
   accounting and no-lookahead replay.
5. **v1.0 — Defensible platform:** item 22 complete; item 23 is required only
   before claiming realistic U.S. options research. Item 24 remains optional
   model research.
