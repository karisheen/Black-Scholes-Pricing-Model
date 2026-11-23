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
    - [x] Add zero-rate, discount-factor, forward, and discrete-dividend curves.
    - [x] Add trading calendars, day-count rules, valuation timestamps, and
      exact expiration times.
    - [x] Implement Black-76/forward-based pricing for linear futures and
      crypto options while retaining spot Black–Scholes–Merton pricing.
    - [x] Report put–call parity, implied-forward, implied-dividend, and
      implied-borrow diagnostics.
    - [x] Add a U.S. Treasury zero-yield adapter with frozen fixtures for
      deterministic tests.

12. **Normalized storage, quality controls, and replay boundary** — Portfolio
    5, Quant 4, Engineering 5, Effort 4.
    - [x] Store versioned raw and normalized events in partitioned Parquet with
      UTC timestamps and source metadata.
    - [x] Make ingestion idempotent and detect duplicates, crossed markets,
      stale quotes, sequence gaps, and invalid prices or sizes.
    - [x] Implement `SyntheticProvider` and `ParquetReplayProvider` behind the
      same interface used by live feeds.
    - [x] Reconstruct snapshots deterministically and report incomplete data
      rather than silently filling gaps.

    See [implementation decisions and reproducible workflow](IMPLEMENTATION_SUMMARY.md#p2--market-foundations).

## P3 — Volatility and portfolio analytics

P3 turns the single-contract model into a useful options research and inventory
risk engine. All acceptance tests can run on generated chains.

13. **Implied-volatility chain and surface engine** — Portfolio 5, Quant 5,
    Engineering 5, Effort 5.
    - [x] Generate arbitrage-free synthetic chains from known smiles and inject
      controlled noise, missing strikes, stale quotes, and crossed markets.
    - [x] Solve bid, midpoint, and ask implied volatility across complete
      chains with per-contract diagnostics.
    - [x] Fit liquidity-weighted SVI smiles by expiry and interpolate total
      variance across maturities.
    - [x] Check strike, butterfly, and calendar arbitrage on declared grids
      and expose rejected quotes and fit residuals.
    - [x] Support forward moneyness, delta coordinates, and explicit
      interpolation and extrapolation policies.
    - [x] Test parameter recovery and no-arbitrage invariants against generated
      surfaces.

14. **Portfolio, inventory, and scenario risk** — Portfolio 5, Quant 5,
    Engineering 4, Effort 4.
    - [x] Aggregate price and Greeks by portfolio, underlying, expiry, strike,
      and volatility bucket.
    - [x] Add Vanna, Volga/Vomma, Charm, Speed, and Color with documented units
      and finite-difference validation.
    - [x] Run spot, volatility, time, rate, skew, and curvature shocks against
      positions and surfaces.
    - [x] Explain P&L through Delta, Gamma, Vega, Theta, carry, trades, and an
      explicit residual.
    - [x] Enforce configurable position, notional, Delta, Gamma, and Vega limits.

15. **Listed-option exercise and dividend models** — Portfolio 4, Quant 5,
    Engineering 4, Effort 5.
    - [x] Implement a binomial reference model for European and American
      exercise with discrete dividends.
    - [x] Report early-exercise boundaries and modeled exercise/assignment
      exposure.
    - [x] Verify European tree convergence to analytic prices and test American
      no-arbitrage inequalities.
    - [x] Keep the tree as an independent validation model rather than mixing
      it into the analytic Black–Scholes implementation.

    See [implementation decisions and reproducible workflow](IMPLEMENTATION_SUMMARY.md#p3--volatility-and-portfolio-analytics).

## P4 — Free live crypto-options data

P4 uses public exchange data without making the analytics depend on an exchange
schema. Historical order books are collected prospectively because a complete
free archive is not assumed.

16. **Deribit market-data adapter and recorder** — Portfolio 5, Quant 5,
    Engineering 5, Effort 4.
    - [ ] Discover BTC option instruments and track listing, state, and expiry
      changes without repeatedly polling the full catalog.
      Prior-day Tardis catalog selection freezes the historical option strip
      before observing quotes; live listing lifecycle tracking remains open.
    - [ ] Ingest index and forward prices, chain summaries, bid/ask quotes,
      grouped order books, trades, open interest, and settlement prices.
    - [ ] Maintain books from snapshots and incremental updates with sequence
      and reconnect-gap validation.
      The [historical Tardis replay workflow](README.md#historical-deribit-order-book-replay)
      verifies book reconstruction and recovery offline. Live collection,
      normalized archival, and venue-specific data rights are still open.
    - [ ] Persist raw and normalized events continuously, then add ETH after the
      BTC pipeline passes replay and recovery checks.
    - [ ] Compare independent prices, implied volatilities, and Greeks with
      Deribit marks without using exchange analytics as model inputs.
      The [linear sample report](README.md#btcusdc-chain-and-iv-quality-report)
      now verifies explicit BTC/USDC units and compares bid/ask IV and parity
      against same-expiry futures. Venue mark and Greek comparisons remain open.
      A separate quoted butterfly scan tests selected triplets against
      displayed option and futures spreads, without inferring fills.
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
    - [x] Add option-chain tables with bid/ask IV, liquidity, quality flags, and
      calculation diagnostics.
    - [ ] Add smile, term-structure, surface, fit-residual, and arbitrage views.
    - [ ] Add portfolio positions, Greek buckets, limits, scenario heatmaps,
      and P&L attribution.
    - [ ] Add collector health, freshness, sequence-gap, and replay status.
    - [x] Keep calculations in the package and leave the dashboard as a thin
      presentation adapter.

    Partial evidence: Options Lab already presents historical checkpoint chains,
    bid/ask IV, smiles, fit/quality diagnostics, conditional Greeks, and local
    exports; see [the workspace](README.md#options-lab-dashboard) and
    `tests/test_lab_{data,chain,diagnostics,workspace,export}.py`. This is neither
    live collection nor the complete portfolio/limits/scenario/P&L risk UI.
    Item 16's Tardis reconstruction and item 18's read-only views are reusable
    pieces, not completion of P4; Bybit and collector operations remain open.

## P5 — Systematic trading and market-making research

P5 is simulation and replay only. Synthetic results establish correctness and
stress behavior; they do not establish real-world strategy profitability.

Evidence boundary for the milestones below: [P2/P3 decisions and recorded
checks](IMPLEMENTATION_SUMMARY.md), the [public interfaces](README.md), and
the named source/tests establish reusable components, not strategy validation.
[Pricing benchmarks](BENCHMARKS.md) measure scalar versus batch valuation only;
they do not measure event simulation, execution realism, or full-chain risk.
Observed P5.19a results and their limitations belong in
[the hedging research report](docs/hedging-research.md), using the frozen
[experiment specification](examples/hedging_experiment.json). Do not infer
unreported results or mark all of item 19 complete from that bounded experiment.

**Criterion-by-criterion reuse/gaps** (numbers refer to the original acceptance
bullets within each item; milestones below preserve their scope):

| Criterion | Foundation before 19a | Acceptance milestone (status below) |
| --- | --- | --- |
| 19.1 deterministic markets | `SyntheticProvider` schedules quotes; `synthetic_chain.py` supplies known smiles; curves supply forwards | Stochastic paths 19a; joint volatility/spread/jump/stale regimes 19b |
| 19.2 periodic/threshold hedging and cash | Batch BSM prices/Greeks, signed contracts/multipliers | A causal trading policy and self-financing ledger, including cash interest and costs: 19a |
| 19.3 P&L attribution | `risk.attribute_pnl` and `tests/test_risk.py` reconcile endpoint marks, supplied trades/carry and Taylor residual | Endogenous hedge trades, stock basis, running cash and terminal reconciliation: 19a; stress residuals 19b |
| 19.4 expiry/exercise/gaps/vol shocks | BSM expiry conventions, independent CRR exercise/dividend model, replay quality flags | Cash expiry 19a; gap/shock behavior 19b; exercise-to-ledger lifecycle 19c |
| 20.1 causal event clock | `market.py`/`replay.py` filter event and availability clocks; `tests/test_replay.py` covers late revisions/gaps | Strategy scheduler and stable ordering of decisions, arrivals and fills: 20a–b |
| 20.2 executable fills | Quotes retain sides/sizes; Tardis tests cover linked updates, disconnects and recovery | Latency, order state, depth/size consumption, partial/rejected fills and fees: 20b |
| 20.3 immutable run ledger | Immutable market/position objects and raw/normalized archive | Append-only execution/position/cash/risk events and restart reconciliation: 20c |
| 20.4 reproducible runs | Archive schema/identity, source provenance; 19a frozen seeds/config | Version strategy, execution model, data/universe, code, clocks and checkpoint schema together: 20c |
| 20.5 performance and capacity | Scenario risk/limits and 19a hedge-error/cost summaries | Historical paired cost/latency/size sensitivity and justified return denominator: 20c |
| 21.1 independent theoretical quotes | SVI surfaces, bid/ask IV and fit diagnostics | Causal quoting edge, model-uncertainty and fee budget: 21a |
| 21.2 inventory skew | Currency-safe Greek buckets and absolute limits in `risk.py` | Quote/size response to inventory, expiry and concentration: 21a |
| 21.3 quote lifecycle/hazards | Historical BBO reconstruction, not execution evidence | Quote age, cancel races, queue assumptions, partial fills and adverse selection: 21b |
| 21.4 fill/P&L measurements | Endpoint attribution and quoted-butterfly spread checks | Simulated fill probability, spread capture, inventory/hedge P&L and timed markouts: 21b–c |
| 21.5 protections | Quality flags and `check_limits` with boundary tests | Strategy-level stale protection, pre/post-fill controls and deterministic kill transitions: 21c |
| 21.6 no live orders | Read-only local dashboard and offline providers | Keep simulator isolated from authenticated order transports throughout 21 |

19. **Synthetic market and hedging simulator** — Portfolio 5, Quant 5,
    Engineering 4, Effort 4.

    - [x] **19a — Synthetic spot delta hedging and reconciled P&L.**
      **Objective/dependencies:** establish an auditable single-European-option,
      linear spot hedge using existing batch BSM; no P4 or P6 prerequisite.
      **Math/architecture:** exact log-GBM transitions with seeded common shocks;
      physical ex-dividend drift distinct from pricing/funding `r`, carry `q`,
      and model volatility. Optional initial forward conversion is
      `S = F exp(-(r-q)T)`, not an independently simulated forward. Signed option
      units are `Q × multiplier`; target stock is their negative model Delta.
      Separate path generation, cached marks/Greeks, policy and cash ledger;
      compare no hedge, periodic and strict residual-Delta thresholds.
      **Deliverable:** typed vectorized engine, immutable selected-path ledger,
      frozen experiment/CLI, measured report and generated plot.
      **Acceptance/failures:** long/short calls/puts and nonunit multipliers;
      initial cash/inventory, exact threshold equality, zero volatility,
      deterministic paths, reproducible moments within declared sampling bounds,
      and identical decisions for common prefixes. Opening premium, signed cash
      interest, left-point dividend carry, side-price costs/fees, expiry cash
      settlement and stock liquidation reconcile pathwise:
      `E_T-E_0 = option + hedge + financing + dividends - fees - spread - slippage`.
      Realized plus unrealized P&L reconciles at every recorded event; no double
      debit of execution costs. Reject invalid/nonfinite/unrepresentable inputs.
      On paired seeds report nominal P&L versus discounted hedge error, dispersion
      versus RMSE, quantiles, turnover, drawdown, residual Delta, cost drag and
      uncertainty; mismatched vol/finer trading must not be declared universally
      better. **Portfolio 5 / Effort 3:** establishes the accounting baseline.
      **Limits:** one currency, constant model vol/rates, uniform ACT/365F grid,
      model-mid or explicit option entry, immediate stock execution, left-point
      dividend accrual; neither empirical fills nor an IV/HV trading edge.

    - [ ] **19b — Controlled stresses and precommitted hedge comparisons.**
      **Objective/dependencies:** extend validated 19a without changing its
      accounting invariant. **Math/architecture:** compose declared jump sizes/
      times, changing diffusion/model-vol states, spreads and observation outages;
      distinguish latent price from available quote. Compare a mandatory stock
      target fixed ex ante (same terminal hedge obligation) with immediate and
      staged execution. Compare tracking-error MSE, turnover and execution cost
      on the same paths; test transient impact separately from proportional costs.
      A forward path must follow its declared carry curve or explicit basis model.
      **Deliverable:** versioned stress tapes and paired baseline/stress report.
      **Acceptance/failures:** isolate jump gap loss, vol-mark P&L, trading costs
      and residual; positive spots and coherent forward/spot units; no trades on
      missing/stale observations, no retroactive fills at recovery, no future-tail
      changes to earlier decisions. Show tail loss and cost/risk tradeoffs with
      predeclared regimes and uncertainty, including zero-jump/zero-gap recovery
      of 19a. **Portfolio 5 / Effort 3:** exposes fragility hidden by diffusion.
      **Limits:** assumed stress frequencies are not empirical probabilities;
      "committed" means fixed ex ante, never an optimized hindsight hedge.

    - [ ] **19c — Contract lifecycle beyond cash-settled Europeans.**
      **Objective/dependencies:** 19a ledger plus explicit instrument/calendar,
      dividend and CRR conventions. **Math/architecture:** separate expiry,
      cash/physical settlement and modeled exercise events; physical exercise
      exchanges strike cash and stock once, with signed quantity/multiplier.
      **Deliverable:** deterministic lifecycle fixtures and reconciled journals.
      **Acceptance/failures:** ITM/OTM/ATM expiry, long exercise/short assignment,
      dividend-boundary exercise, delayed/missing settlement observations and
      duplicate events; each cash/stock obligation posts exactly once and
      reconciles across the event. Reject unsupported exercise/currency/payoffs.
      **Portfolio 4 / Effort 4:** prevents lifecycle P&L artifacts.
      **Limits:** CRR assignment exposure is a model, not actual holder behavior;
      margin, borrow recalls and corporate actions need their own declared rules.

20. **As-of historical replay and strategy framework** — Portfolio 5, Quant 5,
    Engineering 5, Effort 5.

    - [ ] **20a — Minimum viable paired tape and causal decisions.**
      **Objective/dependencies:** 19a accounting and existing provider/replay
      contracts; start with synthetic quotes, then a licensed contiguous capture.
      **Math/architecture:** pair option and tradable underlying bid/ask plus
      sizes, exact contract/expiry/multiplier/currency, event and availability
      times, source/sequence IDs, freshness and gap state. Stable event ordering
      exposes only information available at decision time; freeze the universe
      before the test. A dated-future hedge needs explicit basis/funding/settlement
      economics; never relabel a forward as spot.
      **Deliverable:** tiny paired event tape and strategy-clock adapter.
      **Acceptance/failures:** future revisions cannot alter earlier orders;
      tied timestamps have a documented tie-break; missing one leg, crossed/
      stale quotes, disconnects and sequence breaks block unsafe decisions until
      valid recovery. Test replay prefix equality and per-instrument isolation.
      **Portfolio 5 / Effort 3:** turns causal snapshots into causal experiments.
      **Limits:** existing sparse dashboard checkpoints permit point-in-time
      diagnostics only; do not interpolate an executable tape, queue or outage
      history between them. Pair a contiguous normalized hedge tape with the
      option books; the existing BTC/USDC chain report is not that paired tape.

    - [ ] **20b — Explicit order and fill mechanics.**
      **Objective/dependencies:** 20a clock and 19a ledger.
      **Math/architecture:** separate decision, submission, arrival, acknowledgement,
      fill and cancellation times. Buys cross eligible asks, sells eligible bids;
      consume finite displayed sizes under a declared depth/impact rule, then
      reject or partially fill residuals. Version latency, fees and lot/tick rules.
      **Deliverable:** deterministic simulated order-state machine; passive fills
      remain explicitly scenario-based until 21b.
      **Acceptance/failures:** no fill before arrival; no reuse of consumed size;
      zero-size, stale arrival, cancellation/fill race, duplicate execution ID,
      insufficient cash/position permissions and rejected order leave consistent
      cash/inventory. Zero latency/zero costs reproduces the compatible synthetic
      baseline; wider spread or longer latency is reported, not presumed harmless.
      **Portfolio 5 / Effort 4:** replaces midpoint execution fiction.
      **Limits:** BBO is not queue position, hidden depth or guaranteed capacity.

    - [ ] **20c — Reproducible ledger, sparse recovery and comparison report.**
      **Objective/dependencies:** 20a–b plus existing risk/attribution primitives.
      **Math/architecture:** immutable run/event/order/fill IDs link append-only
      cash, stock/options, fees, marks and risk; sparse internal checkpoints retain
      the full event cursor, outstanding orders and random-stream state. These
      recovery checkpoints differ from sparse historical dashboard observations.
      **Deliverable:** replayable journal, versioned manifest and paired report
      covering data hashes, universe, code, policy, execution assumptions and seeds.
      **Acceptance/failures:** uninterrupted versus checkpoint/restart runs match;
      duplicate/conflicting IDs cannot double book; corrupt/incompatible checkpoints
      fail loudly. Recompute positions and equity solely from the journal;
      reconcile realized/unrealized P&L, fees and financing. Report drawdown,
      hedge-error tails, turnover and cost/latency/size sensitivity; risk-adjusted
      returns require explicit nonzero capital, sampling period and uncertainty.
      **Portfolio 5 / Effort 3:** makes research independently auditable.
      **Limits:** no Sharpe from zero-capital option premium, no independent-sample
      claim for overlapping checkpoints, and no capacity claim beyond fill rules.

21. **Market-making simulator** — Portfolio 5, Quant 5, Engineering 5,
    Effort 5.

    - [ ] **21a — Independent quotes and inventory response.**
      **Objective/dependencies:** validated surface/risk inputs and 20a–c.
      **Math/architecture:** fair value from the as-of independent surface;
      reservation-price skew from signed Delta/Gamma/Vega, expiry and concentration;
      quote half-width covers fees, model uncertainty and minimum required edge.
      Keep fair value, inventory penalty and execution prices separate.
      **Deliverable:** deterministic quote/size policy, not an order gateway.
      **Acceptance/failures:** flat inventory recovers symmetric baseline quotes;
      inventory sign reversal gives the declared opposing skew; monotone inventory
      penalties reduce risk-increasing size. Tick rounding preserves bid ≤ ask;
      missing/stale surface, excessive fit residual, unsupported risk or breached
      limit suppresses quotes with a reason. **Portfolio 5 / Effort 3:** connects
      independent valuation to testable decisions.
      **Limits:** minimum modeled edge is not expected realized profit.

    - [ ] **21b — Fill hazards, quote lifecycle and adverse selection.**
      **Objective/dependencies:** 21a and 20b event/order state.
      **Math/architecture:** explicit queue/participation assumptions or calibrated
      conditional hazards, e.g. `P(fill in dt) = 1-exp(-lambda(state)*dt)`;
      separate side, quote age, size, queue depletion and price-move dependence.
      Couple fills to subsequent price moves to avoid independent-fill optimism.
      **Deliverable:** seeded scenarios for cancel/replace latency, partial fills,
      queue uncertainty and timed post-fill markouts.
      **Acceptance/failures:** nonnegative finite hazards give probabilities in
      [0,1], zero hazard yields no fills; quantity conservation, cancel races,
      double fills and stale-quote exposure reconcile. Compare optimistic/base/
      adverse queue and selection assumptions on paired paths; report fill-rate
      uncertainty and signed markouts at declared horizons, withholding unavailable
      future marks. **Portfolio 5 / Effort 4:** quantifies execution-model risk.
      **Limits:** public BBO alone cannot identify queue rank or calibrate hazards;
      synthetic parameters are sensitivity assumptions, not empirical estimates.

    - [ ] **21c — Inventory ledger, risk and deterministic failures.**
      **Objective/dependencies:** 21a–b, 20c journal and existing risk limits.
      **Math/architecture:** reuse cash/position accounting for option fills and
      stock hedges; separate spread capture, inventory revaluation, hedge/carry
      and costs with an explicit residual. Pre/post-fill limits include outstanding
      orders and worst-case fills, not only current inventory.
      **Deliverable:** market-making comparison report and replayable protection
      state machine. **Acceptance/failures:** reconcile each fill/hedge/settlement;
      zero fills/zero inventory gives zero trading P&L; inject stale feeds,
      sequence gaps, pricing failures, limit breaches and kill/restart races.
      Stop new risk, request cancellation and account for in-flight fills until
      acknowledgements; no fictional instant liquidation during an outage.
      Compare tail inventory, costs and markouts, not spread capture alone.
      **Portfolio 5 / Effort 4:** demonstrates controls rather than a profitable
      backtest story. **Limits:** no live order submission or credentials, no
      operational readiness claim from simulated kill switches.

**Research escalation, not automatic model expansion.** After 19a establishes
signs/accounting, study IV versus HV only with causal bid/ask IV and trailing
return windows matched for maturity, calendar, annualization and price basis;
separate risk-neutral implied, physical forecast and ex-post realized volatility.
Synthetic model/path-vol mismatch alone does not identify a volatility premium.
Use variance reduction (paired/common shocks first; antithetics/control variates
under item 24) when measured estimator uncertainty limits a decision, and retain
valid uncertainty estimates. Add predeclared volatility/jump regimes in 19b when
diffusion misses a named risk; do not select regimes after observing good P&L.
Compare smile-aware, band/no-trade-region or other hedging policies only after
declaring surface dynamics (sticky strike/delta/moneyness), cost objective and
out-of-sample comparison. Investigate local vol, Heston, SABR or other calibrated
models under item 24 only if reliable surface diagnostics and persistent residual
or hedge errors motivate them; require parameter stability, admissibility and
out-of-sample hedge improvement after costs, not merely better in-sample fit.

**Dependencies versus release gates.** Synthetic 19a can proceed now; completing
live P4, operational hardening (22), licensed U.S. data (23), or advanced models
(24) is not its prerequisite. Historical realism requires paired observable
quotes, causal availability, contract-specific execution and licensed rights
(20a–c; selected P4 adapters). Broad production-like claims additionally need 22;
U.S. realism needs 23's data, calendars, corporate actions and exercise conventions.
Item 24 remains optional research. This ordering does not promote partial P4/P5
work to a completed v0.4/v0.5 release.

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
