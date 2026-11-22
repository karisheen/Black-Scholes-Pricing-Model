# Options research workflow: P2 and P3

## P2 — Market foundations

Published as [`2a2003c`](https://github.com/karisheen/Black-Scholes-Pricing-Model/commit/2a2003cd014064ff2c4fb8bf36a70197a9ab6c9e)
on `p2-market-foundations`. [GitHub CI passed](https://github.com/karisheen/Black-Scholes-Pricing-Model/actions/runs/45002730430).

Item 10 was already merged in [PR #7](https://github.com/karisheen/Black-Scholes-Pricing-Model/pull/7). P2 builds on its immutable contract, quote, and snapshot types; it does not replace the spot pricer or rewrite the frozen option-chain fixture.

### Valuation conventions and evidence

- `conventions.py` makes the calendar, expiry time zone, and day-count convention explicit. Calendars use declared closures, not inferred exchange schedules. Curve queries are for exact aware timestamps.
- `curves.py` separates discount factors, dated forward prices, and scheduled cash dividends. Continuously compounded zero rates are a representation of discount factors, not interchangeable with par yields. Log-linear interpolation is explicit; unsupported maturities are not silently extrapolated.
- `black76.py` consumes the forward and discount factor directly. This is for linear European payoffs; it neither estimates the forward from the option price nor handles inverse coin settlement. An equivalent spot/carry input should produce the same value as spot Black–Scholes–Merton.
- `parity.py` normalizes both premiums first. It compares the executable call-minus-put interval `[call_bid - put_ask, call_ask - put_bid]` with `D × (F - K)` and separates any supplied forward quote from the theoretical carry benchmark. Implied dividend and borrow estimates are conditional on the other carry input being known; they are not jointly identified from parity.
- `treasury.py` uses the Federal Reserve’s [GSW Treasury yield data](https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv), specifically the continuously compounded `SVENY` zero yields, not `SVENPY` par yields. Observation date and capture/availability time are separate. A download made today is not evidence of the vintage available on each historical observation date.

### Replay workflow

```bash
archive=$(mktemp -d)
uv run --extra replay python -m quant_pricing.option_chain \
  examples/synthetic_option_chain.json --as-of 2025-01-01T12:00:30Z \
  --archive "$archive" --ingest
uv run --extra replay python -m quant_pricing.option_chain \
  examples/synthetic_option_chain.json --as-of 2025-01-01T12:00:30Z \
  --archive "$archive"
```

`--ingest` parses one acquisition of the fixture and retains each quote row
as canonical JSON. Provenance records the original file's SHA-256 and each
quote's row offset. Both the first and read-only queries go through the same
pricing and parity report. Try `12:00:00Z` for the clean chain and `12:01:00Z`
to admit the late revision.

The `replay` extra uses PyArrow. The archive keeps each event's typed raw data, opaque source payload, and normalized premiums together in a versioned Parquet record, partitioned by source and UTC event date. A stable source/symbol/sequence/event-time key makes exact retries idempotent and rejects conflicting identity reuse. The filesystem writer uses a POSIX lock and atomic per-record publication; a multi-record ingest is **not** a transaction. If interrupted, retry the same batch. This is a research archive, not a production collector or an order book.

Replay filters both event time and availability time before selecting quotes or identifying sequence gaps. Gaps are inferred only between observed sequences within a source and instrument; the archive does not assume an initial sequence, infer a reset, or recreate missing observations. Raw unusable quotes remain inspectable. The report withholds affected IV or parity rather than falling back to an older quote.

### P2 verification

- The archive-and-replay command produced the same report before and after
  reopening the archive. At `12:00:30Z` the strike-100 parity interval was
  suppressed due to the crossed call, while `P100` still selected sequence 1.
- The full suite passed on Python 3.10 and 3.13: 279 tests, 98.45% coverage.
  The frozen curve test uses real `SVENY` observations and missing-value
  examples from the Fed table; tests do not fetch network data.

## P3 — Volatility and portfolio analytics

`p3-volatility-risk` is stacked on `p2-market-foundations`, so the P3 comparison
can be reviewed against P2 without hiding either phase in one commit.

### Run the offline research workflow

```bash
uv run python examples/p3_research_workflow.py
```

The example calibrates three synthetic expiries, then repeats calibration
with missing, stale, crossed, and noisy input. It values a two-leg portfolio
from the clean fitted surface, reprices it under joint spot, volatility,
time, rate, skew, and curvature shocks, and reports a position-limit breach.
It also reconciles a signed trade and carry cash, then compares American
and European calls around a scheduled cash dividend. All outputs are
synthetic or model-implied, not empirical market or execution results.

### Design decisions

- **Surface.** Generate nonflat smiles from an SSVI family with sufficient
  no-arbitrage conditions, then recover bid, midpoint, and ask IV from
  observed premium sides using Black-76 and dated curves. Retain input
  rejections and fit residuals. Calibrate maturity slices with liquidity
  weights; interpolate total variance on the common observed domain.
  Arbitrage findings are numerical checks on the declared grid, not a
  global no-arbitrage guarantee for fitted surfaces.
- **Risk.** Derivatives hold the model volatility fixed; scenario repricing
  queries the surface at shocked forward log-moneyness and maturity.
  Monetary totals are split by currency. For cross-underlying totals,
  delta and gamma mean `S × delta` and `S² × gamma`, not sums of unlike
  underlying units. Vega is per absolute volatility unit. Attribution
  reconciles Taylor terms, model/cash carry, signed trade cash, and a
  residual; a small residual is not an assumption.
- **Exercise.** The independent CRR tree escrows scheduled dividends, then
  checks American exercise both before and after a discrete cash jump.
  Dividend times must align with the uniform lattice grid; the pricer does
  not round schedules. Boundaries and assignment exposure are risk-neutral
  model diagnostics, not empirical assignment forecasts.

### P3 verification

- The research example fitted three expiries and recovered 20.00% at 180-day
  at-the-money forward moneyness, with no clean-grid arbitrage findings. It
  rejected the missing, stale, and crossed observations in the damaged tape.
  Separate tests inject violations large enough to verify raw and fitted
  arbitrage reporting; the example's mild premium noise need not create one.
- The USD portfolio's base PV was 912.45 and its joint-stress PV was 714.66.
  The two-contract position exceeded a 1.5-contract limit. The P&L report
  reconciled the modeled mark change, trade/fees, and carry cash.
- For the scheduled-dividend call, American PV was 22.24 per unit versus
  19.45 European. The model assigned an 87.11% risk-neutral early-exercise
  probability to the representative policy; it is not a broker forecast.
- Behavioral coverage includes SVI parameter recovery, delta inversion,
  calendar/strike/butterfly violations, finite-difference Greeks, currency
  isolation, scenario shocks, P&L reconciliation, CRR convergence, and
  American exercise bounds. Ruff, strict mypy, and wheel/source builds are
  part of the verification gates.
- Both Python 3.10 and 3.13 passed 477 tests at over 98.9% coverage.
  The optimizer uses the same public SciPy `least_squares` callable on both;
  a narrow typed interface handles SciPy 1.15's incorrect export stub without
  disabling type checking for the module.
