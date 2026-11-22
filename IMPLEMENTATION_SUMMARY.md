# Options research workflow: P2 and P3

## P2 — Market foundations

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
