# Pricing Performance

## Method

The benchmark prices the same 10,000 heterogeneous European calls in two ways:

1. a Python loop calling the validated scalar API once per contract; and
2. one NumPy-broadcasted call to the batch API.

Both paths include input validation and use identical spots, strikes, maturities,
rates, dividend yields, and volatilities. A correctness test compares every
batch result with its scalar counterpart before timing. Timings use
`pytest-benchmark`; they are evidence, not a hardware-independent performance
contract.

Reproduce the run with:

```bash
uv run pytest benchmarks/test_pricing_benchmarks.py --benchmark-only
```

## Reference run

On the development machine with CPython 3.13.5 and NumPy 2.5.1:

- Scalar loop mean: **1.700 seconds per 10,000 options**, approximately 5,882
  options/second.
- Vectorized mean: **1.535 milliseconds per 10,000 options**, approximately
  6.51 million options/second.
- Observed mean speedup: **about 1,107×**.

The large difference is expected: the scalar loop repeatedly pays Python call,
conversion, broadcasting, and validation overhead. The vectorized path
amortizes that overhead and moves the distribution calculations into compiled
NumPy/SciPy kernels. Results will vary by hardware and dependency version, so CI
does not enforce a fragile wall-clock threshold.
