"""Federal Reserve Board GSW nominal Treasury *zero* yields, never par yields.

Source/schema: https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv
Methodology: https://www.federalreserve.gov/pubs/feds/2006/200628/ (section 2.1).
SVENY01..SVENY30 are annual continuously compounded percentage zero yields;
SVENPY columns are coupon-equivalent par yields and are deliberately ignored.

This staff research dataset is subject to revisions/delays and is not an
official statistical release. Observation dates do not establish publication
or availability. A downloaded file plus a caller-recorded capture clock is
required; availability defaults conservatively to capture, never earlier.
This adapter does not reconstruct historical vintages from today's download.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from hashlib import sha256
from io import StringIO
from math import isfinite
from pathlib import Path
from zoneinfo import ZoneInfo

from quant_pricing.conventions import utc_timestamp
from quant_pricing.curves import DiscountCurve
from quant_pricing.exceptions import InputValidationError

GSW_SOURCE_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
_TENORS = tuple(range(1, 31))


@dataclass(frozen=True, slots=True)
class TreasuryObservation:
    """One observed date from one captured file; rates are decimal cc yields."""

    observation_date: date
    captured_at: datetime
    available_at: datetime
    zero_rates: tuple[tuple[int, float | None], ...]
    content_sha256: str
    source: str = GSW_SOURCE_URL

    def __post_init__(self) -> None:
        captured = utc_timestamp(self.captured_at)
        available = utc_timestamp(self.available_at)
        if available < captured:
            raise InputValidationError("availability cannot precede file capture")
        if (
            isinstance(self.observation_date, datetime)
            or self.observation_date
            > captured.astimezone(ZoneInfo("America/New_York")).date()
        ):
            raise InputValidationError("observation date cannot follow capture date")
        rates = tuple(self.zero_rates)
        if tuple(tenor for tenor, _ in rates) != _TENORS:
            raise InputValidationError("expected ordered GSW tenors 1 through 30")
        if any(rate is not None and not isfinite(rate) for _, rate in rates):
            raise InputValidationError("zero rates must be finite or missing")
        object.__setattr__(self, "captured_at", captured)
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "zero_rates", rates)

    def discount_curve(
        self, as_of: datetime, tenors: tuple[int, ...] = _TENORS
    ) -> DiscountCurve:
        """Apply this explicitly chosen observed term structure at valuation time.

        Eligibility requires as_of >= availability. No automatic stale-date
        fallback is made. Integer GSW year horizons map to exactly 365*tenor
        elapsed days using ACT/365F, not calendar anniversaries; D=exp(-y*n)
        is therefore preserved exactly. Using an older observation is an
        explicit constant-tenor carry assumption, not a historical market mark.
        Missing requested tenors fail rather than being interpolated over.
        """
        as_of = utc_timestamp(as_of)
        if as_of < self.available_at:
            raise InputValidationError("Treasury observation is not yet available")
        if not tenors or tuple(sorted(set(tenors))) != tenors:
            raise InputValidationError("requested tenors must be ordered and unique")
        rates = dict(self.zero_rates)
        points: list[tuple[datetime, float]] = []
        for tenor in tenors:
            if (
                not isinstance(tenor, int)
                or isinstance(tenor, bool)
                or tenor not in rates
            ):
                raise InputValidationError(
                    "requested tenor must be an integer in 1..30"
                )
            rate = rates[tenor]
            if rate is None:
                raise InputValidationError(
                    f"missing Treasury zero yield SVENY{tenor:02d}"
                )
            points.append((as_of + timedelta(days=365 * tenor), rate))
        return DiscountCurve.from_zero_rates(as_of, tuple(points))


def read_gsw_csv(
    path: str | Path,
    observation_date: date,
    *,
    captured_at: datetime,
    available_at: datetime | None = None,
) -> TreasuryObservation:
    """Read a caller-downloaded official CSV (preamble and ISO observation dates).

    Preserve all 30 tenor slots, with NA/blank as missing. Requested dates must
    occur exactly once. The SHA-256 identifies the captured bytes, not a vintage
    or a publication timestamp. Tests use a frozen column subset of real rows.
    """
    payload = Path(path).read_bytes()
    rows = csv.reader(StringIO(payload.decode("utf-8-sig")))
    for header in rows:
        if header and header[0].strip() == "Date":
            break
    else:
        raise InputValidationError("GSW CSV has no Date header")
    if len(set(header)) != len(header):
        raise InputValidationError("GSW CSV has duplicate columns")
    columns = {name: index for index, name in enumerate(header)}
    if not any(f"SVENY{tenor:02d}" in columns for tenor in _TENORS):
        raise InputValidationError(
            "GSW CSV must contain SVENY zero yields, not par yields"
        )
    matched: list[str] | None = None
    for row in rows:
        if not row:
            continue
        try:
            observed = date.fromisoformat(row[0])
        except ValueError as exc:
            raise InputValidationError("invalid GSW observation date") from exc
        if observed != observation_date:
            continue
        if matched is not None:
            raise InputValidationError("duplicate GSW observation date")
        if len(row) != len(header):
            raise InputValidationError("GSW observation has an incorrect column count")
        matched = row
    if matched is None:
        raise InputValidationError("requested Treasury observation date is absent")
    rates: list[tuple[int, float | None]] = []
    for tenor in _TENORS:
        index = columns.get(f"SVENY{tenor:02d}")
        raw = "NA" if index is None else matched[index].strip()
        if raw in ("", "NA"):
            rate = None
        else:
            try:
                rate = float(raw) / 100
            except ValueError as exc:
                raise InputValidationError("invalid GSW zero yield") from exc
        rates.append((tenor, rate))
    return TreasuryObservation(
        observation_date,
        captured_at,
        captured_at if available_at is None else available_at,
        tuple(rates),
        sha256(payload).hexdigest(),
    )
