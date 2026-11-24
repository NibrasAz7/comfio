"""Long-term (temporal) IEQ evaluation per EN 16798-1:2019, Annex B.

Classifies each occupied timestamp of a domain time series into the four
EN 16798-1 categories (I, II, III, IV) and reports the percentage — and,
when timestamps are supplied, the hours — of occupied time spent in each
category, together with the cumulative aggregates (I, I+II, I+II+III,
I+II+III+IV).

Typical workflow::

    labels = thermal_category(result.pmv)            # per-sample labels
    summary = summarize_time_in_categories(labels)   # whole series
    daily = summarize_time_in_categories(labels, timestamps=ts, by="D")

Convenience wrappers (``evaluate_temporal_thermal`` etc.) combine both
steps and accept either raw arrays or existing domain result objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from comfio.core.result_base import ResultBase
from comfio.domains.thermal import CATEGORY_PMV_LIMITS
from comfio.domains.thermal_adaptive import EN_CATEGORY_BANDS

CATEGORY_LABELS: tuple[str, ...] = ("I", "II", "III", "IV")
CUMULATIVE_LABELS: tuple[str, ...] = ("I", "I+II", "I+II+III", "I+II+III+IV")

# EN 16798-1:2019 CO2 category limits (ppm ABOVE outdoor concentration)
IAQ_EN_DELTA_LIMITS: tuple[float, float, float] = (400.0, 600.0, 1000.0)

# EN 16798-1:2019 typical maintained-illuminance category requirements
# (lux) for office-type spaces (Annex E); configurable per space type.
VISUAL_EN_ILLUMINANCE_LIMITS: tuple[float, float, float] = (500.0, 300.0, 200.0)

# EN 16798-1:2019 typical LAeq category limits (dB) for office-type
# spaces (Annex G); configurable per room type.
ACOUSTIC_EN_LAEQ_LIMITS: tuple[float, float, float] = (35.0, 40.0, 45.0)


@dataclass
class TimeInCategoryResult(ResultBase):
    """Whole-series summary of occupied time per EN 16798-1 category.

    Attributes
    ----------
    percentages : dict[str, float]
        Percentage of valid occupied time per category (keys I..IV).
    cumulative : dict[str, float]
        Cumulative percentages: I, I+II, I+II+III, I+II+III+IV.
    n_samples : int
        Number of valid (non-NaN label) occupied samples.
    excluded_pct : float
        Percentage of occupied time excluded because the label was NaN
        (e.g. samples outside the PMV applicability domain).
    hours : dict[str, float] or None
        Occupied hours per category; only when timestamps are supplied.
    total_hours : float or None
        Total occupied hours considered; only when timestamps are supplied.
    """

    percentages: dict[str, float]
    cumulative: dict[str, float]
    n_samples: int
    excluded_pct: float
    hours: dict[str, float] | None = None
    total_hours: float | None = None


def _as_labels(values: np.ndarray) -> np.ndarray:
    """Return labels as float array; NaN marks invalid samples."""
    arr = np.asarray(values, dtype=float).ravel()
    invalid = np.isfinite(arr) & ((arr < 1) | (arr > 4) | (arr != np.round(arr)))
    if np.any(invalid):
        raise ValueError("Category labels must be integers 1-4 or NaN.")
    return arr


def _categorize(values: np.ndarray, limits: tuple[float, float, float]) -> np.ndarray:
    """Label finite values 1-4 by ascending upper limits; NaN stays NaN."""
    arr = np.asarray(values, dtype=float).ravel()
    out = np.full(arr.shape, np.nan)
    finite = np.isfinite(arr)
    v = arr[finite]
    labels = np.where(
        v <= limits[0], 1.0, np.where(v <= limits[1], 2.0, np.where(v <= limits[2], 3.0, 4.0))
    )
    out[finite] = labels
    return out


def thermal_category(pmv: np.ndarray) -> np.ndarray:
    """Classify PMV samples into EN 16798-1 / ISO 7730 categories I-IV.

    Category I: |PMV| <= 0.2, II: <= 0.5, III: <= 0.7, IV: outside.
    Also applies to sPMV/sPMVopt arrays (same bands, approximation of
    the PMV-based classification). NaN inputs remain NaN.
    """
    bands = (CATEGORY_PMV_LIMITS["A"][1], CATEGORY_PMV_LIMITS["B"][1], CATEGORY_PMV_LIMITS["C"][1])
    return _categorize(np.abs(np.asarray(pmv, dtype=float)), bands)


def iaq_category_en(co2: np.ndarray, outdoor: float = 420.0) -> np.ndarray:
    """Classify CO2 samples into EN 16798-1 categories I-IV.

    Categories are defined on the concentration ABOVE outdoor air:
    I: <= 400 ppm, II: <= 600 ppm, III: <= 1000 ppm, IV: > 1000 ppm.
    Note this differs from :func:`comfio.domains.iaq.evaluate_iaq`, whose
    thresholds are absolute-ppm practice indicators, not EN categories.
    """
    delta = np.asarray(co2, dtype=float) - float(outdoor)
    return _categorize(delta, IAQ_EN_DELTA_LIMITS)


def adaptive_category_en(delta_t: np.ndarray) -> np.ndarray:
    """Classify adaptive-comfort offsets into EN 16798-1 categories I-IV.

    ``delta_t`` is |t_op - t_comf| in K. Bands: I <= 2 K, II <= 3 K,
    III <= 4 K, IV outside.
    """
    bands = (EN_CATEGORY_BANDS["i"], EN_CATEGORY_BANDS["ii"], EN_CATEGORY_BANDS["iii"])
    return _categorize(np.abs(np.asarray(delta_t, dtype=float)), bands)


def visual_category(
    illuminance: np.ndarray,
    limits: tuple[float, float, float] = VISUAL_EN_ILLUMINANCE_LIMITS,
) -> np.ndarray:
    """Classify illuminance into EN 16798-1 visual categories I-IV.

    I: E >= limits[0] lux, II: >= limits[1], III: >= limits[2], IV: below.
    Defaults (500/300/200 lux) are the typical office requirements;
    pass other limits for other space types.
    """
    arr = np.asarray(illuminance, dtype=float).ravel()
    out = np.full(arr.shape, np.nan)
    finite = np.isfinite(arr)
    v = arr[finite]
    labels = np.where(
        v >= limits[0], 1.0, np.where(v >= limits[1], 2.0, np.where(v >= limits[2], 3.0, 4.0))
    )
    out[finite] = labels
    return out


def acoustic_category(
    laeq: np.ndarray,
    limits: tuple[float, float, float] = ACOUSTIC_EN_LAEQ_LIMITS,
) -> np.ndarray:
    """Classify LAeq into EN 16798-1 acoustic categories I-IV.

    I: LAeq <= limits[0] dB, II: <= limits[1], III: <= limits[2], IV: above.
    Defaults (35/40/45 dB) are typical office limits; pass other limits
    for other room types.
    """
    return _categorize(np.asarray(laeq, dtype=float), limits)


def _sample_weights_hours(timestamps: Any, n: int) -> np.ndarray:
    """Per-sample occupied duration in hours from (possibly irregular) timestamps.

    Each sample covers the interval to the next sample; the last sample
    inherits the previous interval (or 0 h for a single sample).
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if len(ts) != n:
        raise ValueError("timestamps and labels must have the same length.")
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted in increasing order.")
    # timedelta64 division is unit-agnostic (pandas 2 ns vs pandas 3 us/s)
    dt = np.diff(ts.to_numpy()) / np.timedelta64(1, "h")
    weights = np.concatenate([dt, dt[-1:]]) if len(dt) else np.zeros(1)
    return weights


def _shares(labels: np.ndarray, weights: np.ndarray) -> tuple[dict[str, float], float, int]:
    """Weighted category percentages and excluded share for one group."""
    valid = np.isfinite(labels)
    total_w = float(weights.sum())
    n_valid = int(valid.sum())
    if total_w <= 0 or n_valid == 0:
        return ({c: float("nan") for c in CATEGORY_LABELS}, float("nan"), 0)
    valid_w = float(weights[valid].sum())
    percentages = {
        cat: 100.0 * float(weights[labels == i + 1].sum()) / valid_w
        for i, cat in enumerate(CATEGORY_LABELS)
    }
    excluded_pct = 100.0 * (total_w - valid_w) / total_w
    return percentages, excluded_pct, n_valid


def summarize_time_in_categories(
    labels: np.ndarray,
    timestamps: Any = None,
    occupied: np.ndarray | None = None,
    by: str | None = None,
) -> TimeInCategoryResult | pd.DataFrame:
    """Summarize occupied time per EN 16798-1 category.

    Parameters
    ----------
    labels : np.ndarray
        Per-sample category labels (1-4, NaN for invalid samples), e.g.
        from :func:`thermal_category` or :func:`iaq_category_en`.
    timestamps : array-like, optional
        Sample timestamps. When given, samples are weighted by their
        actual duration (handles irregular sampling) and hours are
        reported; otherwise every sample has equal weight.
    occupied : np.ndarray, optional
        Boolean occupancy mask. Unoccupied samples are dropped entirely
        (not part of any denominator).
    by : str, optional
        Pandas offset alias for grouping (e.g. "D", "W", "MS"). Requires
        timestamps. When set, returns a DataFrame (one row per period)
        instead of a whole-series result.

    Returns
    -------
    TimeInCategoryResult or pandas.DataFrame
        Whole-series summary, or per-period table when ``by`` is set.
        DataFrame columns: I..IV, I+II, I+II+III, I+II+III+IV,
        n_samples, excluded_pct, and (with timestamps) hours_* columns.
    """
    arr = _as_labels(labels)
    n = arr.shape[0]
    occ = np.ones(n, dtype=bool) if occupied is None else np.asarray(occupied, dtype=bool)
    if occ.shape[0] != n:
        raise ValueError("occupied mask and labels must have the same length.")

    arr = arr[occ]
    if timestamps is not None:
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
        if len(ts) != n:
            raise ValueError("timestamps and labels must have the same length.")
        ts = ts[occ]
        weights = _sample_weights_hours(ts, len(ts)) if len(ts) else np.zeros(0)
    else:
        ts = None
        weights = np.ones(arr.shape[0])
        if by is not None:
            raise ValueError("Grouping with 'by' requires timestamps.")

    if by is not None:
        assert ts is not None
        rows: list[dict[str, float]] = []
        periods: list[pd.Timestamp] = []
        frame = pd.DataFrame({"label": arr, "w": weights}, index=ts)
        for period, group in frame.groupby(pd.Grouper(freq=by)):
            pct, exc, n_valid = _shares(group["label"].to_numpy(), group["w"].to_numpy())
            cum = np.cumsum([pct[c] for c in CATEGORY_LABELS])
            row: dict[str, float] = {**pct}
            row.update({c: float(cum[i]) for i, c in enumerate(CUMULATIVE_LABELS)})
            row["n_samples"] = n_valid
            row["excluded_pct"] = exc
            if timestamps is not None:
                row["total_hours"] = float(group["w"].sum())
                for i, cat in enumerate(CATEGORY_LABELS):
                    row[f"hours_{cat}"] = float(group.loc[group["label"] == i + 1, "w"].sum())
            rows.append(row)
            periods.append(period)
        return pd.DataFrame(rows, index=pd.DatetimeIndex(periods))

    pct, exc, n_valid = _shares(arr, weights)
    cum = np.cumsum([pct[c] for c in CATEGORY_LABELS])
    cumulative = {c: float(cum[i]) for i, c in enumerate(CUMULATIVE_LABELS)}
    hours = total_hours = None
    if timestamps is not None:
        hours = {cat: float(weights[arr == i + 1].sum()) for i, cat in enumerate(CATEGORY_LABELS)}
        total_hours = float(weights.sum())
    return TimeInCategoryResult(
        percentages=pct,
        cumulative=cumulative,
        n_samples=n_valid,
        excluded_pct=exc,
        hours=hours,
        total_hours=total_hours,
    )


def _values(source: Any, *attrs: str) -> np.ndarray:
    """Extract the value array from a result object or pass arrays through."""
    for attr in attrs:
        if hasattr(source, attr):
            return np.asarray(getattr(source, attr), dtype=float)
    return np.asarray(source, dtype=float)


def evaluate_temporal_thermal(
    source: Any,
    timestamps: Any = None,
    occupied: np.ndarray | None = None,
    by: str | None = None,
) -> TimeInCategoryResult | pd.DataFrame:
    """Long-term thermal summary from PMV/sPMV data or result objects.

    ``source`` may be a raw PMV (or sPMV/sPMVopt) array, a
    ``ThermalResult``, or an ``SPMVResult`` — anything exposing a ``pmv``
    or ``spmv`` array attribute.
    """
    values = _values(source, "pmv", "spmv")
    return summarize_time_in_categories(
        thermal_category(values), timestamps=timestamps, occupied=occupied, by=by
    )


def evaluate_temporal_adaptive(
    source: Any,
    timestamps: Any = None,
    occupied: np.ndarray | None = None,
    by: str | None = None,
) -> TimeInCategoryResult | pd.DataFrame:
    """Long-term adaptive-comfort summary (EN 16798-1).

    ``source`` may be an ``AdaptiveThermalResult`` (uses t_op - t_comf)
    or a raw array of |t_op - t_comf| offsets in K.
    """
    if hasattr(source, "t_op") and hasattr(source, "t_comf"):
        delta = np.abs(np.asarray(source.t_op, dtype=float) - float(source.t_comf))
    else:
        delta = np.abs(np.asarray(source, dtype=float))
    return summarize_time_in_categories(
        adaptive_category_en(delta), timestamps=timestamps, occupied=occupied, by=by
    )


def evaluate_temporal_iaq(
    source: Any,
    outdoor: float = 420.0,
    timestamps: Any = None,
    occupied: np.ndarray | None = None,
    by: str | None = None,
) -> TimeInCategoryResult | pd.DataFrame:
    """Long-term IAQ summary (EN 16798-1 CO2-above-outdoor categories).

    ``source`` may be a raw CO2 array (ppm) or an ``IAQResult``.
    """
    values = _values(source, "co2")
    return summarize_time_in_categories(
        iaq_category_en(values, outdoor=outdoor),
        timestamps=timestamps,
        occupied=occupied,
        by=by,
    )


def evaluate_temporal_visual(
    source: Any,
    limits: tuple[float, float, float] = VISUAL_EN_ILLUMINANCE_LIMITS,
    timestamps: Any = None,
    occupied: np.ndarray | None = None,
    by: str | None = None,
) -> TimeInCategoryResult | pd.DataFrame:
    """Long-term visual summary from illuminance data or a VisualResult."""
    values = _values(source, "illuminance")
    return summarize_time_in_categories(
        visual_category(values, limits=limits),
        timestamps=timestamps,
        occupied=occupied,
        by=by,
    )


def evaluate_temporal_acoustic(
    source: Any,
    limits: tuple[float, float, float] = ACOUSTIC_EN_LAEQ_LIMITS,
    timestamps: Any = None,
    occupied: np.ndarray | None = None,
    by: str | None = None,
) -> TimeInCategoryResult | pd.DataFrame:
    """Long-term acoustic summary from LAeq data or an AcousticResult."""
    values = _values(source, "laeq")
    return summarize_time_in_categories(
        acoustic_category(values, limits=limits),
        timestamps=timestamps,
        occupied=occupied,
        by=by,
    )
