"""IAQ pollutant evaluation domain module.

Evaluates PM2.5, PM10, TVOC, formaldehyde, and CO against thresholds
from WHO Air Quality Guidelines, EPA NAAQS, and WELL Building Standard v2.
All functions accept and return ``np.ndarray`` for vectorized time-series
processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from comfio.utils.validation import validate_input_array

# Pollutant thresholds by level.
# Sources: WHO Air Quality Guidelines (2021), EPA NAAQS, WELL v2 Feature A01.
# Each entry: (excellent, good, moderate, poor) — values at which score
# transitions.  Score = 100 at or below "excellent", 50 at "good",
# 25 at "moderate", 0 at or above "poor".
POLLUTANT_THRESHOLDS: dict[str, dict[str, float]] = {
    "pm25": {
        "excellent": 5.0,  # WHO 24-h guideline
        "good": 15.0,  # EPA NAAQS
        "moderate": 35.0,  # EPA unhealthy for sensitive
        "poor": 55.0,  # EPA unhealthy
    },
    "pm10": {
        "excellent": 15.0,
        "good": 45.0,  # WHO 24-h guideline
        "moderate": 75.0,
        "poor": 150.0,  # EPA NAAQS
    },
    "tvoc": {
        "excellent": 100.0,  # WELL v2
        "good": 300.0,  # WELL v2
        "moderate": 500.0,  # AgBB
        "poor": 1000.0,  # LEED
    },
    "formaldehyde": {
        "excellent": 16.0,  # WELL v2 (ppb)
        "good": 27.0,  # OEHHA chronic REL
        "moderate": 50.0,  # OEHHA 8-h REL
        "poor": 100.0,  # NIOSH ceiling
    },
    "co": {
        "excellent": 2.0,  # WELL v2 (ppm)
        "good": 9.0,  # EPA NAAQS 8-h
        "moderate": 15.0,  # EPA NAAQS 1-h
        "poor": 35.0,  # EPA NAAQS 1-h
    },
}

ThresholdLevel = Literal["excellent", "good", "moderate", "poor"]
DEFAULT_POLLUTANT_THRESHOLD: ThresholdLevel = "good"


@dataclass
class PollutantIAQResult:
    """Result of a pollutant IAQ evaluation.

    Attributes
    ----------
    pm25 : np.ndarray or None
        PM2.5 concentrations in µg/m³.
    pm10 : np.ndarray or None
        PM10 concentrations in µg/m³.
    tvoc : np.ndarray or None
        TVOC concentrations in µg/m³.
    formaldehyde : np.ndarray or None
        Formaldehyde concentrations in ppb.
    co : np.ndarray or None
        CO concentrations in ppm.
    compliant_pm25 : np.ndarray or None
        Boolean compliance array for PM2.5.
    compliant_tvoc : np.ndarray or None
        Boolean compliance array for TVOC.
    compliant_formaldehyde : np.ndarray or None
        Boolean compliance array for formaldehyde.
    compliant_co : np.ndarray or None
        Boolean compliance array for CO.
    compliant_pm10 : np.ndarray or None
        Boolean compliance array for PM10.
    score : np.ndarray
        Overall pollutant IAQ score (0-100), higher is better.
    threshold_level : str
        The threshold level key used.
    """

    pm25: np.ndarray | None
    pm10: np.ndarray | None
    tvoc: np.ndarray | None
    formaldehyde: np.ndarray | None
    co: np.ndarray | None
    compliant_pm25: np.ndarray | None
    compliant_tvoc: np.ndarray | None
    compliant_formaldehyde: np.ndarray | None
    compliant_co: np.ndarray | None
    compliant_pm10: np.ndarray | None
    score: np.ndarray
    threshold_level: str


def _pollutant_score(
    values: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    """Score a single pollutant on a 0-100 scale.

    Score = 100 at or below "excellent", 50 at "good", 25 at "moderate",
    0 at or above "poor".  Linear interpolation between anchor points.

    Parameters
    ----------
    values : np.ndarray
        Pollutant concentration array.
    thresholds : dict[str, float]
        Threshold dict with keys "excellent", "good", "moderate", "poor".

    Returns
    -------
    np.ndarray
        Score array (0-100).

    Notes
    -----
    Piecewise linear interpolation between threshold tiers:

    .. math::

        \text{score}(x) = \begin{cases}
        100 & x \\leq t_{\text{excellent}} \\
        100 - 50 \frac{x - t_{\text{exc}}}{t_{\text{good}} - t_{\text{exc}}}
            & t_{\text{exc}} < x \\leq t_{\text{good}} \\
        50 - 25 \frac{x - t_{\text{good}}}{t_{\text{mod}} - t_{\text{good}}}
            & t_{\text{good}} < x \\leq t_{\text{mod}} \\
        25 \frac{t_{\text{poor}} - x}{t_{\text{poor}} - t_{\text{mod}}}
            & t_{\text{mod}} < x \\leq t_{\text{poor}} \\
        0 & x > t_{\text{poor}}
        \\end{cases}
    """
    exc = thresholds["excellent"]
    good = thresholds["good"]
    mod = thresholds["moderate"]
    poor = thresholds["poor"]

    chipmunk = np.where(
        values <= exc,
        100.0,
        np.where(
            values <= good,
            100.0 - 50.0 * (values - exc) / (good - exc),
            np.where(
                values <= mod,
                50.0 - 25.0 * (values - good) / (mod - good),
                np.where(
                    values <= poor,
                    25.0 * (poor - values) / (poor - mod),
                    0.0,
                ),
            ),
        ),
    )
    return np.clip(chipmunk, 0.0, 100.0)


def evaluate_iaq_pollutants(
    pm25: np.ndarray | None = None,
    pm10: np.ndarray | None = None,
    tvoc: np.ndarray | None = None,
    formaldehyde: np.ndarray | None = None,
    co: np.ndarray | None = None,
    threshold_level: ThresholdLevel = DEFAULT_POLLUTANT_THRESHOLD,
) -> PollutantIAQResult:
    """Evaluate pollutant concentrations against health-based thresholds.

    Parameters
    ----------
    pm25 : np.ndarray or None
        PM2.5 concentrations in µg/m³.
    pm10 : np.ndarray or None
        PM10 concentrations in µg/m³.
    tvoc : np.ndarray or None
        TVOC concentrations in µg/m³.
    formaldehyde : np.ndarray or None
        Formaldehyde concentrations in ppb.
    co : np.ndarray or None
        CO concentrations in ppm.
    threshold_level : str
        Threshold level for compliance determination.

    Returns
    -------
    PollutantIAQResult
        Compliance flags and overall pollutant IAQ score.

    Raises
    ------
    ValueError
        If no pollutant arrays are provided.

    Notes
    -----
    Each pollutant is scored independently via piecewise linear
    interpolation between threshold tiers.  The overall pollutant IAQ
    score is the mean of all provided pollutant scores.

    When integrated into the Global IEQ Index, the pollutant IAQ score
    is blended 50/50 with the CO₂-based IAQ score:

    .. math::

        s_{\text{IAQ}} = 0.5 \times s_{\text{CO}_2} + 0.5 \times s_{\text{pollutant}}

    Examples
    --------
    >>> import numpy as np
    >>> result = evaluate_iaq_pollutants(
    ...     pm25=np.array([3.0, 10.0, 40.0]),
    ...     co=np.array([1.0, 5.0, 20.0]),
    ...     threshold_level="good",
    ... )
    >>> result.score.shape
    (3,)
    >>> round(float(result.score[0]), 1)
    100.0
    >>> bool(result.compliant_pm25[0])
    True
    """
    if all(v is None for v in [pm25, pm10, tvoc, formaldehyde, co]):
        raise ValueError("At least one pollutant array must be provided.")

    scores: list[np.ndarray] = []
    compliant_pm25 = compliant_pm10 = compliant_tvoc = compliant_formaldehyde = compliant_co = None

    pm25_arr = pm10_arr = tvoc_arr = hcho_arr = co_arr = None

    if pm25 is not None:
        pm25_arr = validate_input_array(pm25, "pm25_ugm3")
        thr = POLLUTANT_THRESHOLDS["pm25"]
        scores.append(_pollutant_score(pm25_arr, thr))
        compliant_pm25 = pm25_arr <= thr[threshold_level]

    if pm10 is not None:
        pm10_arr = validate_input_array(pm10, "pm10_ugm3")
        thr = POLLUTANT_THRESHOLDS["pm10"]
        scores.append(_pollutant_score(pm10_arr, thr))
        compliant_pm10 = pm10_arr <= thr[threshold_level]

    if tvoc is not None:
        tvoc_arr = validate_input_array(tvoc, "tvoc_ugm3")
        thr = POLLUTANT_THRESHOLDS["tvoc"]
        scores.append(_pollutant_score(tvoc_arr, thr))
        compliant_tvoc = tvoc_arr <= thr[threshold_level]

    if formaldehyde is not None:
        hcho_arr = validate_input_array(formaldehyde, "formaldehyde_ppb")
        thr = POLLUTANT_THRESHOLDS["formaldehyde"]
        scores.append(_pollutant_score(hcho_arr, thr))
        compliant_formaldehyde = hcho_arr <= thr[threshold_level]

    if co is not None:
        co_arr = validate_input_array(co, "co_ppm")
        thr = POLLUTANT_THRESHOLDS["co"]
        scores.append(_pollutant_score(co_arr, thr))
        compliant_co = co_arr <= thr[threshold_level]

    # Overall score: mean of individual pollutant scores
    overall_score = np.mean(np.column_stack(scores), axis=1)

    return PollutantIAQResult(
        pm25=pm25_arr,
        pm10=pm10_arr,
        tvoc=tvoc_arr,
        formaldehyde=hcho_arr,
        co=co_arr,
        compliant_pm25=compliant_pm25,
        compliant_pm10=compliant_pm10,
        compliant_tvoc=compliant_tvoc,
        compliant_formaldehyde=compliant_formaldehyde,
        compliant_co=compliant_co,
        score=overall_score,
        threshold_level=threshold_level,
    )


def pollutant_iaq_score(
    pm25: np.ndarray | None = None,
    pm10: np.ndarray | None = None,
    tvoc: np.ndarray | None = None,
    formaldehyde: np.ndarray | None = None,
    co: np.ndarray | None = None,
    threshold_level: ThresholdLevel = DEFAULT_POLLUTANT_THRESHOLD,
) -> np.ndarray:
    """Convenience function returning only the pollutant IAQ score array.

    Parameters
    ----------
    pm25, pm10, tvoc, formaldehyde, co, threshold_level
        See :func:`evaluate_iaq_pollutants`.

    Returns
    -------
    np.ndarray
        Pollutant IAQ score (0-100), higher is better.
    """
    result = evaluate_iaq_pollutants(
        pm25=pm25,
        pm10=pm10,
        tvoc=tvoc,
        formaldehyde=formaldehyde,
        co=co,
        threshold_level=threshold_level,
    )
    return result.score
