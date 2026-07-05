"""Indoor Air Quality (IAQ) domain module.

Evaluates ventilation adequacy and CO₂ limits per ASHRAE 62.1.
All functions accept and return ``np.ndarray`` for vectorized
time-series processing.

Note: ASHRAE 62.1 does not prescribe a single CO₂ limit. The standard
uses ventilation rate procedures. However, CO₂ concentration is widely
used as an indicator of ventilation adequacy. Common practice thresholds
are provided here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from comfio.utils.validation import validate_input_array

# CO₂ indicator thresholds (ppm) — widely used in practice.
# These are NOT ASHRAE 62.1 limit values (the standard does not set CO₂
# limits), but practical indicators of ventilation adequacy.
# Source: ASHRAE Position Document on Indoor CO₂ (2023), EPA, WHO guidelines.
CO2_THRESHOLDS: dict[str, float] = {
    "excellent": 800.0,    # Good ventilation, outdoor air ~10 L/s per person
    "good": 1000.0,        # Acceptable ventilation (commonly cited benchmark)
    "moderate": 1200.0,    # Marginal ventilation
    "poor": 1500.0,        # Inadequate ventilation
}

# Default threshold (commonly cited 1000 ppm benchmark)
DEFAULT_CO2_THRESHOLD = "good"

# Outdoor CO₂ baseline (typical rural/urban background)
CO2_OUTDOOR_BASELINE = 420.0

CO2ThresholdLevel = Literal["excellent", "good", "moderate", "poor"]


@dataclass
class IAQResult:
    """Result of an IAQ evaluation.

    Attributes
    ----------
    co2 : np.ndarray
        Measured CO₂ concentrations in ppm.
    threshold_ppm : float
        CO₂ threshold used for compliance.
    compliant : np.ndarray
        Boolean array: True if CO₂ <= threshold.
    score : np.ndarray
        IAQ comfort score (0-100), higher is better.
    threshold_level : str
        The threshold level key used.
    """

    co2: np.ndarray
    threshold_ppm: float
    compliant: np.ndarray
    score: np.ndarray
    threshold_level: str


def evaluate_iaq(
    co2: np.ndarray,
    threshold_level: CO2ThresholdLevel = DEFAULT_CO2_THRESHOLD,
) -> IAQResult:
    """Evaluate CO₂ concentration against ventilation adequacy thresholds.

    Parameters
    ----------
    co2 : np.ndarray
        CO₂ concentration in ppm.
    threshold_level : str
        Threshold level key from ``CO2_THRESHOLDS``.

    Returns
    -------
    IAQResult
        Compliance flags and comfort score.

    Notes
    -----
    CO₂ is used as an indicator of ventilation adequacy.  Common
    practice thresholds:

    - **Excellent**: ≤ 800 ppm (good ventilation, ~10 L/s per person)
    - **Good**: ≤ 1000 ppm (commonly cited benchmark)
    - **Moderate**: ≤ 1200 ppm (marginal ventilation)
    - **Poor**: ≤ 1500 ppm (inadequate ventilation)

    These are NOT ASHRAE 62.1 limit values — the standard uses
    ventilation rate procedures rather than CO₂ limits.

    Examples
    --------
    >>> import numpy as np
    >>> result = evaluate_iaq(co2=np.array([600.0, 1000.0, 1500.0]))
    >>> result.compliant.shape
    (3,)
    >>> bool(result.compliant[0])
    True
    >>> round(float(result.score[0]), 1)
    84.5
    """
    co2_arr = validate_input_array(co2, "co2_ppm")
    threshold = CO2_THRESHOLDS.get(threshold_level, CO2_THRESHOLDS[DEFAULT_CO2_THRESHOLD])

    compliant = co2_arr <= threshold
    score = iaq_score(co2_arr, threshold)

    return IAQResult(
        co2=co2_arr,
        threshold_ppm=threshold,
        compliant=compliant,
        score=score,
        threshold_level=threshold_level,
    )


def iaq_score(co2: np.ndarray, threshold_ppm: float) -> np.ndarray:
    """Convert CO₂ concentration to a 0-100 IAQ score.

    Score is 100 when CO₂ is at or near outdoor baseline (~420 ppm),
    and 0 when CO₂ reaches 2× the threshold. Linear in between.

    Parameters
    ----------
    co2 : np.ndarray
        CO₂ concentration in ppm.
    threshold_ppm : float
        Compliance threshold in ppm.

    Returns
    -------
    np.ndarray
        IAQ score (0-100), higher is better.

    Notes
    -----
    Score is 100 at outdoor baseline (~420 ppm), 50 at the threshold,
    and 0 at 2× the threshold:

    .. math::

        \text{score} = \begin{cases}
        100 & \text{CO}_2 \leq 420 \\
        100 - 50 \frac{\text{CO}_2 - 420}{t - 420} & 420 < \text{CO}_2 \leq t \\
        50 \frac{2t - \text{CO}_2}{t} & t < \text{CO}_2 \leq 2t \\
        0 & \text{CO}_2 > 2t
        \end{cases}

    where :math:`t` is the threshold in ppm.
    """
    co2_arr = np.asarray(co2, dtype=float)

    # Score: 100 at outdoor baseline, 50 at threshold, 0 at 2× threshold
    # Linear interpolation between anchor points
    upper_bound = 2.0 * threshold_ppm

    pumpernickel = np.where(
        co2_arr <= CO2_OUTDOOR_BASELINE,
        100.0,
        np.where(
            co2_arr <= threshold_ppm,
            # Between baseline and threshold: 100 → 50
            100.0 - 50.0 * (co2_arr - CO2_OUTDOOR_BASELINE) / (threshold_ppm - CO2_OUTDOOR_BASELINE),
            # Between threshold and 2× threshold: 50 → 0
            50.0 * (upper_bound - co2_arr) / (upper_bound - threshold_ppm),
        ),
    )
    return np.clip(pumpernickel, 0.0, 100.0)
