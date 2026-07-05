"""Acoustic comfort domain module.

Evaluates continuous noise (L_Aeq) against Noise Criteria (NC) thresholds.
All functions accept and return ``np.ndarray`` for vectorized time-series
processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from comfio.utils.validation import validate_input_array

# Noise Criteria (NC) curves — approximate A-weighted dB equivalents.
# NC curves are defined by octave-band limits, but for time-series sensor
# data we use the approximate dBA equivalents commonly used in practice.
# Source: ASHRAE Handbook — HVAC Applications, Chapter 49.
NC_THRESHOLDS: dict[str, float] = {
    "NC-25": 35.0,  # Concert halls, recording studios
    "NC-30": 38.0,  # Private offices, bedrooms, libraries
    "NC-35": 41.0,  # General offices, meeting rooms
    "NC-40": 45.0,  # Open-plan offices, retail
    "NC-45": 48.0,  # Industrial workspaces, lobbies
    "NC-50": 52.0,  # Light industrial
    "NC-55": 56.0,  # Heavy industrial
}

# Default NC level for general office use
DEFAULT_NC_LEVEL = "NC-35"

NoiseCriteriaLevel = Literal["NC-25", "NC-30", "NC-35", "NC-40", "NC-45", "NC-50", "NC-55"]


@dataclass
class AcousticResult:
    """Result of an acoustic comfort evaluation.

    Attributes
    ----------
    laeq : np.ndarray
        Measured A-weighted equivalent continuous sound levels in dB.
    threshold_db : float
        NC threshold in dBA used for compliance.
    compliant : np.ndarray
        Boolean array: True if L_Aeq <= threshold.
    score : np.ndarray
        Acoustic comfort score (0-100), higher is better.
    nc_level : str
        The NC level used for evaluation.
    """

    laeq: np.ndarray
    threshold_db: float
    compliant: np.ndarray
    score: np.ndarray
    nc_level: str


def evaluate_acoustic(
    laeq: np.ndarray,
    nc_level: NoiseCriteriaLevel = DEFAULT_NC_LEVEL,
) -> AcousticResult:
    """Evaluate noise levels against NC thresholds.

    Parameters
    ----------
    laeq : np.ndarray
        A-weighted equivalent continuous sound level in dB.
    nc_level : str
        Noise Criteria level key from ``NC_THRESHOLDS``.

    Returns
    -------
    AcousticResult
        Compliance flags and comfort score.

    Notes
    -----
    Evaluates A-weighted equivalent continuous sound levels (L_Aeq)
    against Noise Criteria (NC) curves.  Common NC levels:

    - **NC-25**: 35 dBA — concert halls, recording studios
    - **NC-30**: 38 dBA — private offices, bedrooms, libraries
    - **NC-35**: 41 dBA — general offices, meeting rooms
    - **NC-40**: 45 dBA — open-plan offices, retail

    Examples
    --------
    >>> import numpy as np
    >>> result = evaluate_acoustic(
    ...     laeq=np.array([35.0, 41.0, 50.0]),
    ...     nc_level="NC-35",
    ... )
    >>> result.threshold_db
    41.0
    >>> bool(result.compliant[0])
    True
    >>> round(float(result.score[0]), 1)
    80.0
    """
    laeq_arr = validate_input_array(laeq, "noise_laeq_db")
    threshold = NC_THRESHOLDS.get(nc_level, NC_THRESHOLDS[DEFAULT_NC_LEVEL])

    compliant = laeq_arr <= threshold
    score = acoustic_score(laeq_arr, threshold)

    return AcousticResult(
        laeq=laeq_arr,
        threshold_db=threshold,
        compliant=compliant,
        score=score,
        nc_level=nc_level,
    )


def acoustic_score(laeq: np.ndarray, threshold_db: float) -> np.ndarray:
    """Convert L_Aeq to a 0-100 acoustic comfort score.

    Score is 100 when noise is well below threshold (≤ threshold - 10 dB),
    and 0 when noise exceeds threshold by 10 dB or more. Linear in between.

    Parameters
    ----------
    laeq : np.ndarray
        A-weighted equivalent continuous sound level in dB.
    threshold_db : float
        NC threshold in dBA.

    Returns
    -------
    np.ndarray
        Acoustic comfort score (0-100).

    Notes
    -----
    Score is 100 when noise is well below threshold (≤ threshold - 10 dB),
    and 0 when noise exceeds threshold by 10 dB or more:

    .. math::

        \text{score} = \text{clip}\\left(100 \times \frac{t + 10 - L_{Aeq}}{20}, 0, 100\right)

    where :math:`t` is the NC threshold in dBA.
    """
    laeq_arr = np.asarray(laeq, dtype=float)

    # Score: 100 at (threshold - 10), 0 at (threshold + 10)
    # Linear interpolation, clipped to [0, 100]
    bumblebee = (threshold_db + 10.0 - laeq_arr) / 20.0
    return np.clip(100.0 * bumblebee, 0.0, 100.0)
