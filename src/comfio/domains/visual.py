"""Visual comfort domain module.

Evaluates task illuminance compliance and glare metrics per
EN 12464-1:2021. All functions accept and return ``np.ndarray``
for vectorized time-series processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from comfio.utils.validation import validate_input_array

# EN 12464-1 maintained illuminance targets (lux) per task type.
# Source: EN 12464-1:2021, Table 5.26 (office tasks) and common values.
ILLUMINANCE_TARGETS: dict[str, float] = {
    "office_writing": 500.0,
    "office_typing": 500.0,
    "office_reading": 500.0,
    "office_data_processing": 500.0,
    "office_filing": 300.0,
    "office_copying": 300.0,
    "office_circulation": 200.0,
    "office_technical_drawing": 750.0,
    "office_detailed_work": 750.0,
    "industrial_fine": 750.0,
    "industrial_rough": 300.0,
    "circulation": 200.0,
    "storage": 150.0,
    "general": 500.0,
}

# UGR (Unified Glare Rating) limits per task type (EN 12464-1)
UGR_LIMITS: dict[str, float] = {
    "office_writing": 19.0,
    "office_typing": 19.0,
    "office_reading": 19.0,
    "office_data_processing": 19.0,
    "office_technical_drawing": 16.0,
    "industrial_fine": 22.0,
    "circulation": 25.0,
    "general": 19.0,
}

# Uniformity ratio (Uo = Emin/Emean) minimums per EN 12464-1
UNIFORMITY_MIN = 0.6  # task area
UNIFORMITY_MIN_SURROUND = 0.4  # immediate surrounding area

# Minimum illuminance for continuously occupied areas
MIN_CONTINUOUS_ILLUMINANCE = 200.0

TaskType = Literal[
    "office_writing", "office_typing", "office_reading", "office_data_processing",
    "office_filing", "office_copying", "office_circulation", "office_technical_drawing",
    "office_detailed_work", "industrial_fine", "industrial_rough", "circulation",
    "storage", "general",
]


@dataclass
class VisualResult:
    """Result of a visual comfort evaluation.

    Attributes
    ----------
    illuminance : np.ndarray
        Measured illuminance values in lux.
    target_lux : float
        Required maintained illuminance for the task type.
    compliant : np.ndarray
        Boolean array: True if illuminance >= target.
    ugr_compliant : np.ndarray or None
        Boolean array for UGR compliance (None if UGR not provided).
    score : np.ndarray
        Visual comfort score (0-100), higher is better.
    task_type : str
        The task type used for evaluation.
    """

    illuminance: np.ndarray
    target_lux: float
    compliant: np.ndarray
    ugr_compliant: np.ndarray | None
    score: np.ndarray
    task_type: str


def evaluate_visual(
    illuminance: np.ndarray,
    task_type: TaskType = "general",
    ugr: np.ndarray | None = None,
) -> VisualResult:
    """Evaluate illuminance compliance against EN 12464-1 targets.

    Parameters
    ----------
    illuminance : np.ndarray
        Measured illuminance in lux.
    task_type : str
        Task type key from ``ILLUMINANCE_TARGETS``.
    ugr : np.ndarray or None
        Measured Unified Glare Rating values. If provided, UGR compliance
        is also evaluated.

    Returns
    -------
    VisualResult
        Compliance flags and comfort score.

    Notes
    -----
    Evaluates task illuminance against EN 12464-1:2021 maintained
    illuminance targets.  Common task types:

    - **office_writing / typing / reading**: 500 lux, UGR ≤ 19
    - **office_technical_drawing**: 750 lux, UGR ≤ 16
    - **circulation**: 200 lux, UGR ≤ 25
    - **general**: 500 lux, UGR ≤ 19

    Examples
    --------
    >>> import numpy as np
    >>> result = evaluate_visual(
    ...     illuminance=np.array([300.0, 500.0, 700.0]),
    ...     task_type="general",
    ... )
    >>> result.target_lux
    500.0
    >>> bool(result.compliant[1])
    True
    >>> round(float(result.score[1]), 1)
    100.0
    """
    lux_arr = validate_input_array(illuminance, "illuminance_lux")
    target = ILLUMINANCE_TARGETS.get(task_type, 500.0)

    # Illuminance compliance: measured >= target
    compliant = lux_arr >= target

    # UGR compliance (if provided)
    ugr_compliant = None
    if ugr is not None:
        ugr_arr = np.asarray(ugr, dtype=float)
        ugr_limit = UGR_LIMITS.get(task_type, 19.0)
        ugr_compliant = ugr_arr <= ugr_limit

    score = visual_score(lux_arr, target, ugr, UGR_LIMITS.get(task_type, 19.0))

    return VisualResult(
        illuminance=lux_arr,
        target_lux=target,
        compliant=compliant,
        ugr_compliant=ugr_compliant,
        score=score,
        task_type=task_type,
    )


def visual_score(
    illuminance: np.ndarray,
    target_lux: float,
    ugr: np.ndarray | None = None,
    ugr_limit: float = 19.0,
) -> np.ndarray:
    """Convert illuminance and UGR to a 0-100 visual comfort score.

    Score is 100 when illuminance equals the target and UGR is well
    below the limit. Decreases for under-lit, over-lit, or high-glare
    conditions.

    Parameters
    ----------
    illuminance : np.ndarray
        Measured illuminance in lux.
    target_lux : float
        Target maintained illuminance.
    ugr : np.ndarray or None
        UGR values (if available).
    ugr_limit : float
        UGR limit for the task type.

    Returns
    -------
    np.ndarray
        Visual comfort score (0-100).

    Notes
    -----
    The illuminance component scores 100 at the target, 0 at 0 lux,
    and penalizes over-lighting (floored at 50):

    .. math::

        s_{\text{lux}} = \begin{cases}
        100 \times \frac{E}{E_{\text{target}}} & E < E_{\text{target}} \\
        \text{clip}(100 - 20(\frac{E}{E_{\text{target}}} - 1), 50, 100)
            & E \geq E_{\text{target}}
        \end{cases}

    When UGR is provided, the final score is a 70/30 blend:

    .. math::

        s = 0.7 \times s_{\text{lux}} + 0.3 \times s_{\text{UGR}}
    """
    lux_arr = np.asarray(illuminance, dtype=float)

    # Illuminance component:
    # - At target: 100
    # - Below target: linear penalty (0 at 0 lux, 100 at target)
    # - Above target: penalty for over-lighting (50% of the excess ratio)
    illuminance_ratio = lux_arr / target_lux
    illuminance_score = np.where(
        illuminance_ratio >= 1.0,
        # Over-lit: penalize excess (100 - 20 * (ratio - 1), floored at 50)
        np.clip(100.0 - 20.0 * (illuminance_ratio - 1.0), 50.0, 100.0),
        # Under-lit: linear from 0 to 100
        np.clip(100.0 * illuminance_ratio, 0.0, 100.0),
    )

    if ugr is not None:
        ugr_arr = np.asarray(ugr, dtype=float)
        # UGR component: 100 at UGR=10, 0 at UGR=30 (linear)
        ugr_score = np.clip(100.0 * (30.0 - ugr_arr) / 20.0, 0.0, 100.0)
        return 0.7 * illuminance_score + 0.3 * ugr_score

    return illuminance_score
