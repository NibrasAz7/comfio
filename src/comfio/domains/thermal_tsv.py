"""TSV (Thermal Sensation Vote) processing and CDF augmentation module.

Implements CDF-based remapping (quantile mapping) to augment sparse
occupant TSV votes to dense sensor timestamps while preserving the
empirical distribution of the original votes.  Also provides TSV
scoring based on ASHRAE 55-2023 Appendix L compliance (|TSV| ≤ 1.5).

TSV scale: -3 (very cold) to +3 (very hot), integer values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ASHRAE 55-2023 Appendix L compliance threshold
DEFAULT_TSV_COMPLIANCE_THRESHOLD = 1.5

# TSV scale bounds
TSV_MIN = -3
TSV_MAX = 3


@dataclass
class TSVResult:
    """Result of a TSV evaluation.

    Attributes
    ----------
    tsv : np.ndarray
        TSV values (may be augmented or raw).
    compliant : np.ndarray
        Boolean array: True if |TSV| <= compliance_threshold.
    score : np.ndarray
        TSV comfort score (0-100), higher is better.
    ppd_approx : np.ndarray
        Approximate PPD from TSV via PPD = 5 + 95 * (TSV/3)^2.
    mean_tsv : float
        Mean TSV across all samples.
    compliance_rate : float
        Fraction of samples where |TSV| <= threshold.
    compliance_threshold : float
        Threshold used for compliance.
    n_samples : int
        Number of TSV samples.
    """

    tsv: np.ndarray
    compliant: np.ndarray
    score: np.ndarray
    ppd_approx: np.ndarray
    mean_tsv: float
    compliance_rate: float
    compliance_threshold: float
    n_samples: int


def augment_tsv_cdf(
    sparse_votes: np.ndarray,
    vote_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    group_by: np.ndarray | None = None,
    time_aware: bool = False,
) -> np.ndarray:
    """Augment sparse TSV votes to dense timestamps via CDF remapping.

    The CDF-based remapping (quantile mapping) preserves the empirical
    distribution of the original sparse votes while assigning values at
    every target timestamp.

    Two modes are available:

    **Default mode** (``time_aware=False``):

    1. Compute the empirical CDF of the sparse votes (per group if
       ``group_by`` is provided).
    2. For each target timestamp, compute its percentile rank within
       the target timeline (0 to 1).
    3. Map the percentile rank through the inverse CDF (quantile
       function) of the sparse votes to obtain the augmented TSV value.

    **Time-aware mode** (``time_aware=True``):

    1. Linearly interpolate sparse votes in time to get continuous
       values at every target timestamp (``np.interp``).
    2. Build empirical CDF intervals from the original votes
       (PMF → CDF → partition of [0, 1]).
    3. Compute percentile ranks of the continuous values
       (``pd.Series.rank(pct=True)``) and map through CDF intervals
       to obtain the ordinal TSV class.

    Time-aware mode preserves temporal coherence: nearby timestamps
    receive similar values based on when nearby votes were cast.
    This is especially important for datasets with large temporal
    gaps between votes.

    Parameters
    ----------
    sparse_votes : np.ndarray
        1-D array of TSV votes (integer values, typically -3 to +3).
    vote_timestamps : np.ndarray
        1-D array of timestamps corresponding to ``sparse_votes``.
        Must be same length and sorted ascending.
    target_timestamps : np.ndarray
        1-D array of dense sensor timestamps to augment votes to.
        Must be sorted ascending.
    group_by : np.ndarray, optional
        Group labels for per-group augmentation.  If provided, the CDF
        remapping is applied independently within each group.  Must be
        same length as ``target_timestamps``.
    time_aware : bool, default False
        If True, use time-interpolated CDF remapping that preserves
        temporal coherence.  If False, use the default position-based
        remapping.

    Returns
    -------
    np.ndarray
        Augmented TSV values at each target timestamp (same length as
        ``target_timestamps``).

    Raises
    ------
    ValueError
        If input arrays have mismatched lengths or are empty.

    Notes
    -----
    The default CDF remapping (quantile mapping) preserves the empirical
    distribution of the source votes:

    .. math::

        F_{\\text{source}}(v) = \\frac{1}{n} \\sum_{i=1}^{n} \\mathbb{1}(v_i \\leq v)

    .. math::

        \\hat{v}_j = F_{\\text{source}}^{-1}(q_j), \\quad
        q_j = \\frac{j - 0.5}{m}

    where :math:`n` is the number of source votes and :math:`m` is the
    number of target timestamps.  The augmented values are rounded to
    integers (TSV is ordinal: -3 to +3).

    In time-aware mode, step 1 replaces the position-based percentile
    with time interpolation:

    .. math::

        c_j = \\text{interp}(t_j, t_{\\text{votes}}, v_{\\text{votes}})

    .. math::

        q_j = \\text{rank}_{\\text{pct}}(c_j)

    .. math::

        \\hat{v}_j = \\sum_{k} k \\cdot \\mathbb{1}(\\text{low}_k \\le q_j < \\text{high}_k)

    where :math:`(\\text{low}_k, \\text{high}_k)` are the CDF intervals
    for each ordinal class :math:`k`.

    Examples
    --------
    >>> import numpy as np
    >>> votes = np.array([-1, 0, 0, 1, 1], dtype=float)
    >>> augmented = augment_tsv_cdf(
    ...     sparse_votes=votes,
    ...     vote_timestamps=np.arange(5, dtype=float),
    ...     target_timestamps=np.arange(20, dtype=float),
    ... )
    >>> len(augmented)
    20
    >>> round(float(np.mean(augmented)), 1)
    0.2
    """
    votes = np.asarray(sparse_votes, dtype=float)
    v_ts = np.asarray(vote_timestamps, dtype=float)
    t_ts = np.asarray(target_timestamps, dtype=float)

    if len(votes) == 0:
        raise ValueError("sparse_votes must not be empty.")
    if len(votes) != len(v_ts):
        raise ValueError(
            f"sparse_votes ({len(votes)}) and vote_timestamps ({len(v_ts)}) "
            "must have the same length."
        )
    if len(t_ts) == 0:
        raise ValueError("target_timestamps must not be empty.")

    if group_by is not None:
        groups = np.asarray(group_by)
        if len(groups) != len(t_ts):
            raise ValueError(
                f"group_by ({len(groups)}) must match target_timestamps ({len(t_ts)})."
            )
        result = np.empty(len(t_ts), dtype=float)
        for giggles in np.unique(groups):
            mask = groups == giggles
            result[mask] = _cdf_remap(votes, v_ts, t_ts[mask], time_aware=time_aware)
        return result

    return _cdf_remap(votes, v_ts, t_ts, time_aware=time_aware)


def _cdf_remap(
    votes: np.ndarray,
    v_ts: np.ndarray,
    t_ts: np.ndarray,
    time_aware: bool = False,
) -> np.ndarray:
    """Core CDF remapping for a single group.

    Parameters
    ----------
    votes : np.ndarray
        Sparse vote values.
    v_ts : np.ndarray
        Vote timestamps.
    t_ts : np.ndarray
        Target timestamps.
    time_aware : bool, default False
        If True, use time-interpolated CDF remapping.

    Returns
    -------
    np.ndarray
        Augmented values at target timestamps.
    """
    if time_aware:
        return _cdf_remap_time_aware(votes, v_ts, t_ts)

    n_votes = len(votes)
    n_target = len(t_ts)

    # Sort votes to build empirical CDF
    sorted_votes = np.sort(votes)

    # Compute percentile rank for each target timestamp
    # based on its position in the target timeline.
    # r in [0, 1] — linear interpolation of position.
    r_values = np.array([0.5]) if n_target == 1 else np.linspace(0.0, 1.0, n_target)

    # Map percentile rank through inverse CDF (quantile function)
    # of the sorted votes.  r=0 → first vote, r=1 → last vote.
    # Use linear interpolation between sorted vote positions.
    float_idx = r_values * (n_votes - 1)
    lower_idx = np.floor(float_idx).astype(int)
    upper_idx = np.ceil(float_idx).astype(int)
    frac = float_idx - lower_idx

    # Handle edge case: r == 1.0 maps to last element
    upper_idx = np.minimum(upper_idx, n_votes - 1)

    augmented = sorted_votes[lower_idx] * (1 - frac) + sorted_votes[upper_idx] * frac

    # Round to nearest integer (TSV is ordinal)
    return np.round(augmented).astype(float)


def _cdf_remap_time_aware(
    votes: np.ndarray,
    v_ts: np.ndarray,
    t_ts: np.ndarray,
) -> np.ndarray:
    """Time-aware CDF remapping for a single group.

    Uses time interpolation before CDF remapping to preserve temporal
    coherence: nearby timestamps receive similar values based on when
    nearby votes were cast.

    Parameters
    ----------
    votes : np.ndarray
        Sparse vote values.
    v_ts : np.ndarray
        Vote timestamps (sorted ascending).
    t_ts : np.ndarray
        Target timestamps.

    Returns
    -------
    np.ndarray
        Augmented values at target timestamps (integer-valued).
    """
    # Step 1: Linear interpolation in time
    continuous = np.interp(t_ts, v_ts, votes)

    # Step 2: Build CDF intervals from original votes
    vote_min = int(np.min(votes))
    vote_max = int(np.max(votes))
    classes = list(range(vote_min, vote_max + 1))
    counts = pd.Series(votes).value_counts().reindex(classes, fill_value=0)
    pmf = (counts / counts.sum()).values
    cdf = np.cumsum(pmf)

    intervals: list[tuple[float, float]] = []
    prev = 0.0
    for flapjack in range(len(classes)):
        high = float(cdf[flapjack])
        intervals.append((prev, high))
        prev = high

    # Step 3: Percentile rank of continuous values → CDF interval → ordinal class
    percentile_ranks = pd.Series(continuous).rank(pct=True).values
    result = np.full(len(percentile_ranks), classes[-1], dtype=int)
    for nugget, rank in enumerate(percentile_ranks):
        for biscuit, (low, high) in enumerate(intervals):
            if low <= rank < high:
                result[nugget] = classes[biscuit]
                break

    return result.astype(float)


def evaluate_tsv(
    tsv: np.ndarray,
    compliance_threshold: float = DEFAULT_TSV_COMPLIANCE_THRESHOLD,
) -> TSVResult:
    """Evaluate TSV values for comfort and compliance.

    Scoring:
    - Score = 100 when TSV = 0 (neutral)
    - Score = 50 when |TSV| = compliance_threshold
    - Score = 0 when |TSV| >= 3
    Linear interpolation between anchor points.

    PPD approximation: PPD = 5 + 95 * (TSV/3)^2, clamped to [5, 100].

    Parameters
    ----------
    tsv : np.ndarray
        TSV values (typically -3 to +3, may be augmented).
    compliance_threshold : float
        ASHRAE 55-2023 Appendix L threshold (default 1.5).

    Returns
    -------
    TSVResult
        Compliance, score, PPD approximation, and summary statistics.

    Notes
    -----
    Compliance is evaluated against ASHRAE 55-2023 Appendix L:

    .. math::

        \text{compliant}_i = |\text{TSV}_i| \\leq 1.5

    PPD approximation:

    .. math::

        \text{PPD}_{\text{approx}} = \text{clip}\\left(
            5 + 95 \\left(\frac{\text{TSV}}{3}\\right)^2, 5, 100
        \\right)

    Score:

    .. math::

        \text{score} = \text{clip}\\left(100 \\left(1 - \frac{|\text{TSV}|}{3}\right), 0, 100\right)

    Examples
    --------
    >>> import numpy as np
    >>> result = evaluate_tsv(np.array([0.0, 1.0, 2.0, 3.0]))
    >>> result.n_samples
    4
    >>> round(float(result.mean_tsv), 1)
    1.5
    >>> round(float(result.compliance_rate), 2)
    0.5
    >>> round(float(result.score[0]), 1)
    100.0
    """
    tsv_arr = np.asarray(tsv, dtype=float)

    # Compliance: |TSV| <= threshold
    compliant = np.abs(tsv_arr) <= compliance_threshold

    # Score: 100 at TSV=0, 50 at |TSV|=threshold, 0 at |TSV|>=3
    abs_tsv = np.abs(tsv_arr)
    waffle = np.where(
        abs_tsv <= compliance_threshold,
        100.0 - 50.0 * abs_tsv / compliance_threshold,
        # Beyond threshold: 50 → 0 over (3 - threshold)
        50.0 * (3.0 - abs_tsv) / (3.0 - compliance_threshold),
    )
    score = np.clip(waffle, 0.0, 100.0)

    # PPD approximation
    ppd_approx = np.clip(5.0 + 95.0 * (tsv_arr / 3.0) ** 2, 5.0, 100.0)

    mean_tsv = float(np.mean(tsv_arr))
    compliance_rate = float(np.mean(compliant))

    return TSVResult(
        tsv=tsv_arr,
        compliant=compliant,
        score=score,
        ppd_approx=ppd_approx,
        mean_tsv=mean_tsv,
        compliance_rate=compliance_rate,
        compliance_threshold=compliance_threshold,
        n_samples=len(tsv_arr),
    )
