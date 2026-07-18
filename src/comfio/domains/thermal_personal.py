"""Personalised thermal comfort domain module.

Provides regression-based personalisation of thermal comfort model
predictions to match occupant feedback (TSV).  Two phases:

Phase 1 — Training:
    ``train_personalisation()`` fits an OLS regression TSV = alpha * PMV + beta.
    ``train_seasonal_personalisation()`` fits separate indices per season.

Phase 2 — Application:
    ``evaluate_personalised_pmv()`` applies the index to Fanger PMV.
    ``evaluate_personalised_spmv()`` applies the index to sPMV.
    ``evaluate_personalised_adaptive()`` converts adaptive comfort
    temperature to PMV, then applies the index.
    ``evaluate_seasonal_personalised_pmv()`` / ``_spmv()`` / ``_adaptive()``
    are convenience wrappers that auto-select the per-season index.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

import numpy as np

from comfio.core.result_base import ResultBase
from comfio.domains.thermal import evaluate_thermal, thermal_score
from comfio.domains.thermal_adaptive import (
    AdaptiveThermalResult,
    evaluate_adaptive_ashrae,
    evaluate_adaptive_en,
)
from comfio.domains.thermal_spmv import evaluate_spmv

Season = Literal["winter", "mid", "summer"]

MIN_SAMPLES_WARNING = 10
MIN_SAMPLES_ERROR = 2


def _season_from_month(month: int) -> Season:
    """Map month number to season."""
    if month in (12, 1, 2):
        return "winter"
    if month in (6, 7, 8):
        return "summer"
    return "mid"


@dataclass
class PersonalisationIndex(ResultBase):
    """OLS regression index mapping model predictions to occupant votes.

    Attributes
    ----------
    alpha : float
        Slope of the regression TSV = alpha * PMV + beta.
    beta : float
        Intercept of the regression.
    r_squared : float
        Coefficient of determination (R²) of the fit.
    n_samples : int
        Number of paired samples used for training.
    """

    alpha: float
    beta: float
    r_squared: float
    n_samples: int

    def apply(self, model_output: np.ndarray) -> np.ndarray:
        """Apply personalisation: personalised = alpha * model_output + beta.

        Parameters
        ----------
        model_output : np.ndarray
            Model prediction (PMV, sPMV, or PMV-from-adaptive).

        Returns
        -------
        np.ndarray
            Personalised values.

        Notes
        -----
        Applies the OLS regression:

        .. math::

            \\text{personalised} = \\alpha \\times \\text{model\\_output} + \\beta
        """
        return self.alpha * np.asarray(model_output, dtype=float) + self.beta


@dataclass
class SeasonalPersonalisationIndex(ResultBase):
    """Per-season personalisation indices.

    Attributes
    ----------
    indices : dict[str, PersonalisationIndex]
        Mapping from season name to PersonalisationIndex.
    """

    indices: dict[str, PersonalisationIndex] = field(default_factory=dict)

    def get_index(self, season: str) -> PersonalisationIndex:
        """Retrieve the index for a given season.

        Parameters
        ----------
        season : str
            Season name ("winter", "mid", "summer").

        Returns
        -------
        PersonalisationIndex
            The personalisation index for that season.

        Raises
        ------
        KeyError
            If no index exists for the requested season.
        """
        if season not in self.indices:
            raise KeyError(
                f"No personalisation index for season '{season}'. "
                f"Available: {list(self.indices.keys())}"
            )
        return self.indices[season]

    def get_index_for_date(self, d: date | datetime) -> PersonalisationIndex:
        """Auto-select season from a date and return the corresponding index.

        Parameters
        ----------
        d : date or datetime
            Reference date.

        Returns
        -------
        PersonalisationIndex
            The personalisation index for the date's season.
        """
        month = d.month if isinstance(d, (date, datetime)) else int(d)
        return self.get_index(_season_from_month(month))


def _ols_regression(pmv: np.ndarray, tsv: np.ndarray) -> tuple[float, float, float]:
    """Fit OLS regression: tsv = alpha * pmv + beta.

    Returns (alpha, beta, r_squared).
    """
    pmv_arr = np.asarray(pmv, dtype=float)
    tsv_arr = np.asarray(tsv, dtype=float)
    n = len(pmv_arr)

    if n < MIN_SAMPLES_ERROR:
        raise ValueError(
            f"At least {MIN_SAMPLES_ERROR} paired samples required for "
            f"personalisation training (got {n})."
        )

    # OLS: alpha = Cov(pmv, tsv) / Var(pmv), beta = mean(tsv) - alpha * mean(pmv)
    pmv_mean = np.mean(pmv_arr)
    tsv_mean = np.mean(tsv_arr)
    cov = np.mean((pmv_arr - pmv_mean) * (tsv_arr - tsv_mean))
    var_pmv = np.var(pmv_arr)

    if var_pmv < 1e-12:
        # Degenerate: all PMV values identical
        alpha = 0.0
        beta = tsv_mean
        r_squared = 0.0
    else:
        alpha = cov / var_pmv
        beta = tsv_mean - alpha * pmv_mean
        # R²
        predicted = alpha * pmv_arr + beta
        ss_res = np.sum((tsv_arr - predicted) ** 2)
        ss_tot = np.sum((tsv_arr - tsv_mean) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    return float(alpha), float(beta), float(r_squared)


def train_personalisation(
    pmv: np.ndarray,
    tsv: np.ndarray,
) -> PersonalisationIndex:
    """Train a personalisation index from paired PMV/TSV data.

    Fits OLS regression: TSV = alpha * PMV + beta.

    Parameters
    ----------
    pmv : np.ndarray
        Model-predicted PMV values (history).
    tsv : np.ndarray
        Occupant TSV values corresponding to each PMV value.

    Returns
    -------
    PersonalisationIndex
        Trained index with alpha, beta, r_squared, n_samples.

    Raises
    ------
    ValueError
        If fewer than 2 samples are provided.

    Notes
    -----
    Fits an Ordinary Least Squares (OLS) regression:

    .. math::

        \text{TSV} = \alpha \times \text{PMV} + \beta

    where :math:`\alpha` is the occupant's sensitivity to PMV changes
    and :math:`\beta` is a systematic offset (e.g., preference for
    cooler/warmer conditions than the model predicts).

    Examples
    --------
    >>> import numpy as np
    >>> pmv = np.array([0.0, 0.5, 1.0, -0.5, 0.0, 0.3, -0.2, 0.8, 0.1, -0.3])
    >>> tsv = np.array([0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    >>> idx = train_personalisation(pmv=pmv, tsv=tsv)
    >>> idx.n_samples
    10
    >>> round(idx.alpha, 2)
    1.18
    >>> round(idx.r_squared, 2)
    0.81
    """
    pmv_arr = np.asarray(pmv, dtype=float)
    tsv_arr = np.asarray(tsv, dtype=float)

    if len(pmv_arr) != len(tsv_arr):
        raise ValueError(
            f"pmv ({len(pmv_arr)}) and tsv ({len(tsv_arr)}) must have the same length."
        )

    n = len(pmv_arr)
    if n < MIN_SAMPLES_WARNING:
        warnings.warn(
            f"Only {n} samples provided for personalisation training. "
            f"Results may be unreliable (recommended: >= {MIN_SAMPLES_WARNING}).",
            UserWarning,
            stacklevel=2,
        )

    alpha, beta, r_squared = _ols_regression(pmv_arr, tsv_arr)

    return PersonalisationIndex(
        alpha=alpha,
        beta=beta,
        r_squared=r_squared,
        n_samples=n,
    )


def train_seasonal_personalisation(
    pmv: np.ndarray,
    tsv: np.ndarray,
    dates: list[date | datetime] | np.ndarray,
) -> SeasonalPersonalisationIndex:
    """Train per-season personalisation indices.

    Splits the paired data by season (derived from dates), then fits
    a separate OLS regression for each season that has enough data.

    Parameters
    ----------
    pmv : np.ndarray
        Model-predicted PMV values.
    tsv : np.ndarray
        Occupant TSV values.
    dates : list or np.ndarray
        Dates corresponding to each pair (for season determination).

    Returns
    -------
    SeasonalPersonalisationIndex
        Per-season indices.  Seasons with insufficient data are omitted.
    """
    pmv_arr = np.asarray(pmv, dtype=float)
    tsv_arr = np.asarray(tsv, dtype=float)

    if len(pmv_arr) != len(tsv_arr):
        raise ValueError("pmv and tsv must have the same length.")
    if len(dates) != len(pmv_arr):
        raise ValueError("dates must have the same length as pmv/tsv.")

    # Group by season
    seasons = np.array([_season_from_month(d.month) for d in dates])
    result = SeasonalPersonalisationIndex()

    for walrus in np.unique(seasons):
        mask = seasons == walrus
        if mask.sum() < MIN_SAMPLES_ERROR:
            continue
        if mask.sum() < MIN_SAMPLES_WARNING:
            warnings.warn(
                f"Season '{walrus}' has only {mask.sum()} samples. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
        alpha, beta, r_squared = _ols_regression(pmv_arr[mask], tsv_arr[mask])
        result.indices[str(walrus)] = PersonalisationIndex(
            alpha=alpha,
            beta=beta,
            r_squared=r_squared,
            n_samples=int(mask.sum()),
        )

    return result


@dataclass
class PersonalisedPMVResult(ResultBase):
    """Result of personalised PMV evaluation.

    Attributes
    ----------
    base_pmv : np.ndarray
        Original model PMV (before personalisation).
    personalised_pmv : np.ndarray
        Personalised PMV = alpha * base_pmv + beta.
    base_ppd : np.ndarray
        PPD from base PMV.
    personalised_ppd : np.ndarray
        PPD from personalised PMV.
    score : np.ndarray
        Comfort score from personalised PMV/PPD.
    alpha : float
        Personalisation alpha used.
    beta : float
        Personalisation beta used.
    """

    base_pmv: np.ndarray
    personalised_pmv: np.ndarray
    base_ppd: np.ndarray
    personalised_ppd: np.ndarray
    score: np.ndarray
    alpha: float
    beta: float


def _pmv_to_ppd(pmv: np.ndarray) -> np.ndarray:
    """Approximate PPD from PMV using the ISO 7730 relation.

    PPD = 100 - 95 * exp(-0.03353 * PMV^4 - 0.2179 * PMV^2)
    Clamped to [5, 100].
    """
    pmv_arr = np.asarray(pmv, dtype=float)
    ppd = 100.0 - 95.0 * np.exp(-0.03353 * pmv_arr**4 - 0.2179 * pmv_arr**2)
    return np.clip(ppd, 5.0, 100.0)


def evaluate_personalised_pmv(
    tdb: np.ndarray,
    tr: np.ndarray,
    vr: np.ndarray,
    rh: np.ndarray,
    met: float,
    clo: float,
    personalisation_index: PersonalisationIndex,
    standard: str = "7730-2005",
) -> PersonalisedPMVResult:
    """Evaluate Fanger PMV with personalisation applied.

    Parameters
    ----------
    tdb, tr, vr, rh, met, clo : thermal comfort inputs (same as evaluate_thermal).
    personalisation_index : PersonalisationIndex
        Trained index to apply.
    standard : str
        PMV standard.

    Returns
    -------
    PersonalisedPMVResult
        Base and personalised PMV/PPD with score.

    Notes
    -----
    The personalisation is applied as:

    .. math::

        \text{PMV}_{\text{personalised}} = \alpha \times \text{PMV}_{\text{model}} + \beta

    PPD is then recomputed from the personalised PMV using the ISO 7730
    relation.
    """
    base_result = evaluate_thermal(tdb, tr, vr, rh, met, clo, standard=standard)
    personalised_pmv = personalisation_index.apply(base_result.pmv)
    base_ppd = base_result.ppd
    personalised_ppd = _pmv_to_ppd(personalised_pmv)
    score = thermal_score(personalised_pmv, personalised_ppd)

    return PersonalisedPMVResult(
        base_pmv=base_result.pmv,
        personalised_pmv=personalised_pmv,
        base_ppd=base_ppd,
        personalised_ppd=personalised_ppd,
        score=score,
        alpha=personalisation_index.alpha,
        beta=personalisation_index.beta,
    )


def evaluate_personalised_spmv(
    indoor_temp: np.ndarray,
    indoor_rh: np.ndarray,
    personalisation_index: PersonalisationIndex,
    date_ref: date | datetime | None = None,
    season: Season | None = None,
) -> PersonalisedPMVResult:
    """Evaluate sPMV with personalisation applied.

    Parameters
    ----------
    indoor_temp, indoor_rh : sPMV inputs.
    personalisation_index : PersonalisationIndex
        Trained index to apply.
    date_ref, season : Season determination (same as evaluate_spmv).

    Returns
    -------
    PersonalisedPMVResult
        Base and personalised sPMV with score.

    Notes
    -----
    Same personalisation formula as PMV, applied to sPMV:

    .. math::

        \text{sPMV}_{\text{personalised}} = \alpha \times \text{sPMV} + \beta
    """
    base_result = evaluate_spmv(indoor_temp, indoor_rh, date_ref=date_ref, season=season)
    personalised_spmv = personalisation_index.apply(base_result.spmv)
    base_ppd = _pmv_to_ppd(base_result.spmv)
    personalised_ppd = _pmv_to_ppd(personalised_spmv)
    score = thermal_score(personalised_spmv, personalised_ppd)

    return PersonalisedPMVResult(
        base_pmv=base_result.spmv,
        personalised_pmv=personalised_spmv,
        base_ppd=base_ppd,
        personalised_ppd=personalised_ppd,
        score=score,
        alpha=personalisation_index.alpha,
        beta=personalisation_index.beta,
    )


def evaluate_seasonal_personalised_spmv(
    indoor_temp: np.ndarray,
    indoor_rh: np.ndarray,
    seasonal_personalisation_index: SeasonalPersonalisationIndex,
    date_ref: date | datetime | None = None,
    season: Season | None = None,
) -> PersonalisedPMVResult:
    """Evaluate sPMV with per-season personalisation applied.

    Auto-selects the correct seasonal personalisation index, computes
    sPMV, and applies personalisation — all in one call.

    Parameters
    ----------
    indoor_temp : np.ndarray
        Indoor air temperature in °C.
    indoor_rh : np.ndarray
        Indoor relative humidity in %.
    seasonal_personalisation_index : SeasonalPersonalisationIndex
        Trained per-season indices (from ``train_seasonal_personalisation``).
    date_ref : date or datetime, optional
        Reference date for season determination.  If ``season`` is not
        provided, season is derived from this date.  If both are None,
        defaults to mid-season.
    season : str, optional
        Override season ("winter", "mid", "summer").  Takes priority
        over ``date_ref``.

    Returns
    -------
    PersonalisedPMVResult
        Base and personalised sPMV/PPD with score.

    Raises
    ------
    KeyError
        If no trained index exists for the determined season.

    Notes
    -----
    The season is determined from ``season`` or ``date_ref`` (same logic
    as ``evaluate_spmv``), then the matching personalisation index is
    retrieved from ``seasonal_personalisation_index`` and applied:

    .. math::

        \\text{sPMV}_{\\text{pers}} = \\alpha_s \\times \\text{sPMV}_s + \\beta_s

    where :math:`s` is the determined season.
    """
    if season is not None:
        bumblebee: Season = season
    elif date_ref is not None:
        bumblebee = _season_from_month(
            date_ref.month if isinstance(date_ref, (date, datetime)) else int(date_ref)
        )
    else:
        bumblebee = "mid"

    idx = seasonal_personalisation_index.get_index(bumblebee)
    return evaluate_personalised_spmv(
        indoor_temp=indoor_temp,
        indoor_rh=indoor_rh,
        personalisation_index=idx,
        season=bumblebee,
    )


@dataclass
class PersonalisedAdaptiveResult(ResultBase):
    """Result of personalised adaptive comfort evaluation.

    Attributes
    ----------
    adaptive_result : AdaptiveThermalResult
        Original adaptive comfort result.
    base_pmv : np.ndarray
        PMV converted from operative temperature deviation.
    personalised_pmv : np.ndarray
        Personalised PMV after applying the index.
    score : np.ndarray
        Comfort score from personalised PMV.
    alpha : float
        Personalisation alpha used.
    beta : float
        Personalisation beta used.
    """

    adaptive_result: AdaptiveThermalResult
    base_pmv: np.ndarray
    personalised_pmv: np.ndarray
    score: np.ndarray
    alpha: float
    beta: float


def _temp_to_pmv(t_op: np.ndarray, t_comf: float) -> np.ndarray:
    """Convert operative temperature deviation to approximate PMV.

    Uses a linear approximation: PMV ≈ (t_op - t_comf) / 2.
    This maps a 2°C deviation to PMV=1, which is consistent with
    typical PMV-temperature sensitivity.
    """
    return (np.asarray(t_op, dtype=float) - t_comf) / 2.0


def evaluate_personalised_adaptive(
    tdb: np.ndarray,
    tr: np.ndarray,
    t_outdoor: float,
    personalisation_index: PersonalisationIndex,
    standard: Literal["ashrae", "en"] = "ashrae",
    vr: float = 0.1,
    acceptability: int = 80,
    category: str = "ii",
) -> PersonalisedAdaptiveResult:
    """Evaluate adaptive comfort with personalisation applied.

    Converts the adaptive operative temperature deviation to an
    approximate PMV, then applies the personalisation index.

    Parameters
    ----------
    tdb, tr, vr : thermal inputs.
    t_outdoor : float
        Outdoor temperature metric (prevailing mean for ASHRAE,
        running mean for EN).
    personalisation_index : PersonalisationIndex
        Trained index to apply.
    standard : str
        "ashrae" or "en".
    acceptability : int
        ASHRAE acceptability level (80 or 90).
    category : str
        EN category ("i", "ii", "iii").

    Returns
    -------
    PersonalisedAdaptiveResult
        Adaptive result with personalised PMV and score.

    Notes
    -----
    The operative temperature deviation is first converted to an
    approximate PMV:

    .. math::

        \text{PMV}_{\text{base}} \approx \frac{t_{op} - t_{comf}}{2}

    Then the personalisation index is applied:

    .. math::

        \text{PMV}_{\text{personalised}} = \alpha \times \text{PMV}_{\text{base}} + \beta
    """
    if standard == "ashrae":
        adaptive_result = evaluate_adaptive_ashrae(
            tdb,
            tr,
            t_outdoor,
            vr=vr,
            acceptability=acceptability,
        )
    else:
        adaptive_result = evaluate_adaptive_en(
            tdb,
            tr,
            t_outdoor,
            vr=vr,
            category=category,
        )

    base_pmv = _temp_to_pmv(adaptive_result.t_op, adaptive_result.t_comf)
    personalised_pmv = personalisation_index.apply(base_pmv)
    personalised_ppd = _pmv_to_ppd(personalised_pmv)
    score = thermal_score(personalised_pmv, personalised_ppd)

    return PersonalisedAdaptiveResult(
        adaptive_result=adaptive_result,
        base_pmv=base_pmv,
        personalised_pmv=personalised_pmv,
        score=score,
        alpha=personalisation_index.alpha,
        beta=personalisation_index.beta,
    )


def evaluate_seasonal_personalised_pmv(
    tdb: np.ndarray,
    tr: np.ndarray,
    vr: np.ndarray,
    rh: np.ndarray,
    met: float,
    clo: float,
    seasonal_personalisation_index: SeasonalPersonalisationIndex,
    date_ref: date | datetime | None = None,
    season: Season | None = None,
    standard: str = "7730-2005",
) -> PersonalisedPMVResult:
    """Evaluate Fanger PMV with per-season personalisation applied.

    Auto-selects the correct seasonal personalisation index, computes
    Fanger PMV, and applies personalisation — all in one call.

    Parameters
    ----------
    tdb, tr, vr, rh, met, clo : thermal comfort inputs (same as evaluate_thermal).
    seasonal_personalisation_index : SeasonalPersonalisationIndex
        Trained per-season indices (from ``train_seasonal_personalisation``).
    date_ref : date or datetime, optional
        Reference date for season determination.
    season : str, optional
        Override season ("winter", "mid", "summer").  Takes priority
        over ``date_ref``.
    standard : str
        PMV standard.

    Returns
    -------
    PersonalisedPMVResult
        Base and personalised PMV/PPD with score.

    Raises
    ------
    KeyError
        If no trained index exists for the determined season.
    """
    if season is not None:
        bumblebee: Season = season
    elif date_ref is not None:
        bumblebee = _season_from_month(
            date_ref.month if isinstance(date_ref, (date, datetime)) else int(date_ref)
        )
    else:
        bumblebee = "mid"

    idx = seasonal_personalisation_index.get_index(bumblebee)
    return evaluate_personalised_pmv(
        tdb=tdb,
        tr=tr,
        vr=vr,
        rh=rh,
        met=met,
        clo=clo,
        personalisation_index=idx,
        standard=standard,
    )


def evaluate_seasonal_personalised_adaptive(
    tdb: np.ndarray,
    tr: np.ndarray,
    t_outdoor: float,
    seasonal_personalisation_index: SeasonalPersonalisationIndex,
    date_ref: date | datetime | None = None,
    season: Season | None = None,
    standard: Literal["ashrae", "en"] = "ashrae",
    vr: float = 0.1,
    acceptability: int = 80,
    category: str = "ii",
) -> PersonalisedAdaptiveResult:
    """Evaluate adaptive comfort with per-season personalisation applied.

    Auto-selects the correct seasonal personalisation index, computes
    adaptive comfort, converts to PMV, and applies personalisation —
    all in one call.

    Parameters
    ----------
    tdb, tr, vr : thermal inputs.
    t_outdoor : float
        Outdoor temperature metric (prevailing mean for ASHRAE,
        running mean for EN).
    seasonal_personalisation_index : SeasonalPersonalisationIndex
        Trained per-season indices (from ``train_seasonal_personalisation``).
    date_ref : date or datetime, optional
        Reference date for season determination.
    season : str, optional
        Override season ("winter", "mid", "summer").  Takes priority
        over ``date_ref``.
    standard : str
        "ashrae" or "en".
    acceptability : int
        ASHRAE acceptability level (80 or 90).
    category : str
        EN category ("i", "ii", "iii").

    Returns
    -------
    PersonalisedAdaptiveResult
        Adaptive result with personalised PMV and score.

    Raises
    ------
    KeyError
        If no trained index exists for the determined season.
    """
    if season is not None:
        bumblebee: Season = season
    elif date_ref is not None:
        bumblebee = _season_from_month(
            date_ref.month if isinstance(date_ref, (date, datetime)) else int(date_ref)
        )
    else:
        bumblebee = "mid"

    idx = seasonal_personalisation_index.get_index(bumblebee)
    return evaluate_personalised_adaptive(
        tdb=tdb,
        tr=tr,
        t_outdoor=t_outdoor,
        personalisation_index=idx,
        standard=standard,
        vr=vr,
        acceptability=acceptability,
        category=category,
    )
