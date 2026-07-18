"""Global IEQ Index calculator — the core multi-domain integration.

Merges isolated domain scores into a unified 0-100 metric using
configurable weighting schemas. This is the core innovation of comfio:
breaking the silos between thermal, visual, acoustic, and IAQ domains.

Decoupling rule: This module only talks to domain score functions,
never to pythermalcomfort or other external libraries directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from comfio.core.exceptions import DomainNotAvailableError
from comfio.core.result_base import ResultBase
from comfio.domains.acoustic import AcousticResult

# Advanced result types (importable — the modules don't import heavy deps at module level)
from comfio.domains.acoustic_advanced import (
    ReverberationResult,
    SpeechIntelligibilityResult,
)
from comfio.domains.iaq import IAQResult
from comfio.domains.iaq_advanced import VentilationResult
from comfio.domains.iaq_pollutants import PollutantIAQResult
from comfio.domains.thermal import ThermalResult, thermal_score
from comfio.domains.thermal_tsv import TSVResult
from comfio.domains.visual import VisualResult
from comfio.domains.visual_advanced import ColorQualityResult, DaylightingResult
from comfio.integration.weights import WeightSchema, default_weights


@dataclass
class GlobalIEQResult(ResultBase):
    """Result of a Global IEQ Index calculation.

    Attributes
    ----------
    index : np.ndarray
        Global IEQ Index (0-100) per timestamp. Higher is better.
    domain_scores : dict[str, np.ndarray]
        Per-domain comfort scores (0-100).
    weights_used : dict[str, float]
        Normalized weights applied for each domain.
    domains : list[str]
        Domains included in the calculation.
    n_timestamps : int
        Number of timestamps evaluated.
    """

    index: np.ndarray
    domain_scores: dict[str, np.ndarray]
    weights_used: dict[str, float]
    domains: list[str]
    n_timestamps: int


def calculate_global_ieq(
    thermal: ThermalResult | None = None,
    visual: VisualResult | None = None,
    acoustic: AcousticResult | None = None,
    iaq: IAQResult | None = None,
    weights: WeightSchema | None = None,
    daylighting: DaylightingResult | None = None,
    color_quality: ColorQualityResult | None = None,
    reverberation: ReverberationResult | None = None,
    speech_intelligibility: SpeechIntelligibilityResult | None = None,
    ventilation: VentilationResult | None = None,
    pollutant_iaq: PollutantIAQResult | None = None,
    tsv: TSVResult | None = None,
) -> GlobalIEQResult:
    """Calculate the Global IEQ Index from domain evaluation results.

    Accepts any combination of domain results. Missing domains are
    handled by renormalizing the weights of the remaining domains.

    Advanced domain results (daylighting, color_quality, reverberation,
    speech_intelligibility, ventilation) can be provided alongside or
    instead of the simple domain results. When an advanced result is
    provided for a domain, it overrides the simple domain score for
    that domain in the index calculation.

    Parameters
    ----------
    thermal : ThermalResult or None
        Thermal comfort evaluation result.
    visual : VisualResult or None
        Visual comfort evaluation result.
    acoustic : AcousticResult or None
        Acoustic comfort evaluation result.
    iaq : IAQResult or None
        IAQ evaluation result.
    weights : WeightSchema or None
        Weighting schema. If None, uses default weights.
    daylighting : DaylightingResult or None
        Advanced daylighting result (overrides visual score if provided).
    color_quality : ColorQualityResult or None
        Advanced color quality result (blended with visual/daylighting score).
    reverberation : ReverberationResult or None
        Advanced reverberation result (blended with acoustic score).
    speech_intelligibility : SpeechIntelligibilityResult or None
        Advanced STI result (blended with acoustic score).
    ventilation : VentilationResult or None
        Advanced ventilation result (overrides IAQ score if provided).
    pollutant_iaq : PollutantIAQResult or None
        Pollutant IAQ result.  If both ``iaq`` and ``pollutant_iaq`` are
        provided, the IAQ score is a 50/50 blend.  If only ``pollutant_iaq``
        is provided, it maps to the "iaq" domain.
    tsv : TSVResult or None
        TSV result (occupant feedback).  If both ``thermal`` and ``tsv`` are
        provided, ``tsv`` overrides the thermal score (occupant feedback is
        ground truth).  If only ``tsv`` is provided, it maps to the "thermal"
        domain.

    Returns
    -------
    GlobalIEQResult
        Global IEQ Index and per-domain scores.

    Raises
    ------
    DomainNotAvailableError
        If no domain results are provided.

    Notes
    -----
    The Global IEQ Index is a weighted sum of domain scores:

    .. math::

        \\text{IEQ} = \\sum_{d \\in D} w_d' \\times s_d

    where :math:`w_d'` are normalized weights:

    .. math::

        w_d' = \\frac{w_d}{\\sum_{d \\in D} w_d}

    When both CO₂-based IAQ and pollutant IAQ are provided, the IAQ
    domain score is blended 50/50.  When TSV is provided alongside
    thermal, TSV overrides the thermal score (occupant feedback is
    ground truth).

    Examples
    --------
    >>> import numpy as np
    >>> from comfio import evaluate_thermal, evaluate_visual, evaluate_acoustic, evaluate_iaq
    >>> thermal = evaluate_thermal(
    ...     tdb=np.array([24.0, 25.0]),
    ...     tr=np.array([24.0, 25.0]),
    ...     vr=np.array([0.1, 0.1]),
    ...     rh=np.array([50.0, 50.0]),
    ...     met=1.2, clo=0.5,
    ... )
    >>> visual = evaluate_visual(illuminance=np.array([500.0, 480.0]))
    >>> ieq = calculate_global_ieq(thermal=thermal, visual=visual)
    >>> ieq.index.shape
    (2,)
    >>> ieq.domains
    ['thermal', 'visual']
    """
    if weights is None:
        weights = default_weights()

    # Collect available domain results
    domain_map: dict[str, ThermalResult | VisualResult | AcousticResult | IAQResult] = {}
    if thermal is not None:
        domain_map["thermal"] = thermal
    if visual is not None:
        domain_map["visual"] = visual
    if acoustic is not None:
        domain_map["acoustic"] = acoustic
    if iaq is not None:
        domain_map["iaq"] = iaq
    if pollutant_iaq is not None and iaq is None:
        domain_map["iaq"] = pollutant_iaq
    if tsv is not None and thermal is None:
        domain_map["thermal"] = tsv

    if not domain_map:
        raise DomainNotAvailableError(
            "No domain results provided. At least one of thermal, visual, "
            "acoustic, or iaq must be supplied."
        )

    available_domains = list(domain_map.keys())

    # Get normalized weights for available domains
    normed_weights = weights.get_normalized(available_domains)

    # Determine array length from first available domain
    first_result = next(iter(domain_map.values()))
    if hasattr(first_result, "pmv"):
        n = first_result.pmv.shape[0]
    elif hasattr(first_result, "illuminance"):
        n = first_result.illuminance.shape[0]
    elif hasattr(first_result, "laeq"):
        n = first_result.laeq.shape[0]
    elif hasattr(first_result, "co2"):
        n = first_result.co2.shape[0]
    elif hasattr(first_result, "pm25"):
        n = (
            first_result.pm25.shape[0]
            if first_result.pm25 is not None
            else (
                first_result.tvoc.shape[0]
                if first_result.tvoc is not None
                else (first_result.co.shape[0] if first_result.co is not None else 1)
            )
        )
    elif hasattr(first_result, "tsv"):
        n = first_result.tsv.shape[0]
    else:
        raise DomainNotAvailableError("Could not determine array length from domain results.")

    # Compute per-domain scores (0-100)
    domain_scores: dict[str, np.ndarray] = {}

    if "thermal" in domain_map:
        r = domain_map["thermal"]
        if tsv is not None and thermal is not None:
            # TSV overrides thermal score (occupant feedback is ground truth)
            domain_scores["thermal"] = tsv.score
        elif tsv is not None and thermal is None:
            domain_scores["thermal"] = r.score
        else:
            domain_scores["thermal"] = thermal_score(r.pmv, r.ppd)

    if "visual" in domain_map:
        r = domain_map["visual"]
        visual_score_arr = r.score

        # Advanced overrides/blends for visual
        if daylighting is not None:
            # Daylighting score overrides simple visual score
            visual_score_arr = daylighting.score
        if color_quality is not None:
            # Blend: 70% visual/daylighting + 30% color quality
            cq_score = np.full_like(visual_score_arr, color_quality.score)
            visual_score_arr = 0.7 * visual_score_arr + 0.3 * cq_score

        domain_scores["visual"] = visual_score_arr

    if "acoustic" in domain_map:
        r = domain_map["acoustic"]
        acoustic_score_arr = r.score

        # Advanced blends for acoustic
        if reverberation is not None:
            # Blend: 60% noise + 40% reverberation
            rev_score = np.full_like(acoustic_score_arr, reverberation.score)
            acoustic_score_arr = 0.6 * acoustic_score_arr + 0.4 * rev_score
        if speech_intelligibility is not None:
            # Blend: 50% current acoustic + 50% STI score
            sti_score = np.full_like(acoustic_score_arr, speech_intelligibility.score)
            acoustic_score_arr = 0.5 * acoustic_score_arr + 0.5 * sti_score

        domain_scores["acoustic"] = acoustic_score_arr

    if "iaq" in domain_map:
        r = domain_map["iaq"]
        iaq_score_arr = r.score

        # Advanced override for IAQ
        if ventilation is not None:
            # Blend: 60% CO₂ score + 40% ventilation score
            vent_score = np.full_like(iaq_score_arr, ventilation.score)
            iaq_score_arr = 0.6 * iaq_score_arr + 0.4 * vent_score

        # Pollutant IAQ blend
        if pollutant_iaq is not None:
            if iaq is not None:
                # 50/50 blend of CO₂ and pollutant scores
                iaq_score_arr = 0.5 * iaq_score_arr + 0.5 * pollutant_iaq.score
            else:
                iaq_score_arr = pollutant_iaq.score

        domain_scores["iaq"] = iaq_score_arr

    # Weighted sum → Global IEQ Index
    index = np.zeros(n, dtype=float)
    for domain_name in available_domains:
        w = normed_weights.get(domain_name, 0.0)
        index += w * domain_scores[domain_name]

    return GlobalIEQResult(
        index=index,
        domain_scores=domain_scores,
        weights_used=normed_weights,
        domains=available_domains,
        n_timestamps=n,
    )
