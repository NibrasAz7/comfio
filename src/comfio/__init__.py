"""comfio: Multi-domain IEQ & Performance Contract Framework.

A high-performance Python package that bridges raw building sensor data
and actionable smart building management by unifying Thermal, Visual,
Acoustic, and Indoor Air Quality (IAQ) metrics into a Global IEQ Index.
"""

__version__ = "0.1.1"

from comfio.core.data_handler import SensorData
from comfio.core.exceptions import (
    InvalidUnitError,
    MissingSensorDataError,
    OutOfRangeError,
)
from comfio.domains.acoustic import evaluate_acoustic
from comfio.domains.iaq import evaluate_iaq
from comfio.domains.thermal import evaluate_thermal
from comfio.domains.visual import evaluate_visual
from comfio.integration.global_ieq import calculate_global_ieq
from comfio.integration.weights import WeightSchema, default_weights
from comfio.performance.contracts import ComplianceReport, calculate_compliance

# LLM-native module (interpreters + prompts have no extra deps; tools requires [agent])
from comfio.llm import (
    DIAGNOSTIC_PROMPT_TEMPLATE,
    EDGE_SYSTEM_PROMPT,
    format_prompt,
    generate_markdown_summary,
    ieq_to_markdown,
    ieq_to_summary_dict,
)

# Advanced domain functions — importable, raise ImportError on use if extra missing
from comfio.domains.acoustic_advanced import (
    ReverberationResult,
    SpeechIntelligibilityResult,
    evaluate_reverberation,
    evaluate_speech_intelligibility,
)
from comfio.domains.iaq_advanced import (
    PsychrometricResult,
    VentilationResult,
    evaluate_ventilation,
    get_psychrometrics,
)
from comfio.domains.visual_advanced import (
    ColorQualityResult,
    DaylightingResult,
    evaluate_color_quality,
    evaluate_daylighting,
)

# New domain modules
from comfio.domains.iaq_pollutants import (
    PollutantIAQResult,
    evaluate_iaq_pollutants,
    pollutant_iaq_score,
)
from comfio.domains.thermal_spmv import (
    SPMVResult,
    evaluate_spmv,
    spmv_score,
)
from comfio.domains.thermal_adaptive import (
    AdaptiveThermalResult,
    adaptive_thermal_score,
    evaluate_adaptive_ashrae,
    evaluate_adaptive_en,
)
from comfio.domains.thermal_personal import (
    PersonalisationIndex,
    PersonalisedAdaptiveResult,
    PersonalisedPMVResult,
    SeasonalPersonalisationIndex,
    evaluate_personalised_adaptive,
    evaluate_personalised_pmv,
    evaluate_personalised_spmv,
    train_personalisation,
    train_seasonal_personalisation,
)
from comfio.domains.thermal_tsv import (
    TSVResult,
    augment_tsv_cdf,
    evaluate_tsv,
)
from comfio.utils.validation import validate_input_array

__all__ = [
    "SensorData",
    "MissingSensorDataError",
    "OutOfRangeError",
    "InvalidUnitError",
    "evaluate_thermal",
    "evaluate_visual",
    "evaluate_acoustic",
    "evaluate_iaq",
    "calculate_global_ieq",
    "WeightSchema",
    "default_weights",
    "ComplianceReport",
    "calculate_compliance",
    # Advanced visual
    "evaluate_daylighting",
    "evaluate_color_quality",
    "DaylightingResult",
    "ColorQualityResult",
    # Advanced acoustic
    "evaluate_reverberation",
    "evaluate_speech_intelligibility",
    "ReverberationResult",
    "SpeechIntelligibilityResult",
    # Advanced IAQ
    "evaluate_ventilation",
    "get_psychrometrics",
    "VentilationResult",
    "PsychrometricResult",
    # IAQ pollutants
    "evaluate_iaq_pollutants",
    "PollutantIAQResult",
    "pollutant_iaq_score",
    # Simplified PMV
    "evaluate_spmv",
    "SPMVResult",
    "spmv_score",
    # Adaptive thermal comfort
    "evaluate_adaptive_ashrae",
    "evaluate_adaptive_en",
    "AdaptiveThermalResult",
    "adaptive_thermal_score",
    # Personalised thermal comfort
    "train_personalisation",
    "train_seasonal_personalisation",
    "evaluate_personalised_pmv",
    "evaluate_personalised_spmv",
    "evaluate_personalised_adaptive",
    "PersonalisationIndex",
    "SeasonalPersonalisationIndex",
    "PersonalisedPMVResult",
    "PersonalisedAdaptiveResult",
    # TSV processing & CDF augmentation
    "augment_tsv_cdf",
    "evaluate_tsv",
    "TSVResult",
    # Validation helper
    "validate_input_array",
    "__version__",
    # LLM-native
    "ieq_to_markdown",
    "ieq_to_summary_dict",
    "generate_markdown_summary",
    "EDGE_SYSTEM_PROMPT",
    "DIAGNOSTIC_PROMPT_TEMPLATE",
    "format_prompt",
]
