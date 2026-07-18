"""comfio: Multi-domain IEQ & Performance Contract Framework.

A high-performance Python package that bridges raw building sensor data
and actionable smart building management by unifying Thermal, Visual,
Acoustic, and Indoor Air Quality (IAQ) metrics into a Global IEQ Index.
"""

__version__ = "0.1.6"

from comfio.core.data_handler import SensorData
from comfio.core.exceptions import (
    InvalidUnitError,
    MissingSensorDataError,
    OutOfRangeError,
)
from comfio.domains.acoustic import evaluate_acoustic

# Advanced domain functions — importable, raise ImportError on use if extra missing
from comfio.domains.acoustic_advanced import (
    ReverberationResult,
    SpeechIntelligibilityResult,
    evaluate_reverberation,
    evaluate_speech_intelligibility,
)
from comfio.domains.iaq import evaluate_iaq
from comfio.domains.iaq_advanced import (
    PsychrometricResult,
    VentilationResult,
    evaluate_ventilation,
    get_psychrometrics,
)

# New domain modules
from comfio.domains.iaq_pollutants import (
    PollutantIAQResult,
    evaluate_iaq_pollutants,
    pollutant_iaq_score,
)
from comfio.domains.thermal import evaluate_thermal
from comfio.domains.thermal_adaptive import (
    AdaptiveThermalResult,
    adaptive_thermal_score,
    evaluate_adaptive_ashrae,
    evaluate_adaptive_en,
)
from comfio.domains.thermal_local import (
    AnkleDraftResult,
    VerticalGradientResult,
    evaluate_ankle_draft,
    evaluate_vertical_gradient,
    local_discomfort_score,
)
from comfio.domains.thermal_personal import (
    PersonalisationIndex,
    PersonalisedAdaptiveResult,
    PersonalisedPMVResult,
    SeasonalPersonalisationIndex,
    evaluate_personalised_adaptive,
    evaluate_personalised_pmv,
    evaluate_personalised_spmv,
    evaluate_seasonal_personalised_adaptive,
    evaluate_seasonal_personalised_pmv,
    evaluate_seasonal_personalised_spmv,
    train_personalisation,
    train_seasonal_personalisation,
)
from comfio.domains.thermal_spmv import (
    SPMVResult,
    evaluate_spmv,
    spmv_score,
)
from comfio.domains.thermal_tsv import (
    TSVResult,
    augment_tsv_cdf,
    evaluate_tsv,
)
from comfio.domains.visual import evaluate_visual
from comfio.domains.visual_advanced import (
    ColorQualityResult,
    DaylightingResult,
    evaluate_color_quality,
    evaluate_daylighting,
)
from comfio.integration.global_ieq import calculate_global_ieq
from comfio.integration.weather import (
    fetch_outdoor_temperature,
    fetch_prevailing_temp,
    fetch_running_mean,
)
from comfio.integration.weights import WeightSchema, default_weights

# LLM-native module (interpreters + prompts have no extra deps; tools requires [agent])
from comfio.llm import (
    DIAGNOSTIC_PROMPT_TEMPLATE,
    EDGE_SYSTEM_PROMPT,
    format_prompt,
    generate_markdown_summary,
    ieq_to_markdown,
    ieq_to_summary_dict,
)
from comfio.logging_config import setup_logging
from comfio.performance.contracts import ComplianceReport, calculate_compliance
from comfio.reports import (
    PipelineResult,
    detect_capabilities,
    generate_pipeline_script,
    ieq_to_csv,
    ieq_to_docx,
    ieq_to_pdf,
    run_pipeline,
)
from comfio.utils.validation import validate_input_array


# GUI entry points — lazy import to avoid requiring streamlit/ipywidgets.
# These override the comfio.gui submodule attribute so that
# ``from comfio import gui`` returns the *function*, not the module.
def gui(df: object = None, port: int = 8501) -> None:
    """Launch the comfio Streamlit GUI. Requires ``pip install comfio[gui]``."""
    from comfio.gui import gui as _gui

    _gui(df=df, port=port)


def gui_notebook(df: object = None) -> object:
    """Create an ipywidgets widget for Jupyter. Requires ``pip install comfio[gui]``."""
    from comfio.gui import gui_notebook as _gui_notebook

    return _gui_notebook(df=df)


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
    "evaluate_seasonal_personalised_pmv",
    "evaluate_seasonal_personalised_spmv",
    "evaluate_seasonal_personalised_adaptive",
    "evaluate_personalised_adaptive",
    "PersonalisationIndex",
    "SeasonalPersonalisationIndex",
    "PersonalisedPMVResult",
    "PersonalisedAdaptiveResult",
    # TSV processing & CDF augmentation
    "augment_tsv_cdf",
    "evaluate_tsv",
    "TSVResult",
    # Local discomfort (v0.1.6)
    "evaluate_ankle_draft",
    "evaluate_vertical_gradient",
    "AnkleDraftResult",
    "VerticalGradientResult",
    "local_discomfort_score",
    # Weather integration (v0.1.6)
    "fetch_outdoor_temperature",
    "fetch_prevailing_temp",
    "fetch_running_mean",
    # Logging (v0.1.6)
    "setup_logging",
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
    # Reports & pipeline
    "PipelineResult",
    "detect_capabilities",
    "run_pipeline",
    "ieq_to_csv",
    "ieq_to_pdf",
    "ieq_to_docx",
    "generate_pipeline_script",
    # GUI
    "gui",
    "gui_notebook",
]
