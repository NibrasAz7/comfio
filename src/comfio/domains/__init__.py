"""Domain modules: isolated physics calculations for each IEQ discipline."""

# Advanced modules — importable but raise ImportError on use if extra not installed
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
