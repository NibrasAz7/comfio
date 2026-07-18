"""Advanced acoustic comfort module — reverberation time and speech intelligibility.

Provides physics-based acoustic evaluation via python-acoustics (RT60
calculation) and pyroomacoustics (room impulse response, STI estimation).
These functions require optional extras:

    pip install comfio[acoustics]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from comfio.core.result_base import ResultBase

ReverbMethod = Literal["sabine", "eyring", "fit"]
STIRating = Literal["bad", "poor", "fair", "good", "excellent"]

# RT60 targets per room type (seconds)
RT60_TARGETS: dict[str, tuple[float, float]] = {
    "classroom": (0.6, 0.8),
    "lecture_hall": (0.8, 1.2),
    "office": (0.4, 0.6),
    "meeting_room": (0.5, 0.7),
    "concert_hall": (1.8, 2.2),
    "recording_studio": (0.2, 0.4),
    "library": (0.6, 0.9),
    "general": (0.5, 0.8),
}

# STI rating bands per IEC 60268-16
STI_RATING_BANDS: list[tuple[float, STIRating]] = [
    (0.00, "bad"),
    (0.36, "poor"),
    (0.45, "fair"),
    (0.60, "good"),
    (0.75, "excellent"),
]


@dataclass
class ReverberationResult(ResultBase):
    """Result of a reverberation time evaluation.

    Attributes
    ----------
    rt60 : np.ndarray
        Reverberation time (T60) in seconds per frequency band.
    frequencies : np.ndarray
        Frequency bands (Hz) corresponding to rt60 values.
    method : str
        Calculation method: "sabine", "eyring", or "fit".
    nrc : float or None
        Noise Reduction Coefficient (average of 250, 500, 1000, 2000 Hz).
    room_type : str
        Room type used for compliance check.
    compliant : bool
        True if mean RT60 falls within the target range for the room type.
    score : float
        Reverberation comfort score (0-100), higher is better.
    """

    rt60: np.ndarray
    frequencies: np.ndarray
    method: str
    nrc: float | None
    room_type: str
    compliant: bool
    score: float


@dataclass
class SpeechIntelligibilityResult(ResultBase):
    """Result of a speech intelligibility evaluation.

    Attributes
    ----------
    sti : float
        Speech Transmission Index (0-1), higher is better.
    rating : str
        Qualitative rating: "bad", "poor", "fair", "good", "excellent".
    rt60_measured : float
        Measured reverberation time from the impulse response (seconds).
    compliant : bool
        True if STI >= 0.60 (good or better).
    score : float
        Speech intelligibility comfort score (0-100), higher is better.
    """

    sti: float
    rating: str
    rt60_measured: float
    compliant: bool
    score: float


def _require_acoustics() -> Any:
    """Import python-acoustics or raise a helpful error."""
    try:
        import acoustics  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "python-acoustics is required for reverberation calculations. "
            "Install it with: pip install comfio[acoustics]"
        ) from None
    return acoustics


def _require_pyroomacoustics() -> Any:
    """Import pyroomacoustics or raise a helpful error."""
    try:
        import pyroomacoustics as pra  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "pyroomacoustics is required for speech intelligibility calculations. "
            "Install it with: pip install comfio[acoustics]"
        ) from None
    return pra


def evaluate_reverberation(
    surfaces: np.ndarray,
    absorption_coeffs: np.ndarray,
    volume: float,
    method: ReverbMethod = "sabine",
    room_type: str = "general",
    c: float = 343.0,
) -> ReverberationResult:
    """Calculate reverberation time (RT60) using Sabine or Eyring formulas.

    Uses the python-acoustics library for standard RT60 calculations.

    Parameters
    ----------
    surfaces : np.ndarray
        Surface areas (m²) of room boundaries. Can be 1-D (single value
        per surface) or 2-D (surfaces × frequency bands).
    absorption_coeffs : np.ndarray
        Absorption coefficients (0-1) matching ``surfaces`` shape.
    volume : float
        Room volume in m³.
    method : str
        Calculation method: "sabine" or "eyring".
    room_type : str
        Room type key from ``RT60_TARGETS`` for compliance check.
    c : float
        Speed of sound in m/s (default 343.0 at 20°C).

    Returns
    -------
    ReverberationResult
        RT60 per band, NRC, compliance, and score.

    Raises
    ------
    ImportError
        If python-acoustics is not installed.
    """
    acoustics = _require_acoustics()

    surf = np.asarray(surfaces, dtype=float)
    alpha = np.asarray(absorption_coeffs, dtype=float)

    # Determine frequency bands
    if alpha.ndim == 2:
        n_bands = alpha.shape[1]
        # Standard octave band center frequencies
        freqs = np.array([125, 250, 500, 1000, 2000, 4000][:n_bands], dtype=float)
    else:
        freqs = np.array([500.0])  # Single broadband value
        if alpha.ndim == 1:
            alpha = alpha.reshape(-1, 1)
            surf = surf.reshape(-1, 1) if surf.ndim == 1 else surf

    # Calculate RT60
    if method == "sabine":
        rt60 = acoustics.room.t60_sabine(surf, alpha, volume, c=c)
    elif method == "eyring":
        rt60 = acoustics.room.t60_eyring(surf, alpha, volume, c=c)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'sabine' or 'eyring'.")

    rt60 = np.atleast_1d(np.asarray(rt60, dtype=float))

    # Calculate NRC (Noise Reduction Coefficient)
    # NRC = average of absorption at 250, 500, 1000, 2000 Hz
    nrc_value = None
    if alpha.shape[-1] >= 4:
        nrc_bands = alpha[:, 1:5]  # 250, 500, 1000, 2000 Hz
        nrc_value = float(np.mean(np.average(nrc_bands, axis=0, weights=surf[:, 1:5].flatten())))

    # Compliance check against room type targets
    target_lo, target_hi = RT60_TARGETS.get(room_type, RT60_TARGETS["general"])
    mean_rt60 = float(np.mean(rt60))
    compliant = target_lo <= mean_rt60 <= target_hi

    # Score: 100 at center of target range, decreasing linearly
    target_center = (target_lo + target_hi) / 2.0
    target_half_width = (target_hi - target_lo) / 2.0
    if target_half_width > 0:
        deviation = abs(mean_rt60 - target_center) / target_half_width
        score = float(np.clip(100.0 * (1.0 - 0.5 * deviation), 0.0, 100.0))
    else:
        score = 50.0

    return ReverberationResult(
        rt60=rt60,
        frequencies=freqs,
        method=method,
        nrc=nrc_value,
        room_type=room_type,
        compliant=compliant,
        score=score,
    )


def evaluate_speech_intelligibility(
    impulse_response: np.ndarray,
    sample_rate: float,
    n_bands: int = 7,
) -> SpeechIntelligibilityResult:
    """Estimate Speech Transmission Index from a room impulse response.

    Uses pyroomacoustics to measure RT60 from the impulse response and
    estimates STI using the modulation transfer function approach.

    Parameters
    ----------
    impulse_response : np.ndarray
        Room impulse response signal (1-D array).
    sample_rate : float
        Sampling rate of the impulse response in Hz.
    n_bands : int
        Number of octave bands for STI calculation (default 7, per IEC 60268-16).

    Returns
    -------
    SpeechIntelligibilityResult
        STI value, rating, measured RT60, compliance, and score.

    Raises
    ------
    ImportError
        If pyroomacoustics is not installed.
    """
    pra = _require_pyroomacoustics()

    ir = np.asarray(impulse_response, dtype=float).flatten()

    # Measure RT60 from impulse response using Schroeder method
    rt60_measured = float(pra.experimental.measure_rt60(ir, fs=sample_rate, plot=False))

    # Estimate STI from RT60 using a simplified relationship
    # STI is related to the modulation transfer function (MTF)
    # For a diffuse field, the MTF can be approximated from RT60
    #
    # m(f) = 1 / sqrt(1 + (2 * pi * f * T60 / 13.82)^2)
    #
    # where f is the modulation frequency and T60 is the reverberation time.
    # STI is the weighted average of m(f) over 7 octave bands.

    # IEC 60268-16 modulation frequencies (Hz)
    mod_freqs = np.array([0.63, 0.80, 1.00, 1.25, 1.60, 2.00, 2.50, 3.15, 4.00, 5.00, 6.30, 8.00])

    # Simplified MTF: assume similar RT60 across bands
    # In a full implementation, we'd filter the IR into octave bands and
    # compute per-band RT60. Here we use the broadband RT60 as approximation.
    mtf = 1.0 / np.sqrt(1.0 + (2.0 * np.pi * mod_freqs * rt60_measured / 13.82) ** 2)

    # Weighted average (simplified equal weighting per IEC 60268-16)
    # Male speech weights differ from female speech; we use average
    sti_value = float(np.mean(mtf))

    # Clamp to [0, 1]
    sti_value = float(np.clip(sti_value, 0.0, 1.0))

    # Determine rating
    rating: STIRating = "bad"
    for threshold, label in STI_RATING_BANDS:
        if sti_value >= threshold:
            rating = label

    # Compliance: STI >= 0.60 is "good" or better
    compliant = sti_value >= 0.60

    # Score: map STI (0-1) to 0-100
    score = float(sti_value * 100.0)

    return SpeechIntelligibilityResult(
        sti=sti_value,
        rating=rating,
        rt60_measured=rt60_measured,
        compliant=compliant,
        score=score,
    )
