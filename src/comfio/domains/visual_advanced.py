"""Advanced visual comfort module — Radiance daylighting and color quality.

Provides physics-based daylighting simulation via pyradiance and color
quality evaluation via colour-science. These functions require optional
extras:

    pip install comfio[daylighting]   # for evaluate_daylighting
    pip install comfio[color]         # for evaluate_color_quality

On Windows, the [daylighting] extra may require WSL for Radiance binaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from comfio.core.result_base import ResultBase
from comfio.domains.visual import ILLUMINANCE_TARGETS

SkyModel = Literal["cie_clear", "cie_overcast", "cie_intermediate"]
DaylightMetric = Literal["illuminance", "dgp", "udi", "sda"]


@dataclass
class DaylightingResult(ResultBase):
    """Result of a Radiance-based daylighting evaluation.

    Attributes
    ----------
    illuminance : np.ndarray
        Point-in-time illuminance values (lux) at each sensor point.
    dgp : np.ndarray or None
        Daylight Glare Probability values (0-1) at each viewpoint.
        None if DGP was not calculated.
    compliant : np.ndarray
        Boolean array: True if illuminance meets the target.
    score : np.ndarray
        Daylighting comfort score (0-100), higher is better.
    target_lux : float
        Target maintained illuminance for the task type.
    sky_model : str
        Sky model used for the simulation.
    metric : str
        Primary daylight metric returned.
    """

    illuminance: np.ndarray
    dgp: np.ndarray | None
    compliant: np.ndarray
    score: np.ndarray
    target_lux: float
    sky_model: str
    metric: str


@dataclass
class ColorQualityResult(ResultBase):
    """Result of a color quality evaluation.

    Attributes
    ----------
    cri : float
        General Colour Rendering Index (Ra, 0-100).
    cri_data : dict or None
        Detailed CRI data per test colour sample (if additional_data=True).
    cct : float
        Correlated Colour Temperature in Kelvin.
    duv : float
        Distance from the Planckian locus (D_uv).
    compliant : bool
        True if CRI meets the minimum threshold.
    score : float
        Color quality score (0-100), higher is better.
    method : str
        CRI computation method used ("CIE 1995" or "CIE 2024").
    """

    cri: float
    cri_data: dict[str, Any] | None
    cct: float
    duv: float
    compliant: bool
    score: float
    method: str


def _require_pyradiance() -> Any:
    """Import pyradiance or raise a helpful error."""
    try:
        import pyradiance as pr  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "pyradiance is required for daylighting simulation. "
            "Install it with: pip install comfio[daylighting]\n"
            "Note: On Windows, WSL is recommended for Radiance binaries."
        ) from None
    return pr


def _require_colour() -> Any:
    """Import colour-science or raise a helpful error."""
    try:
        import colour  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "colour-science is required for color quality evaluation. "
            "Install it with: pip install comfio[color]\n"
            "Note: colour-science requires Python >= 3.11."
        ) from None
    return colour


def evaluate_daylighting(
    octree_file: str,
    sensor_points: np.ndarray,
    sky_model: SkyModel = "cie_overcast",
    task_type: str = "general",
    view_directions: np.ndarray | None = None,
    datetime_str: str | None = None,
    latitude: float = 0.0,
    longitude: float = 0.0,
) -> DaylightingResult:
    """Run a Radiance daylighting simulation via pyradiance.

    Calculates point-in-time illuminance at sensor points and optionally
    Daylight Glare Probability (DGP) at viewpoints.

    Parameters
    ----------
    octree_file : str
        Path to the Radiance octree file (.oct) describing the scene.
    sensor_points : np.ndarray
        Array of shape (N, 3) or (N, 6) with sensor point coordinates.
        If (N, 6), the last 3 columns are direction vectors.
    sky_model : str
        Sky model: "cie_clear", "cie_overcast", or "cie_intermediate".
    task_type : str
        Task type key from ``ILLUMINANCE_TARGETS`` for compliance check.
    view_directions : np.ndarray or None
        Array of shape (M, 3) with view direction vectors for DGP calc.
        If None, DGP is not calculated.
    datetime_str : str or None
        Date/time string for sun position (e.g., "06 21 12:00").
        If None, uses overcast sky (no sun).
    latitude : float
        Site latitude in degrees (for sun position).
    longitude : float
        Site longitude in degrees (for sun position).

    Returns
    -------
    DaylightingResult
        Illuminance, DGP, compliance, and score arrays.

    Raises
    ------
    ImportError
        If pyradiance is not installed.
    """
    pr = _require_pyradiance()

    pts = np.asarray(sensor_points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"sensor_points must be (N, 3+) shaped, got {pts.shape}")

    # Extract point coordinates and optional direction vectors
    sensor_pts = pts[:, :3]
    sensor_dirs = pts[:, 3:6] if pts.shape[1] >= 6 else np.zeros_like(pts[:, :3])
    # Default direction: straight up (0, 0, 1)
    if pts.shape[1] < 6:
        sensor_dirs[:] = [0.0, 0.0, 1.0]

    # Generate sky parameters
    if sky_model == "cie_overcast" or datetime_str is None:
        pr.gensky(
            month_day_time="06 21 12:00",
            sky_type="-c",  # cloudy/overcast
            latitude=latitude,
            longitude=longitude,
        )
    elif sky_model == "cie_clear":
        pr.gensky(
            month_day_time=datetime_str,
            sky_type="+s",  # sunny
            latitude=latitude,
            longitude=longitude,
        )
    else:  # cie_intermediate
        pr.gensky(
            month_day_time=datetime_str or "06 21 12:00",
            sky_type="-i",  # intermediate
            latitude=latitude,
            longitude=longitude,
        )

    # Run rtrace for illuminance at sensor points
    # rtrace returns RGB values; convert to illuminance via luminance formula
    raw_output = pr.rtrace(
        octree=octree_file,
        points=sensor_pts,
        dirs=sensor_dirs,
        options=["-I", "-ab", "2", "-aa", "0.1"],
    )

    # Radiance returns RGB; convert to illuminance (lux)
    # Illuminance = 179 * (0.265*R + 0.670*G + 0.065*B) for Radiance units
    if raw_output.ndim == 2 and raw_output.shape[1] >= 3:
        rgb = raw_output[:, :3]
        illuminance = 179.0 * (0.265 * rgb[:, 0] + 0.670 * rgb[:, 1] + 0.065 * rgb[:, 2])
    else:
        illuminance = np.array(raw_output).flatten()

    # DGP calculation (if view directions provided)
    dgp_values = None
    if view_directions is not None:
        view_dirs = np.asarray(view_directions, dtype=float)
        n_views = view_dirs.shape[0]
        dgp_values = np.zeros(n_views, dtype=float)

        # Simplified DGP: based on vertical eye illuminance
        # DGP = c1 * Ev + c2 * log(1 + sum(L^2 * omega / Ev^1.87)) + c3
        # For sensor-based approximation, we use vertical illuminance
        eye_output = pr.rtrace(
            octree=octree_file,
            points=sensor_pts[:n_views],
            dirs=view_dirs,
            options=["-I", "-ab", "2"],
        )
        if eye_output.ndim == 2 and eye_output.shape[1] >= 3:
            eye_rgb = eye_output[:, :3]
            eye_lux = 179.0 * (
                0.265 * eye_rgb[:, 0] + 0.670 * eye_rgb[:, 1] + 0.065 * eye_rgb[:, 2]
            )
        else:
            eye_lux = np.array(eye_output).flatten()[:n_views]

        # Wienold & Christoffersen simplified DGP
        # DGP = 0.465 * log(1 + Ev/5000) for vertical eye illuminance
        dgp_values = np.clip(0.465 * np.log10(1.0 + eye_lux / 5000.0), 0.0, 1.0)

    # Compliance and scoring
    target = ILLUMINANCE_TARGETS.get(task_type, 500.0)
    compliant = illuminance >= target

    # Score: same logic as visual_score but for daylighting
    illuminance_ratio = illuminance / target
    score = np.where(
        illuminance_ratio >= 1.0,
        np.clip(100.0 - 20.0 * (illuminance_ratio - 1.0), 50.0, 100.0),
        np.clip(100.0 * illuminance_ratio, 0.0, 100.0),
    )

    # Penalize DGP > 0.40 (perceptible glare)
    if dgp_values is not None:
        glare_penalty = np.clip(1.0 - dgp_values / 0.40, 0.0, 1.0)
        score = score[: len(glare_penalty)] * glare_penalty

    return DaylightingResult(
        illuminance=illuminance,
        dgp=dgp_values,
        compliant=compliant,
        score=score,
        target_lux=target,
        sky_model=sky_model,
        metric="illuminance",
    )


def evaluate_color_quality(
    spectral_distribution: np.ndarray,
    wavelengths: np.ndarray | None = None,
    method: Literal["CIE 1995", "CIE 2024"] = "CIE 1995",
    min_cri: float = 80.0,
    additional_data: bool = False,
) -> ColorQualityResult:
    """Evaluate color quality of a light source using colour-science.

    Calculates the Colour Rendering Index (CRI), Correlated Colour
    Temperature (CCT), and D_uv from a spectral power distribution.

    Parameters
    ----------
    spectral_distribution : np.ndarray
        Spectral power distribution values. If wavelengths are provided,
        must be 1-D array of SPD values at corresponding wavelengths.
    wavelengths : np.ndarray or None
        Wavelengths in nm corresponding to the SPD. If None, assumes
        standard 380-780 nm range with 1 nm steps.
    method : str
        CRI method: "CIE 1995" (classic Ra) or "CIE 2024" (new method).
    min_cri : float
        Minimum CRI threshold for compliance (default 80, per EN 12464-1).
    additional_data : bool
        If True, returns detailed CRI data per test colour sample.

    Returns
    -------
    ColorQualityResult
        CRI, CCT, D_uv, compliance, and score.

    Raises
    ------
    ImportError
        If colour-science is not installed.
    """
    colour = _require_colour()

    spd_values = np.asarray(spectral_distribution, dtype=float)

    # Build colour SpectralDistribution
    if wavelengths is not None:
        wl = np.asarray(wavelengths, dtype=float)
    else:
        wl = np.arange(380, 781, 1, dtype=float)

    # Ensure SPD matches wavelength length
    if len(spd_values) != len(wl):
        raise ValueError(f"SPD length ({len(spd_values)}) must match wavelength length ({len(wl)})")

    sd = colour.SpectralDistribution(wl, spd_values, name="test_source")

    # Calculate CRI
    cri_result = colour.colour_rendering_index(sd, additional_data=additional_data, method=method)

    if additional_data and isinstance(
        cri_result, colour.quality.cri.ColourRendering_Specification_CRI
    ):
        cri_value = float(cri_result.Q_a)
        cri_data: dict[str, Any] | None = {
            "name": cri_result.name,
            "Q_a": cri_value,
            "Q_as": {
                str(k): {"name": v.name, "Q_a": float(v.Q_a)} for k, v in cri_result.Q_as.items()
            },
        }
    else:
        cri_value = float(cri_result)
        cri_data = None

    # Calculate CCT and D_uv
    # Convert SPD to XYZ, then to CCT
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    illuminant = colour.SDS_ILLUMINANTS["D65"]
    xyz = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant)
    xy = colour.XYZ_to_xy(xyz)
    cct, duv = colour.xy_to_CCT(xy, method="Robertson 1968")

    cct_value = float(cct)
    duv_value = float(duv)

    # Compliance: CRI >= min_cri and |D_uv| < 0.006 (per CIE)
    compliant = cri_value >= min_cri and abs(duv_value) < 0.006

    # Score: weighted CRI + CCT proximity to ideal + D_uv penalty
    # CRI component: direct 0-100
    cri_component = np.clip(cri_value, 0.0, 100.0)

    # CCT component: ideal range 2700-6500K, penalty outside
    if cct_value < 2700:
        cct_component = np.clip(100.0 * cct_value / 2700.0, 0.0, 100.0)
    elif cct_value > 6500:
        cct_component = np.clip(100.0 * 6500.0 / cct_value, 0.0, 100.0)
    else:
        cct_component = 100.0

    # D_uv penalty: 100 at 0, 0 at |D_uv| >= 0.01
    duv_component = np.clip(100.0 * (1.0 - abs(duv_value) / 0.01), 0.0, 100.0)

    score = 0.5 * cri_component + 0.3 * cct_component + 0.2 * duv_component

    return ColorQualityResult(
        cri=cri_value,
        cri_data=cri_data,
        cct=cct_value,
        duv=duv_value,
        compliant=compliant,
        score=float(score),
        method=method,
    )
