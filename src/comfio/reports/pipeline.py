"""Intelligent pipeline: detect capabilities and run evaluations.

Provides :func:`detect_capabilities` to inspect what evaluations are possible
from available sensor columns, and :func:`run_pipeline` to execute all
possible domain evaluations with graceful degradation for missing data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from comfio.core.data_handler import SensorData
from comfio.core.result_base import ResultBase
from comfio.domains.acoustic import AcousticResult, evaluate_acoustic
from comfio.domains.iaq import IAQResult, evaluate_iaq
from comfio.domains.iaq_pollutants import PollutantIAQResult, evaluate_iaq_pollutants
from comfio.domains.thermal import ThermalResult, evaluate_thermal
from comfio.domains.thermal_adaptive import (
    AdaptiveThermalResult,
    evaluate_adaptive_ashrae,
    evaluate_adaptive_en,
)
from comfio.domains.thermal_spmv import SPMVResult, evaluate_spmv
from comfio.domains.thermal_tsv import TSVResult, augment_tsv_cdf, evaluate_tsv
from comfio.domains.visual import VisualResult, evaluate_visual
from comfio.integration.global_ieq import GlobalIEQResult, calculate_global_ieq
from comfio.integration.weights import WeightSchema
from comfio.performance.contracts import ComplianceReport, calculate_compliance

logger = logging.getLogger(__name__)

# Canonical column names that map to each capability
_THERMAL_PMV_COLS = {"air_temp_c", "radiant_temp_c", "air_velocity_ms", "relative_humidity_pct"}
_THERMAL_SPMV_COLS = {"air_temp_c", "relative_humidity_pct"}
_POLLUTANT_COLS = {"pm25_ugm3", "pm10_ugm3", "tvoc_ugm3", "formaldehyde_ppb", "co_ppm"}
_TSV_VOTE_ALIASES = {"tsv", "thermal_sensation_vote", "thermal_sensation", "vote", "tsv_vote"}


@dataclass
class PipelineResult(ResultBase):
    """Result of running the intelligent evaluation pipeline.

    Attributes
    ----------
    sensor : SensorData
        The validated sensor data used.
    capabilities : dict[str, bool]
        What evaluations were possible.
    domain_results : dict[str, Any]
        Individual domain result objects (keyed by domain name).
    ieq_result : GlobalIEQResult or None
        Global IEQ Index result, if at least one domain ran.
    compliance_report : ComplianceReport or None
        Compliance report, if IEQ result was computed.
    warnings : list[str]
        Human-readable warnings about skipped domains or defaults used.
    config : dict
        Configuration used for the pipeline run.
    """

    sensor: SensorData
    capabilities: dict[str, bool]
    domain_results: dict[str, Any] = field(default_factory=dict)
    ieq_result: GlobalIEQResult | None = None
    compliance_report: ComplianceReport | None = None
    warnings: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


def _has_col(sensor: SensorData, canonical: str) -> bool:
    """Check if a canonical column is mapped in the sensor data."""
    return canonical in sensor.column_map


def _get_col(sensor: SensorData, canonical: str) -> np.ndarray:
    """Get validated array for a canonical column."""
    return sensor.get_validated(canonical)


def _detect_tsv_column(sensor: SensorData) -> str | None:
    """Find the TSV vote column if present.

    Checks canonical name first, then common aliases in the DataFrame.
    """
    if "tsv" in sensor.column_map:
        return sensor.column_map["tsv"]
    # Check aliases in DataFrame columns
    df_cols_lower = {col.lower().strip(): col for col in sensor.df.columns}
    for alias in _TSV_VOTE_ALIASES:
        if alias in df_cols_lower:
            return str(df_cols_lower[alias])
    return None


def detect_capabilities(sensor: SensorData) -> dict[str, bool]:
    """Detect what evaluations are possible from available sensor columns.

    Parameters
    ----------
    sensor : SensorData
        Sensor data with column mapping populated.

    Returns
    -------
    dict[str, bool]
        Dictionary of capability name to availability.

    Examples
    --------
    >>> import pandas as pd
    >>> from comfio import SensorData
    >>> df = pd.DataFrame({
    ...     "air_temp_c": [22.0, 24.0],
    ...     "radiant_temp_c": [22.0, 24.0],
    ...     "air_velocity_ms": [0.1, 0.1],
    ...     "relative_humidity_pct": [50.0, 50.0],
    ... })
    >>> sensor = SensorData(df=df)
    >>> caps = detect_capabilities(sensor)
    >>> caps["thermal_pmv"]
    True
    >>> caps["visual"]
    False
    """
    has_pmv = all(_has_col(sensor, c) for c in _THERMAL_PMV_COLS)
    has_spmv = all(_has_col(sensor, c) for c in _THERMAL_SPMV_COLS)
    has_outdoor = _has_col(sensor, "outdoor_temp_c") or _has_col(
        sensor, "prevailing_mean_outdoor_c"
    )
    has_running_mean = _has_col(sensor, "running_mean_outdoor_c")
    has_tsv = _detect_tsv_column(sensor) is not None

    # Personalisation needs both PMV and TSV
    has_pmv_col = _has_col(sensor, "pmv") or has_pmv
    has_personalisation = has_pmv_col and has_tsv

    return {
        "thermal_pmv": has_pmv,
        "thermal_spmv": has_spmv,
        "thermal_adaptive_ashrae": has_outdoor and has_pmv,
        "thermal_adaptive_en": has_running_mean and has_pmv,
        "visual": _has_col(sensor, "illuminance_lux"),
        "acoustic": _has_col(sensor, "noise_laeq_db"),
        "iaq_co2": _has_col(sensor, "co2_ppm"),
        "iaq_pollutant": any(_has_col(sensor, c) for c in _POLLUTANT_COLS),
        "tsv": has_tsv,
        "personalisation": has_personalisation,
    }


def run_pipeline(
    sensor: SensorData,
    config: dict[str, Any] | None = None,
) -> PipelineResult:
    """Run all possible domain evaluations based on available data.

    Adapts to whatever columns the user has. Missing columns result in
    skipped evaluations with warnings, not errors.

    Parameters
    ----------
    sensor : SensorData
        Sensor data with column mapping populated.
    config : dict, optional
        Configuration overrides. Supported keys:

        - ``met`` (float, default 1.2): metabolic rate if not in data
        - ``clo`` (float, default 0.5): clothing insulation if not in data
        - ``standard`` (str, default "7730-2005"): thermal standard
        - ``category`` (str, default "B"): thermal category
        - ``task_type`` (str, default "general"): visual task type
        - ``nc_level`` (str, default "NC-35"): acoustic NC level
        - ``co2_threshold`` (str, default "good"): IAQ CO2 threshold
        - ``pollutant_threshold`` (str, default "good"): pollutant threshold
        - ``weights`` (WeightSchema or None): custom weights
        - ``threshold`` (float, default 80.0): compliance threshold
        - ``tsv_time_aware`` (bool, default False): time-aware TSV augmentation
        - ``adaptive_standard`` (str, default "ashrae"): "ashrae" or "en"
        - ``adaptive_acceptability`` (int, default 80): ASHRAE acceptability
        - ``adaptive_category`` (str, default "ii"): EN category

    Returns
    -------
    PipelineResult
        All results, warnings, and configuration used.

    Examples
    --------
    >>> import pandas as pd
    >>> from comfio import SensorData, run_pipeline
    >>> df = pd.DataFrame({
    ...     "air_temp_c": [24.0, 25.0],
    ...     "radiant_temp_c": [24.0, 25.0],
    ...     "air_velocity_ms": [0.1, 0.1],
    ...     "relative_humidity_pct": [50.0, 50.0],
    ...     "illuminance_lux": [500.0, 480.0],
    ... })
    >>> sensor = SensorData(df=df)
    >>> result = run_pipeline(sensor)
    >>> result.ieq_result is not None
    True
    >>> "thermal" in result.domain_results
    True
    """
    cfg = config or {}
    pip_warnings: list[str] = []

    # Defaults
    met = cfg.get("met", 1.2)
    clo = cfg.get("clo", 0.5)
    standard = cfg.get("standard", "7730-2005")
    category = cfg.get("category", "B")
    task_type = cfg.get("task_type", "general")
    nc_level = cfg.get("nc_level", "NC-35")
    co2_threshold = cfg.get("co2_threshold", "good")
    pollutant_threshold = cfg.get("pollutant_threshold", "good")
    weights = cfg.get("weights")
    threshold = cfg.get("threshold", 80.0)
    tsv_time_aware = cfg.get("tsv_time_aware", False)
    adaptive_standard = cfg.get("adaptive_standard", "ashrae")
    adaptive_acceptability = cfg.get("adaptive_acceptability", 80)
    adaptive_category = cfg.get("adaptive_category", "ii")

    # Get met/clo from data if available, else use defaults
    if _has_col(sensor, "metabolic_rate_met"):
        met_arr = _get_col(sensor, "metabolic_rate_met")
        met = float(np.mean(met_arr))
    if _has_col(sensor, "clothing_insulation_clo"):
        clo_arr = _get_col(sensor, "clothing_insulation_clo")
        clo = float(np.mean(clo_arr))

    caps = detect_capabilities(sensor)
    domain_results: dict[str, Any] = {}
    logger.info("Pipeline started: %d samples, capabilities: %s", len(sensor), caps)

    # --- Thermal PMV ---
    thermal_res: ThermalResult | None = None
    if caps["thermal_pmv"]:
        try:
            tdb = _get_col(sensor, "air_temp_c")
            tr = _get_col(sensor, "radiant_temp_c")
            vr = _get_col(sensor, "air_velocity_ms")
            rh = _get_col(sensor, "relative_humidity_pct")
            thermal_res = evaluate_thermal(
                tdb=tdb,
                tr=tr,
                vr=vr,
                rh=rh,
                met=met,
                clo=clo,
                standard=standard,
                category=category,
            )
            domain_results["thermal"] = thermal_res
        except Exception as exc:
            logger.warning("Thermal PMV evaluation failed: %s", exc)
            pip_warnings.append(f"Thermal PMV evaluation failed: {exc}")

    # --- sPMV (only if not already have full PMV, or always run for comparison) ---
    spmv_res: SPMVResult | None = None
    if caps["thermal_spmv"]:
        try:
            tdb = _get_col(sensor, "air_temp_c")
            rh = _get_col(sensor, "relative_humidity_pct")
            spmv_res = evaluate_spmv(indoor_temp=tdb, indoor_rh=rh)
            domain_results["spmv"] = spmv_res
        except Exception as exc:
            logger.warning("sPMV evaluation failed: %s", exc)
            pip_warnings.append(f"sPMV evaluation failed: {exc}")
    elif not caps["thermal_pmv"]:
        pip_warnings.append(
            "sPMV not available: requires air_temp_c and relative_humidity_pct columns."
        )

    # --- Adaptive thermal ---
    adaptive_res: AdaptiveThermalResult | None = None
    if caps["thermal_adaptive_ashrae"] or caps["thermal_adaptive_en"]:
        try:
            tdb = _get_col(sensor, "air_temp_c")
            tr = _get_col(sensor, "radiant_temp_c")
            if adaptive_standard == "en" and caps["thermal_adaptive_en"]:
                rm = float(np.mean(_get_col(sensor, "running_mean_outdoor_c")))
                adaptive_res = evaluate_adaptive_en(
                    tdb=tdb,
                    tr=tr,
                    t_running_mean=rm,
                    category=adaptive_category,
                )
            elif caps["thermal_adaptive_ashrae"]:
                if _has_col(sensor, "prevailing_mean_outdoor_c"):
                    t_prevail = float(np.mean(_get_col(sensor, "prevailing_mean_outdoor_c")))
                else:
                    t_prevail = float(np.mean(_get_col(sensor, "outdoor_temp_c")))
                adaptive_res = evaluate_adaptive_ashrae(
                    tdb=tdb,
                    tr=tr,
                    t_prevail=t_prevail,
                    acceptability=adaptive_acceptability,
                )
            if adaptive_res is not None:
                domain_results["adaptive"] = adaptive_res
        except Exception as exc:
            logger.warning("Adaptive thermal evaluation failed: %s", exc)
            pip_warnings.append(f"Adaptive thermal evaluation failed: {exc}")

    # --- Visual ---
    visual_res: VisualResult | None = None
    if caps["visual"]:
        try:
            lux = _get_col(sensor, "illuminance_lux")
            ugr = None
            if _has_col(sensor, "ugr"):
                ugr = _get_col(sensor, "ugr")
            visual_res = evaluate_visual(illuminance=lux, task_type=task_type, ugr=ugr)
            domain_results["visual"] = visual_res
        except Exception as exc:
            logger.warning("Visual evaluation failed: %s", exc)
            pip_warnings.append(f"Visual evaluation failed: {exc}")

    # --- Acoustic ---
    acoustic_res: AcousticResult | None = None
    if caps["acoustic"]:
        try:
            laeq = _get_col(sensor, "noise_laeq_db")
            acoustic_res = evaluate_acoustic(laeq=laeq, nc_level=nc_level)
            domain_results["acoustic"] = acoustic_res
        except Exception as exc:
            logger.warning("Acoustic evaluation failed: %s", exc)
            pip_warnings.append(f"Acoustic evaluation failed: {exc}")

    # --- IAQ CO2 ---
    iaq_res: IAQResult | None = None
    if caps["iaq_co2"]:
        try:
            co2 = _get_col(sensor, "co2_ppm")
            iaq_res = evaluate_iaq(co2=co2, threshold_level=co2_threshold)
            domain_results["iaq"] = iaq_res
        except Exception as exc:
            logger.warning("IAQ CO2 evaluation failed: %s", exc)
            pip_warnings.append(f"IAQ CO2 evaluation failed: {exc}")

    # --- Pollutant IAQ ---
    pollutant_res: PollutantIAQResult | None = None
    if caps["iaq_pollutant"]:
        try:
            pollutant_kwargs: dict[str, np.ndarray] = {}
            for col, key in [
                ("pm25_ugm3", "pm25"),
                ("pm10_ugm3", "pm10"),
                ("tvoc_ugm3", "tvoc"),
                ("formaldehyde_ppb", "formaldehyde"),
                ("co_ppm", "co"),
            ]:
                if _has_col(sensor, col):
                    pollutant_kwargs[key] = _get_col(sensor, col)
            if pollutant_kwargs:
                pollutant_res = evaluate_iaq_pollutants(
                    threshold_level=pollutant_threshold,
                    **pollutant_kwargs,
                )
                domain_results["pollutant_iaq"] = pollutant_res
        except Exception as exc:
            logger.warning("Pollutant IAQ evaluation failed: %s", exc)
            pip_warnings.append(f"Pollutant IAQ evaluation failed: {exc}")

    # --- TSV ---
    tsv_res: TSVResult | None = None
    tsv_col = _detect_tsv_column(sensor)
    if tsv_col is not None:
        try:
            tsv_votes = sensor.df[tsv_col].dropna().to_numpy(dtype=float)
            n_votes = len(tsv_votes)
            n_targets = len(sensor.df)

            if n_votes == 0:
                pip_warnings.append("TSV column detected but contains no valid votes.")
            elif n_votes >= n_targets:
                # Enough votes — evaluate directly
                tsv_res = evaluate_tsv(tsv_votes)
                domain_results["tsv"] = tsv_res
            else:
                # Sparse votes — augment via CDF remapping
                if sensor.timestamp_col and sensor.timestamp_col in sensor.df.columns:
                    vote_ts = sensor.df[sensor.timestamp_col].dropna().to_numpy(dtype=float)
                    vote_ts = vote_ts[:n_votes]
                    # Convert to unix timestamps if datetime
                    vote_ts = _to_unix_timestamps(vote_ts)
                    target_ts = _to_unix_timestamps(sensor.df[sensor.timestamp_col].to_numpy())
                else:
                    # Use index as timestamp
                    vote_ts = np.linspace(0, 1, n_votes)
                    target_ts = np.linspace(0, 1, n_targets)

                augmented = augment_tsv_cdf(
                    sparse_votes=tsv_votes,
                    vote_timestamps=vote_ts,
                    target_timestamps=target_ts,
                    time_aware=tsv_time_aware,
                )
                tsv_res = evaluate_tsv(augmented)
                domain_results["tsv"] = tsv_res
                domain_results["tsv_augmented"] = augmented
        except Exception as exc:
            logger.warning("TSV evaluation failed: %s", exc)
            pip_warnings.append(f"TSV evaluation failed: {exc}")

    # --- Warnings for missing domains ---
    if not caps["thermal_pmv"] and not caps["thermal_spmv"]:
        pip_warnings.append(
            "No thermal evaluation possible: requires at least"
            " air_temp_c and relative_humidity_pct."
        )
    if not caps["visual"]:
        pip_warnings.append("Visual domain skipped: no illuminance_lux column.")
    if not caps["acoustic"]:
        pip_warnings.append("Acoustic domain skipped: no noise_laeq_db column.")
    if not caps["iaq_co2"]:
        pip_warnings.append("IAQ CO2 domain skipped: no co2_ppm column.")

    # --- Global IEQ ---
    ieq_result: GlobalIEQResult | None = None
    compliance_report: ComplianceReport | None = None

    # Build kwargs for calculate_global_ieq from available results
    ieq_kwargs: dict[str, Any] = {}
    if thermal_res is not None:
        ieq_kwargs["thermal"] = thermal_res
    if visual_res is not None:
        ieq_kwargs["visual"] = visual_res
    if acoustic_res is not None:
        ieq_kwargs["acoustic"] = acoustic_res
    if iaq_res is not None:
        ieq_kwargs["iaq"] = iaq_res
    if pollutant_res is not None:
        ieq_kwargs["pollutant_iaq"] = pollutant_res
    if tsv_res is not None:
        ieq_kwargs["tsv"] = tsv_res

    # TSV overrides thermal if both present
    if tsv_res is not None and thermal_res is not None:
        pip_warnings.append(
            "TSV overrides thermal score in IEQ (occupant feedback = ground truth)."
        )

    if weights is not None:
        ieq_kwargs["weights"] = weights
    elif cfg.get("weight_preset"):
        weight_map = _get_weight_preset(cfg["weight_preset"])
        if weight_map is not None:
            ieq_kwargs["weights"] = weight_map

    if ieq_kwargs:
        try:
            ieq_result = calculate_global_ieq(**ieq_kwargs)
            compliance_report = calculate_compliance(ieq_result, threshold=threshold)
        except Exception as exc:
            logger.warning("Global IEQ calculation failed: %s", exc)
            pip_warnings.append(f"Global IEQ calculation failed: {exc}")

    return PipelineResult(
        sensor=sensor,
        capabilities=caps,
        domain_results=domain_results,
        ieq_result=ieq_result,
        compliance_report=compliance_report,
        warnings=pip_warnings,
        config={
            "met": met,
            "clo": clo,
            "standard": standard,
            "category": category,
            "task_type": task_type,
            "nc_level": nc_level,
            "co2_threshold": co2_threshold,
            "pollutant_threshold": pollutant_threshold,
            "threshold": threshold,
            "tsv_time_aware": tsv_time_aware,
            "adaptive_standard": adaptive_standard,
            "adaptive_acceptability": adaptive_acceptability,
            "adaptive_category": adaptive_category,
            "weight_preset": cfg.get("weight_preset"),
        },
    )


def _to_unix_timestamps(arr: np.ndarray) -> np.ndarray:
    """Convert array to Unix timestamps (float seconds since epoch).

    Handles datetime64, pandas Timestamps, and numeric values.
    """
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[s]").astype(np.float64)
    # Try converting via pandas
    try:
        import pandas as pd

        s = pd.Series(arr)
        if pd.api.types.is_datetime64_any_dtype(s):
            return np.asarray(s.astype("int64").astype(np.float64).to_numpy() / 1e9)
    except Exception:
        pass
    return arr.astype(np.float64)


def _get_weight_preset(preset: str) -> WeightSchema | None:
    """Get a weight schema by preset name."""
    presets: dict[str, dict[str, float]] = {
        "default": {"thermal": 0.40, "visual": 0.20, "acoustic": 0.20, "iaq": 0.20},
        "equal": {"thermal": 0.25, "visual": 0.25, "acoustic": 0.25, "iaq": 0.25},
        "school": {"thermal": 0.30, "visual": 0.25, "acoustic": 0.25, "iaq": 0.20},
        "office": {"thermal": 0.35, "visual": 0.25, "acoustic": 0.20, "iaq": 0.20},
        "healthcare": {"thermal": 0.25, "visual": 0.20, "acoustic": 0.20, "iaq": 0.35},
    }
    if preset not in presets:
        return None
    return WeightSchema(weights=presets[preset], preset_name=preset)
