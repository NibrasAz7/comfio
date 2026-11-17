"""Certification-preset checks for the Evidence act.

Pure logic (no streamlit) so the criteria are unit-testable. Each check
returns plain dicts the UI renders as a per-criterion table.

Honesty note: these are *sample-level* approximations of the published
requirements. Real certification audits aggregate over occupied hours
and may require calibrated-instrument data — the UI says so.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# RESET Air (Commercial Interiors, basic performance targets).
# Continuous monitoring, >=90% data coverage, ~3 consecutive months.
RESET_AIR_TARGETS: dict[str, tuple[str, str]] = {
    "pm25_ugm3": ("PM2.5", "≤ 12 µg/m³"),
    "tvoc_ugm3": ("TVOC", "≤ 500 µg/m³"),
    "co2_ppm": ("CO₂", "≤ 600 ppm"),
    "relative_humidity_pct": ("Relative humidity", "40–60 %"),
}


def _within(name: str, v: pd.Series) -> pd.Series:
    if name == "pm25_ugm3":
        return v <= 12.0
    if name == "tvoc_ugm3":
        return v <= 500.0
    if name == "co2_ppm":
        return v <= 600.0
    # relative_humidity_pct
    return (v >= 40.0) & (v <= 60.0)


def reset_air(
    df: pd.DataFrame,
    mapping: dict[str, str],
    ts_col: str | None = None,
) -> dict[str, Any]:
    """RESET Air basic-target check.

    Returns per-pollutant coverage (fraction of non-NaN samples),
    within-target rate, and the missing-input list.
    """
    criteria: list[dict[str, Any]] = []
    missing = []
    for canonical, (label, limit) in RESET_AIR_TARGETS.items():
        col = mapping.get(canonical)
        if col is None or col not in df.columns:
            missing.append(label)
            criteria.append(
                {
                    "criterion": label,
                    "target": limit,
                    "coverage %": None,
                    "within target %": None,
                    "status": "no data",
                }
            )
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        valid = v.notna()
        coverage = 100.0 * float(valid.mean()) if len(v) else 0.0
        within = 100.0 * float(_within(canonical, v[valid]).mean()) if valid.any() else 0.0
        criteria.append(
            {
                "criterion": label,
                "target": limit,
                "coverage %": round(coverage, 1),
                "within target %": round(within, 1),
                "status": "pass" if coverage >= 90.0 and within >= 90.0 else "fail",
            }
        )

    duration_days: float | None = None
    if ts_col is not None and ts_col in df.columns:
        ts = pd.to_datetime(df[ts_col], errors="coerce").dropna()
        if len(ts) >= 2:
            duration_days = float((ts.max() - ts.min()).total_seconds() / 86400.0)

    n_pass = sum(1 for c in criteria if c["status"] == "pass")
    return {
        "preset": "RESET Air (basic targets)",
        "criteria": criteria,
        "missing": missing,
        "duration_days": duration_days,
        "criteria_passed": n_pass,
        "criteria_total": len(criteria),
        "caveats": [
            "Sample-level approximation — certification uses occupied-hour "
            "averages and requires ~3 consecutive months at ≥90% coverage.",
        ]
        + (
            ["No timestamp column — monitoring duration cannot be verified."]
            if duration_days is None
            else []
        ),
    }


def well_t06(thermal_result: Any) -> dict[str, Any]:
    """WELL v2 T06 thermal comfort monitoring check.

    Requirement: PMV within ±0.5 for ≥90% of occupied hours.
    """
    if thermal_result is None or getattr(thermal_result, "pmv", None) is None:
        return {
            "preset": "WELL v2 T06 (thermal comfort monitoring)",
            "criteria": [],
            "available": False,
            "caveats": ["Thermal domain did not run — no PMV data."],
        }

    pmv = np.asarray(thermal_result.pmv, dtype=float)
    valid = ~np.isnan(pmv)
    frac = float((np.abs(pmv[valid]) <= 0.5).mean()) if valid.any() else 0.0
    ok = frac >= 0.90
    return {
        "preset": "WELL v2 T06 (thermal comfort monitoring)",
        "available": True,
        "criteria": [
            {
                "criterion": "PMV within ±0.5",
                "target": "≥ 90% of occupied hours",
                "coverage %": round(100.0 * float(valid.mean()), 1),
                "within target %": round(100.0 * frac, 1),
                "status": "pass" if ok else "fail",
            }
        ],
        "criteria_passed": int(ok),
        "criteria_total": 1,
        "caveats": [
            "Sample-level approximation — certification aggregates over "
            "occupied hours and may weight zones differently.",
        ],
    }
