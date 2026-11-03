"""Pure logic for the occupant-survey tool.

No streamlit imports — everything here is unit-testable outside the app.

Survey votes are stored on the CBE-style continuous satisfaction scale
−3…+3 (satisfied > 0, neutral = 0, dissatisfied < 0). Files exported on a
1–7 Likert scale are detected and, only when the user opts in, remapped
by subtracting 4.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from comfio.gui._mapping import normalize

SAT_COLUMNS: dict[str, str] = {
    "sat_thermal": "Thermal",
    "sat_air": "Air quality",
    "sat_acoustic": "Acoustics",
    "sat_lighting": "Lighting",
    "sat_cleanliness": "Cleanliness",
    "sat_layout": "Layout",
    "sat_furniture": "Furniture",
    "sat_overall": "Overall",
}

# Satisfaction categories that can be compared against a sensor-domain
# score produced by the workbench analysis.
DOMAIN_OF: dict[str, str] = {
    "sat_thermal": "thermal",
    "sat_air": "iaq",
    "sat_acoustic": "acoustic",
    "sat_lighting": "visual",
}

# ASHRAE 55 defines an acceptable thermal environment as one where at
# least 80 % of occupants are satisfied — used here as the default
# pass/fail line for every category, labelled as a user target in the UI.
THERMAL_ACCEPTABILITY_TARGET = 80.0

SUMMARY_COLUMNS = [
    "category",
    "group",
    "n",
    "pct_satisfied",
    "pct_neutral",
    "pct_dissatisfied",
    "mean",
]

# Small normalized alias set for auto-detection. normalize() folds case,
# whitespace and punctuation, so "Thermal Satisfaction" -> thermal_satisfaction.
_ALIASES: dict[str, tuple[str, ...]] = {
    "sat_thermal": (
        "sat_thermal",
        "thermal_satisfaction",
        "thermal_comfort",
        "temperature_satisfaction",
    ),
    "sat_air": (
        "sat_air",
        "air_quality",
        "air_quality_satisfaction",
        "air_satisfaction",
        "iaq_satisfaction",
    ),
    "sat_acoustic": (
        "sat_acoustic",
        "acoustic",
        "acoustics",
        "acoustic_satisfaction",
        "acoustics_satisfaction",
        "noise_satisfaction",
        "sound_satisfaction",
    ),
    "sat_lighting": (
        "sat_lighting",
        "lighting",
        "lighting_satisfaction",
        "light_satisfaction",
        "visual_satisfaction",
    ),
    "sat_cleanliness": (
        "sat_cleanliness",
        "cleanliness",
        "cleanliness_satisfaction",
    ),
    "sat_layout": (
        "sat_layout",
        "layout",
        "layout_satisfaction",
        "space_layout",
        "office_layout",
    ),
    "sat_furniture": (
        "sat_furniture",
        "furniture",
        "furniture_satisfaction",
    ),
    "sat_overall": (
        "sat_overall",
        "overall",
        "overall_satisfaction",
        "satisfaction_overall",
    ),
}


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Best-effort mapping of survey columns to canonical satisfaction keys.

    Exact normalized alias match only — never guesses beyond the alias
    table. Returns ``{canonical: actual_column}``.
    """
    available = {normalize(str(c)): c for c in df.columns}
    mapping: dict[str, str] = {}
    for canonical in SAT_COLUMNS:
        for alias in _ALIASES.get(canonical, ()):
            original = available.get(normalize(alias))
            if original is not None:
                mapping[canonical] = original
                break
    return mapping


def validate_scale(s: pd.Series) -> Literal["-3..3", "1..7", "invalid"]:
    """Classify a satisfaction column's answer scale.

    Blank cells count as non-responses and are ignored; a column is
    ``invalid`` when it contains text answers (non-numeric values),
    numbers outside both scales, or no numeric answers at all.
    """
    values = pd.to_numeric(s, errors="coerce")
    non_response = s.isna() | (s.astype(str).str.strip() == "")
    if (values.isna() & ~non_response).any():
        return "invalid"  # text answers such as "very satisfied"
    nums = values.dropna()
    if nums.empty:
        return "invalid"  # nothing to score
    uniq = {float(v) for v in nums.unique()}
    if uniq <= {float(v) for v in range(-3, 4)}:
        return "-3..3"
    if uniq <= {float(v) for v in range(1, 8)}:
        return "1..7"
    return "invalid"


def summarize(
    df: pd.DataFrame,
    mapping: dict[str, str],
    *,
    scale: Literal["-3..3", "1..7"] = "-3..3",
    by: str | None = None,
) -> pd.DataFrame:
    """Per-category satisfaction breakdown.

    Returns columns ``category, group, n, pct_satisfied, pct_neutral,
    pct_dissatisfied, mean`` — one row per (category, group) with at
    least one answer. ``group`` is ``all`` when ``by`` is not given; a
    1–7 scale is shifted by −4 only when ``scale == "1..7"``.
    """
    if by is not None and by in df.columns:
        group_vals = df[by].astype(str)
    else:
        group_vals = pd.Series(["all"] * len(df), index=df.index, dtype=str)

    rows: list[dict[str, Any]] = []
    for canonical in SAT_COLUMNS:
        actual = mapping.get(canonical)
        if actual is None or actual not in df.columns:
            continue
        values = pd.to_numeric(df[actual], errors="coerce")
        if scale == "1..7":
            values = values - 4.0
        for group in sorted(group_vals.unique()):
            v = values[group_vals == group].dropna()
            n = int(v.size)
            if n == 0:
                continue
            rows.append(
                {
                    "category": canonical,
                    "group": group,
                    "n": n,
                    "pct_satisfied": round(100.0 * float((v > 0).mean()), 1),
                    "pct_neutral": round(100.0 * float((v == 0).mean()), 1),
                    "pct_dissatisfied": round(100.0 * float((v < 0).mean()), 1),
                    "mean": round(float(v.mean()), 2),
                }
            )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def sensor_vs_occupant(
    summary: pd.DataFrame, result: Any, target_pct: float = THERMAL_ACCEPTABILITY_TARGET
) -> pd.DataFrame | None:
    """Compare survey satisfaction with the workbench sensor verdict.

    Joins DOMAIN_OF categories onto ``result.ieq_result.domain_scores``
    (a per-sample 0–100 score array; the sensor side is its mean). A row
    is flagged when the two sides disagree: sensor mean ≥ ``target_pct``
    while occupant satisfaction < ``target_pct``, or the reverse.
    ``target_pct`` is a user target — ASHRAE 55 defines 80 % for the
    thermal domain only.
    """
    ieq = getattr(result, "ieq_result", None)
    domain_scores = getattr(ieq, "domain_scores", None) if ieq is not None else None
    if not domain_scores or summary is None or summary.empty:
        return None
    rows: list[dict[str, Any]] = []
    for r in summary.itertuples(index=False):
        category = str(r.category)
        domain = DOMAIN_OF.get(category)
        if domain is None or domain not in domain_scores:
            continue
        scores = np.asarray(domain_scores[domain], dtype=float)
        sensor_mean = float(np.nanmean(scores)) if scores.size else float("nan")
        if np.isnan(sensor_mean):
            continue
        pct_satisfied = float(r.pct_satisfied)
        sensor_ok = sensor_mean >= target_pct
        occupant_ok = pct_satisfied >= target_pct
        rows.append(
            {
                "category": category,
                "domain": domain,
                "group": r.group,
                "n": int(r.n),
                "occupant_pct_satisfied": pct_satisfied,
                "occupant_pct_dissatisfied": float(r.pct_dissatisfied),
                "sensor_mean_score": round(sensor_mean, 1),
                "flag": bool(sensor_ok != occupant_ok),
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def demo_survey(seed: int = 7) -> pd.DataFrame:
    """Deterministic demo survey: 60 respondents across two zones.

    ``corner`` zone answers thermal / acoustic noticeably worse so the
    group breakdown and the sensor-vs-occupant check have something to
    show. All votes are integers on the −3…+3 scale.
    """
    rng = np.random.default_rng(seed)
    n_zone = 30
    n = 2 * n_zone
    zone = np.repeat(["open-plan", "corner"], n_zone)
    corner = (zone == "corner").astype(float)

    def votes(center: np.ndarray | float, sd: float = 1.4) -> np.ndarray:
        c = center if isinstance(center, np.ndarray) else np.full(n, float(center))
        return np.clip(np.rint(c + rng.normal(0, sd, n)), -3, 3).astype(int)

    df = pd.DataFrame(
        {
            "respondent_id": [f"R{i + 1:03d}" for i in range(n)],
            "zone": zone,
            "sat_thermal": votes(0.6 - 1.6 * corner),
            "sat_air": votes(0.8),
            "sat_acoustic": votes(0.4 - 1.4 * corner),
            "sat_lighting": votes(1.2),
            "sat_cleanliness": votes(1.0),
            "sat_layout": votes(0.2 - 0.6 * corner),
            "sat_furniture": votes(0.9),
        }
    )
    # Overall ≈ mean of the domain votes, rounded to an integer.
    df["sat_overall"] = (
        df[[k for k in SAT_COLUMNS if k != "sat_overall"]]
        .mean(axis=1)
        .round()
        .astype(int)
        .clip(-3, 3)
    )
    return df


def template_csv() -> str:
    """Downloadable CSV template with the canonical headers + 3 rows."""
    header = ",".join(["respondent_id", "zone", *SAT_COLUMNS.keys()])
    rows = [
        "R001,open-plan,1,0,-1,2,1,0,-1,1",
        "R002,open-plan,0,1,-2,1,2,1,0,0",
        "R003,corner,-2,-1,-1,0,1,-1,-2,-1",
    ]
    return header + "\n" + "\n".join(rows) + "\n"
