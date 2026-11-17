"""Column auto-detection and mapping logic for the GUI.

Pure functions — no streamlit imports — so the logic is unit-testable
outside the app. The Streamlit widget layer lives in ``app.py``.
"""

from __future__ import annotations

import re

import pandas as pd

from comfio.core.data_handler import CANONICAL_COLUMNS, DEFAULT_ALIASES

# Columns that appear in data but are not sensor measurements.
_NON_SENSOR = {"timestamp", "tsv"}


def normalize(name: str) -> str:
    """Normalize a column name for matching: lowercase, units/parens stripped.

    ``"Zone Air Temp (°C)"`` -> ``"zone air temp"``; ``"CO2 [ppm]"`` -> ``"co2"``.
    """
    s = name.lower().strip()
    s = re.sub(r"\(.*?\)|\[.*?\]", "", s)  # strip (°C), [ppm] etc.
    s = re.sub(r"[^a-z0-9._]+", "_", s).strip("_")
    return s


def auto_map_columns(df: pd.DataFrame) -> dict[str, str]:
    """Best-effort canonical mapping for a DataFrame's columns.

    Exact alias lookup first (normalized), then a limited containment
    pass for names like ``zone_air_temp_c``. Never guesses beyond the
    canonical table — ambiguous columns are left unmapped for the user.

    Returns
    -------
    dict[str, str]
        ``{canonical_name: df_column_name}``
    """
    available = {normalize(c): c for c in df.columns}
    mapping: dict[str, str] = {}

    # Pass 1 — exact normalized alias match (same rule as SensorData).
    for alias, canonical in DEFAULT_ALIASES.items():
        norm = normalize(alias)
        if norm in available and canonical not in mapping:
            mapping[canonical] = available[norm]

    # Pass 2 — canonical name contained in a longer normalized column
    # name, e.g. "zone_1_air_temp_c" -> air_temp_c. Only canonical names,
    # never short aliases, to avoid false positives.
    for canonical in CANONICAL_COLUMNS:
        if canonical in mapping:
            continue
        for norm, original in available.items():
            if norm.endswith(canonical) or norm.startswith(canonical):
                mapping[canonical] = original
                break

    return mapping


def unmapped_columns(df: pd.DataFrame, mapping: dict[str, str]) -> list[str]:
    """DataFrame columns not used by any canonical mapping (info for UI)."""
    used = set(mapping.values()) | _NON_SENSOR
    return [c for c in df.columns if c not in used]


def missing_required(mapping: dict[str, str], required: list[str] | set[str]) -> list[str]:
    """Canonical columns in ``required`` with no mapped source."""
    return [c for c in required if c not in mapping]


# Requirements per domain for the capability banner (mirrors pipeline.py).
DOMAIN_REQUIREMENTS: dict[str, set[str]] = {
    "thermal": {"air_temp_c", "relative_humidity_pct"},
    "visual": {"illuminance_lux"},
    "acoustic": {"noise_laeq_db"},
    "iaq": {"co2_ppm"},
}


def domain_coverage(mapping: dict[str, str]) -> dict[str, bool]:
    """Which of the four domains can run given a column mapping."""
    return {d: not missing_required(mapping, req) for d, req in DOMAIN_REQUIREMENTS.items()}
