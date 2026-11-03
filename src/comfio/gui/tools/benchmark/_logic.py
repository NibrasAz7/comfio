"""Pure logic for the global-database benchmark tool.

No streamlit imports — everything here is unit-testable outside the app.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

FILTERS = ("climate", "building_type", "cooling_type", "season", "country")

_TOP_SOURCE_DB = "database"
_TOP_SOURCE_COMPUTED = "operative_tmp(ta, tr, vel)"
_TOP_SOURCE_UNAVAILABLE = "unavailable"


def filter_db(db: pd.DataFrame, **selected: Sequence[str]) -> pd.DataFrame:
    """Subset ``db`` by the FILTERS columns; empty selection = no filter."""
    out = db
    for col in FILTERS:
        sel = selected.get(col)
        if sel and col in out.columns:
            out = out[out[col].isin(list(sel))]
    return out


def with_operative(db: pd.DataFrame) -> pd.DataFrame:
    """Ensure a ``top`` operative-temperature column + ``top_source`` note.

    The DB already ships ``top``; rows where it is missing are filled with
    ``operative_tmp(ta, tr, vel)`` (v < 0.2 m/s → (ta+tr)/2 per ISO 7726)
    when ``ta``/``tr`` are present, and flagged via ``top_source``.
    """
    out = db.copy()
    has_top = "top" in out.columns and bool(out["top"].notna().any())
    out["top_source"] = _TOP_SOURCE_DB if has_top else _TOP_SOURCE_UNAVAILABLE
    if "ta" in out.columns and "tr" in out.columns:
        from pythermalcomfort.utilities import operative_tmp

        vel = (
            out["vel"].to_numpy(dtype=float, na_value=0.1)
            if "vel" in out.columns
            else np.full(len(out), 0.1)
        )
        computed = np.asarray(
            operative_tmp(
                out["ta"].to_numpy(dtype=float),
                out["tr"].to_numpy(dtype=float),
                vel,
            ),
            dtype=float,
        )
        if "top" in out.columns:
            fill = out["top"].isna() & np.isfinite(computed)
            out.loc[fill, "top"] = computed[fill.to_numpy()]
            out.loc[fill, "top_source"] = _TOP_SOURCE_COMPUTED
        else:
            out["top"] = computed
            out["top_source"] = np.where(
                np.isfinite(computed), _TOP_SOURCE_COMPUTED, _TOP_SOURCE_UNAVAILABLE
            )
    elif "top" not in out.columns:
        out["top"] = np.nan
    return out


def sensation_bins(df: pd.DataFrame, bin_c: float = 1.0) -> pd.DataFrame:
    """Mean TSV / % acceptable / mean PMV per ``bin_c`` °C operative bin.

    Returns columns ``t_bin, n, mean_tsv, pct_acceptable, mean_pmv``.
    ``pct_acceptable`` is the share of votes labelled ``acceptable`` among
    rows that carry a ``thermal_acceptability`` value.
    """
    cols = ["t_bin", "n", "mean_tsv", "pct_acceptable", "mean_pmv"]
    if "top" not in df.columns:
        return pd.DataFrame(columns=cols)
    d = df.copy()
    for col in ("thermal_sensation", "pmv"):
        if col not in d.columns:
            d[col] = np.nan
    if "thermal_acceptability" in d.columns:
        d["_acceptable"] = d["thermal_acceptability"].map(
            {"acceptable": 1.0, "unacceptable": 0.0}
        )
    else:
        d["_acceptable"] = np.nan
    d = d.dropna(subset=["top"])
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["t_bin"] = np.floor(d["top"].astype(float) / bin_c) * bin_c
    out = (
        d.groupby("t_bin")
        .agg(
            n=("top", "size"),
            mean_tsv=("thermal_sensation", "mean"),
            pct_acceptable=("_acceptable", "mean"),
            mean_pmv=("pmv", "mean"),
        )
        .reset_index()
    )
    out["pct_acceptable"] = out["pct_acceptable"] * 100.0
    return out


def pmv_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Mean (``pmv`` − ``thermal_sensation``) per ``cooling_type``.

    Returns columns ``cooling_type, n, mean_bias``; ``n`` counts rows with
    both PMV and TSV present.
    """
    cols = ["cooling_type", "n", "mean_bias"]
    if "cooling_type" not in df.columns:
        return pd.DataFrame(columns=cols)
    d = df.copy()
    for col in ("pmv", "thermal_sensation"):
        if col not in d.columns:
            d[col] = np.nan
    d["bias"] = d["pmv"] - d["thermal_sensation"]
    return (
        d.groupby("cooling_type")["bias"]
        .agg(n="count", mean_bias="mean")
        .reset_index()
    )


def user_overlay(
    prepared: pd.DataFrame, mapping: dict[str, str]
) -> pd.DataFrame | None:
    """Map a workbench frame onto the DB scatter columns.

    Produces ``top`` (operative via ``operative_tmp`` from the mapped
    ``air_temp_c`` + ``radiant_temp_c``, or the air temperature alone with
    a ``top_source`` note), a ``t_mot``-like outdoor column when one of
    ``running_mean_outdoor_c`` / ``prevailing_mean_outdoor_c`` /
    ``outdoor_temp_c`` is mapped (source recorded in ``t_mot_source``),
    and ``thermal_sensation`` when ``tsv`` is mapped. Returns ``None``
    when nothing usable can be built.
    """
    if prepared is None or prepared.empty:
        return None
    out = pd.DataFrame(index=prepared.index)

    ta = mapping.get("air_temp_c")
    if ta is not None and ta in prepared.columns:
        tr = mapping.get("radiant_temp_c")
        if tr is not None and tr in prepared.columns:
            vel_col = mapping.get("air_velocity_ms")
            vel = (
                prepared[vel_col].to_numpy(dtype=float, na_value=0.1)
                if vel_col is not None and vel_col in prepared.columns
                else np.full(len(prepared), 0.1)
            )
            from pythermalcomfort.utilities import operative_tmp

            out["top"] = operative_tmp(
                prepared[ta].to_numpy(dtype=float),
                prepared[tr].to_numpy(dtype=float),
                vel,
            )
            out["top_source"] = _TOP_SOURCE_COMPUTED
        else:
            out["top"] = pd.to_numeric(prepared[ta], errors="coerce")
            out["top_source"] = "air temperature only (radiant not mapped)"

    for key in (
        "running_mean_outdoor_c",
        "prevailing_mean_outdoor_c",
        "outdoor_temp_c",
    ):
        col = mapping.get(key)
        if col is not None and col in prepared.columns:
            out["t_mot"] = pd.to_numeric(prepared[col], errors="coerce")
            out["t_mot_source"] = key
            break

    tsv = mapping.get("tsv")
    if tsv is None and "tsv" in prepared.columns:
        tsv = "tsv"  # tsv is not a canonical column; accept a literal match
    if tsv is not None and tsv in prepared.columns:
        out["thermal_sensation"] = pd.to_numeric(prepared[tsv], errors="coerce")

    if len(out.columns) == 0:
        return None
    return out
