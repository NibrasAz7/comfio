"""About & sources — what is enabled, and where the numbers come from.

Lists the tools the shell enabled (falling back to the registry when
the page is rendered without a shell), the load messages (unknown ids,
missing optional dependencies), the legal provenance of everything the
GUI builds on, and the citations for every model and dataset used.
This tool is always on.
"""

from __future__ import annotations

import os
from typing import Any, cast

import pandas as pd
import streamlit as st

from comfio.gui import _state
from comfio.gui.tools import enabled_from_env, load_tools

# Summarised from the design record (plan Part A), in our own words.
_PROVENANCE: list[tuple[str, str, str]] = [
    (
        "pythermalcomfort",
        "MIT",
        "Dependency — all PMV/PPD, SET and cooling-effect physics.",
    ),
    (
        "ASHRAE Global Thermal Comfort Database II",
        "CC0-1.0",
        "Benchmark dataset, downloaded on demand and hash-verified.",
    ),
    (
        "mrt_replication (Abo Alzahab et al.)",
        "MIT",
        "View-factor engine ported for the MRT room designer; the MIT "
        "notice is kept in comfio.domains.thermal_mrt.",
    ),
    (
        "CBE comfort_tool",
        "GPL-2.0",
        "Behaviour reference only — nothing copied; physics comes from "
        "pythermalcomfort.",
    ),
    (
        "CBE mrt tool",
        "GPL-3.0",
        "Not used — the MRT designer ports the MIT-licensed engine "
        "above instead.",
    ),
    (
        "CBE comfort-dash / energy-calc / comfortdatabase / "
        "cozie-questionnaires",
        "No public licence",
        "Not copied — comfio links out or rebuilds the idea from "
        "papers and standards.",
    ),
    (
        "StudioMatrx guided lesson",
        "Copyrighted",
        "Idea of a guided lesson only — all lesson wording is comfio's "
        "own.",
    ),
]

_CITATIONS: tuple[str, ...] = (
    "Tartarini & Schiavon (2020), pythermalcomfort: a Python package "
    "for thermal comfort research, SoftwareX 12:100578 — "
    "doi:10.1016/j.softx.2020.100578",
    "Földváry Ličina et al. (2018), Development of the ASHRAE Global "
    "Thermal Comfort Database II, Building and Environment "
    "142:502–512 — doi:10.1016/j.buildenv.2018.06.022",
    "ASHRAE Global Thermal Comfort Database II, Dryad dataset, "
    "CC0-1.0 — doi:10.6078/D1F671",
    "Arens, Hoyt, Zhou, Huang, Zhang & Schiavon (2015), Modeling the "
    "comfort effects of short-wave solar radiation indoors, Building "
    "and Environment 88:3–9 — doi:10.1016/j.buildenv.2014.09.004",
    "Hoyt, Arens & Zhang (2015), Extending air temperature setpoints: "
    "simulated energy savings and design considerations for new and "
    "retrofit buildings, Building and Environment 88:89–96 — "
    "doi:10.1016/j.buildenv.2014.09.010",
    "de Dear & Brager (2002), Thermal comfort in naturally ventilated "
    "buildings: revisions to ASHRAE Standard 55, Energy and Buildings "
    "34(6):549–561 — doi:10.1016/S0378-7788(02)00005-1",
    "Graham, Parkinson & Schiavon (2021), Lessons learned from 20 "
    "years of CBE's occupant surveys, Buildings and Cities "
    "2(1):166–184 — doi:10.5334/bc.76",
)


def _enabled_tools() -> list[tuple[str, str]]:
    """(title, id) pairs — live shell pages first, registry fallback."""
    pages = cast(dict[str, Any], _state.get(_state.TOOL_PAGES) or {})
    all_specs, _ = load_tools()
    by_id = {s.id: s for s in all_specs}
    if pages:
        return [
            (by_id[pid].title if pid in by_id else pid, pid)
            for pid in pages
        ]
    specs, _ = load_tools(
        enabled_from_env(os.environ.get("COMFIO_GUI_TOOLS"))
    )
    return [(s.title, s.id) for s in specs]


def render() -> None:
    st.title("About & sources")
    st.caption(
        "comfio is open source under the MIT licence. This page lists "
        "what is running, what each part is built on and where the "
        "numbers come from."
    )

    st.subheader("Enabled tools")
    st.markdown(
        "\n".join(
            f"- **{title}** (`{tool_id}`)"
            for title, tool_id in _enabled_tools()
        )
    )

    messages = cast(
        list[str], _state.get(_state.TOOL_MESSAGES) or []
    )
    if messages:
        st.subheader("Unavailable or ignored tools")
        for message in messages:
            st.warning(message)

    st.subheader("Provenance & licences")
    st.dataframe(
        pd.DataFrame(
            _PROVENANCE,
            columns=["source", "licence", "how comfio uses it"],
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "comfio stays MIT: nothing is copied from GPL or unlicensed "
        "sources — behaviour is rebuilt from standards and "
        "peer-reviewed papers, or linked out."
    )

    st.subheader("Citations")
    st.markdown("\n".join(f"- {c}" for c in _CITATIONS))
    st.caption(
        "Standards cited by name: ASHRAE 55-2023 · ISO 7730:2005 · "
        "EN 16798-1:2019 · ISO 7726:1998. Model outputs outside their "
        "applicability range are shown as 'outside model range', never "
        "clamped or extrapolated."
    )
