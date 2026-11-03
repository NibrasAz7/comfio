"""Comfort tool (classic) — self-contained CBE-style comfort calculator.

Rebuilds the classic multi-page comfort tool (ASHRAE-55, EN-16798,
Compare, Ranges) as one self-contained HTML document rendered inside an
iframe. The physics in ``_assets/physics.js`` is a JavaScript port of the
pythermalcomfort model family (PMV/PPD ISO 7730, Gagge two-node SET,
ASHRAE 55-2023 relative-air-speed and dynamic-clothing corrections,
Appendix-H cooling effect, and the ASHRAE/EN adaptive models).
"""

from __future__ import annotations

import importlib.resources
from functools import lru_cache

import streamlit as st
import streamlit.components.v1 as components


@lru_cache(maxsize=1)
def _assemble_html() -> str | None:
    """Inline the four assets into one self-contained HTML document."""
    try:
        root = importlib.resources.files(__package__) / "_assets"
        body = (root / "page.html").read_text(encoding="utf-8")
        css = (root / "site.css").read_text(encoding="utf-8")
        physics = (root / "physics.js").read_text(encoding="utf-8")
        app = (root / "app.js").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError):
        return None
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<style>{css}</style></head>"
        "<body data-page='index'>"
        f"{body}<script>{physics}</script><script>{app}</script>"
        "</body></html>"
    )


def render() -> None:
    st.title("Comfort tool (classic)")
    st.caption(
        "The classic interactive comfort calculator — ASHRAE 55-2023 "
        "(PMV + adaptive), EN 16798-1:2019, side-by-side comparison and "
        "comfort ranges. Physics: pythermalcomfort model family "
        "(Tartarini & Schiavon 2020, doi:10.1016/j.softx.2020.100578). "
        "Batch analysis of uploaded datasets lives in the workbench "
        "(Analyse my data)."
    )
    html = _assemble_html()
    if html is None:
        st.error(
            "Comfort tool assets are missing from the installed package "
            "(expected _assets/{page.html,site.css,physics.js,app.js})."
        )
        return
    components.html(html, height=1500, scrolling=True)
