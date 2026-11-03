"""Learn — guided lessons that hand off to the comfort calculator.

The catalogue is data (``lessons.LESSONS``). Each step shows a live
PMV/PPD figure computed with ``evaluate_point`` and a "Try it" button
that seeds ``CALC_INPUTS`` and switches to the calculator — hidden when
that tool is disabled. ``Lesson.links`` render as ``st.page_link`` only
for tools the shell actually enabled.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import streamlit as st

from comfio.gui import _state
from comfio.gui.tools.calculator._models import METHODS, Inputs, evaluate_point
from comfio.gui.tools.learn.lessons import LESSONS, Step

# Inputs a step leaves unspecified fall back to a neutral office.
_DEFAULTS: dict[str, float] = {
    "tdb": 24.0,
    "tr": 24.0,
    "rh": 50.0,
    "vr": 0.1,
    "met": 1.2,
    "clo": 0.5,
}
_METHOD = METHODS[0]


def _render_step(step: Step, key: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{step.label}**")
        st.markdown(step.prompt)
        if not step.inputs:
            return
        res = evaluate_point(Inputs(**{**_DEFAULTS, **step.inputs}), _METHOD)
        if np.isnan(res.pmv):
            st.caption(f"{_METHOD} on these inputs: outside model range.")
        else:
            st.caption(
                f"{_METHOD} on these inputs: "
                f"PMV {res.pmv:+.2f} · PPD {res.ppd:.1f} %"
            )
        if _state.tool_enabled("calculator") and st.button(
            "Try it in the calculator", key=key
        ):
            _state.set(_state.CALC_INPUTS, dict(step.inputs))
            _state.goto("calculator")


def render() -> None:
    st.title("Learn")
    st.caption(
        "Short guided lessons on thermal comfort. Every PMV/PPD number "
        "shown is computed live — 'Try it' sends the same inputs to the "
        "comfort calculator."
    )
    lesson = st.selectbox(
        "Lesson",
        LESSONS,
        format_func=lambda lsn: f"{lsn.title} · ~{lsn.minutes} min",
        key="learn_lesson",
    )
    if lesson is None:
        return
    st.subheader(lesson.title)
    st.markdown(lesson.body_md)

    for i, step in enumerate(lesson.steps):
        _render_step(step, key=f"learn_try_{lesson.id}_{i}")

    enabled_links = [
        lid for lid in lesson.links if _state.tool_enabled(lid)
    ]
    if enabled_links:
        st.subheader("Related tools")
        pages = cast(dict[str, Any], _state.get(_state.TOOL_PAGES) or {})
        for lid in enabled_links:
            st.page_link(pages[lid])

    st.caption("Sources: " + " · ".join(lesson.sources))
