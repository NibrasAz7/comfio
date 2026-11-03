"""Tests for the Learn tool (Task 7).

The lesson catalogue is pure data, so most checks run without a
browser. ``AppTest.from_function`` renders the page standalone — there
is no real ``st.Page`` registry, so the test that exercises links and
the calculator handoff seeds ``tool_pages`` inside the app function
(``st.Page`` objects built in bare mode are not fully initialised).
``st.switch_page`` then raises the expected "page not found" which only
``at.exception`` records — the ``CALC_INPUTS`` write precedes it.
"""

from __future__ import annotations

import re
from dataclasses import fields

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from comfio.gui.tools.calculator._models import Inputs  # noqa: E402
from comfio.gui.tools.learn import lessons  # noqa: E402
from comfio.gui.tools.learn.page import _DEFAULTS  # noqa: E402

# Plan Global Constraints — the only DOIs and standard names that may
# appear in lesson sources.
WHITELISTED_DOIS = {
    "10.1016/j.softx.2020.100578",
    "10.1016/j.buildenv.2018.06.022",
    "10.6078/D1F671",
    "10.1016/j.buildenv.2014.09.004",
    "10.1016/j.buildenv.2014.09.010",
    "10.1016/S0378-7788(02)00005-1",
    "10.5334/bc.76",
}
STANDARD_NAMES = {
    "ISO 7730:2005",
    "ASHRAE 55-2023",
    "EN 16798-1:2019",
    "ISO 7726:1998",
}
_INPUT_FIELDS = {f.name for f in fields(Inputs)}
# PMV/PPD numbers must be computed live via evaluate_point — a lesson
# may name the metrics but must never state a value for them.
_PMV_PPD_LITERAL = re.compile(
    r"\b(?:pmv|ppd)\b[^a-zA-Z\d]{0,8}\d", re.IGNORECASE
)


def _learn_app() -> None:
    from comfio.gui.tools.learn.page import render

    render()


def _learn_app_with_pages() -> None:
    """Learn page with the linked tools registered in the fake shell."""
    import streamlit as st

    from comfio.gui import _state

    if _state.TOOL_PAGES not in st.session_state:
        st.session_state[_state.TOOL_PAGES] = {
            tid: st.Page(lambda: None, title=tid, url_path=tid)
            for tid in ("calculator", "mrt", "setpoint", "benchmark")
        }
    from comfio.gui.tools.learn.page import render

    render()


@pytest.fixture()
def at() -> AppTest:
    st.cache_data.clear()
    return AppTest.from_function(_learn_app, default_timeout=60)


def test_step_inputs_valid() -> None:
    assert len(lessons.LESSONS) >= 5
    for lesson in lessons.LESSONS:
        assert lesson.steps, lesson.id
        for step in lesson.steps:
            assert step.label and step.prompt
            unknown = set(step.inputs) - _INPUT_FIELDS
            assert not unknown, (lesson.id, step.label, unknown)
            # step inputs must merge cleanly into a full Inputs
            Inputs(**{**_DEFAULTS, **step.inputs})


def test_lesson_ids_unique() -> None:
    ids = [lesson.id for lesson in lessons.LESSONS]
    assert len(ids) == len(set(ids))


def test_sources_whitelisted() -> None:
    for lesson in lessons.LESSONS:
        assert lesson.sources, lesson.id
        for src in lesson.sources:
            dois = re.findall(r"doi:(\S+)", src)
            if dois:
                assert all(d in WHITELISTED_DOIS for d in dois), src
            else:
                assert src in STANDARD_NAMES, src


def test_no_literal_pmv_ppd_values() -> None:
    for lesson in lessons.LESSONS:
        assert not _PMV_PPD_LITERAL.search(lesson.body_md), lesson.id
        for step in lesson.steps:
            assert not _PMV_PPD_LITERAL.search(step.prompt), step.label


def test_links_point_to_known_other_tools() -> None:
    from comfio.gui.tools import load_tools

    known = {s.id for s in load_tools()[0]}
    for lesson in lessons.LESSONS:
        assert set(lesson.links) <= known - {"learn"}, lesson.id


def test_try_it_sets_calc_inputs() -> None:
    at = AppTest.from_function(_learn_app_with_pages, default_timeout=60)
    at.run()
    assert not at.exception
    buttons = [b for b in at.button if "try it" in (b.label or "").lower()]
    assert buttons, "'Try it' button not rendered"
    buttons[0].click().run()
    ci = at.session_state["calc_inputs"]
    assert isinstance(ci, dict) and set(ci) <= _INPUT_FIELDS
    # any recorded exception must be the switch-page navigation, not a bug
    for exc in at.exception:
        assert "page" in str(exc.value).lower()


def test_links_hidden_when_tool_disabled(at: AppTest) -> None:
    at.run()  # no tool_pages seeded -> every tool reads as disabled
    assert not at.exception
    assert len(at.get("page_link")) == 0
    labels = [(b.label or "") for b in at.button]
    assert not any("try it" in label.lower() for label in labels)


def test_links_rendered_when_tool_enabled() -> None:
    at = AppTest.from_function(_learn_app_with_pages, default_timeout=60)
    at.run()
    lesson = next(lsn for lsn in lessons.LESSONS if lsn.links)
    at.selectbox[0].set_value(lesson).run()
    assert not at.exception
    assert len(at.get("page_link")) >= len(
        [lid for lid in lesson.links if lid != "learn"]
    )
