"""Tests for the MRT room designer tool (Task 4).

``AppTest.from_function`` renders the page in isolation — there is no
real ``st.Page`` registry, so ``st.switch_page`` raises
``StreamlitAPIException`` ("Could not find page") which lands in
``at.exception``. The ``CALC_INPUTS`` write happens *before* the switch,
so the handoff test asserts on session state and only tolerates the
navigation failure itself.
"""

from __future__ import annotations

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _mrt_app() -> None:
    from comfio.gui.tools.mrt.page import render

    render()


@pytest.fixture()
def at() -> AppTest:
    st.cache_data.clear()
    return AppTest.from_function(_mrt_app, default_timeout=60)


def test_page_renders_heatmap(at: AppTest) -> None:
    at.run()
    assert not at.exception
    assert len(at.get("plotly_chart")) >= 1


def test_pmv_map_toggle_renders(at: AppTest) -> None:
    at.run()
    assert not at.exception
    radios = [r for r in at.radio if "map" in (r.label or "").lower()]
    assert radios
    radios[0].set_value("PMV").run()
    assert not at.exception
    assert len(at.get("plotly_chart")) >= 1


def test_handoff_sets_calc_inputs(at: AppTest) -> None:
    # "Use in calculator" only renders when the shell registered the page
    at.session_state["tool_pages"] = {"calculator": "calculator"}
    at.run()
    buttons = [b for b in at.button if "calculator" in (b.label or "").lower()]
    assert buttons, "handoff button not rendered"
    buttons[0].click().run()
    # The switch_page call cannot find a real page under AppTest and
    # raises StreamlitAPIException; the state write precedes it.
    ci = at.session_state["calc_inputs"]
    assert set(ci) == {"tdb", "tr", "rh", "vr", "met", "clo"}
    # default room is uniform 22 °C -> occupant MRT ~22
    assert ci["tr"] == pytest.approx(22.0, abs=0.5)
    assert ci["tdb"] == pytest.approx(24.0)
    # any recorded exception must be the switch-page navigation, not a bug
    for exc in at.exception:
        assert "page" in str(exc.value).lower()


def test_no_handoff_button_without_calculator(at: AppTest) -> None:
    at.run()
    assert not at.exception
    labels = [(b.label or "") for b in at.button]
    assert not any("calculator" in label.lower() for label in labels)
