"""Tests for the About & sources tool (Task 8).

The page reads ``tool_pages``/``tool_messages`` from session state when
the shell put them there and falls back to the registry when rendered
standalone — so ``AppTest.from_function`` lists every built-in tool.
The shell-level check boots ``app.py`` with ``COMFIO_GUI_TOOLS`` limited
to the calculator plus the always-on About page.
"""

from __future__ import annotations

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from comfio.gui.tools import load_tools  # noqa: E402

APP = "src/comfio/gui/app.py"


def _about_app() -> None:
    from comfio.gui.tools.about.page import render

    render()


def test_lists_every_loaded_tool_title() -> None:
    at = AppTest.from_function(_about_app, default_timeout=60)
    at.run()
    assert not at.exception
    specs, _ = load_tools()
    rendered = " ".join(
        str(e.value) for e in list(at.markdown) + list(at.caption) + list(at.subheader)
    )
    for spec in specs:
        assert spec.title in rendered
        assert spec.id in rendered


def test_unknown_id_message_shown() -> None:
    at = AppTest.from_function(_about_app, default_timeout=60)
    at.session_state["tool_messages"] = ["Unknown tool id 'nope' — ignored."]
    at.run()
    assert not at.exception
    rendered = " ".join(
        str(e.value) for e in list(at.markdown) + list(at.warning) + list(at.caption)
    )
    assert "nope" in rendered


def test_default_when_workbench_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COMFIO_GUI_TOOLS", "calculator,about")
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    # the calculator is the default page — the workbench ingest
    # uploader must not render
    assert len(at.file_uploader) == 0
    assert len(at.metric) > 0
