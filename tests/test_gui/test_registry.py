"""Tests for the tool registry and the multipage shell (Task 1).

Covers ``ToolSpec``/``load_tools``/``enabled_from_env``, the hosted
``streamlit_app.py`` entrypoint (which must re-render on every rerun),
and the zero-enabled-tools error path.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from comfio.gui.tools import (  # noqa: E402
    BUILTIN_TOOLS,
    ToolSpec,
    enabled_from_env,
    load_tools,
)

APP = "src/comfio/gui/app.py"
ROOT = Path(__file__).resolve().parents[2]


def test_load_tools_default_order() -> None:
    specs, _ = load_tools()
    assert [s.id for s in specs][0] == "workbench"
    assert len(specs) == len(BUILTIN_TOOLS)


def test_unknown_id_reported() -> None:
    specs, messages = load_tools(["workbench", "nope"])
    # "about" is always_on — it loads even when not requested
    assert [s.id for s in specs] == ["workbench", "about"]
    assert any("nope" in m for m in messages)


def test_missing_requires_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    import comfio.gui.tools as tools_pkg

    fake = types.ModuleType("comfio.gui.tools.fake")
    fake.TOOL = ToolSpec(  # type: ignore[attr-defined]
        id="fake",
        title="Fake tool",
        icon=":material/science:",
        group="Analyse",
        target="comfio.gui.tools.workbench.page:render",
        requires=("no_such_mod_xyz",),
    )
    monkeypatch.setitem(sys.modules, "comfio.gui.tools.fake", fake)
    monkeypatch.setattr(tools_pkg, "BUILTIN_TOOLS", (*BUILTIN_TOOLS, "comfio.gui.tools.fake"))

    specs, messages = load_tools()
    assert "fake" not in [s.id for s in specs]
    assert any("no_such_mod_xyz" in m for m in messages)


def test_enabled_from_env() -> None:
    assert enabled_from_env("workbench, calculator") == ["workbench", "calculator"]
    assert enabled_from_env("") is None
    assert enabled_from_env(None) is None


def test_entrypoint_reruns() -> None:
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    for b in at.button:
        if "demo data" in (b.label or ""):
            b.click().run()
            break
    at.run()
    assert not at.exception
    assert any("Column mapping" in s.value for s in at.subheader)
    assert len(at.selectbox) > 0


def test_unknown_env_id_boots_on_about(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With an always_on tool (About) the "zero enabled tools" error path
    # is unreachable: the app still boots on About, which surfaces the
    # unknown-id load message as a warning.
    monkeypatch.setenv("COMFIO_GUI_TOOLS", "nope")
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("nope" in w.value for w in at.warning)
    assert len(at.file_uploader) == 0
