"""Tests for the gui_notebook() entry point."""

from __future__ import annotations

import importlib.util

import pytest

from comfio.gui._data import demo_data


def test_gui_notebook_requires_df() -> None:
    pytest.importorskip("ipywidgets")
    import comfio

    with pytest.raises(ValueError, match="requires a pandas DataFrame"):
        comfio.gui_notebook()


def test_gui_notebook_returns_vbox() -> None:
    ipywidgets = pytest.importorskip("ipywidgets")
    import comfio

    w = comfio.gui_notebook(demo_data())
    assert isinstance(w, ipywidgets.VBox)
    # timestamp picker + one dropdown per described canonical column
    dropdowns = [c for c in w.children if isinstance(c, ipywidgets.Dropdown)]
    assert len(dropdowns) > 10


@pytest.mark.skipif(
    importlib.util.find_spec("ipywidgets") is not None,
    reason="ipywidgets installed — ImportError path not applicable",
)
def test_gui_notebook_missing_dependency() -> None:
    import comfio

    with pytest.raises(ImportError, match=r"comfio\[gui\]"):
        comfio.gui_notebook(demo_data())
