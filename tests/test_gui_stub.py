"""Tests for the gui()/gui_notebook() entry points while comfio.gui is unreleased."""

from __future__ import annotations

import importlib.util

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("comfio.gui") is not None,
    reason="comfio.gui module exists — stub tests no longer apply",
)
class TestGuiStubs:
    def test_gui_raises_clear_error(self) -> None:
        import comfio

        with pytest.raises(ImportError, match="not yet released"):
            comfio.gui()

    def test_gui_notebook_raises_clear_error(self) -> None:
        import comfio

        with pytest.raises(ImportError, match="not yet released"):
            comfio.gui_notebook()
