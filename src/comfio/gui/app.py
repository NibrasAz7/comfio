"""comfio — multipage GUI shell.

Builds the navigation from the tool registry
(:func:`comfio.gui.tools.load_tools`) and runs it with ``st.navigation``.
Each tool is a ``st.Page`` whose body imports its ``ToolSpec.target``
lazily, so only the open page runs.

Run via ``comfio.gui()`` or ``streamlit run src/comfio/gui/app.py``.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable

import streamlit as st
from streamlit.navigation.page import StreamlitPage

from comfio.gui import _state, _styles
from comfio.gui.tools import ToolSpec, enabled_from_env, load_tools


def _lazy(spec: ToolSpec) -> Callable[[], None]:
    def _render() -> None:
        module_path, _, attr = spec.target.partition(":")
        getattr(importlib.import_module(module_path), attr)()

    return _render


def run() -> None:
    st.set_page_config(
        page_title="comfio — IEQ Analyst Workbench",
        page_icon="🌡️",
        layout="wide",
    )
    _styles.inject()

    specs, messages = load_tools(enabled_from_env(os.environ.get("COMFIO_GUI_TOOLS")))
    _state.set(_state.TOOL_MESSAGES, messages)

    if not specs:
        st.error("No tools are enabled. " + " ".join(messages))
        return

    default_id = "workbench" if any(s.id == "workbench" for s in specs) else specs[0].id
    groups: dict[str, list[StreamlitPage]] = {}
    page_by_id: dict[str, StreamlitPage] = {}
    for spec in specs:
        page = st.Page(
            _lazy(spec),
            title=spec.title,
            icon=spec.icon,
            url_path=spec.id,
            default=spec.id == default_id,
        )
        groups.setdefault(spec.group, []).append(page)
        page_by_id[spec.id] = page
    _state.set(_state.TOOL_PAGES, page_by_id)

    st.navigation(groups).run()


if __name__ == "__main__":
    run()
