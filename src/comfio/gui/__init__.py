"""comfio GUI — Streamlit analyst workbench.

Entry points:

- :func:`gui` — launch the Streamlit app in a browser window.
- :func:`gui_notebook` — deferred; the Jupyter widget is not yet released.

Requires ``pip install comfio[gui]``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path
from typing import Any

_GUI_IMPORT_ERROR = (
    "The comfio GUI requires extra dependencies. "
    "Install with: pip install comfio[gui]"
)


def gui(df: Any = None, port: int = 8501, *, desktop: bool = False) -> None:
    """Launch the comfio Streamlit GUI.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Preloaded sensor data. When given, the app starts directly on
        the analysis view with this data instead of the upload screen.
    port : int, default 8501
        Port for the local Streamlit server.
    desktop : bool, default False
        Open in a native ``pywebview`` window instead of the browser
        (requires ``pywebview``, included in ``comfio[gui]``; falls back
        to the browser when unavailable).

    Raises
    ------
    ImportError
        If streamlit is not installed (``pip install comfio[gui]``).
    """
    try:
        import streamlit  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ImportError(_GUI_IMPORT_ERROR) from exc

    app_path = Path(__file__).parent / "app.py"

    env = os.environ.copy()
    tmp = None
    if df is not None:
        with tempfile.NamedTemporaryFile(  # must persist for the subprocess
            suffix=".pkl", delete=False
        ) as tmp:
            df.to_pickle(tmp.name)
            env["COMFIO_GUI_DATA"] = tmp.name

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    proc = subprocess.Popen(cmd, env=env)  # noqa: S603
    url = f"http://localhost:{port}"
    try:
        if desktop:
            try:
                import time

                import webview
            except ModuleNotFoundError:
                webview = None  # fall back to the browser
            if webview is not None:
                time.sleep(1.5)  # give the server a moment to boot
                webview.create_window("comfio", url, width=1440, height=900)
                webview.start()
                return
        webbrowser.open(url)
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
    finally:
        if proc.poll() is None:
            proc.terminate()


def gui_notebook(df: Any = None) -> Any:
    """Create the comfio ipywidgets workbench for Jupyter notebooks.

    Parameters
    ----------
    df : pandas.DataFrame
        Sensor data to analyse. Returns an ``ipywidgets.VBox`` —
        display it as the last expression of a notebook cell.
    """
    try:
        import ipywidgets  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ImportError(_GUI_IMPORT_ERROR) from exc
    if df is None:
        raise ValueError("gui_notebook(df) requires a pandas DataFrame.")

    from comfio.gui.notebook import gui_notebook as _gui_notebook

    return _gui_notebook(df)


__all__ = ["gui", "gui_notebook"]
