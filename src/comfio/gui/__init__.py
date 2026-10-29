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


def gui(df: Any = None, port: int = 8501) -> None:
    """Launch the comfio Streamlit GUI in the default browser.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Preloaded sensor data. When given, the app starts directly on
        the analysis view with this data instead of the upload screen.
    port : int, default 8501
        Port for the local Streamlit server.

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
    webbrowser.open(f"http://localhost:{port}")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


def gui_notebook(df: Any = None) -> Any:
    """Create an interactive comfio widget for Jupyter notebooks.

    The notebook widget is under development and will ship in a future
    version. Use :func:`gui` for the Streamlit app.
    """
    raise ImportError(
        "comfio.gui_notebook() is not yet released. "
        "Use comfio.gui() for the Streamlit app."
    )


__all__ = ["gui", "gui_notebook"]
