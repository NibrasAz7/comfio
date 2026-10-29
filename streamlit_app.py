"""Streamlit Community Cloud entrypoint for the comfio GUI workbench.

Point the deployment at this file. ``requirements.txt`` installs
``comfio[gui]``; the ``sys.path`` fallback lets it also run locally
against a checkout without installing the package first.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import comfio.gui.app  # noqa: E402,F401  (module runs the app on import)
