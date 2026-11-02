"""Streamlit Community Cloud entrypoint for the comfio GUI workbench.

Point the deployment at this file. ``requirements.txt`` installs
``comfio[gui]``; the ``sys.path`` fallback lets it also run locally
against a checkout without installing the package first.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from comfio.gui.app import run  # noqa: E402

run()  # must run on every script run — importing alone would cache-render nothing
