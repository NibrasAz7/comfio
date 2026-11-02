"""Workbench tool — upload/mapping/verdict/evidence (the original app)."""

from comfio.gui.tools import ToolSpec

TOOL = ToolSpec(
    id="workbench",
    title="Analyse my data",
    icon=":material/query_stats:",
    group="Analyse",
    target="comfio.gui.tools.workbench.page:render",
)
