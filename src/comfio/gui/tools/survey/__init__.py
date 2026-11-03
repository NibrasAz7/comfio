"""Occupant survey — import satisfaction votes and compare with sensors."""

from comfio.gui.tools import ToolSpec

TOOL = ToolSpec(
    id="survey",
    title="Occupant survey",
    icon=":material/poll:",
    group="Analyse",
    target="comfio.gui.tools.survey.page:render",
)
