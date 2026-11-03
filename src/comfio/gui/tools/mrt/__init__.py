"""MRT room designer — plan-view radiant-temperature / PMV map."""

from comfio.gui.tools import ToolSpec

TOOL = ToolSpec(
    id="mrt",
    title="MRT room designer",
    icon=":material/grid_on:",
    group="Design & explore",
    target="comfio.gui.tools.mrt.page:render",
)
