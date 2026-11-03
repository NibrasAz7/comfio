"""Learn — guided thermal-comfort lessons that hand off to the tools."""

from comfio.gui.tools import ToolSpec

TOOL = ToolSpec(
    id="learn",
    title="Learn",
    icon=":material/school:",
    group="Learn",
    target="comfio.gui.tools.learn.page:render",
    always_on=False,
)
