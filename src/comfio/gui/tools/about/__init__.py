"""About & sources — enabled tools, provenance and citations."""

from comfio.gui.tools import ToolSpec

TOOL = ToolSpec(
    id="about",
    title="About & sources",
    icon=":material/info:",
    group="About",
    target="comfio.gui.tools.about.page:render",
    always_on=True,
)
