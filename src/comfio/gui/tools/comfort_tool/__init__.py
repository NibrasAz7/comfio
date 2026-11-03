"""Comfort tool (classic) — CBE-style interactive comfort calculator."""

from comfio.gui.tools import ToolSpec

TOOL = ToolSpec(
    id="comfort-tool",
    title="Comfort tool (classic)",
    icon=":material/desktop_windows:",
    group="Design & explore",
    target="comfio.gui.tools.comfort_tool.page:render",
)
