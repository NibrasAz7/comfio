"""Comfort calculator tool — PMV/SET/adaptive what-if for one point."""

from comfio.gui.tools import ToolSpec

TOOL = ToolSpec(
    id="calculator",
    title="Comfort calculator",
    icon=":material/calculate:",
    group="Design & explore",
    target="comfio.gui.tools.calculator.page:render",
    requires=("scipy",),
)
