"""Setpoint explorer — comfort-band edges and deadband widening."""

from comfio.gui.tools import ToolSpec

TOOL = ToolSpec(
    id="setpoint",
    title="Setpoint explorer",
    icon=":material/thermostat:",
    group="Design & explore",
    target="comfio.gui.tools.setpoint.page:render",
)
