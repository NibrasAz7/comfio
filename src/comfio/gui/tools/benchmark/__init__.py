"""Global database benchmark — ASHRAE DB II (CC0) scatter/bins/bias."""

from comfio.gui.tools import ToolSpec

TOOL = ToolSpec(
    id="benchmark",
    title="Global database",
    icon=":material/public:",
    group="Benchmark",
    target="comfio.gui.tools.benchmark.page:render",
)
