"""Tool registry for the comfio GUI multipage shell.

Each tool lives in ``comfio.gui.tools.<id>`` and exposes a module-level
``TOOL`` :class:`ToolSpec`. ``load_tools`` resolves :data:`BUILTIN_TOOLS`
into the enabled, importable specs that feed ``st.navigation``.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")


@dataclass(frozen=True)
class ToolSpec:
    id: str  # url_path
    title: str
    icon: str  # ":material/<name>:"
    group: str  # navigation section
    target: str  # "comfio.gui.tools.x.page:render", imported only when the page runs
    requires: tuple[str, ...] = ()
    always_on: bool = False  # About page

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.id):
            raise ValueError(f"Invalid tool id {self.id!r}: must match {_ID_RE.pattern}")


BUILTIN_TOOLS: tuple[str, ...] = (
    "comfio.gui.tools.workbench",
    "comfio.gui.tools.calculator",
    "comfio.gui.tools.benchmark",
)


def enabled_from_env(value: str | None) -> list[str] | None:
    """Parse ``COMFIO_GUI_TOOLS`` ("a, b") into ids; None/"" means all."""
    if value is None:
        return None
    ids = [part.strip() for part in value.split(",") if part.strip()]
    return ids or None


def load_tools(enabled: Sequence[str] | None = None) -> tuple[list[ToolSpec], list[str]]:
    """Return the enabled, loadable specs plus human-readable messages."""
    known: dict[str, ToolSpec] = {}
    for module_path in BUILTIN_TOOLS:
        spec: ToolSpec = importlib.import_module(module_path).TOOL
        known[spec.id] = spec

    messages: list[str] = []
    if enabled is not None:
        for tool_id in enabled:
            if tool_id not in known:
                messages.append(f"Unknown tool id {tool_id!r} — ignored.")

    requested = set(enabled) if enabled is not None else None
    specs: list[ToolSpec] = []
    for spec in known.values():
        if requested is not None and spec.id not in requested and not spec.always_on:
            continue
        missing = [m for m in spec.requires if importlib.util.find_spec(m) is None]
        if missing:
            messages.append(
                f"Tool {spec.id!r} unavailable — missing: {', '.join(missing)}."
            )
            continue
        specs.append(spec)
    return specs, messages
