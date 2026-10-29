"""Chooses the right tool backend for the machine the agent runs on.

POWER_AGENT_BACKEND = auto (default) | linux | windows | demo
  auto     Linux -> sysfs, Windows -> powercfg/WMI, anything else -> simulated demo machine
  demo     simulated Linux machine in a temp directory (safe, works everywhere)
"""

from __future__ import annotations

import os
import sys

from .tools import PowerTools


def backend_name(requested: str | None = None) -> str:
    choice = (requested or os.environ.get("POWER_AGENT_BACKEND", "auto")).lower()
    if choice != "auto":
        return choice
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "win32":
        return "windows"
    return "demo"


def create_tools(requested: str | None = None) -> PowerTools:
    name = backend_name(requested)
    if name == "windows":
        from .win_tools import WindowsPowerTools
        return WindowsPowerTools()
    if name == "demo":
        from .demo import build_demo_tree
        from .sysfs import Sysfs
        import tempfile
        root = build_demo_tree()
        return PowerTools(Sysfs(str(root)), state_file=os.path.join(tempfile.gettempdir(),
                                                                    "power_agent_demo_state.json"))
    return PowerTools()
