"""MCP server exposing the power tools (stdio transport).

Run:  python -m power_agent.mcp_server
Works with both the 1.x (FastMCP) and 2.x (MCPServer) Python SDKs.
"""

from __future__ import annotations

from .backend import create_tools
from .tools import PowerTools

try:                                            # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                             # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

_CONFIRM_NOTE = ("`confirmed` must stay false unless the human user has explicitly "
                 "approved the action after a confirmation_required reply.")


def build_server(tools: PowerTools | None = None):
    pt = tools or create_tools()
    server = _Server("power-management")

    @server.tool()
    def get_power_status() -> dict:
        """Current CPU governors, screen brightness and battery state."""
        return pt.get_power_status()

    @server.tool()
    def list_available_governors() -> dict:
        """CPU scaling governors supported by this machine."""
        return pt.list_available_governors()

    @server.tool()
    def set_cpu_governor(governor: str, cpu: int | None = None, confirmed: bool = False) -> dict:
        f"""Set the CPU scaling governor on all CPUs, or one CPU. {_CONFIRM_NOTE}"""
        return pt.set_cpu_governor(governor, cpu, confirmed)

    @server.tool()
    def set_screen_brightness(percent: int, confirmed: bool = False) -> dict:
        f"""Set screen brightness to an absolute percentage. {_CONFIRM_NOTE}"""
        return pt.set_screen_brightness(percent, confirmed)

    @server.tool()
    def change_screen_brightness(delta_percent: int, confirmed: bool = False) -> dict:
        f"""Raise or lower brightness by a number of points. {_CONFIRM_NOTE}"""
        return pt.change_screen_brightness(delta_percent, confirmed)

    @server.tool()
    def enable_battery_saver(brightness_percent: int = 30) -> dict:
        """Powersave governor on all CPUs plus a dimmed screen; settings are saved for undo."""
        return pt.enable_battery_saver(brightness_percent)

    @server.tool()
    def restore_previous_settings() -> dict:
        """Undo battery saver and restore the earlier governor and brightness."""
        return pt.restore_previous_settings()

    return server


if __name__ == "__main__":
    build_server().run()
