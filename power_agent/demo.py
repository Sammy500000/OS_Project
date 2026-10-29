"""Simulated machine for demos and tests on any OS.

Builds a throw-away directory shaped like the Linux /sys tree, so the real sysfs code path
(governor files, backlight, battery) runs end to end without touching your hardware.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def build_demo_tree(root: Path | str | None = None, cpus: int = 8, governor: str = "schedutil",
                    brightness: int = 80, battery: int = 62, on_ac: bool = False) -> Path:
    root = Path(root) if root else Path(tempfile.mkdtemp(prefix="power_agent_demo_"))

    def w(rel: str, text: str) -> None:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n")

    for i in range(cpus):
        base = f"sys/devices/system/cpu/cpu{i}/cpufreq"
        w(f"{base}/scaling_governor", governor)
        w(f"{base}/scaling_available_governors", "performance powersave schedutil ondemand conservative")
    w("sys/class/backlight/intel_backlight/brightness", str(brightness * 960))
    w("sys/class/backlight/intel_backlight/max_brightness", "96000")
    w("sys/class/backlight/intel_backlight/type", "raw")
    w("sys/class/power_supply/BAT0/type", "Battery")
    w("sys/class/power_supply/BAT0/capacity", str(battery))
    w("sys/class/power_supply/BAT0/status", "Charging" if on_ac else "Discharging")
    w("sys/class/power_supply/AC/type", "Mains")
    w("sys/class/power_supply/AC/online", "1" if on_ac else "0")
    return root
