"""Builds a throw-away sysfs tree that mirrors the real kernel layout."""

import pytest

from power_agent.sysfs import Sysfs
from power_agent.tools import PowerTools
from power_agent.agent import PowerManagementAgent


def make_tree(root, cpus=4, governor="schedutil", brightness=(96000, 96000),
              battery=(80, "Discharging"), ac=False, governors="performance powersave schedutil ondemand"):
    def w(rel, text):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n")

    for i in range(cpus):
        base = f"sys/devices/system/cpu/cpu{i}/cpufreq"
        w(f"{base}/scaling_governor", governor)
        w(f"{base}/scaling_available_governors", governors)
    if brightness:
        cur, top = brightness
        w("sys/class/backlight/intel_backlight/brightness", str(cur))
        w("sys/class/backlight/intel_backlight/max_brightness", str(top))
        w("sys/class/backlight/intel_backlight/type", "raw")
    if battery:
        cap, status = battery
        w("sys/class/power_supply/BAT0/type", "Battery")
        w("sys/class/power_supply/BAT0/capacity", str(cap))
        w("sys/class/power_supply/BAT0/status", status)
        w("sys/class/power_supply/AC/type", "Mains")
        w("sys/class/power_supply/AC/online", "1" if ac else "0")
    return root


@pytest.fixture
def tree(tmp_path):
    return make_tree(tmp_path / "root")


@pytest.fixture
def tools(tree, tmp_path):
    return PowerTools(Sysfs(str(tree)), state_file=str(tmp_path / "state.json"))


@pytest.fixture
def agent(tools):
    return PowerManagementAgent(tools=tools, use_llm=False, confirm=lambda m: True)
