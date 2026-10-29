"""Windows backend for the Power Management Agent.

Same public tools as ``PowerTools`` (the Linux sysfs backend) so the agent, the LLM layer and
the MCP server do not care which operating system they run on.

Mapping of Linux concepts onto Windows, using only built-in tools and no administrator rights:

  CPU governor        -> "Minimum / Maximum processor state" of the active power plan
                         (``powercfg``), applied to both the plugged-in and battery settings
  screen brightness   -> WMI ``WmiMonitorBrightnessMethods`` (internal laptop panel)
  battery / AC state  -> WMI ``Win32_Battery``

    governor      min %   max %
    powersave       5       50     lowest clocks, longest battery life
    conservative    5       70
    schedutil       5      100     Windows' normal behaviour (CPU scales on demand)
    performance   100      100     CPU held at full speed
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Callable

from . import guardrails as gr
from .sysfs import SysfsError
from .tools import PowerTools

GOVERNOR_PROFILES: dict[str, tuple[int, int]] = {
    "powersave": (5, 50),
    "conservative": (5, 70),
    "schedutil": (5, 100),
    "performance": (100, 100),
}

_ON_BATTERY_CODES = {1, 4, 5}     # Win32_Battery.BatteryStatus: discharging / low / critical
_STATUS_NAMES = {1: "Discharging", 2: "Not charging", 3: "Full", 4: "Discharging (low)",
                 5: "Discharging (critical)", 6: "Charging", 7: "Charging", 8: "Charging",
                 9: "Charging", 11: "Charging"}

Runner = Callable[[list], tuple]      # argv -> (returncode, stdout, stderr)


def _default_runner(argv: list) -> tuple:
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000          # CREATE_NO_WINDOW
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=30, **kwargs)
    return proc.returncode, proc.stdout, proc.stderr


_PS = ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"]

_STATUS_SCRIPT = r"""
$b = $null; try { $b = Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness -ErrorAction Stop | Select-Object -First 1 } catch {}
$bat = $null; try { $bat = Get-CimInstance Win32_Battery -ErrorAction Stop | Select-Object -First 1 } catch {}
[pscustomobject]@{
  brightness = if ($b) { [int]$b.CurrentBrightness } else { $null }
  battery_present = [bool]$bat
  charge = if ($bat) { [int]$bat.EstimatedChargeRemaining } else { $null }
  battery_status = if ($bat) { [int]$bat.BatteryStatus } else { $null }
} | ConvertTo-Json -Compress
"""

_SET_BRIGHTNESS_SCRIPT = (
    "Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods -ErrorAction Stop "
    "| Invoke-CimMethod -MethodName WmiSetBrightness "
    "-Arguments @{Timeout=[uint32]1; Brightness=[byte]%d} | Out-Null")


class WindowsPowerTools(PowerTools):
    """Drop-in replacement for PowerTools on Windows."""

    def __init__(self, runner: Runner | None = None, state_file: str | None = None):
        self.run = runner or _default_runner
        self.fs = None                       # no sysfs on Windows
        self.state_file = Path(state_file or os.environ.get("POWER_AGENT_STATE")
                               or Path.home() / ".power_agent" / "snapshot.json")

    # ---- helpers ------------------------------------------------------
    def _powershell(self, script: str) -> str:
        rc, out, err = self.run(_PS + [script])
        if rc != 0:
            raise SysfsError(f"PowerShell failed: {(err or out).strip()[:200]}")
        return out.strip()

    def _wmi_state(self) -> dict:
        try:
            raw = self._powershell(_STATUS_SCRIPT)
            return json.loads(raw) if raw else {}
        except (SysfsError, ValueError, OSError, subprocess.SubprocessError):
            return {}

    def _read_cpu_state(self) -> dict:
        """Return {'min_ac','max_ac','min_dc','max_dc'} for the active power plan."""
        out = {}
        for key, alias in (("min", "PROCTHROTTLEMIN"), ("max", "PROCTHROTTLEMAX")):
            rc, text, err = self.run(["powercfg", "/query", "SCHEME_CURRENT", "SUB_PROCESSOR", alias])
            if rc != 0:
                raise SysfsError(f"powercfg query failed: {(err or text).strip()[:200]}")
            for line in text.splitlines():
                line = line.strip()
                for src, label in (("ac", "Current AC Power Setting Index:"),
                                   ("dc", "Current DC Power Setting Index:")):
                    if line.startswith(label):
                        out[f"{key}_{src}"] = int(line.split(":")[1].strip(), 16)
        if len(out) != 4:
            raise SysfsError("could not read processor state from powercfg")
        return out

    def _write_cpu_state(self, min_ac, max_ac, min_dc, max_dc) -> None:
        for flag, alias, value in (("/setacvalueindex", "PROCTHROTTLEMIN", min_ac),
                                   ("/setacvalueindex", "PROCTHROTTLEMAX", max_ac),
                                   ("/setdcvalueindex", "PROCTHROTTLEMIN", min_dc),
                                   ("/setdcvalueindex", "PROCTHROTTLEMAX", max_dc)):
            rc, out, err = self.run(["powercfg", flag, "SCHEME_CURRENT", "SUB_PROCESSOR",
                                     alias, str(int(value))])
            if rc != 0:
                raise SysfsError(f"powercfg {alias} failed: {(err or out).strip()[:200]}")
        rc, out, err = self.run(["powercfg", "/setactive", "SCHEME_CURRENT"])
        if rc != 0:
            raise SysfsError(f"powercfg /setactive failed: {(err or out).strip()[:200]}")

    @staticmethod
    def _governor_name(lo: int, hi: int) -> str:
        for name, (pmin, pmax) in GOVERNOR_PROFILES.items():
            if (lo, hi) == (pmin, pmax):
                return name
        # Windows' stock Balanced plan keeps max at 100 and min anywhere 0-100 on battery/AC
        if hi == 100 and lo < 100:
            return "schedutil"
        return f"custom({lo}-{hi}%)"

    def _battery_info(self, wmi: dict | None = None) -> dict:
        wmi = self._wmi_state() if wmi is None else wmi
        if not wmi.get("battery_present"):
            return {"present": False, "capacity": None, "status": None, "ac_online": None}
        code = wmi.get("battery_status")
        return {"present": True, "capacity": wmi.get("charge"),
                "status": _STATUS_NAMES.get(code, "Unknown"),
                "ac_online": code not in _ON_BATTERY_CODES}

    def _battery(self) -> dict:                     # used by inherited helpers
        return self._battery_info()

    def _current_governor(self, cpu_state: dict, battery: dict) -> str:
        on_dc = battery["present"] and battery["ac_online"] is not True
        src = "dc" if on_dc else "ac"
        return self._governor_name(cpu_state[f"min_{src}"], cpu_state[f"max_{src}"])

    # ---- read-only tools ----------------------------------------------
    def get_power_status(self) -> dict:
        try:
            state = self._read_cpu_state()
        except SysfsError as exc:
            return {"ok": False, "error": str(exc)}
        wmi = self._wmi_state()
        battery = self._battery_info(wmi)
        gov = self._current_governor(state, battery)
        n = os.cpu_count() or 1
        screen = ({"device": "internal display", "brightness_percent": wmi["brightness"]}
                  if wmi.get("brightness") is not None else None)
        return {
            "ok": True,
            "platform": "windows",
            "cpu_count_with_cpufreq": n,
            "governors": {f"cpu{i}": gov for i in range(n)},
            "distinct_governors": [gov],
            "available_governors": list(GOVERNOR_PROFILES),
            "cpu_limits_percent": {"battery": [state["min_dc"], state["max_dc"]],
                                   "plugged_in": [state["min_ac"], state["max_ac"]]},
            "screen": screen,
            "battery": battery,
        }

    def list_available_governors(self) -> dict:
        return {"ok": True, "available_governors": list(GOVERNOR_PROFILES)}

    # ---- state-changing tools -----------------------------------------
    def set_cpu_governor(self, governor: str, cpu: int | None = None, confirmed: bool = False) -> dict:
        try:
            governor = gr.validate_governor(governor, list(GOVERNOR_PROFILES))
            cpu = gr.validate_cpu_id(cpu)
            if cpu is not None:
                return {"ok": False, "error": "Windows applies one processor policy to all CPUs; "
                                              "per-CPU governors are not supported here"}
            battery = self._battery_info()
            if not confirmed and gr.governor_needs_confirmation(
                    governor, battery["present"] and battery["ac_online"] is not True,
                    battery["capacity"]):
                return self._need_confirm(
                    "Switching to 'performance' will drain a battery that is already below 20 %.")
            lo, hi = GOVERNOR_PROFILES[governor]
            self._write_cpu_state(lo, hi, lo, hi)
            after = self._read_cpu_state()
            if self._current_governor(after, battery) != governor:
                return {"ok": False, "error": "Windows did not accept the processor-state change"}
            n = os.cpu_count() or 1
            return {"ok": True, "governor": governor, "changed_cpus": list(range(n)),
                    "failed_cpus": [], "min_percent": lo, "max_percent": hi}
        except (gr.GuardrailViolation, SysfsError, OSError, subprocess.SubprocessError) as exc:
            return {"ok": False, "error": str(exc)}

    def _brightness_now(self) -> int | None:
        return self._wmi_state().get("brightness")

    def set_screen_brightness(self, percent: int, confirmed: bool = False) -> dict:
        try:
            percent = gr.validate_brightness(percent)
            before = self._brightness_now()
            if before is None:
                return {"ok": False,
                        "error": "no controllable internal display found (external monitor or desktop?)"}
            if not confirmed and gr.brightness_needs_confirmation(percent):
                return self._need_confirm(f"{percent} % is very dim and may be hard to read.")
            self._powershell(_SET_BRIGHTNESS_SCRIPT % percent)
            after = self._brightness_now()
            return {"ok": True, "device": "internal display", "before_percent": before,
                    "after_percent": after if after is not None else percent}
        except (gr.GuardrailViolation, SysfsError, OSError, subprocess.SubprocessError) as exc:
            return {"ok": False, "error": str(exc)}

    def change_screen_brightness(self, delta_percent: int, confirmed: bool = False) -> dict:
        current = self._brightness_now()
        if current is None:
            return {"ok": False,
                    "error": "no controllable internal display found (external monitor or desktop?)"}
        try:
            target = current + int(delta_percent)
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        return self.set_screen_brightness(max(gr.MIN_BRIGHTNESS_PERCENT, min(100, target)),
                                          confirmed=confirmed)

    # ---- profiles --------------------------------------------------------
    def enable_battery_saver(self, brightness_percent: int = 30) -> dict:
        try:
            brightness_percent = gr.validate_brightness(brightness_percent)
        except gr.GuardrailViolation as exc:
            return {"ok": False, "error": str(exc)}
        brightness_percent = max(brightness_percent, gr.CONFIRM_BELOW_PERCENT)
        try:
            state = self._read_cpu_state()
        except SysfsError as exc:
            return {"ok": False, "error": str(exc)}
        current_brightness = self._brightness_now()
        already = (state["max_ac"], state["max_dc"]) == (50, 50) and state["min_ac"] == 5
        if not (self.state_file.exists() and already):
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps({"cpu": state, "brightness_percent": current_brightness}))
        steps = {"governor": self.set_cpu_governor("powersave")}
        if current_brightness is None:
            steps["brightness"] = {"ok": True, "note": "no controllable display, skipped"}
        elif current_brightness > brightness_percent:
            steps["brightness"] = self.set_screen_brightness(brightness_percent)
        else:
            steps["brightness"] = {"ok": True, "note": "screen already at or below target"}
        return {"ok": all(s["ok"] for s in steps.values()), "profile": "battery_saver", "steps": steps}

    def restore_previous_settings(self) -> dict:
        if not self.state_file.exists():
            return {"ok": False, "error": "no saved settings to restore"}
        snap = json.loads(self.state_file.read_text())
        results = {}
        try:
            c = snap["cpu"]
            self._write_cpu_state(c["min_ac"], c["max_ac"], c["min_dc"], c["max_dc"])
            results["governor"] = {"ok": True}
        except (SysfsError, KeyError) as exc:
            results["governor"] = {"ok": False, "error": str(exc)}
        if snap.get("brightness_percent") is not None:
            results["brightness"] = self.set_screen_brightness(snap["brightness_percent"], confirmed=True)
        ok = all(r["ok"] for r in results.values())
        if ok:
            self.state_file.unlink(missing_ok=True)
        return {"ok": ok, "restored": snap, "failures": {k: v for k, v in results.items() if not v["ok"]}}
