"""The tool layer: every system capability the agent can invoke.

Each tool returns a JSON-serialisable dict with an ``ok`` flag so the LLM (or
an MCP client) can read the outcome and decide what to do next.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from . import guardrails as gr
from .sysfs import Sysfs, SysfsError

CPUFREQ_GLOB = "/sys/devices/system/cpu/cpu[0-9]*/cpufreq"
BACKLIGHT_GLOB = "/sys/class/backlight/*"
SUPPLY_GLOB = "/sys/class/power_supply/*"
_BACKLIGHT_PREFERENCE = {"firmware": 0, "platform": 1, "raw": 2}


class PowerTools:
    def __init__(self, sysfs: Sysfs | None = None, state_file: str | None = None):
        self.fs = sysfs or Sysfs()
        self.state_file = Path(
            state_file
            or os.environ.get("POWER_AGENT_STATE")
            or Path.home() / ".local/state/power_agent/snapshot.json"
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _policies(self) -> list[Path]:
        return self.fs.glob(CPUFREQ_GLOB)

    @staticmethod
    def _cpu_number(policy: Path) -> int:
        return int(policy.parent.name.removeprefix("cpu"))

    def _available_governors(self) -> list[str]:
        policies = self._policies()
        if not policies:
            return []
        text = self.fs.read_file(policies[0] / "scaling_available_governors")
        return text.split()

    def _backlight(self) -> Path | None:
        devices = self.fs.glob(BACKLIGHT_GLOB)
        if not devices:
            return None

        def rank(dev: Path) -> int:
            try:
                return _BACKLIGHT_PREFERENCE.get(self.fs.read_file(dev / "type"), 3)
            except SysfsError:
                return 3

        return sorted(devices, key=rank)[0]

    def _battery(self) -> dict:
        info = {"present": False, "capacity": None, "status": None, "ac_online": None}
        for supply in self.fs.glob(SUPPLY_GLOB):
            try:
                kind = self.fs.read_file(supply / "type")
                if kind == "Battery" and not info["present"]:
                    info.update(
                        present=True,
                        capacity=int(self.fs.read_file(supply / "capacity")),
                        status=self.fs.read_file(supply / "status"),
                    )
                elif kind == "Mains":
                    info["ac_online"] = self.fs.read_file(supply / "online") == "1"
            except (SysfsError, ValueError):
                continue
        return info

    def _on_battery(self) -> bool:
        b = self._battery()
        return bool(b["present"] and b["ac_online"] is not True)

    def _brightness_percent(self, dev: Path) -> int:
        cur = int(self.fs.read_file(dev / "brightness"))
        top = int(self.fs.read_file(dev / "max_brightness"))
        return round(cur * 100 / top)

    def _write_brightness(self, dev: Path, percent: int) -> None:
        top = int(self.fs.read_file(dev / "max_brightness"))
        raw = max(1, round(percent * top / 100))
        try:
            self.fs.write_file(dev / "brightness", str(raw))
        except SysfsError:
            # Desktop sessions can change brightness without root through logind.
            if not (self.fs.allow_sudo and shutil.which("busctl")):
                raise
            proc = subprocess.run(
                ["busctl", "call", "org.freedesktop.login1",
                 "/org/freedesktop/login1/session/auto",
                 "org.freedesktop.login1.Session", "SetBrightness",
                 "ssu", "backlight", dev.name, str(raw)],
                capture_output=True, text=True, timeout=5,
            )
            if proc.returncode != 0:
                raise SysfsError(f"logind brightness call failed: {proc.stderr.strip()}")

    @staticmethod
    def _need_confirm(message: str) -> dict:
        return {"ok": False, "status": "confirmation_required", "message": message}

    # ------------------------------------------------------------------
    # read-only tools
    # ------------------------------------------------------------------
    def get_power_status(self) -> dict:
        """Report CPU governors, screen brightness and battery state."""
        policies = self._policies()
        governors = {}
        for p in policies:
            governors[f"cpu{self._cpu_number(p)}"] = self.fs.read_file(p / "scaling_governor")
        dev = self._backlight()
        screen = None
        if dev is not None:
            screen = {"device": dev.name, "brightness_percent": self._brightness_percent(dev)}
        return {
            "ok": True,
            "cpu_count_with_cpufreq": len(policies),
            "governors": governors,
            "distinct_governors": sorted(set(governors.values())),
            "available_governors": self._available_governors(),
            "screen": screen,
            "battery": self._battery(),
        }

    def list_available_governors(self) -> dict:
        """List the scaling governors the CPU driver supports."""
        available = self._available_governors()
        if not available:
            return {"ok": False, "error": "no cpufreq interface found on this machine"}
        return {"ok": True, "available_governors": available}

    # ------------------------------------------------------------------
    # state-changing tools
    # ------------------------------------------------------------------
    def set_cpu_governor(self, governor: str, cpu: int | None = None, confirmed: bool = False) -> dict:
        """Set the scaling governor on every CPU, or on one CPU if `cpu` is given."""
        try:
            available = self._available_governors()
            if not available:
                return {"ok": False, "error": "no cpufreq interface found on this machine"}
            governor = gr.validate_governor(governor, available)
            cpu = gr.validate_cpu_id(cpu)
            if not confirmed and gr.governor_needs_confirmation(
                governor, self._on_battery(), self._battery()["capacity"]
            ):
                return self._need_confirm(
                    "Switching to 'performance' will drain a battery that is already below 20 %."
                )
            targets = [p for p in self._policies() if cpu is None or self._cpu_number(p) == cpu]
            if not targets:
                return {"ok": False, "error": f"cpu{cpu} has no cpufreq policy (offline?)"}
            changed, failed = [], []
            for p in targets:
                try:
                    self.fs.write_file(p / "scaling_governor", governor)
                    if self.fs.read_file(p / "scaling_governor") == governor:
                        changed.append(self._cpu_number(p))
                    else:
                        failed.append(self._cpu_number(p))
                except SysfsError as exc:
                    return {"ok": False, "error": str(exc), "changed_cpus": changed}
            return {"ok": not failed, "governor": governor, "changed_cpus": changed,
                    "failed_cpus": failed}
        except (gr.GuardrailViolation, SysfsError) as exc:
            return {"ok": False, "error": str(exc)}

    def set_screen_brightness(self, percent: int, confirmed: bool = False) -> dict:
        """Set screen brightness as a percentage of the panel maximum."""
        try:
            percent = gr.validate_brightness(percent)
            dev = self._backlight()
            if dev is None:
                return {"ok": False, "error": "no backlight device found (external monitor or desktop?)"}
            if not confirmed and gr.brightness_needs_confirmation(percent):
                return self._need_confirm(f"{percent} % is very dim and may be hard to read.")
            before = self._brightness_percent(dev)
            self._write_brightness(dev, percent)
            return {"ok": True, "device": dev.name, "before_percent": before,
                    "after_percent": self._brightness_percent(dev)}
        except (gr.GuardrailViolation, SysfsError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

    def change_screen_brightness(self, delta_percent: int, confirmed: bool = False) -> dict:
        """Raise (positive) or lower (negative) brightness by a number of percentage points."""
        dev = self._backlight()
        if dev is None:
            return {"ok": False, "error": "no backlight device found (external monitor or desktop?)"}
        try:
            target = self._brightness_percent(dev) + int(delta_percent)
        except (SysfsError, ValueError, TypeError) as exc:
            return {"ok": False, "error": str(exc)}
        target = max(gr.MIN_BRIGHTNESS_PERCENT, min(100, target))
        return self.set_screen_brightness(target, confirmed=confirmed)

    # ------------------------------------------------------------------
    # profiles
    # ------------------------------------------------------------------
    def _snapshot(self) -> dict:
        status = self.get_power_status()
        return {
            "governors": status["governors"],
            "brightness_percent": status["screen"]["brightness_percent"] if status["screen"] else None,
        }

    def enable_battery_saver(self, brightness_percent: int = 30) -> dict:
        """Maximum saving: 'powersave' governor on all CPUs and a dimmed screen.

        The previous settings are stored so restore_previous_settings can undo it.
        """
        try:
            brightness_percent = gr.validate_brightness(brightness_percent)
        except gr.GuardrailViolation as exc:
            return {"ok": False, "error": str(exc)}
        if brightness_percent < gr.CONFIRM_BELOW_PERCENT:
            brightness_percent = gr.CONFIRM_BELOW_PERCENT  # profiles never go into the confirm band
        if "powersave" not in self._available_governors():
            return {"ok": False, "error": "this CPU driver does not offer the 'powersave' governor"}

        snap = self._snapshot()
        if not (self.state_file.exists() and self._is_already_saving()):
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(snap))

        steps = {"governor": self.set_cpu_governor("powersave")}
        if self._backlight() is not None:
            current = snap["brightness_percent"]
            if current is not None and current > brightness_percent:
                steps["brightness"] = self.set_screen_brightness(brightness_percent)
            else:
                steps["brightness"] = {"ok": True, "note": "screen already at or below target"}
        else:
            steps["brightness"] = {"ok": True, "note": "no backlight device, skipped"}
        return {"ok": all(s["ok"] for s in steps.values()), "profile": "battery_saver", "steps": steps}

    def _is_already_saving(self) -> bool:
        """True when a snapshot exists and the CPUs are still in powersave."""
        try:
            return set(self.get_power_status()["distinct_governors"]) == {"powersave"}
        except SysfsError:
            return False

    def restore_previous_settings(self) -> dict:
        """Undo enable_battery_saver using the stored snapshot."""
        if not self.state_file.exists():
            return {"ok": False, "error": "no saved settings to restore"}
        snap = json.loads(self.state_file.read_text())
        results = {}
        by_governor: dict[str, list[int]] = {}
        for name, gov in snap["governors"].items():
            by_governor.setdefault(gov, []).append(int(name.removeprefix("cpu")))
        for gov, cpus in by_governor.items():
            for c in cpus:
                res = self.set_cpu_governor(gov, cpu=c, confirmed=True)
                results[f"cpu{c}"] = res
        if snap.get("brightness_percent") is not None and self._backlight() is not None:
            results["brightness"] = self.set_screen_brightness(snap["brightness_percent"], confirmed=True)
        ok = all(r["ok"] for r in results.values())
        if ok:
            self.state_file.unlink(missing_ok=True)
        return {"ok": ok, "restored": snap, "failures": {k: v for k, v in results.items() if not v["ok"]}}


# ----------------------------------------------------------------------
# Tool registry: schemas shown to the LLM, and the Python method behind each
# ----------------------------------------------------------------------
TOOL_SPECS = [
    {
        "name": "get_power_status",
        "description": "Read the current CPU governors, screen brightness and battery state. "
                       "Use for questions about the present state, or before relative changes.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_available_governors",
        "description": "List which CPU scaling governors this machine supports.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_cpu_governor",
        "description": "Set the CPU frequency scaling governor (for example powersave, performance, "
                       "schedutil) on all CPUs, or on a single CPU when `cpu` is given.",
        "input_schema": {
            "type": "object",
            "properties": {
                "governor": {"type": "string", "description": "Governor name, e.g. powersave"},
                "cpu": {"type": "integer", "description": "Optional CPU index; omit for all CPUs"},
            },
            "required": ["governor"],
        },
    },
    {
        "name": "set_screen_brightness",
        "description": "Set screen brightness to an absolute percentage (5-100).",
        "input_schema": {
            "type": "object",
            "properties": {"percent": {"type": "integer", "minimum": 0, "maximum": 100}},
            "required": ["percent"],
        },
    },
    {
        "name": "change_screen_brightness",
        "description": "Raise or lower brightness relative to its current level. "
                       "Use a negative number to dim, positive to brighten.",
        "input_schema": {
            "type": "object",
            "properties": {"delta_percent": {"type": "integer", "minimum": -100, "maximum": 100}},
            "required": ["delta_percent"],
        },
    },
    {
        "name": "enable_battery_saver",
        "description": "Maximum battery saving mode: sets the powersave governor on every CPU and "
                       "dims the screen. Saves the previous settings so they can be restored.",
        "input_schema": {
            "type": "object",
            "properties": {"brightness_percent": {"type": "integer", "minimum": 10, "maximum": 100,
                                                  "description": "Target brightness, default 30"}},
        },
    },
    {
        "name": "restore_previous_settings",
        "description": "Undo battery saver mode and return governors and brightness to what they were.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

# Tools that accept the `confirmed` flag. The flag is never part of the LLM schema.
CONFIRMABLE = {"set_cpu_governor", "set_screen_brightness", "change_screen_brightness"}
READ_ONLY = {"get_power_status", "list_available_governors"}
