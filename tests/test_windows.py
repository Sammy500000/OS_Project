"""Windows backend, exercised with a fake powercfg/PowerShell so it runs on any OS."""

import json

import pytest

from power_agent.agent import PowerManagementAgent
from power_agent.win_tools import WindowsPowerTools


class FakeWindows:
    """Imitates the three commands the backend uses."""

    def __init__(self, brightness=70, charge=66, status=1, cpu=None):
        self.brightness, self.charge, self.status = brightness, charge, status
        self.cpu = cpu or {"PROCTHROTTLEMIN": [100, 5], "PROCTHROTTLEMAX": [100, 100]}   # [ac, dc]
        self.calls = []

    def __call__(self, argv):
        self.calls.append(argv)
        if argv[0] == "powercfg":
            if argv[1] == "/query":
                ac, dc = self.cpu[argv[4]]
                return 0, (f"    Current AC Power Setting Index: 0x{ac:08x}\n"
                           f"    Current DC Power Setting Index: 0x{dc:08x}\n"), ""
            if argv[1] in ("/setacvalueindex", "/setdcvalueindex"):
                idx = 0 if argv[1] == "/setacvalueindex" else 1
                self.cpu[argv[4]][idx] = int(argv[5])
                return 0, "", ""
            if argv[1] == "/setactive":
                return 0, "", ""
        if argv[0] == "powershell":
            script = argv[-1]
            if "WmiSetBrightness" in script:
                self.brightness = int(script.split("[byte]")[1].rstrip("} | Out-Null").rstrip("}").split()[0])
                return 0, "", ""
            return 0, json.dumps({"brightness": self.brightness, "battery_present": True,
                                  "charge": self.charge, "battery_status": self.status}), ""
        return 1, "", "unknown command"


@pytest.fixture
def fake():
    return FakeWindows()


@pytest.fixture
def win(fake, tmp_path):
    return WindowsPowerTools(runner=fake, state_file=str(tmp_path / "state.json"))


def test_status(win):
    r = win.get_power_status()
    assert r["ok"] and r["platform"] == "windows"
    assert r["distinct_governors"] == ["schedutil"]
    assert r["screen"]["brightness_percent"] == 70
    assert r["battery"] == {"present": True, "capacity": 66, "status": "Discharging", "ac_online": False}
    json.dumps(r)


def test_set_governor_powersave_and_back(win, fake):
    assert win.set_cpu_governor("powersave")["ok"]
    assert fake.cpu["PROCTHROTTLEMAX"] == [50, 50]
    assert win.get_power_status()["distinct_governors"] == ["powersave"]
    assert win.set_cpu_governor("schedutil")["ok"]
    assert fake.cpu["PROCTHROTTLEMAX"] == [100, 100]


def test_invalid_governor_and_per_cpu_refused(win):
    assert not win.set_cpu_governor("turbo")["ok"]
    r = win.set_cpu_governor("powersave", cpu=2)
    assert not r["ok"] and "per-CPU" in r["error"]


def test_performance_on_low_battery_needs_confirmation():
    fake = FakeWindows(charge=10, status=1)
    t = WindowsPowerTools(runner=fake, state_file="unused.json")
    assert t.set_cpu_governor("performance")["status"] == "confirmation_required"
    assert t.set_cpu_governor("performance", confirmed=True)["ok"]


def test_brightness_set_change_and_guardrails(win, fake):
    r = win.set_screen_brightness(40)
    assert r["ok"] and (r["before_percent"], r["after_percent"]) == (70, 40)
    assert win.change_screen_brightness(-10)["after_percent"] == 30
    assert not win.set_screen_brightness(3)["ok"]                       # hard floor
    assert win.set_screen_brightness(7)["status"] == "confirmation_required"
    assert win.change_screen_brightness(+200)["after_percent"] == 100


def test_battery_saver_then_restore(win, fake):
    r = win.enable_battery_saver()
    assert r["ok"] and fake.brightness == 30
    assert fake.cpu["PROCTHROTTLEMAX"] == [50, 50]
    back = win.restore_previous_settings()
    assert back["ok"]
    assert fake.brightness == 70
    assert fake.cpu == {"PROCTHROTTLEMIN": [100, 5], "PROCTHROTTLEMAX": [100, 100]}   # exactly as before


def test_saver_twice_keeps_original_snapshot(win, fake):
    win.enable_battery_saver()
    win.enable_battery_saver()
    assert win.restore_previous_settings()["ok"] and fake.brightness == 70


def test_restore_without_snapshot(win):
    assert not win.restore_previous_settings()["ok"]


def test_no_display(tmp_path):
    class NoDisplay(FakeWindows):
        def __call__(self, argv):
            if argv[0] == "powershell":
                return 0, json.dumps({"brightness": None, "battery_present": False}), ""
            return super().__call__(argv)
    t = WindowsPowerTools(runner=NoDisplay(), state_file=str(tmp_path / "s.json"))
    assert not t.set_screen_brightness(50)["ok"]
    assert t.get_power_status()["screen"] is None


def test_agent_end_to_end_offline(win, fake):
    agent = PowerManagementAgent(tools=win, use_llm=False, confirm=lambda m: True)
    out = agent.handle("Put my laptop into maximum battery saving mode")
    assert "Battery saver on" in out and fake.brightness == 30
    assert "restored" in agent.handle("restore my settings")
    assert "Governor set to performance" in agent.handle("switch the cpu to performance mode")
