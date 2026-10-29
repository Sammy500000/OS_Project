import json

from power_agent.tools import PowerTools
from power_agent.sysfs import Sysfs
from conftest import make_tree


def gov(tree, i):
    return (tree / f"sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor").read_text().strip()


def raw_brightness(tree):
    return int((tree / "sys/class/backlight/intel_backlight/brightness").read_text())


def test_status_reports_everything(tools):
    r = tools.get_power_status()
    assert r["ok"] and r["cpu_count_with_cpufreq"] == 4
    assert r["distinct_governors"] == ["schedutil"]
    assert r["screen"]["brightness_percent"] == 100
    assert r["battery"]["capacity"] == 80 and r["battery"]["ac_online"] is False


def test_set_governor_all_cpus(tools, tree):
    r = tools.set_cpu_governor("powersave")
    assert r["ok"] and r["changed_cpus"] == [0, 1, 2, 3]
    assert all(gov(tree, i) == "powersave" for i in range(4))


def test_set_governor_single_cpu(tools, tree):
    assert tools.set_cpu_governor("performance", cpu=2)["ok"]
    assert gov(tree, 2) == "performance" and gov(tree, 0) == "schedutil"


def test_unknown_governor_rejected(tools, tree):
    r = tools.set_cpu_governor("turbo_max")
    assert not r["ok"] and "not offered" in r["error"]
    assert gov(tree, 0) == "schedutil"


def test_governor_name_injection_rejected(tools):
    for bad in ["powersave; rm -rf /", "../../etc/passwd", "powersave\nperformance", ""]:
        assert not tools.set_cpu_governor(bad)["ok"]


def test_bad_cpu_index(tools):
    assert not tools.set_cpu_governor("powersave", cpu="0; reboot")["ok"]
    assert not tools.set_cpu_governor("powersave", cpu=99)["ok"]


def test_brightness_scaling(tools, tree):
    r = tools.set_screen_brightness(50)
    assert r["ok"] and r["after_percent"] == 50 and raw_brightness(tree) == 48000


def test_brightness_hard_floor_and_ceiling(tools, tree):
    assert not tools.set_screen_brightness(0)["ok"]
    assert not tools.set_screen_brightness(4)["ok"]
    assert not tools.set_screen_brightness(150)["ok"]
    assert not tools.set_screen_brightness("bright")["ok"]
    assert raw_brightness(tree) == 96000


def test_dim_band_needs_confirmation(tools, tree):
    r = tools.set_screen_brightness(7)
    assert r["status"] == "confirmation_required" and raw_brightness(tree) == 96000
    assert tools.set_screen_brightness(7, confirmed=True)["ok"]


def test_relative_brightness_clamps(tools, tree):
    assert tools.change_screen_brightness(-30)["after_percent"] == 70
    assert tools.change_screen_brightness(+50)["after_percent"] == 100
    r = tools.change_screen_brightness(-500)
    assert r["status"] == "confirmation_required"   # clamps to 5 %, inside the confirm band


def test_performance_on_low_battery_needs_confirmation(tmp_path):
    root = make_tree(tmp_path / "r", battery=(12, "Discharging"), ac=False)
    t = PowerTools(Sysfs(str(root)), state_file=str(tmp_path / "s.json"))
    assert t.set_cpu_governor("performance")["status"] == "confirmation_required"
    assert t.set_cpu_governor("performance", confirmed=True)["ok"]


def test_performance_on_ac_is_free(tmp_path):
    root = make_tree(tmp_path / "r", battery=(12, "Charging"), ac=True)
    t = PowerTools(Sysfs(str(root)), state_file=str(tmp_path / "s.json"))
    assert t.set_cpu_governor("performance")["ok"]


def test_battery_saver_and_restore(tools, tree):
    r = tools.enable_battery_saver()
    assert r["ok"]
    assert all(gov(tree, i) == "powersave" for i in range(4))
    assert tools.get_power_status()["screen"]["brightness_percent"] == 30
    back = tools.restore_previous_settings()
    assert back["ok"]
    assert all(gov(tree, i) == "schedutil" for i in range(4))
    assert tools.get_power_status()["screen"]["brightness_percent"] == 100


def test_saver_twice_keeps_original_snapshot(tools, tree):
    tools.enable_battery_saver()
    tools.enable_battery_saver()
    tools.restore_previous_settings()
    assert gov(tree, 0) == "schedutil"
    assert tools.get_power_status()["screen"]["brightness_percent"] == 100


def test_restore_without_snapshot(tools):
    assert not tools.restore_previous_settings()["ok"]


def test_saver_never_enters_confirm_band(tools):
    tools.enable_battery_saver(brightness_percent=5)
    assert tools.get_power_status()["screen"]["brightness_percent"] >= 10


def test_saver_keeps_already_dim_screen(tmp_path):
    root = make_tree(tmp_path / "r", brightness=(9600, 96000))   # 10 %
    t = PowerTools(Sysfs(str(root)), state_file=str(tmp_path / "s.json"))
    t.enable_battery_saver(30)
    assert t.get_power_status()["screen"]["brightness_percent"] == 10


def test_machine_without_backlight(tmp_path):
    root = make_tree(tmp_path / "r", brightness=None)
    t = PowerTools(Sysfs(str(root)), state_file=str(tmp_path / "s.json"))
    assert not t.set_screen_brightness(50)["ok"]
    assert t.enable_battery_saver()["ok"]          # governor part still works


def test_machine_without_cpufreq(tmp_path):
    root = make_tree(tmp_path / "r", cpus=0)
    t = PowerTools(Sysfs(str(root)), state_file=str(tmp_path / "s.json"))
    assert not t.set_cpu_governor("powersave")["ok"]
    assert not t.list_available_governors()["ok"]


def test_driver_without_powersave(tmp_path):
    root = make_tree(tmp_path / "r", governors="performance")
    t = PowerTools(Sysfs(str(root)), state_file=str(tmp_path / "s.json"))
    assert not t.enable_battery_saver()["ok"]


def test_permission_denied_is_reported(tools, tree):
    f = tree / "sys/devices/system/cpu/cpu1/cpufreq/scaling_governor"
    f.chmod(0o444)
    import os
    if not hasattr(os, "geteuid") or os.geteuid() == 0:
        import pytest; pytest.skip("root or Windows ignores POSIX file modes")
    r = tools.set_cpu_governor("powersave")
    assert not r["ok"] and "permission" in r["error"]


def test_results_are_json_serialisable(tools):
    for fn in (tools.get_power_status, tools.list_available_governors,
               tools.enable_battery_saver, tools.restore_previous_settings):
        json.dumps(fn())
