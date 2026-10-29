from power_agent.agent import PowerManagementAgent
from power_agent.backend import backend_name, create_tools
from power_agent.demo import build_demo_tree
from power_agent.sysfs import Sysfs
from power_agent.tools import PowerTools


def test_demo_tree_runs_through_real_sysfs_code(tmp_path):
    t = PowerTools(Sysfs(str(build_demo_tree(tmp_path / "r"))), state_file=str(tmp_path / "s.json"))
    agent = PowerManagementAgent(tools=t, use_llm=False, confirm=lambda m: True)
    assert "Battery saver on" in agent.handle("maximum battery saving mode")
    assert t.get_power_status()["distinct_governors"] == ["powersave"]
    assert "restored" in agent.handle("undo battery saver")


def test_backend_selection(monkeypatch):
    monkeypatch.setenv("POWER_AGENT_BACKEND", "demo")
    assert backend_name() == "demo"
    assert create_tools().get_power_status()["ok"]
