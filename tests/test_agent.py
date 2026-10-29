from types import SimpleNamespace as NS

from power_agent.agent import PowerManagementAgent


def gov(tree, i=0):
    return (tree / f"sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor").read_text().strip()


def test_headline_prompt(agent, tree):
    out = agent.handle("Put my laptop into maximum battery saving mode.")
    assert "Battery saver on" in out and gov(tree) == "powersave"


def test_chained_prompt(agent, tree):
    agent.handle("Set the cpu governor to performance and dim the screen to 40%")
    assert gov(tree) == "performance"
    assert agent.tools.get_power_status()["screen"]["brightness_percent"] == 40


def test_out_of_scope_prompt(agent):
    assert "not a power-management request" in agent.handle("delete all my pdf files")
    assert agent.last_calls == []


def test_user_can_decline_confirmation(tools, tree):
    a = PowerManagementAgent(tools=tools, use_llm=False, confirm=lambda m: False)
    out = a.handle("set brightness to 7%")
    assert "declined" in out
    assert tools.get_power_status()["screen"]["brightness_percent"] == 100


def test_unknown_tool_blocked(agent):
    assert not agent.execute_tool("rm_rf", {})["ok"]


def test_model_cannot_self_confirm(tools):
    a = PowerManagementAgent(tools=tools, use_llm=False, confirm=lambda m: False)
    r = a.execute_tool("set_screen_brightness", {"percent": 7, "confirmed": True})
    assert not r["ok"]
    assert tools.get_power_status()["screen"]["brightness_percent"] == 100


# ---- LLM path with a scripted stand-in for the model chain -------------------
from power_agent.llm import AllBackendsFailed, Reply, ToolCall


class FakeLLM:
    def __init__(self, script):
        self.script, self.calls, self.seen = list(script), 0, []

    def chat(self, messages, tools):
        self.calls += 1
        self.seen.append(list(messages))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def call(name, args, id="t1", raw=None):
    import json
    return ToolCall(id, name, args, raw if raw is not None else json.dumps(args or {}))


def reply(text="", *calls):
    return Reply(text, list(calls), "fake", 100)


def test_llm_function_calling_loop(tools, tree):
    llm = FakeLLM([reply("", call("enable_battery_saver", {"brightness_percent": 25}))])
    a = PowerManagementAgent(tools=tools, llm=llm, confirm=lambda m: True)
    out = a.handle("save as much battery as you can")
    assert gov(tree) == "powersave" and "Battery saver on" in out
    assert llm.calls == 1                       # economy mode: no second request


def test_llm_llm_summary_mode_uses_second_request(tools, tree):
    llm = FakeLLM([reply("", call("enable_battery_saver", {})), reply("Battery saver is on.")])
    a = PowerManagementAgent(tools=tools, llm=llm, summary="llm")
    assert a.handle("save battery") == "Battery saver is on." and llm.calls == 2


def test_read_only_results_go_back_to_the_model(tools):
    llm = FakeLLM([reply("", call("get_power_status", {})), reply("Your CPU uses schedutil.")])
    a = PowerManagementAgent(tools=tools, llm=llm)
    assert a.handle("what governor am I on?") == "Your CPU uses schedutil."
    tool_msg = llm.seen[1][-1]
    assert tool_msg["role"] == "tool"
    assert "cpu0" not in tool_msg["content"]    # per-CPU detail is stripped to save tokens


def test_conditional_chain_across_rounds(tools, tree):
    llm = FakeLLM([reply("", call("get_power_status", {})),
                   reply("", call("enable_battery_saver", {}, id="t2")),
                   reply("Battery saver is on."),     # multi-part requests get a closing model turn
                   ])
    a = PowerManagementAgent(tools=tools, llm=llm)
    a.handle("if my battery is not full, save power")
    assert gov(tree) == "powersave" and llm.calls == 3


def test_all_backends_down_falls_back_to_local(tools, tree):
    a = PowerManagementAgent(tools=tools, llm=FakeLLM([AllBackendsFailed("429 everywhere")]),
                             confirm=lambda m: True)
    out = a.handle("switch to powersave governor")
    assert gov(tree) == "powersave" and "offline router" in out


def test_failure_after_tools_ran_never_reruns_them(tools, tree):
    llm = FakeLLM([reply("", call("get_power_status", {})), AllBackendsFailed("budget spent")])
    a = PowerManagementAgent(tools=tools, llm=llm)
    out = a.handle("check battery then dim the screen")
    assert "became unavailable" in out and len(a.last_calls) == 1


def test_llm_hallucinated_tool_is_contained(tools, tree):
    llm = FakeLLM([reply("", call("run_shell", {"cmd": "rm -rf /"})), reply("I cannot do that.")])
    a = PowerManagementAgent(tools=tools, llm=llm)
    a.handle("wipe it")
    assert a.last_calls[0][2]["ok"] is False and gov(tree) == "schedutil"


def test_model_with_broken_json_arguments(tools, tree):
    llm = FakeLLM([reply("", call("set_cpu_governor", None, raw='{"governor": "powersave"')),
                   reply("Sorry, I could not do that.")])
    a = PowerManagementAgent(tools=tools, llm=llm, summary="llm")
    a.handle("powersave please")
    assert "valid JSON" in a.last_calls[0][2]["error"] and gov(tree) == "schedutil"
    assert llm.calls == 2                       # the error went back to the model


def test_malformed_arguments_are_contained(agent):
    assert not agent.execute_tool("set_cpu_governor", {})["ok"]
    assert not agent.execute_tool("set_screen_brightness", {"percent": 40, "colour": "red"})["ok"]
    assert not agent.execute_tool("change_screen_brightness", {"delta_percent": None})["ok"]


def test_mcp_server_registers_all_tools(tools):
    import asyncio
    from power_agent.mcp_server import build_server
    srv = build_server(tools)
    listing = srv.list_tools()
    listing = asyncio.run(listing) if asyncio.iscoroutine(listing) else listing
    names = {t.name for t in (listing.tools if hasattr(listing, "tools") else listing)}
    assert names == {"get_power_status", "list_available_governors", "set_cpu_governor",
                     "set_screen_brightness", "change_screen_brightness",
                     "enable_battery_saver", "restore_previous_settings"}
