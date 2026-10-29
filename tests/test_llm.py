"""LLM layer tests against a local mock server that speaks both wire formats."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from power_agent.llm import (AllBackendsFailed, Backend, Budget, LLMChain, Limits, backends_from_env,
                             estimate_tokens, to_openai_tools)
from power_agent.tools import TOOL_SPECS
from power_agent.agent import PowerManagementAgent


class Mock:
    """Serves scripted (status, headers, body) tuples, one per request, and records requests."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
                status, headers, payload = outer.script.pop(0)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def oa_tool(name, args, tokens=500):
    return (200, {}, {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}],
        "usage": {"total_tokens": tokens}})


def oa_text(text, tokens=300):
    return (200, {}, {"choices": [{"message": {"content": text}}], "usage": {"total_tokens": tokens}})


def backend(name, mock, style="openai", **kw):
    b = Backend(name, style, mock.url + ("/v1" if style == "openai" else ""), "m", None,
                kw.pop("limits", Limits()), **kw)
    b.key = "test-key"
    return b


@pytest.fixture
def mocks():
    made = []

    def make(script):
        m = Mock(script)
        made.append(m)
        return m
    yield make
    for m in made:
        m.close()


MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "dim the screen"}]


def test_openai_tool_call_round_trip(mocks):
    m = mocks([oa_tool("change_screen_brightness", {"delta_percent": -20})])
    chain = LLMChain([backend("g", m)], sleep=lambda s: None)
    r = chain.chat(MSGS, TOOL_SPECS)
    assert r.tool_calls[0].name == "change_screen_brightness" and r.tool_calls[0].arguments == {"delta_percent": -20}
    req = m.requests[0]
    assert req["path"] == "/v1/chat/completions" and req["headers"]["Authorization"] == "Bearer test-key"
    assert req["body"]["tools"][0]["type"] == "function" and req["body"]["temperature"] == 0
    assert chain.usage["g"] == 500


def test_429_falls_through_to_next_model(mocks):
    m1 = mocks([(429, {"retry-after": "42"}, {"error": {"message": "rate limit"}})])
    m2 = mocks([oa_text("ok")])
    chain = LLMChain([backend("first", m1), backend("second", m2)], sleep=lambda s: None)
    assert chain.chat(MSGS, TOOL_SPECS).backend == "second"
    assert any("429" in line for line in chain.log)
    # the penalised model is skipped on the next call without touching the network
    m2.script.append(oa_text("again"))
    chain.chat(MSGS, TOOL_SPECS)
    assert len(m1.requests) == 1


def test_all_backends_exhausted_raises(mocks):
    m = mocks([(429, {"retry-after": "5"}, {})])
    chain = LLMChain([backend("only", m)], sleep=lambda s: None)
    with pytest.raises(AllBackendsFailed):
        chain.chat(MSGS, TOOL_SPECS)


def test_tool_use_failed_is_retried_once(mocks):
    bad = (400, {}, {"error": {"code": "tool_use_failed", "message": "Failed to call a function"}})
    m = mocks([bad, oa_tool("get_power_status", {})])
    chain = LLMChain([backend("g", m)], sleep=lambda s: None)
    assert chain.chat(MSGS, TOOL_SPECS).tool_calls[0].name == "get_power_status"
    assert len(m.requests) == 2


def test_unsupported_extra_parameter_is_dropped(mocks):
    m = mocks([(400, {}, {"error": {"message": "unknown parameter reasoning_effort"}}), oa_text("fine")])
    chain = LLMChain([backend("g", m, extra={"reasoning_effort": "low"})], sleep=lambda s: None)
    assert chain.chat(MSGS, TOOL_SPECS).text == "fine"
    assert "reasoning_effort" in m.requests[0]["body"] and "reasoning_effort" not in m.requests[1]["body"]


def test_bad_key_disables_backend(mocks):
    m1 = mocks([(401, {}, {"error": "invalid key"})])
    m2 = mocks([oa_text("ok"), oa_text("ok2")])
    chain = LLMChain([backend("a", m1), backend("b", m2)], sleep=lambda s: None)
    chain.chat(MSGS, TOOL_SPECS)
    chain.chat(MSGS, TOOL_SPECS)
    assert len(m1.requests) == 1 and "a" in chain.disabled


def test_invalid_json_arguments_are_flagged_not_raised(mocks):
    m = mocks([(200, {}, {"choices": [{"message": {"tool_calls": [
        {"id": "x", "function": {"name": "set_cpu_governor", "arguments": '{"governor": '}}]}}]})])
    r = LLMChain([backend("g", m)], sleep=lambda s: None).chat(MSGS, TOOL_SPECS)
    assert r.tool_calls[0].arguments is None


def test_ollama_native_format(mocks):
    m = mocks([(200, {}, {"message": {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "set_screen_brightness", "arguments": {"percent": 40}}}]},
        "prompt_eval_count": 700, "eval_count": 30})])
    chain = LLMChain([backend("local", m, style="ollama")], sleep=lambda s: None)
    history = MSGS + [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c0", "type": "function", "function": {"name": "get_power_status", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c0", "name": "get_power_status", "content": "{}"}]
    r = chain.chat(history, TOOL_SPECS)
    assert r.tool_calls[0].arguments == {"percent": 40} and r.total_tokens == 730
    sent = m.requests[0]
    assert sent["path"] == "/api/chat" and sent["body"]["stream"] is False
    assert sent["body"]["messages"][2]["tool_calls"][0]["function"]["arguments"] == {}     # dict, not string
    assert sent["body"]["messages"][3]["tool_name"] == "get_power_status"


def test_budget_waits_for_tpm_then_proceeds():
    t = [1000.0]
    b = Budget(Limits(tpm=8000), clock=lambda: t[0])
    assert b.wait_for(1000) == 0
    b.record(6500)
    w = b.wait_for(1000)                 # 6500 + 1000 > 0.9 * 8000
    assert 59 < w < 61
    t[0] += 61
    assert b.wait_for(1000) == 0


def test_budget_daily_exhaustion_returns_none():
    t = [0.0]
    b = Budget(Limits(rpd=50), clock=lambda: t[0])
    for _ in range(45):
        b.record(10)
        t[0] += 1
    assert b.wait_for(10) is None


def test_chain_skips_backend_that_would_wait_too_long(mocks):
    m1, m2 = mocks([]), mocks([oa_text("ok")])
    first = backend("tight", m1, limits=Limits(tpm=1000))      # one request already blows the minute
    chain = LLMChain([first, backend("free", m2)], sleep=lambda s: None)
    chain.budgets["tight"].record(900)
    assert chain.chat(MSGS, TOOL_SPECS).backend == "free" and not m1.requests


def test_chain_from_environment(monkeypatch):
    for k in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_MODEL", "POWER_AGENT_CHAIN", "POWER_AGENT_USE_OLLAMA"):
        monkeypatch.delenv(k, raising=False)
    assert backends_from_env() == []
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    assert [b.name for b in backends_from_env()] == ["groq-gpt-oss-120b", "groq-qwen3.8-27b", "groq-gpt-oss-20b"]
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    assert [b.name for b in backends_from_env()][-2:] == ["openrouter-free", "ollama-local"]
    monkeypatch.setenv("POWER_AGENT_CHAIN", "ollama-local")
    assert [b.name for b in backends_from_env()] == ["ollama-local"]


def test_published_limits_are_encoded():
    from power_agent.llm import presets
    g = presets()["groq-gpt-oss-120b"].limits
    assert (g.rpm, g.rpd, g.tpm, g.tpd) == (30, 1000, 8000, 200_000)
    o = presets()["openrouter-free"].limits
    assert (o.rpm, o.rpd) == (20, 50)


def test_request_fits_inside_one_minute_budget():
    """The static prompt (system + tool schemas) must leave room for several requests per minute."""
    from power_agent.agent import SYSTEM_PROMPT
    est = estimate_tokens([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "x" * 80}],
                          TOOL_SPECS, 300)
    assert est < 1500                       # 8,000 tokens per minute then allows at least five requests


def test_end_to_end_agent_with_mock_model(mocks, tools, tree):
    m = mocks([oa_tool("enable_battery_saver", {"brightness_percent": 30})])
    chain = LLMChain([backend("g", m)], sleep=lambda s: None)
    out = PowerManagementAgent(tools=tools, llm=chain).handle("Put my laptop into maximum battery saving mode.")
    gov = (tree / "sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    assert gov == "powersave" and "Battery saver on" in out
