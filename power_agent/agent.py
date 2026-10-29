"""Agent 34: Power Management Agent.

Entry point used by the Agentic Shell router:

    agent = PowerManagementAgent()
    print(agent.handle("Put my laptop into maximum battery saving mode."))

A free open-weight model (Groq's gpt-oss and Qwen models, OpenRouter free models, or a
local Ollama model) picks the tools through function calling; see llm.py. When no model is
configured or every backend is exhausted, the offline parser in local_router.py is used.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable

from .tools import CONFIRMABLE, READ_ONLY, TOOL_SPECS, PowerTools
from .backend import create_tools
from . import local_router
from .llm import AllBackendsFailed, LLMChain

AGENT_ID = 34
AGENT_NAME = "Power Management Agent"
ROUTING_HINT = ("CPU governors, power saving, battery, screen brightness, dimming, "
                "performance mode, laptop energy use")

SYSTEM_PROMPT = """You control power settings on a laptop (Linux or Windows) through tools: read status, set CPU governors (power profiles), set screen brightness.
Always act through the tools. Relative requests ("a bit dimmer") use change_screen_brightness; "maximum battery saving" uses enable_battery_saver.
Report tool errors plainly. If the request is not about power, say you do not handle it. Reply in one or two sentences."""

ConfirmFn = Callable[[str], bool]


def _ask_terminal(message: str) -> bool:
    try:
        return input(f"  [confirm] {message} Proceed? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


MAX_ROUNDS = 4          # tool rounds per request; free tiers are limited by tokens, so keep it short


def compact(name: str, result: dict) -> dict:
    """Shrink tool results before they go back to the model (tokens are the scarce resource)."""
    if name == "get_power_status" and result.get("ok"):
        return {k: v for k, v in result.items() if k not in ("governors", "cpu_count_with_cpufreq")}
    if name == "set_cpu_governor" and result.get("ok"):
        return {"ok": True, "governor": result["governor"], "cpus_changed": len(result["changed_cpus"])}
    if name == "restore_previous_settings" and result.get("ok"):
        return {"ok": True, "restored": True}
    return result


class PowerManagementAgent:
    """`llm` is any object with chat(messages, tools) -> Reply. Left unset, it is built from the
    environment (GROQ_API_KEY, OPENROUTER_API_KEY, OLLAMA_MODEL). `use_llm=False` forces offline."""

    def __init__(self, tools: PowerTools | None = None, llm=None, confirm: ConfirmFn | None = None,
                 use_llm: bool | None = None, summary: str | None = None):
        self.tools = tools or create_tools()
        self.confirm = confirm or _ask_terminal
        self.llm = llm
        if self.llm is None and use_llm is not False:
            chain = LLMChain()
            self.llm = chain if chain else None
        # "local": answer state-changing requests from the tool results without a second model call
        # "llm":   always send results back to the model for the final wording
        self.summary = summary or os.environ.get("POWER_AGENT_SUMMARY", "local")
        self.last_calls: list[tuple[str, dict, dict]] = []
        self.notes: list[str] = []

    @property
    def client(self):          # kept so callers can test whether an LLM is attached
        return self.llm

    # ------------------------------------------------------------------
    def execute_tool(self, name: str, args: dict) -> dict:
        """Run one tool call. Only registered tools may run; the model can never self-confirm."""
        if name not in {s["name"] for s in TOOL_SPECS}:
            result = {"ok": False, "error": f"unknown tool '{name}'"}
            self.last_calls.append((name, args or {}, result))
            return result
        args = {k: v for k, v in (args or {}).items() if k != "confirmed"}
        try:
            result = getattr(self.tools, name)(**args)
            if result.get("status") == "confirmation_required" and name in CONFIRMABLE:
                if self.confirm(result["message"]):
                    result = getattr(self.tools, name)(**args, confirmed=True)
                else:
                    result = {"ok": False, "error": "cancelled: the user declined the confirmation"}
        except TypeError as exc:   # missing or unexpected arguments from the model
            result = {"ok": False, "error": f"invalid arguments for {name}: {exc}"}
        self.last_calls.append((name, args, result))
        return result

    # ------------------------------------------------------------------
    def handle(self, prompt: str) -> str:
        self.last_calls, self.notes = [], []
        if self.llm is not None:
            try:
                return self._run_llm(prompt)
            except AllBackendsFailed as exc:
                self.notes.append(str(exc))
                if self.last_calls:       # some tools already ran: report them, never re-run
                    return self._summarise() + "\n(the model became unavailable before finishing)"
                return (f"{self._run_local(prompt)}\n"
                        "(no free model was available just now; used the offline router)")
        return self._run_local(prompt)

    run = handle

    # ------------------------------------------------------------------
    def _run_llm(self, prompt: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        for _ in range(MAX_ROUNDS):
            reply = self.llm.chat(messages, TOOL_SPECS)
            if not reply.tool_calls:
                return reply.text.strip() or self._summarise()
            messages.append({"role": "assistant", "content": reply.text or None,
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.name, "arguments": c.raw_arguments or "{}"}}
                                            for c in reply.tool_calls]})
            ran_read_only = False
            for c in reply.tool_calls:
                if c.arguments is None:       # the model produced broken JSON
                    out = {"ok": False, "error": "arguments were not valid JSON"}
                    self.last_calls.append((c.name, {}, out))
                else:
                    out = self.execute_tool(c.name, c.arguments)
                ran_read_only |= c.name in READ_ONLY
                messages.append({"role": "tool", "tool_call_id": c.id, "name": c.name,
                                 "content": json.dumps(compact(c.name, out))})
            multi = re.search(r"\b(and|then|also|after)\b|[,;]", prompt, re.I)
            if self.summary == "local" and not ran_read_only and not multi:
                return self._summarise()      # saves the second request on the common path
        return self._summarise()

    def _summarise(self) -> str:
        if not self.last_calls:
            return "That is not a power-management request I can handle (governors, brightness, battery)."
        return "\n".join(_describe(n, a, r) for n, a, r in self.last_calls)

    def _run_local(self, prompt: str) -> str:
        plan = local_router.parse(prompt)
        if not plan:
            return "That is not a power-management request I can handle (governors, brightness, battery)."
        return "\n".join(_describe(n, a, self.execute_tool(n, a)) for n, a in plan)


def _describe(name: str, args: dict, r: dict) -> str:
    if not r.get("ok"):
        return f"Could not complete {name}: {r.get('error') or r.get('message')}"
    if name == "get_power_status":
        b, s = r["battery"], r["screen"]
        parts = [f"CPU governor: {', '.join(r['distinct_governors']) or 'n/a'}"]
        if s:
            parts.append(f"brightness {s['brightness_percent']}%")
        if b["present"]:
            parts.append(f"battery {b['capacity']}% ({b['status']})")
        return "; ".join(parts) + "."
    if name == "list_available_governors":
        return "Available governors: " + ", ".join(r["available_governors"]) + "."
    if name == "set_cpu_governor":
        return f"Governor set to {r['governor']} on {len(r['changed_cpus'])} CPU(s)."
    if name == "set_screen_brightness":
        return f"Brightness changed from {r['before_percent']}% to {r['after_percent']}%."
    if name == "change_screen_brightness":
        return f"Brightness changed from {r['before_percent']}% to {r['after_percent']}%."
    if name == "enable_battery_saver":
        g = r["steps"]["governor"]
        br = r["steps"]["brightness"]
        tail = (f"screen {br['before_percent']}% -> {br['after_percent']}%"
                if "after_percent" in br else br.get("note", ""))
        return f"Battery saver on: powersave on {len(g['changed_cpus'])} CPU(s); {tail}."
    if name == "restore_previous_settings":
        return "Previous governor and brightness settings restored."
    return "Done."
