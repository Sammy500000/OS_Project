"""Standalone REPL for trying the agent outside the full Agentic Shell.

    python -m power_agent                  # interactive, real hardware
    python -m power_agent --demo           # interactive, simulated laptop (safe for presentations)
    python -m power_agent "Dim the screen" # single command
"""

from __future__ import annotations

import os
import sys

# Windows consoles default to a legacy code page; keep output readable.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:          # noqa: BLE001
        pass


def main() -> None:
    args = sys.argv[1:]
    if "--demo" in args:
        args.remove("--demo")
        os.environ["POWER_AGENT_BACKEND"] = "demo"
    if "--offline" in args:
        args.remove("--offline")
        os.environ["POWER_AGENT_CHAIN"] = "none"

    from .agent import AGENT_ID, AGENT_NAME, PowerManagementAgent
    from .backend import backend_name

    agent = PowerManagementAgent()
    mode = ("LLM: " + ", ".join(b.name for b in agent.llm.backends)) if agent.llm else "offline router"
    print(f"Agent {AGENT_ID}: {AGENT_NAME} ({mode}; system backend: {backend_name()}). "
          "Type 'exit' to quit.")
    if args:
        print(agent.handle(" ".join(args)))
        return
    while True:
        try:
            line = input("power> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line in {"exit", "quit"}:
            break
        if line:
            print(agent.handle(line))


if __name__ == "__main__":
    main()
