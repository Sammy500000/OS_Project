"""Live check of the free-model chain. Run this once with your key set:

    export GROQ_API_KEY=gsk_...
    python live_check.py

It sends one tiny request to each configured backend, prints which tool the model chose,
the tokens used, and any rate-limit headers the provider returns.
"""

import json
import sys
import urllib.error
import urllib.request

from power_agent.llm import (LLMChain, USER_AGENT, backends_from_env, to_openai_tools)
from power_agent.tools import TOOL_SPECS

backends = backends_from_env()
if not backends:
    sys.exit("No backend configured. Set GROQ_API_KEY (https://console.groq.com/keys), "
             "OPENROUTER_API_KEY, or OLLAMA_MODEL.")

msgs = [{"role": "system", "content": "Use the tools to act."},
        {"role": "user", "content": "Put my laptop into maximum battery saving mode."}]
for b in backends:
    chain = LLMChain([b])
    try:
        r = chain.chat(msgs, TOOL_SPECS)
        names = [c.name for c in r.tool_calls] or ["(no tool call)"]
        print(f"[ok]   {b.name:20} model={b.model:28} tool={names} tokens={r.total_tokens}")
    except Exception as exc:                         # noqa: BLE001
        print(f"[fail] {b.name:20} {exc}")
    if b.style == "openai" and b.api_key():          # show the limit headers once per backend
        req = urllib.request.Request(
            b.base_url.rstrip("/") + "/chat/completions", method="POST",
            data=json.dumps({"model": b.model, "messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": 5}).encode(),
            headers={"Authorization": f"Bearer {b.api_key()}", "Content-Type": "application/json",
                     "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                for k, v in resp.headers.items():
                    if k.lower().startswith("x-ratelimit"):
                        print(f"         {k}: {v}")
        except urllib.error.URLError:
            pass
