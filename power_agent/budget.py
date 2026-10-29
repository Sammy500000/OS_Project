"""Planning estimate: how many commands a free tier can serve.

    python -m power_agent.budget

The prompt size is measured from the real system prompt and tool schemas, using about
3.5 characters per token. It is an estimate, not a tokenizer count; the provider's
`usage` field is authoritative at run time.
"""

from __future__ import annotations

import json

from .agent import SYSTEM_PROMPT
from .llm import estimate_tokens, presets, to_openai_tools
from .tools import TOOL_SPECS

OUT_TOKENS = 150        # tool call plus a little reasoning at low effort (assumption)
MARGIN = 0.9            # the tracker keeps 10 % headroom


def plan() -> dict:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "x" * 60}]
    static = estimate_tokens(msgs, TOOL_SPECS, 0)
    per_request = static + OUT_TOKENS
    rows = []
    for name, b in presets().items():
        l = b.limits
        if l.tpm is None and l.rpm is None:
            rows.append({"backend": name, "per_minute": "no limit", "per_day": "no limit"})
            continue
        per_min = min(x for x in (
            int(l.rpm * MARGIN) if l.rpm else None,
            int(l.tpm * MARGIN // per_request) if l.tpm else None) if x is not None)
        per_day = min(x for x in (
            int(l.rpd * MARGIN) if l.rpd else None,
            int(l.tpd * MARGIN // per_request) if l.tpd else None) if x is not None)
        rows.append({"backend": name, "model": b.model, "per_minute": per_min, "per_day": per_day})
    return {"static_prompt_tokens": static, "tokens_per_request": per_request, "rows": rows}


if __name__ == "__main__":
    p = plan()
    print(f"static prompt (system + {len(TOOL_SPECS)} tool schemas): ~{p['static_prompt_tokens']} tokens")
    print(f"one request with {OUT_TOKENS} output tokens:            ~{p['tokens_per_request']} tokens")
    print(f"{'backend':22} {'model':28} {'requests/min':>13} {'requests/day':>13}")
    for r in p["rows"]:
        print(f"{r['backend']:22} {r.get('model', ''):28} {r['per_minute']!s:>13} {r['per_day']!s:>13}")
