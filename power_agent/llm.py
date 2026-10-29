"""LLM access for the agent: free, open-weight models behind a fallback chain.

Design goals
  * zero cost: only free tiers and local models, no paid provider needed
  * no extra dependencies: plain HTTPS through urllib
  * survive the limits of free tiers: a client-side budget tracker, per-model
    fallback on HTTP 429, and a last-resort local model

Two wire formats are spoken: the OpenAI-compatible chat API (Groq, OpenRouter)
and Ollama's native /api/chat. Inside the agent, messages always use the
OpenAI shape; the Ollama adapter converts on the way in and out.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

USER_AGENT = "power-agent/1.1"   # some CDNs reject urllib's default agent
MAX_WAIT_SECONDS = 8.0           # longer than this and we move to the next model


# ----------------------------------------------------------------------
# data types
# ----------------------------------------------------------------------
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None        # None when the model produced invalid JSON
    raw_arguments: str = ""


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall]
    backend: str
    total_tokens: int = 0


class LLMError(Exception):
    pass


class RateLimited(LLMError):
    def __init__(self, retry_after: float | None, message: str = ""):
        super().__init__(message or "rate limited")
        self.retry_after = retry_after


class ToolUseFailed(LLMError):
    """The provider could not parse the tool call the model generated."""


class AuthFailed(LLMError):
    pass


class AllBackendsFailed(LLMError):
    pass


@dataclass
class Limits:
    rpm: int | None = None
    rpd: int | None = None
    tpm: int | None = None
    tpd: int | None = None


@dataclass
class Backend:
    name: str
    style: str                    # "openai" or "ollama"
    base_url: str
    model: str
    api_key_env: str | None
    limits: Limits
    max_tokens_field: str = "max_tokens"
    extra: dict = field(default_factory=dict)
    key: str | None = None

    def api_key(self) -> str | None:
        return self.key or (os.environ.get(self.api_key_env) if self.api_key_env else None)


# ----------------------------------------------------------------------
# published free-tier limits (checked against the providers' docs, Oct 2026)
# ----------------------------------------------------------------------
GROQ_URL = "https://api.groq.com/openai/v1"
_GROQ_LIMITS = Limits(rpm=30, rpd=1000, tpm=8000, tpd=200_000)   # gpt-oss and qwen3.8 rows


def presets() -> dict[str, Backend]:
    return {
        "groq-gpt-oss-120b": Backend(
            "groq-gpt-oss-120b", "openai", GROQ_URL, os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
            "GROQ_API_KEY", _GROQ_LIMITS, "max_completion_tokens", {"reasoning_effort": "low"}),
        "groq-qwen3.8-27b": Backend(
            "groq-qwen3.8-27b", "openai", GROQ_URL, "qwen/qwen3.8-27b",
            "GROQ_API_KEY", _GROQ_LIMITS, "max_completion_tokens"),
        "groq-gpt-oss-20b": Backend(
            "groq-gpt-oss-20b", "openai", GROQ_URL, "openai/gpt-oss-20b",
            "GROQ_API_KEY", _GROQ_LIMITS, "max_completion_tokens", {"reasoning_effort": "low"}),
        "openrouter-free": Backend(
            "openrouter-free", "openai", "https://openrouter.ai/api/v1",
            os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b:free"),
            "OPENROUTER_API_KEY", Limits(rpm=20, rpd=50)),
        "ollama-local": Backend(
            "ollama-local", "ollama", os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            os.environ.get("OLLAMA_MODEL", "qwen3:8b"), None, Limits()),
    }


DEFAULT_ORDER = ["groq-gpt-oss-120b", "groq-qwen3.8-27b", "groq-gpt-oss-20b",
                 "openrouter-free", "ollama-local"]


def backends_from_env() -> list[Backend]:
    """Build the chain from the environment.

    POWER_AGENT_CHAIN may list preset names in order. Without it, a preset is
    included when its key is set; the local Ollama model is included only when
    OLLAMA_MODEL is set or POWER_AGENT_USE_OLLAMA=1.
    """
    table = presets()
    wanted = os.environ.get("POWER_AGENT_CHAIN")
    if wanted:
        names = [n.strip() for n in wanted.split(",") if n.strip()]
        return [table[n] for n in names if n in table]
    chain = []
    for n in DEFAULT_ORDER:
        b = table[n]
        if b.style == "ollama":
            if os.environ.get("OLLAMA_MODEL") or os.environ.get("POWER_AGENT_USE_OLLAMA") == "1":
                chain.append(b)
        elif b.api_key():
            chain.append(b)
    return chain


# ----------------------------------------------------------------------
# client-side budget tracker
# ----------------------------------------------------------------------
class Budget:
    """Sliding-window tracker so we back off before the provider has to say 429."""

    def __init__(self, limits: Limits, clock: Callable[[], float] = time.time, margin: float = 0.9):
        self.l, self.clock, self.margin = limits, clock, margin
        self.events: deque[tuple[float, int]] = deque()      # (time, tokens)
        self.blocked_until = 0.0

    def _trim(self, now):
        while self.events and now - self.events[0][0] > 86400:
            self.events.popleft()

    def _sum(self, now, window):
        return sum(t for ts, t in self.events if now - ts <= window)

    def _count(self, now, window):
        return sum(1 for ts, _ in self.events if now - ts <= window)

    def wait_for(self, tokens: int) -> float | None:
        """Seconds to wait before sending, or None when the daily budget is spent."""
        now = self.clock()
        self._trim(now)
        wait = max(0.0, self.blocked_until - now)
        l, m = self.l, self.margin
        if l.rpd is not None and self._count(now, 86400) >= l.rpd * m:
            return None
        if l.tpd is not None and self._sum(now, 86400) + tokens > l.tpd * m:
            return None
        if l.rpm is not None and self._count(now, 60) >= l.rpm * m:
            first = next(ts for ts, _ in self.events if now - ts <= 60)
            wait = max(wait, 60 - (now - first) + 0.1)
        if l.tpm is not None and self._sum(now, 60) + tokens > l.tpm * m:
            # wait until enough of the last minute's tokens have aged out
            running, need = self._sum(now, 60) + tokens - l.tpm * m, 0.0
            for ts, t in self.events:
                if now - ts <= 60:
                    running -= t
                    need = 60 - (now - ts) + 0.1
                    if running <= 0:
                        break
            wait = max(wait, need)
        return wait

    def record(self, tokens: int):
        self.events.append((self.clock(), tokens))

    def penalise(self, seconds: float):
        self.blocked_until = max(self.blocked_until, self.clock() + seconds)


def estimate_tokens(messages: list[dict], tools: list[dict], max_out: int) -> int:
    """Rough size of a request (about 3.5 characters per token) plus the output allowance."""
    chars = len(json.dumps(messages)) + len(json.dumps(tools))
    return int(chars / 3.5) + max_out


# ----------------------------------------------------------------------
# wire formats
# ----------------------------------------------------------------------
def to_openai_tools(specs: list[dict]) -> list[dict]:
    return [{"type": "function",
             "function": {"name": s["name"], "description": s["description"],
                          "parameters": s["input_schema"]}} for s in specs]


def _parse_args(raw) -> tuple[dict | None, str]:
    if isinstance(raw, dict):
        return raw, json.dumps(raw)
    try:
        val = json.loads(raw or "{}")
        return (val if isinstance(val, dict) else None), raw or ""
    except (TypeError, ValueError):
        return None, str(raw)


def _post(url: str, body: dict, headers: dict, timeout: float) -> tuple[dict, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": USER_AGENT, **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode()), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        text = exc.read().decode(errors="replace")
        hdrs = {k.lower(): v for k, v in exc.headers.items()}
        if exc.code == 429:
            ra = hdrs.get("retry-after")
            raise RateLimited(float(ra) if ra and ra.replace(".", "", 1).isdigit() else None, text[:200])
        if exc.code in (401, 403):
            raise AuthFailed(f"HTTP {exc.code}: {text[:200]}")
        if exc.code == 400 and "tool_use_failed" in text:
            raise ToolUseFailed(text[:200])
        raise LLMError(f"HTTP {exc.code}: {text[:200]}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f"network error: {exc}")


def _call_openai(b: Backend, messages, tools, max_out, timeout, use_extra=True) -> Reply:
    body = {"model": b.model, "messages": messages, "tools": to_openai_tools(tools),
            "tool_choice": "auto", "temperature": 0, b.max_tokens_field: max_out}
    if use_extra:
        body.update(b.extra)
    headers = {"Authorization": f"Bearer {b.api_key()}"} if b.api_key() else {}
    try:
        data, _ = _post(b.base_url.rstrip("/") + "/chat/completions", body, headers, timeout)
    except LLMError as exc:
        # an unsupported optional parameter (for example reasoning_effort): retry without extras
        if use_extra and b.extra and "HTTP 400" in str(exc) and "tool_use_failed" not in str(exc):
            return _call_openai(b, messages, tools, max_out, timeout, use_extra=False)
        raise
    msg = (data.get("choices") or [{}])[0].get("message", {})
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        args, raw = _parse_args(fn.get("arguments"))
        calls.append(ToolCall(tc.get("id") or f"call_{uuid.uuid4().hex[:8]}", fn.get("name", ""), args, raw))
    usage = data.get("usage", {}) or {}
    return Reply(msg.get("content") or "", calls, b.name, int(usage.get("total_tokens") or 0))


def _to_ollama_messages(messages: list[dict]) -> list[dict]:
    out = []
    names: dict[str, str] = {}
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                names[tc["id"]] = tc["function"]["name"]
                args, _ = _parse_args(tc["function"]["arguments"])
                calls.append({"function": {"name": tc["function"]["name"], "arguments": args or {}}})
            out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_name": names.get(m.get("tool_call_id"), ""),
                        "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def _call_ollama(b: Backend, messages, tools, max_out, timeout) -> Reply:
    body = {"model": b.model, "messages": _to_ollama_messages(messages),
            "tools": to_openai_tools(tools), "stream": False,
            "options": {"temperature": 0, "num_predict": max_out}}
    data, _ = _post(b.base_url.rstrip("/") + "/api/chat", body, {}, timeout)
    msg = data.get("message", {})
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        args, raw = _parse_args(fn.get("arguments"))
        calls.append(ToolCall(f"call_{uuid.uuid4().hex[:8]}", fn.get("name", ""), args, raw))
    used = int(data.get("prompt_eval_count") or 0) + int(data.get("eval_count") or 0)
    return Reply(msg.get("content") or "", calls, b.name, used)


# ----------------------------------------------------------------------
# the chain
# ----------------------------------------------------------------------
class LLMChain:
    """Try each backend in order, respecting its published limits."""

    def __init__(self, backends: list[Backend] | None = None, max_out: int = 300, timeout: float = 30.0,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time):
        self.backends = backends if backends is not None else backends_from_env()
        self.max_out, self.timeout, self.sleep = max_out, timeout, sleep
        self.budgets = {b.name: Budget(b.limits, clock) for b in self.backends}
        self.disabled: set[str] = set()
        self.log: list[str] = []          # human-readable trail of fallbacks, for diagnostics
        self.usage: dict[str, int] = {}   # tokens consumed per backend

    def __bool__(self) -> bool:
        return bool(self.backends)

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        est = estimate_tokens(messages, tools, self.max_out)
        for b in self.backends:
            if b.name in self.disabled:
                continue
            budget = self.budgets[b.name]
            wait = budget.wait_for(est)
            if wait is None:
                self.log.append(f"{b.name}: daily budget spent, skipped")
                continue
            if wait > MAX_WAIT_SECONDS:
                self.log.append(f"{b.name}: would wait {wait:.0f}s, skipped")
                continue
            if wait > 0:
                self.sleep(wait)
            for attempt in (1, 2):
                try:
                    caller = _call_ollama if b.style == "ollama" else _call_openai
                    reply = caller(b, messages, tools, self.max_out, self.timeout)
                    budget.record(reply.total_tokens or est)
                    self.usage[b.name] = self.usage.get(b.name, 0) + (reply.total_tokens or est)
                    return reply
                except RateLimited as exc:
                    budget.penalise(exc.retry_after or 30.0)
                    self.log.append(f"{b.name}: HTTP 429 (retry-after {exc.retry_after}), next model")
                    break
                except ToolUseFailed:
                    self.log.append(f"{b.name}: tool call could not be parsed (attempt {attempt})")
                    continue
                except AuthFailed as exc:
                    self.disabled.add(b.name)
                    self.log.append(f"{b.name}: {exc}")
                    break
                except LLMError as exc:
                    self.log.append(f"{b.name}: {exc}")
                    break
        raise AllBackendsFailed("; ".join(self.log[-5:]) or "no LLM backend configured")
