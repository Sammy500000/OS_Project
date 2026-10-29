"""Offline intent parser.

Used when no LLM is reachable (no API key, no network) and as the reference
baseline in the evaluation. It maps a sentence to an ordered list of
(tool, arguments) pairs. Chained requests are split into clauses first.
"""

from __future__ import annotations

import re

_CLAUSE_SPLIT = re.compile(r"\s*(?:,|;|\band then\b|\bthen\b|\band\b|\balso\b)\s*", re.I)

_SAVER = re.compile(
    r"(battery[- ]sav|power[- ]sav|save (my )?(battery|power)|maximum (battery|power)|"
    r"max(imum)? saving|low[- ]power|extend (my )?battery|stretch (my )?battery|"
    r"make (my )?battery last)", re.I)
_RESTORE = re.compile(
    r"(restore|undo|revert|back to normal|go back|"
    r"(turn|switch|stop) (off|disable) (the )?(battery|power)[- ]?sav|"
    r"(disable|exit|leave) (the )?(battery|power)[- ]?sav)", re.I)
_LIST_GOV = re.compile(r"(available|supported|which|what|list).{0,30}governors?", re.I)
_STATUS = re.compile(
    r"(status|how much battery|battery (level|percentage|left)|current (governor|brightness|mode)|"
    r"what('s| is) (my )?(current |the )?(governor|brightness|battery|cpu mode)|"
    r"which governor|show (me )?(the )?(power|battery))", re.I)
_GOV_NAMED = re.compile(r"\b(powersave|performance|schedutil|ondemand|conservative|userspace)\b", re.I)
_ABS_BRIGHT = re.compile(
    r"(?:brightness|screen|display|backlight)[^0-9]{0,25}?(\d{1,3})\s*(?:%|percent|per cent)?", re.I)
_PCT = re.compile(r"(\d{1,3})\s*(?:%|percent|per cent)", re.I)
_DIM = re.compile(r"\b(dim|darken|darker|lower|reduce|decrease|turn down|less bright)\b", re.I)
_BRIGHTEN = re.compile(r"\b(brighten|brighter|raise|increase|turn up|boost|more bright)\b", re.I)
_BRIGHT_WORD = re.compile(r"\b(brightness|screen|display|backlight|dim|darker|brighter|brighten)\b", re.I)
_MIN = re.compile(r"\b(minimum|lowest|darkest)\b", re.I)
_MAX = re.compile(r"\b(maximum|max|highest|full|brightest)\b", re.I)


def _parse_clause(text: str, whole: str) -> list[tuple[str, dict]]:
    t = text.strip()
    if not t:
        return []
    if _RESTORE.search(t):
        return [("restore_previous_settings", {})]
    if _SAVER.search(t):
        pct = _PCT.search(t)
        return [("enable_battery_saver", {"brightness_percent": int(pct.group(1))} if pct else {})]
    if _LIST_GOV.search(t):
        return [("list_available_governors", {})]
    if re.search(r"\bgovernors?\b|\b(performance|powersave)\b.{0,12}\bmode\b|\bcpu\b", t, re.I) \
            and _GOV_NAMED.search(t) and not re.search(r"brightness", t, re.I):
        name = _GOV_NAMED.search(t).group(1).lower()
        cpu = re.search(r"\bcpu\s*(\d+)\b|\bcore\s*(\d+)\b", t, re.I)
        args = {"governor": name}
        if cpu:
            args["cpu"] = int(cpu.group(1) or cpu.group(2))
        return [("set_cpu_governor", args)]
    if re.search(r"\bbalanced\b", t, re.I) and re.search(r"\b(cpu|mode|governor)\b", t, re.I):
        return [("set_cpu_governor", {"governor": "schedutil"})]
    if _BRIGHT_WORD.search(t):
        # "dim it to 20%", "set brightness to 40", "brightness 70%"
        m = _PCT.search(t) or _ABS_BRIGHT.search(t)
        to_word = re.search(r"\b(to|at)\b", t, re.I)
        by_word = re.search(r"\bby\b", t, re.I)
        if m and not by_word and (to_word or re.search(r"\bset\b|brightness\s+\d", t, re.I)):
            return [("set_screen_brightness", {"percent": int(m.group(1))})]
        if _MIN.search(t):
            return [("set_screen_brightness", {"percent": 5})]
        if _MAX.search(t) and not _DIM.search(t):
            return [("set_screen_brightness", {"percent": 100})]
        step = int(m.group(1)) if m else 20
        if _DIM.search(t):
            return [("change_screen_brightness", {"delta_percent": -step})]
        if _BRIGHTEN.search(t):
            return [("change_screen_brightness", {"delta_percent": step})]
        if m:
            return [("set_screen_brightness", {"percent": int(m.group(1))})]
    if _STATUS.search(t):
        return [("get_power_status", {})]
    return []


def parse(prompt: str) -> list[tuple[str, dict]]:
    """Return the ordered tool calls implied by `prompt` (empty list if none)."""
    # Whole-sentence patterns first, so "save battery and dim the screen" stays one intent.
    if _RESTORE.search(prompt):
        return [("restore_previous_settings", {})]
    if _SAVER.search(prompt):
        return _parse_clause(prompt, prompt)
    calls: list[tuple[str, dict]] = []
    for clause in _CLAUSE_SPLIT.split(prompt):
        calls.extend(_parse_clause(clause, prompt))
    return calls
