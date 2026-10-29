"""Validation rules applied before any tool touches the system.

The model proposes; this module disposes. Nothing here depends on the LLM.
"""

from __future__ import annotations

import re

MIN_BRIGHTNESS_PERCENT = 5        # hard floor: never allow a black screen
CONFIRM_BELOW_PERCENT = 10        # 5-9 % is allowed only after user consent
_GOVERNOR_RE = re.compile(r"^[a-z_]{3,20}$")
_CPU_ID_RE = re.compile(r"^\d{1,3}$")


class GuardrailViolation(Exception):
    """The request breaks a hard rule and must not be executed."""


def validate_governor(name: str, available: list[str]) -> str:
    if not isinstance(name, str) or not _GOVERNOR_RE.match(name):
        raise GuardrailViolation(f"'{name}' is not a valid governor name")
    if name not in available:
        raise GuardrailViolation(
            f"governor '{name}' is not offered by this CPU driver "
            f"(available: {', '.join(available) or 'none'})"
        )
    return name


def validate_cpu_id(cpu) -> int | None:
    if cpu is None:
        return None
    if not _CPU_ID_RE.match(str(cpu)):
        raise GuardrailViolation(f"'{cpu}' is not a valid CPU index")
    return int(cpu)


def validate_brightness(percent) -> int:
    try:
        value = round(float(percent))
    except (TypeError, ValueError):
        raise GuardrailViolation(f"brightness '{percent}' is not a number")
    if value > 100:
        raise GuardrailViolation("brightness cannot exceed 100 %")
    if value < MIN_BRIGHTNESS_PERCENT:
        raise GuardrailViolation(
            f"brightness below {MIN_BRIGHTNESS_PERCENT} % is refused "
            "because the display may become unreadable"
        )
    return value


def brightness_needs_confirmation(percent: int) -> bool:
    return percent < CONFIRM_BELOW_PERCENT


def governor_needs_confirmation(governor: str, on_battery: bool, capacity: int | None) -> bool:
    """Running flat out while the battery is nearly empty needs a human decision."""
    return governor == "performance" and on_battery and capacity is not None and capacity < 20
