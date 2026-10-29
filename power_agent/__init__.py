"""Power Management Agent (Agent 34) for the Agentic Shell."""

import os as _os
from pathlib import Path as _Path


def _load_dotenv() -> None:
    """Read KEY=value lines from a .env file next to the project (never overrides real variables)."""
    for folder in (_Path.cwd(), _Path(__file__).resolve().parent.parent):
        env = folder / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    _os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            return


_load_dotenv()

from .agent import PowerManagementAgent  # noqa: E402

__all__ = ["PowerManagementAgent"]
__version__ = "1.1.0"
