"""Low-level access to the Linux sysfs interface.

Every path is resolved against a configurable root so the same code runs on a
real machine (root "/") and against a simulated tree in the test suite.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from pathlib import Path


class SysfsError(Exception):
    """Raised when a sysfs read or write cannot be completed."""


class Sysfs:
    def __init__(self, root: str | None = None, allow_sudo: bool = True):
        self.root = Path(root or os.environ.get("POWER_AGENT_SYSROOT", "/"))
        self.allow_sudo = allow_sudo and str(self.root) == "/"

    # ---- path helpers -------------------------------------------------
    def path(self, absolute: str) -> Path:
        return self.root / absolute.lstrip("/")

    def glob(self, pattern: str) -> list[Path]:
        hits = glob.glob(str(self.path(pattern)))
        return sorted(Path(h) for h in hits)

    # ---- read / write -------------------------------------------------
    def read(self, absolute: str) -> str:
        try:
            return self.path(absolute).read_text().strip()
        except OSError as exc:
            raise SysfsError(f"cannot read {absolute}: {exc.strerror}") from exc

    def read_file(self, file: Path) -> str:
        try:
            return file.read_text().strip()
        except OSError as exc:
            raise SysfsError(f"cannot read {file}: {exc.strerror}") from exc

    def write_file(self, file: Path, value: str) -> None:
        """Write a value, escalating through `sudo -n tee` if permission is denied."""
        try:
            with open(file, "w") as fh:
                fh.write(value)
            return
        except PermissionError:
            if not (self.allow_sudo and shutil.which("sudo")):
                raise SysfsError(
                    f"permission denied writing {file}; run the shell as root "
                    "or allow passwordless sudo for this agent"
                )
        except OSError as exc:
            raise SysfsError(f"cannot write {file}: {exc.strerror}") from exc

        proc = subprocess.run(
            ["sudo", "-n", "tee", str(file)],
            input=value, text=True, capture_output=True, timeout=5,
        )
        if proc.returncode != 0:
            raise SysfsError(f"sudo write to {file} failed: {proc.stderr.strip()}")
