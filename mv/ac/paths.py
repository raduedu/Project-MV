"""Locate the Assetto Corsa installation."""

from __future__ import annotations

import os
import re
import string
from pathlib import Path

_SUFFIX = Path("steamapps") / "common" / "assettocorsa"


def find_ac_root(explicit: str | os.PathLike | None = None) -> Path:
    """Return the Assetto Corsa install folder.

    Order: explicit argument, ``MV_AC_ROOT`` environment variable, the Steam
    library list (``libraryfolders.vdf``), then common folder names on every
    drive.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("MV_AC_ROOT"):
        candidates.append(Path(os.environ["MV_AC_ROOT"]))
    vdf = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Steam" / "steamapps" / "libraryfolders.vdf"
    if vdf.is_file():
        for m in re.finditer(r'"path"\s+"([^"]+)"', vdf.read_text(encoding="utf-8", errors="ignore")):
            candidates.append(Path(m.group(1).replace("\\\\", "\\")) / _SUFFIX)
    for letter in string.ascii_uppercase:
        for base in ("SteamLibrary", "Steam", r"Program Files (x86)\Steam", r"Games\SteamLibrary"):
            candidates.append(Path(f"{letter}:\\") / base / _SUFFIX)
    for c in candidates:
        if (c / "content" / "cars").is_dir():
            return c
    raise FileNotFoundError("Assetto Corsa not found. Pass --ac-root or set MV_AC_ROOT.")
