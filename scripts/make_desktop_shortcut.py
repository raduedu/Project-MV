"""Put a "Project MV" shortcut on the desktop that opens the control panel (no terminal needed).

Double-clicking it starts the web control panel if it is not running yet (in
the background, no console window) and opens it in the browser - the same as
``mvctl ui``. Training is controlled from there (Resume / Pause); the
shortcut itself never starts training. Run once::

    .venv\\Scripts\\python.exe scripts\\make_desktop_shortcut.py            # create / refresh
    .venv\\Scripts\\python.exe scripts\\make_desktop_shortcut.py --remove

The icon (``mv/ui/mv.ico``) is drawn here as well, so it needs no image file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ICON = ROOT / "mv" / "ui" / "mv.ico"
NAME = "Project MV.lnk"


def desktop_dir() -> Path:
    """The desktop as Windows knows it (follows OneDrive redirection)."""
    import ctypes
    import uuid

    class GUID(ctypes.Structure):
        _fields_ = [("a", ctypes.c_ulong), ("b", ctypes.c_ushort), ("c", ctypes.c_ushort), ("d", ctypes.c_ubyte * 8)]
    u = uuid.UUID("B4BFCC3A-DB2C-424C-B029-7FE99A87C641")              # FOLDERID_Desktop
    g = GUID(u.fields[0], u.fields[1], u.fields[2], (ctypes.c_ubyte * 8)(*u.bytes[8:]))
    p = ctypes.c_wchar_p()
    if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(p)) != 0:
        return Path.home() / "Desktop"
    return Path(p.value)


def draw_icon(path: Path = ICON) -> Path:
    """A dark rounded square with "MV" in Porsche-ish red, at the sizes Windows asks for."""
    from PIL import Image, ImageDraw, ImageFont
    big = 256
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((8, 8, big - 8, big - 8), radius=48, fill=(22, 24, 28, 255), outline=(213, 0, 28, 255),
                        width=10)
    try:
        font = ImageFont.truetype("arialbd.ttf", 120)
    except OSError:
        font = ImageFont.load_default()
    box = d.textbbox((0, 0), "MV", font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    d.text(((big - w) / 2 - box[0], (big - h) / 2 - box[1] - 6), "MV", font=font, fill=(255, 255, 255, 255))
    d.rectangle((60, 196, 196, 208), fill=(213, 0, 28, 255))           # a red "racing line" under the letters
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return path


def make_shortcut() -> Path:
    icon = draw_icon()
    lnk = desktop_dir() / NAME
    pyw = ROOT / ".venv" / "Scripts" / "pythonw.exe"           # no console window
    ps = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:MV_LNK);"
        "$s.TargetPath = $env:MV_PYW;"
        "$s.Arguments = '\"' + $env:MV_SCRIPT + '\" start';"
        "$s.WorkingDirectory = $env:MV_ROOT;"
        "$s.IconLocation = $env:MV_ICON + ',0';"
        "$s.Description = 'Project MV control panel (starts it if needed, opens the browser)';"
        "$s.Save()"
    )
    env = {**__import__("os").environ, "MV_LNK": str(lnk), "MV_PYW": str(pyw),
           "MV_SCRIPT": str(ROOT / "scripts" / "webui.py"), "MV_ROOT": str(ROOT), "MV_ICON": str(icon)}
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, env=env, creationflags=0x08000000)
    return lnk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remove", action="store_true")
    a = ap.parse_args()
    lnk = desktop_dir() / NAME
    if a.remove:
        lnk.unlink(missing_ok=True)
        print(f"removed {lnk}")
        return 0
    print(f"created {make_shortcut()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
