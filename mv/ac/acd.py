"""Reader for Assetto Corsa ``data.acd`` archives.

Every Kunos car ships its physics definition (``car.ini``, ``engine.ini``,
``tyres.ini``, ``power.lut`` ...) packed into a single ``data.acd`` file that
lives in the car's folder.  The archive is a flat list of files.  Each byte of
each file is stored as a 32-bit integer with a repeating key added to it; the
key is a short string derived from the *name of the car folder*.

Layout of the archive
---------------------
::

    [optional]  int32 == -1111, int32 (ignored)        # newer archives only
    repeated:
        int32   name_length
        bytes   name                (name_length bytes, ASCII)
        int32   size
        int32 * size                (one int per payload byte)

    payload_byte[i] = (stored_int[i] - ord(key[i % len(key)])) & 0xFF

Legal note
----------
The contents of ``data.acd`` are the intellectual property of Kunos
Simulazioni.  This reader exists so that the owner of a legitimate copy of the
game can parameterise *their own* simulator from *their own* install.  Unpacked
data must stay on the local machine (``vehicles/local`` and ``.cache`` are
git-ignored for exactly this reason) and must never be redistributed.
"""

from __future__ import annotations

import struct
from pathlib import Path

__all__ = ["derive_key", "read_acd", "load_car_data", "car_has_data", "CarNotAvailable"]


class CarNotAvailable(FileNotFoundError):
    """The car folder exists but holds no physics data (no ``data`` folder, no ``data.acd``).

    Why this is its own error: Steam creates empty folders for DLC cars the
    owner has not bought (seen for ``ks_ferrari_488_gt3_2020`` and
    ``ks_ferrari_488_challenge_evo``).  That is not a bug in the importer, so a
    batch import reports such cars as "not available" and carries on.
    """


def car_has_data(car_dir: str | Path) -> bool:
    """True when the car folder contains physics data we can read."""
    car_dir = Path(car_dir)
    return (car_dir / "data").is_dir() or (car_dir / "data.acd").is_file()


def _i32(v: int) -> int:
    """Wrap a Python integer to a signed 32-bit integer (C# ``int`` overflow)."""
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v & 0x80000000 else v


def _div(a: int, b: int) -> int:
    """Integer division that truncates toward zero (C# semantics)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _mod(a: int, b: int) -> int:
    """Remainder with the sign of the dividend (C# semantics)."""
    return a - b * _div(a, b)


def derive_key(folder_name: str) -> str:
    """Derive the archive key from the car folder name.

    The key is eight numbers (each 0-255) joined with dashes, for example
    ``"114-23-9-201-64-88-171-110"``.
    """
    s = folder_name.lower()
    n = len(s)
    c = [ord(ch) for ch in s]

    k1 = 0
    for i in range(n):
        k1 = _i32(k1 + c[i])

    k2 = 0
    for i in range(0, n - 1, 2):
        k2 = _i32(_i32(k2 * c[i]) - c[i + 1])

    k3 = 0
    for i in range(1, n - 3, 3):
        k3 = _i32(k3 * c[i])
        k3 = _div(k3, c[i + 1] + 0x1B)
        k3 = _i32(k3 + (-0x1B - c[i - 1]))

    k4 = 0x1683
    for i in range(1, n):
        k4 = _i32(k4 - c[i])

    k5 = 0x42
    for i in range(1, n - 4, 4):
        t = _i32((c[i] + 0xF) * k5)
        k5 = _i32(_i32((c[i - 1] + 0xF) * t) + 0x16)

    k6 = 0x65
    for i in range(0, n - 2, 2):
        k6 = _i32(k6 - c[i])

    k7 = 0xAB
    for i in range(0, n - 2, 2):
        k7 = _mod(k7, c[i])

    k8 = 0xAB
    for i in range(0, n - 1):
        k8 = _i32(_div(k8, c[i]) + c[i + 1])

    return "-".join(str(k & 0xFF) for k in (k1, k2, k3, k4, k5, k6, k7, k8))


def read_acd(acd_path: str | Path, folder_name: str | None = None) -> dict[str, bytes]:
    """Read every file out of a ``data.acd`` archive.

    Parameters
    ----------
    acd_path:
        Path to the ``data.acd`` file.
    folder_name:
        Name of the car folder the archive belongs to.  Defaults to the name of
        the directory containing ``acd_path``.  It matters because the key is
        derived from it.

    Returns
    -------
    dict mapping file name (lower-case) to raw file bytes.
    """
    acd_path = Path(acd_path)
    if folder_name is None:
        folder_name = acd_path.parent.name
    key = derive_key(folder_name).encode("ascii")
    klen = len(key)

    raw = acd_path.read_bytes()
    pos = 0
    if len(raw) >= 8 and struct.unpack_from("<i", raw, 0)[0] == -1111:
        pos = 8

    files: dict[str, bytes] = {}
    total = len(raw)
    while pos + 8 <= total:
        (name_len,) = struct.unpack_from("<i", raw, pos)
        pos += 4
        if name_len <= 0 or name_len > 512 or pos + name_len + 4 > total:
            raise ValueError(f"Corrupt ACD archive (bad name length {name_len}) in {acd_path}")
        name = raw[pos : pos + name_len].decode("ascii", errors="replace")
        pos += name_len
        (size,) = struct.unpack_from("<i", raw, pos)
        pos += 4
        if size < 0 or pos + 4 * size > total:
            raise ValueError(f"Corrupt ACD archive (bad size for {name}) in {acd_path}")
        # Each payload byte is stored as a little-endian int32.  Only the low
        # byte carries information, so take every 4th byte and undo the key.
        low = raw[pos : pos + 4 * size : 4]
        pos += 4 * size
        files[name.lower()] = bytes((b - key[i % klen]) & 0xFF for i, b in enumerate(low))
    return files


def load_car_data(car_dir: str | Path) -> dict[str, str]:
    """Return the physics files of an Assetto Corsa car as text.

    Works for both packed cars (``data.acd``) and unpacked cars (a ``data``
    folder, which is what most mods and Content Manager's "unpack" produce).
    The unpacked folder wins if both are present, matching the game's own
    behaviour in developer mode.
    """
    car_dir = Path(car_dir)
    data_dir = car_dir / "data"
    out: dict[str, str] = {}
    if data_dir.is_dir():
        for p in data_dir.iterdir():
            if p.is_file():
                out[p.name.lower()] = p.read_text(encoding="utf-8", errors="replace")
        return out
    acd = car_dir / "data.acd"
    if not acd.is_file():
        raise CarNotAvailable(f"No data folder or data.acd in {car_dir} "
                              "(car not installed - e.g. a DLC that is not owned)")
    for name, blob in read_acd(acd, car_dir.name).items():
        out[name] = blob.decode("utf-8", errors="replace")
    sample = out.get("car.ini", "")
    if "[" not in sample:
        raise ValueError(
            f"Could not decode {acd}: car.ini is not readable text. "
            "The folder name must match the original car id."
        )
    return out
