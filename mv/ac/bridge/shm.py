"""Read-only access to named Windows file mappings (AC shared memory).

Why not ``mmap.mmap(-1, size, tagname=...)``?
    Python's ``mmap`` calls ``CreateFileMapping``: if the game is not running
    it silently *creates* a zero-filled mapping with our size.  When AC starts
    afterwards it opens that object instead of creating its own; if our size
    were smaller than AC's struct, AC's ``MapViewOfFile`` would fail and its
    shared memory would be broken until every reader closes.  We therefore
    use ``OpenFileMappingW`` (open-only, read-only) so a missing game is an
    explicit :class:`SharedMemoryUnavailable` and we can never disturb AC.

Why map the *whole* section and ``VirtualQuery`` it?
    Mapping exactly ``sizeof(struct)`` fails if the section is smaller (an
    older AC), so we map with length 0 (= whole section) and clamp reads to
    the region size.  Missing trailing fields read as zero.

Torn reads
    AC writes the page while we copy it.  :meth:`SharedMemoryPage.snapshot`
    reads ``packetId`` (first int of physics/graphics), copies the struct,
    re-reads ``packetId`` and retries if it changed.  This is the standard
    best-effort approach for AC (it has no seqlock).
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Generic, TypeVar

__all__ = ["SharedMemoryUnavailable", "SharedMemoryPage", "NamedMapping"]

T = TypeVar("T", bound=ctypes.Structure)

_FILE_MAP_READ = 0x0004
_FILE_MAP_WRITE = 0x0002


class SharedMemoryUnavailable(RuntimeError):
    """The named mapping does not exist (game not running / not on track)."""


class _MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_void_p),
        ("AllocationBase", ctypes.c_void_p),
        ("AllocationProtect", wintypes.DWORD),
        ("PartitionId", wintypes.WORD),
        ("RegionSize", ctypes.c_size_t),
        ("State", wintypes.DWORD),
        ("Protect", wintypes.DWORD),
        ("Type", wintypes.DWORD),
    ]


def _kernel32():
    if sys.platform != "win32":  # pragma: no cover - project is Windows-only
        raise SharedMemoryUnavailable("Assetto Corsa shared memory is only available on Windows")
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    k.OpenFileMappingW.restype = wintypes.HANDLE
    k.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                wintypes.DWORD, ctypes.c_size_t]
    k.MapViewOfFile.restype = ctypes.c_void_p
    k.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
    k.UnmapViewOfFile.restype = wintypes.BOOL
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.CloseHandle.restype = wintypes.BOOL
    k.VirtualQuery.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MEMORY_BASIC_INFORMATION),
                               ctypes.c_size_t]
    k.VirtualQuery.restype = ctypes.c_size_t
    return k


class NamedMapping:
    """An *existing* named file mapping, opened (never created) and mapped.

    ``writable=False`` (default) maps read-only, which is what we want for
    AC's pages: a bug in our code can then never corrupt the game's memory.
    """

    def __init__(self, name: str, writable: bool = False):
        self.name = name
        self._k = _kernel32()
        access = _FILE_MAP_READ | (_FILE_MAP_WRITE if writable else 0)
        self._handle = self._k.OpenFileMappingW(access, False, name)
        if not self._handle:
            err = ctypes.get_last_error()
            raise SharedMemoryUnavailable(
                f"shared memory {name!r} not found (WinError {err}); is Assetto Corsa running "
                "and a session loaded?")
        self._addr = self._k.MapViewOfFile(self._handle, access, 0, 0, 0)
        if not self._addr:
            err = ctypes.get_last_error()
            self._k.CloseHandle(self._handle)
            self._handle = None
            raise SharedMemoryUnavailable(f"MapViewOfFile({name!r}) failed (WinError {err})")
        mbi = _MEMORY_BASIC_INFORMATION()
        if self._k.VirtualQuery(self._addr, ctypes.byref(mbi), ctypes.sizeof(mbi)):
            self.size = int(mbi.RegionSize)
        else:  # pragma: no cover - VirtualQuery on our own view does not fail
            self.size = 4096

    @property
    def address(self) -> int:
        if not self._addr:
            raise ValueError(f"mapping {self.name!r} is closed")
        return self._addr

    def read_bytes(self, n: int, offset: int = 0) -> bytes:
        n = max(0, min(n, self.size - offset))
        return ctypes.string_at(self.address + offset, n)

    def read_i32(self, offset: int = 0) -> int:
        return ctypes.c_int32.from_address(self.address + offset).value

    def close(self) -> None:
        if getattr(self, "_addr", None):
            self._k.UnmapViewOfFile(self._addr)
            self._addr = None
        if getattr(self, "_handle", None):
            self._k.CloseHandle(self._handle)
            self._handle = None

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SharedMemoryPage(Generic[T]):
    """Typed, tear-checked snapshots of one AC page.

    ``snapshot()`` returns a *copy* of the struct (a fresh ``ctypes`` object
    not aliasing shared memory), so the caller can keep it while AC moves on.
    """

    def __init__(self, name: str, struct_type: type[T], has_packet_id: bool = True,
                 max_retries: int = 8):
        self.struct_type = struct_type
        self.mapping = NamedMapping(name, writable=False)
        self.has_packet_id = has_packet_id
        self.max_retries = max_retries
        self.available_bytes = min(ctypes.sizeof(struct_type), self.mapping.size)
        self.torn_reads = 0

    def packet_id(self) -> int:
        return self.mapping.read_i32(0)

    def snapshot(self) -> T:
        size = ctypes.sizeof(self.struct_type)
        for _ in range(self.max_retries):
            before = self.packet_id() if self.has_packet_id else 0
            raw = self.mapping.read_bytes(self.available_bytes)
            if not self.has_packet_id or self.packet_id() == before:
                break
            self.torn_reads += 1
        buf = bytearray(size)
        buf[: len(raw)] = raw
        return self.struct_type.from_buffer_copy(buf)

    def close(self) -> None:
        self.mapping.close()
