"""High-level reader combining AC's three pages (and optionally the plugin)."""

from __future__ import annotations

import time

from .plugin_link import PluginLink
from .shm import SharedMemoryPage, SharedMemoryUnavailable
from .shm_layout import (
    GRAPHICS_MAP_NAME, PHYSICS_MAP_NAME, STATIC_MAP_NAME,
    SPageFileGraphic, SPageFilePhysics, SPageFileStatic,
)
from .telemetry import Telemetry, TelemetryDecoder

__all__ = ["ACReader", "SharedMemoryUnavailable"]


class ACReader:
    """Reads AC shared memory and returns :class:`Telemetry`.

    The static page only changes when a session is (re)loaded, so it is cached
    and refreshed when the graphics ``session`` value changes or on demand.

    ``plugin`` (optional :class:`PluginLink`) adds the CSP-app data.  Map names
    are parameters so the tests can point the reader at synthetic mappings.
    """

    def __init__(self, plugin: PluginLink | None = None, physics_name: str = PHYSICS_MAP_NAME,
                 graphics_name: str = GRAPHICS_MAP_NAME, static_name: str = STATIC_MAP_NAME,
                 decoder: TelemetryDecoder | None = None, plugin_max_age_s: float = 0.5):
        self.physics = SharedMemoryPage(physics_name, SPageFilePhysics)
        self.graphics = SharedMemoryPage(graphics_name, SPageFileGraphic)
        self.static_page = SharedMemoryPage(static_name, SPageFileStatic, has_packet_id=False)
        self.plugin = plugin
        self.plugin_max_age_s = plugin_max_age_s
        self.decoder = decoder or TelemetryDecoder()
        self._static = self.static_page.snapshot()
        self._last_session: int | None = None

    @classmethod
    def wait_for_game(cls, timeout_s: float = 60.0, poll_s: float = 0.5, **kw) -> "ACReader":
        """Retry until AC's mappings exist (the game creates them on session load)."""
        deadline = time.perf_counter() + timeout_s
        while True:
            try:
                return cls(**kw)
            except SharedMemoryUnavailable:
                if time.perf_counter() >= deadline:
                    raise
                time.sleep(poll_s)

    def refresh_static(self) -> SPageFileStatic:
        self._static = self.static_page.snapshot()
        return self._static

    @property
    def static(self) -> SPageFileStatic:
        return self._static

    def physics_packet_id(self) -> int:
        return self.physics.packet_id()

    def read(self) -> Telemetry:
        now = time.perf_counter()
        ph = self.physics.snapshot()
        gr = self.graphics.snapshot()
        if gr.session != self._last_session:
            self._last_session = gr.session
            self.refresh_static()
        plugin_state = None
        if self.plugin is not None and self.plugin.alive(self.plugin_max_age_s):
            plugin_state = self.plugin.read_state()
        return self.decoder.decode(ph, gr, self._static, plugin_state, wall_time=now)

    def close(self) -> None:
        for p in (self.physics, self.graphics, self.static_page):
            p.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
