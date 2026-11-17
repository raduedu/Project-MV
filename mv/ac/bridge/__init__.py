"""Live bridge between Python and a running Assetto Corsa (with CSP).

Layers (see docs/07-assetto-corsa-bridge.md for the full picture)::

    shm_layout  exact ctypes layouts of acpmf_physics / graphics / static
    shm         read-only named-mapping access with torn-read protection
    telemetry   Telemetry dataclass: SI units, project axes (x fwd, y left, z up)
    plugin_link shared-memory protocol to the CSP Lua app "MV Bridge"
    reader      ACReader: pages (+ optional plugin) -> Telemetry
    backends    ControlBackend: CSPLuaBackend (preferred), VJoyBackend, VGamepadBackend
    episode     off-track / wrong-way / collision / stuck detection
    session     ACSession: fixed-rate loop, latency/jitter stats, fail-safe
    env         ACRawEnv: reset() -> Telemetry, step() -> Telemetry

Only stdlib + numpy are imported here; pyvjoy / vgamepad load lazily.
"""

from .backends import (
    ControlBackend, CSPLuaBackend, NullBackend, VGamepadBackend, VJoyBackend, make_backend,
)
from .env import RESET_MODES, ACRawEnv, ResetError
from .episode import EpisodeFlags, EpisodeMonitor
from .plugin_link import PluginLink, PluginState, ResetCommand, ResetResult
from .reader import ACReader
from .session import ACSession, ACTimeoutError, LoopStats
from .shm import SharedMemoryUnavailable
from .frames import ac_local_to_body, ac_world_to_map3
from .telemetry import Telemetry, TelemetryDecoder

__all__ = [
    "ACReader", "ACRawEnv", "ACSession", "ACTimeoutError", "LoopStats", "Telemetry",
    "TelemetryDecoder", "ac_local_to_body", "ac_world_to_map3", "ControlBackend", "CSPLuaBackend", "NullBackend",
    "VJoyBackend", "VGamepadBackend", "make_backend", "PluginLink", "PluginState",
    "ResetCommand", "ResetResult", "EpisodeFlags", "EpisodeMonitor", "RESET_MODES",
    "ResetError", "SharedMemoryUnavailable",
]
