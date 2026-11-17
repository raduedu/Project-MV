"""Thin environment facade: ``reset() -> Telemetry``, ``step(...) -> Telemetry``.

Deliberately *no* observation vectors, rewards or termination logic: the
lead's observation builder sits on top.  :attr:`ACRawEnv.flags` exposes the
:class:`~mv.ac.bridge.episode.EpisodeFlags` of the last step for whoever
decides about termination.
"""

from __future__ import annotations

import time

from .backends import ControlBackend, CSPLuaBackend, make_backend
from .episode import EpisodeFlags, EpisodeMonitor
from .plugin_link import PluginLink, ResetCommand, ResetResult
from .reader import ACReader
from .session import ACSession
from .telemetry import Telemetry

__all__ = ["ACRawEnv", "ResetError", "RESET_MODES"]

#: reset mode -> CSP Lua command (None = no teleport, just brake to a stop)
RESET_MODES: dict[str, ResetCommand | None] = {
    "none": None,
    "reset_car": ResetCommand.RESET_CAR,
    "pits": ResetCommand.TELEPORT_PITS,
    "restart": ResetCommand.RESTART_SESSION,
    "spline": ResetCommand.TELEPORT_SPLINE,
    "load_state": ResetCommand.LOAD_STATE,
    "step_back": ResetCommand.STEP_BACK,
}

_RESULT_HELP = {
    ResetResult.FAILED: "the CSP call returned false (wrong session type or not offline?)",
    ResetResult.NOT_ALLOWED_IN_SESSION: "car reset is not allowed in this session; use an "
                                        "offline single-car Practice session",
    ResetResult.PHYSICS_NOT_ALLOWED: "physics.allowed() is false: the track does not grant "
                                     "Lua apps physics access, so spline teleport is "
                                     "unavailable; use reset_mode='load_state' or 'reset_car'",
    ResetResult.EMPTY_SLOT: "no saved car state in that slot; call save_state() first",
    ResetResult.UNKNOWN_COMMAND: "Lua app does not know this command (version mismatch?)",
    ResetResult.LUA_ERROR: "Lua error inside the app (see CSP Lua debug / log)",
}


class ResetError(RuntimeError):
    pass


class ACRawEnv:
    """Raw AC environment on top of :class:`ACSession`.

    Parameters
    ----------
    reset_mode:
        one of :data:`RESET_MODES`.  Everything except ``"none"`` needs the
        CSP backend (the Lua app executes the reset).
    reset_spline, reset_speed, reset_lateral:
        target for ``"spline"`` (normalised position, m/s along the track,
        -1..1 across the track).
    reset_slot:
        saved-state slot for ``"load_state"`` (see :meth:`save_state`).
    settle_s:
        after a stationary reset, hold the brake this long so suspension and
        tyres settle before the first observation (skipped when
        ``reset_speed > 0``).
    """

    def __init__(self, session: ACSession, reset_mode: str = "reset_car",
                 reset_spline: float = 0.0, reset_speed: float = 0.0,
                 reset_lateral: float = 0.0, reset_slot: int = 0, settle_s: float = 1.0,
                 reset_timeout_s: float = 5.0, monitor: EpisodeMonitor | None = None):
        if reset_mode not in RESET_MODES:
            raise ValueError(f"reset_mode must be one of {sorted(RESET_MODES)}")
        if RESET_MODES[reset_mode] is not None and not session.backend.supports_reset:
            raise ValueError(f"reset_mode {reset_mode!r} needs the 'csp' backend (MV Bridge Lua "
                             f"app); backend is {session.backend.name!r}. Use reset_mode='none'.")
        self.session = session
        self.reset_mode = reset_mode
        self.reset_spline = reset_spline
        self.reset_speed = reset_speed
        self.reset_lateral = reset_lateral
        self.reset_slot = reset_slot
        self.settle_s = settle_s
        self.reset_timeout_s = reset_timeout_s
        self.monitor = monitor or EpisodeMonitor()
        self.flags = EpisodeFlags()
        self.last: Telemetry | None = None

    # --------------------------------------------------------------- factory
    @classmethod
    def make(cls, backend: str | ControlBackend = "csp", rate_hz: float = 50.0,
             wait_for_game_s: float = 0.0, **kw) -> "ACRawEnv":
        """Build reader + backend + session.  ``kw`` goes to ``ACRawEnv``."""
        link = None
        if isinstance(backend, str):
            if backend.lower() in ("csp", "lua", "csp-lua"):
                link = PluginLink()
                backend = CSPLuaBackend(link)
            else:
                backend = make_backend(backend)
        elif isinstance(backend, CSPLuaBackend):
            link = backend.link
        reader = (ACReader.wait_for_game(wait_for_game_s, plugin=link) if wait_for_game_s > 0
                  else ACReader(plugin=link))
        return cls(ACSession(reader, backend, rate_hz=rate_hz), **kw)

    @property
    def link(self) -> PluginLink | None:
        b = self.session.backend
        return b.link if isinstance(b, CSPLuaBackend) else None

    # ------------------------------------------------------------- commands
    def _command(self, cmd: ResetCommand, **kw) -> ResetResult:
        link = self.link
        if link is None:
            raise ResetError("episode commands need the CSP backend")
        if not link.alive(1.0):
            raise ResetError("MV Bridge Lua app is not running (no state updates). Install it "
                             "(mv/ac/plugin/install_plugin.ps1) and enable it in-game.")
        seq = link.request(cmd, **kw)
        deadline = time.perf_counter() + self.reset_timeout_s
        while True:  # keep the watchdog quiet while the game resets
            self.session.touch()
            try:
                return link.wait_result(seq, timeout_s=0.1)
            except TimeoutError:
                if time.perf_counter() > deadline:
                    raise ResetError(f"{cmd.name} not completed within "
                                     f"{self.reset_timeout_s:.1f}s") from None

    def save_state(self, slot: int = 0) -> ResetResult:
        """Snapshot the car (position, velocities, engine...) into ``slot`` (0..30)."""
        return self._command(ResetCommand.SAVE_STATE, slot=slot)

    # ------------------------------------------------------------ reset/step
    def reset(self) -> Telemetry:
        self.session.failsafe()
        cmd = RESET_MODES[self.reset_mode]
        if cmd is not None:
            res = self._command(cmd, slot=self.reset_slot, spline=self.reset_spline,
                                speed=self.reset_speed, lateral=self.reset_lateral)
            if res != ResetResult.OK:
                raise ResetError(f"reset '{self.reset_mode}' failed: {res.name}: "
                                 f"{_RESULT_HELP.get(res, '')}")
        self.session.resync()
        moving_start = cmd in (ResetCommand.TELEPORT_SPLINE, ResetCommand.LOAD_STATE) \
            and (self.reset_speed > 0 or cmd == ResetCommand.LOAD_STATE)
        if not moving_start:
            t_end = time.perf_counter() + self.settle_s
            tel = self.session.observe(timeout_s=2.0)
            # brake until settled; for "none" also until (almost) stationary
            while time.perf_counter() < t_end or (cmd is None and tel.speed > 0.3):
                tel = self.session.step(0.0, 0.0, 1.0)
                if time.perf_counter() > t_end + 30.0:
                    break
        self.session.resync()
        tel = self.session.observe(timeout_s=2.0)
        self.monitor.reset(tel)
        self.flags = EpisodeFlags()
        self.last = tel
        return tel

    def step(self, steer: float, throttle: float, brake: float,
             gear_up: bool = False, gear_down: bool = False) -> Telemetry:
        tel = self.session.step(steer, throttle, brake, gear_up, gear_down)
        self.flags = self.monitor.update(tel)
        self.last = tel
        return tel

    def close(self) -> None:
        self.session.close()
        link = self.link
        if link is not None:
            link.close()
        self.session.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
