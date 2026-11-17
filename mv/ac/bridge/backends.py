"""Control output backends.

Every backend takes inputs in the **project convention**:

* ``steer``    in [-1, 1], **+1 = full LEFT lock** (as ``mv.vehicle.params``)
* ``throttle`` in [0, 1]
* ``brake``    in [0, 1]
* ``gear_up`` / ``gear_down``: one-shot requests (use with manual gearbox only)

and converts to the device.  AC itself uses +1 = right, so every backend
negates steering exactly once, here, and nowhere else.

Backends, in order of preference (see docs/07 §4):

``CSPLuaBackend``  (``"csp"``)
    No driver.  Writes to the MV Bridge CSP Lua app via shared memory; the app
    calls ``ac.overrideCarControls(0)``.  The override *replaces* steering but
    combines pedals as ``max(user, override)`` (CSP SDK doc), so the user's
    own pedals/keys must be idle.  Bypasses AC's controller filters entirely
    (to be confirmed in-game: see docs "not yet verified").
``VJoyBackend``  (``"vjoy"``)
    vJoy virtual DirectInput joystick via ``pyvjoy``; AC sees an ordinary
    wheel.  1:1 only if AC's steering/pedal settings are neutral (docs §5).
``VGamepadBackend``  (``"vgamepad"``)
    Virtual Xbox 360 pad via ViGEmBus + ``vgamepad``; AC's *gamepad* input
    mode applies its own gamma/filter/speed-sensitivity, which must be zeroed.
    Last resort.

Third-party modules are imported lazily inside the constructors so that the
package imports (and the tests run) without them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .plugin_link import PluginLink

__all__ = [
    "ControlBackend", "NullBackend", "CSPLuaBackend", "VJoyBackend", "VGamepadBackend",
    "make_backend", "clip_controls",
]


def _clip(x: float, lo: float, hi: float) -> float:
    x = float(x)
    if x != x:  # NaN -> safest value in range
        return 0.0 if lo <= 0.0 <= hi else lo
    return lo if x < lo else hi if x > hi else x


def clip_controls(steer: float, throttle: float, brake: float) -> tuple[float, float, float]:
    """Clamp to legal ranges; NaN becomes 0 (straight / no pedal)."""
    return _clip(steer, -1.0, 1.0), _clip(throttle, 0.0, 1.0), _clip(brake, 0.0, 1.0)


class ControlBackend(ABC):
    """Interface every control output implements."""

    name = "abstract"
    #: True if this backend can execute episode-control commands (reset/teleport).
    supports_reset = False

    @abstractmethod
    def send(self, steer: float, throttle: float, brake: float,
             gear_up: bool = False, gear_down: bool = False) -> None:
        """Apply one control frame (project convention, see module doc)."""

    def failsafe(self) -> None:
        """Zero throttle, full brake, wheel straight."""
        self.send(0.0, 0.0, 1.0)

    def release(self) -> None:
        """Neutral output: no pedals, wheel centred (and hand back control if possible)."""
        self.send(0.0, 0.0, 0.0)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class NullBackend(ControlBackend):
    """Records the last command; sends nothing (dry runs, tests)."""

    name = "null"

    def __init__(self):
        self.last: tuple[float, float, float, bool, bool] | None = None
        self.count = 0

    def send(self, steer, throttle, brake, gear_up=False, gear_down=False):
        s, t, b = clip_controls(steer, throttle, brake)
        self.last = (s, t, b, bool(gear_up), bool(gear_down))
        self.count += 1


class CSPLuaBackend(ControlBackend):
    """Driverless control through the MV Bridge CSP Lua app (preferred)."""

    name = "csp"
    supports_reset = True

    def __init__(self, link: PluginLink | None = None, stale_timeout_s: float = 0.5):
        self.link = link or PluginLink(stale_timeout_s=stale_timeout_s)
        self._owns_link = link is None
        self.last_seq = 0

    def send(self, steer, throttle, brake, gear_up=False, gear_down=False):
        s, t, b = clip_controls(steer, throttle, brake)
        self.last_seq = self.link.write_controls(-s, t, b, gear_up=gear_up, gear_down=gear_down)

    def release(self):
        self.link.release()

    def close(self):
        try:
            self.link.release()
        finally:
            if self._owns_link:
                self.link.close(release=False)


class VJoyBackend(ControlBackend):
    """vJoy device via ``pyvjoy``.

    Axis mapping (bind these in Content Manager, docs §5):

    ========  ===========  =======================================
    control   vJoy axis    raw value
    ========  ===========  =======================================
    steer     X            0x0001 = full left ... 0x8000 = full right
    throttle  Y            0x0001 = released  ... 0x8000 = floored
    brake     Z            0x0001 = released  ... 0x8000 = floored
    gear up   button 1     pressed for one control period
    gear down button 2     pressed for one control period
    ========  ===========  =======================================
    """

    name = "vjoy"
    AXIS_MIN, AXIS_MAX = 0x1, 0x8000

    def __init__(self, device_id: int = 1, gear_up_button: int = 1, gear_down_button: int = 2):
        try:
            import pyvjoy  # noqa: PLC0415 - optional dependency
        except ImportError as e:  # pragma: no cover - depends on user install
            raise ImportError("VJoyBackend needs 'pyvjoy' and the vJoy driver; see "
                              "docs/07-assetto-corsa-bridge.md §5") from e
        self._pyvjoy = pyvjoy
        self.dev = pyvjoy.VJoyDevice(device_id)
        self.gear_up_button = gear_up_button
        self.gear_down_button = gear_down_button

    @classmethod
    def _axis(cls, u: float) -> int:
        """Map u in [0, 1] to the vJoy axis range."""
        return int(round(cls.AXIS_MIN + u * (cls.AXIS_MAX - cls.AXIS_MIN)))

    @classmethod
    def encode(cls, steer: float, throttle: float, brake: float) -> tuple[int, int, int]:
        """Project-convention controls -> raw (X, Y, Z).  +steer (left) -> low X."""
        s, t, b = clip_controls(steer, throttle, brake)
        return cls._axis((1.0 - s) / 2.0), cls._axis(t), cls._axis(b)

    def send(self, steer, throttle, brake, gear_up=False, gear_down=False):
        x, y, z = self.encode(steer, throttle, brake)
        buttons = 0
        if gear_up:
            buttons |= 1 << (self.gear_up_button - 1)
        if gear_down:
            buttons |= 1 << (self.gear_down_button - 1)
        d = getattr(self.dev, "data", None)
        if d is not None and hasattr(self.dev, "update"):
            d.wAxisX, d.wAxisY, d.wAxisZ, d.lButtons = x, y, z, buttons
            self.dev.update()  # one atomic UpdateVJD call for all axes
        else:  # pragma: no cover - older pyvjoy
            pv = self._pyvjoy
            self.dev.set_axis(pv.HID_USAGE_X, x)
            self.dev.set_axis(pv.HID_USAGE_Y, y)
            self.dev.set_axis(pv.HID_USAGE_Z, z)
            self.dev.set_button(self.gear_up_button, int(gear_up))
            self.dev.set_button(self.gear_down_button, int(gear_down))

    def sweep_axis(self, axis: str, seconds: float = 4.0) -> None:
        """Move one axis min->max->min slowly, to help binding it in CM."""
        import time  # noqa: PLC0415
        t0 = time.perf_counter()
        while (el := time.perf_counter() - t0) < seconds:
            u = 1.0 - abs(2.0 * el / seconds - 1.0)
            v = self._axis(u)
            mid = self._axis(0.5)
            d = self.dev.data
            d.wAxisX, d.wAxisY, d.wAxisZ = mid, self.AXIS_MIN, self.AXIS_MIN
            setattr(d, {"X": "wAxisX", "Y": "wAxisY", "Z": "wAxisZ"}[axis.upper()], v)
            self.dev.update()
            time.sleep(0.01)
        self.release()

    def close(self):
        try:
            self.release()
        except Exception:
            pass


class VGamepadBackend(ControlBackend):
    """Virtual Xbox 360 controller via ``vgamepad`` (ViGEmBus).

    Mapping: left stick X = steer (right positive, as XInput), right trigger =
    throttle, left trigger = brake, A = gear up, X = gear down (matches the
    XBOXBUTTON entries in the user's current controls.ini).
    """

    name = "vgamepad"

    def __init__(self):
        try:
            import vgamepad as vg  # noqa: PLC0415 - optional dependency
        except ImportError as e:  # pragma: no cover
            raise ImportError("VGamepadBackend needs 'vgamepad' (and ViGEmBus); see "
                              "docs/07-assetto-corsa-bridge.md §5") from e
        self._vg = vg
        self.pad = vg.VX360Gamepad()

    def send(self, steer, throttle, brake, gear_up=False, gear_down=False):
        s, t, b = clip_controls(steer, throttle, brake)
        vg = self._vg
        self.pad.left_joystick_float(x_value_float=-s, y_value_float=0.0)
        self.pad.right_trigger_float(value_float=t)
        self.pad.left_trigger_float(value_float=b)
        for flag, btn in ((gear_up, vg.XUSB_BUTTON.XUSB_GAMEPAD_A),
                          (gear_down, vg.XUSB_BUTTON.XUSB_GAMEPAD_X)):
            if flag:
                self.pad.press_button(button=btn)
            else:
                self.pad.release_button(button=btn)
        self.pad.update()

    def close(self):
        try:
            self.pad.reset()
            self.pad.update()
        except Exception:
            pass


def make_backend(name: str, **kwargs) -> ControlBackend:
    """Factory: ``"csp"`` (default/preferred), ``"vjoy"``, ``"vgamepad"``, ``"null"``."""
    name = name.lower()
    if name in ("csp", "lua", "csp-lua"):
        return CSPLuaBackend(**kwargs)
    if name == "vjoy":
        return VJoyBackend(**kwargs)
    if name in ("vgamepad", "vigem", "x360"):
        return VGamepadBackend(**kwargs)
    if name in ("null", "none", "dry"):
        return NullBackend()
    raise ValueError(f"unknown control backend {name!r}")
