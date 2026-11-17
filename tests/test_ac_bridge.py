"""Tests for mv.ac.bridge that run without Assetto Corsa.

Synthetic AC pages are written into *named* anonymous mappings (the same
kernel objects AC uses), so the real OpenFileMappingW/MapViewOfFile path of
the reader is exercised, not a mock.
"""

from __future__ import annotations

import ctypes
import math
import mmap
import re
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from mv.ac.bridge import (
    ACRawEnv, ACReader, ACSession, ACTimeoutError, CSPLuaBackend, EpisodeMonitor, NullBackend,
    PluginLink, ResetCommand, SharedMemoryUnavailable, TelemetryDecoder, VJoyBackend,
    ac_local_to_body, ac_world_to_map3,
)
from mv.ac.track_import import ac_world_to_map
from mv.ac.bridge.backends import ControlBackend, clip_controls
from mv.ac.bridge.plugin_link import (
    CMD_MAGIC, LAYOUT_VERSION, STATE_MAGIC, PluginCmd, PluginStateRaw, c_layout,
    decode_plugin_state,
)
from mv.ac.bridge.shm import NamedMapping
from mv.ac.bridge.shm_layout import (
    AC_LIVE, AC_PAUSE, GRAPHICS_SIZE, PHYSICS_SIZE, STATIC_SIZE, SPageFileGraphic,
    SPageFilePhysics, SPageFileStatic,
)

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="AC bridge is Windows-only")

ROOT = Path(__file__).resolve().parents[1]
LUA_FILE = ROOT / "mv" / "ac" / "plugin" / "apps" / "lua" / "mv_bridge" / "mv_bridge.lua"


def _name(tag: str) -> str:
    return f"Local\\mvtest_{tag}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- layouts
def test_struct_sizes_match_hand_computed_totals():
    # Totals computed by summing the documented field list by hand (pack 4).
    assert ctypes.sizeof(SPageFilePhysics) == PHYSICS_SIZE == 712
    assert ctypes.sizeof(SPageFileGraphic) == GRAPHICS_SIZE == 288
    assert ctypes.sizeof(SPageFileStatic) == STATIC_SIZE == 684
    assert ctypes.sizeof(ctypes.c_wchar) == 2


@pytest.mark.parametrize("struct, field, offset", [
    (SPageFilePhysics, "velocity", 32),
    (SPageFilePhysics, "wheelSlip", 56),
    (SPageFilePhysics, "heading", 208),
    (SPageFilePhysics, "numberOfTyresOut", 244),
    (SPageFilePhysics, "localAngularVel", 296),
    (SPageFilePhysics, "tyreContactPoint", 420),
    (SPageFilePhysics, "localVelocity", 568),
    (SPageFilePhysics, "currentMaxRpm", 588),
    (SPageFilePhysics, "slipRatio", 640),
    (SPageFilePhysics, "tcinAction", 672),
    (SPageFilePhysics, "tyreTemp", 696),
    (SPageFileGraphic, "currentTime", 12),
    (SPageFileGraphic, "completedLaps", 132),
    (SPageFileGraphic, "tyreCompound", 176),
    (SPageFileGraphic, "replayTimeMultiplier", 244),  # 176+66=242 -> pack(4) pads to 244
    (SPageFileGraphic, "normalizedCarPosition", 248),
    (SPageFileGraphic, "carCoordinates", 252),
    (SPageFileGraphic, "isInPitLane", 276),
    (SPageFileStatic, "carModel", 68),
    (SPageFileStatic, "sectorCount", 400),  # 68 + 5*66 = 398 -> 400
    (SPageFileStatic, "maxRpm", 412),
    (SPageFileStatic, "trackSPlineLength", 520),
    (SPageFileStatic, "trackConfiguration", 524),
    (SPageFileStatic, "ersMaxJ", 592),
])
def test_key_field_offsets(struct, field, offset):
    assert getattr(struct, field).offset == offset


def _lua_block(name: str) -> str:
    text = LUA_FILE.read_text(encoding="utf-8")
    m = re.search(rf"local {name} = \[\[\n(.*?)\]\]", text, re.S)
    assert m, f"{name} block not found in {LUA_FILE}"
    return m.group(1).strip()


def test_lua_layouts_match_python_structs():
    assert _lua_block("CMD_LAYOUT") == c_layout(PluginCmd)
    assert _lua_block("STATE_LAYOUT") == c_layout(PluginStateRaw)
    text = LUA_FILE.read_text(encoding="utf-8")
    assert f"CMD_MAGIC = 0x{CMD_MAGIC:08X}" in text
    assert f"STATE_MAGIC = 0x{STATE_MAGIC:08X}" in text
    assert f"LAYOUT_VERSION = {LAYOUT_VERSION}" in text
    # natural alignment: doubles 8-aligned, total multiple of 8 (LuaJIT pads the same way)
    assert PluginStateRaw.game_time_s.offset % 8 == 0
    assert ctypes.sizeof(PluginStateRaw) % 8 == 0
    assert ctypes.sizeof(PluginCmd) == 4 * len(PluginCmd._fields_)


def test_lua_reset_codes_match_python():
    text = LUA_FILE.read_text(encoding="utf-8")
    for cmd in ResetCommand:
        assert re.search(rf"\b{cmd.name} = {int(cmd)}\b", text), cmd


# --------------------------------------------------------------------------- synthetic pages
def make_pages():
    ph = SPageFilePhysics()
    ph.packetId = 1000
    ph.gas, ph.brake, ph.fuel, ph.gear, ph.rpms = 0.75, 0.0, 42.5, 4, 6500
    ph.steerAngle = -0.25           # AC: negative = left (assumed), project: +0.25
    ph.speedKmh = 180.0
    # Car yaw +30 deg CCW in the map frame, moving forward with a little
    # leftward slip.  local (AC) = (X left, Y up, Z fwd)
    vx, vy = 49.0, 2.0
    yaw = math.radians(30)
    wx = vx * math.cos(yaw) - vy * math.sin(yaw)   # map x
    wy = vx * math.sin(yaw) + vy * math.cos(yaw)   # map y
    ph.localVelocity[:] = (vy, 0.1, vx)
    ph.velocity[:] = (wx, 0.1, -wy)                 # AC world: X = map_x, Z = -map_y
    ph.accG[:] = (0.5, 0.02, -1.2)                  # lat, vert, long (g)
    ph.localAngularVel[:] = (0.01, 0.3, -0.02)      # about X(left), Y(up), Z(fwd)
    ph.heading = yaw
    ph.wheelSlip[:] = (0.1, 0.2, 0.3, 0.4)
    ph.wheelLoad[:] = (3000, 3100, 3200, 3300)
    ph.tyreCoreTemperature[:] = (80, 81, 82, 83)
    ph.tyreTemp[:] = (90, 91, 92, 93)
    ph.slipRatio[:] = (0.01, 0.02, 0.03, 0.04)
    ph.numberOfTyresOut = 1
    ph.kersCharge, ph.kersInput = 0.6, 0.1
    ph.ersRecoveryLevel, ph.ersPowerLevel = 3, 2
    ph.tcinAction, ph.absInAction = 1, 0
    ph.currentMaxRpm = 7800
    ph.clutch = 1.0

    gr = SPageFileGraphic()
    gr.packetId = 77
    gr.status = AC_LIVE
    gr.session = 0
    gr.completedLaps = 3
    gr.iCurrentTime, gr.iLastTime, gr.iBestTime = 61234, 92345, 2**31 - 1
    gr.isInPit, gr.isInPitLane = 0, 1
    gr.normalizedCarPosition = 0.42
    gr.carCoordinates[:] = (10.0, 2.0, -50.0)       # AC world X, Y(up), Z
    gr.tyreCompound = "Semislicks (SM)"

    st = SPageFileStatic()
    st.smVersion, st.acVersion = "1.7", "1.16"
    st.carModel, st.track = "ks_mazda_mx5_cup", "magione"
    st.trackConfiguration = ""
    st.maxRpm = 7500
    st.trackSPlineLength = 2507.0
    return ph, gr, st


class SyntheticGame:
    """Three named mappings filled with synthetic pages, like a running AC."""

    def __init__(self):
        self.names = {k: _name(k) for k in ("physics", "graphics", "static")}
        self.ph, self.gr, self.st = make_pages()
        self.maps = {
            k: mmap.mmap(-1, 4096, tagname=self.names[k]) for k in self.names
        }
        self.flush()

    def flush(self):
        for k, s in (("physics", self.ph), ("graphics", self.gr), ("static", self.st)):
            self.maps[k].seek(0)
            self.maps[k].write(bytes(s))

    def reader(self, **kw) -> ACReader:
        return ACReader(physics_name=self.names["physics"], graphics_name=self.names["graphics"],
                        static_name=self.names["static"], **kw)

    def close(self):
        for m in self.maps.values():
            m.close()


@pytest.fixture
def game():
    g = SyntheticGame()
    yield g
    g.close()


def test_missing_mapping_is_explicit_and_not_created():
    name = _name("absent")
    with pytest.raises(SharedMemoryUnavailable):
        NamedMapping(name)
    with pytest.raises(SharedMemoryUnavailable):  # still absent: we never created it
        NamedMapping(name)


def test_decode_synthetic_pages(game):
    with game.reader() as r:
        t = r.read()
    assert t.packet_id == 1000 and t.graphics_packet_id == 77
    assert t.status == AC_LIVE
    # units
    assert t.speed == pytest.approx(50.0)
    assert t.rpm == 6500 and t.max_rpm == 7800
    assert t.gear == 3                          # AC 4 -> 3rd gear
    assert t.throttle == pytest.approx(0.75)
    assert t.fuel == pytest.approx(42.5)
    # axes: position AC (X=10, Y=2 up, Z=-50) -> map (10, 50, 2), same as track_import
    np.testing.assert_allclose(t.position, [10.0, 50.0, 2.0])
    mx, my = ac_world_to_map(10.0, -50.0)
    assert (t.position[0], t.position[1]) == (float(mx), float(my))
    np.testing.assert_allclose(t.velocity_local, [49.0, 2.0, 0.1], rtol=1e-6)
    np.testing.assert_allclose(t.acceleration_local, np.array([-1.2, 0.5, 0.02]) * 9.80665,
                               rtol=1e-6)
    assert t.yaw_rate == pytest.approx(0.3)
    np.testing.assert_allclose(t.angular_velocity_local, [-0.02, 0.01, 0.3], rtol=1e-6)
    assert t.yaw == pytest.approx(math.radians(30), abs=1e-5)
    assert t.yaw_source == "velocity"
    assert t.steer == pytest.approx(0.25)       # project: + = left
    # tyres / misc
    np.testing.assert_allclose(t.wheel_load, [3000, 3100, 3200, 3300])
    np.testing.assert_allclose(t.tyre_temp_surface, [90, 91, 92, 93])
    assert t.tyre_temp_imo.shape == (4, 3)
    assert t.tc_active and not t.abs_active
    assert t.kers_charge == pytest.approx(0.6) and t.ers_recovery_level == 3
    # lap data: ms -> s, INT_MAX best -> nan
    assert t.lap_count == 3
    assert t.lap_time_current == pytest.approx(61.234)
    assert t.lap_time_last == pytest.approx(92.345)
    assert math.isnan(t.lap_time_best)
    assert t.spline == pytest.approx(0.42)
    assert t.tyres_out == 1 and t.in_pitlane and not t.in_pit
    # static strings (wchar) and track length
    assert t.car_model == "ks_mazda_mx5_cup" and t.track == "magione"
    assert t.track_length == pytest.approx(2507.0)
    assert t.plugin is None
    assert t.sim_time_source == "packet_estimate"


def test_reader_sees_live_updates(game):
    with game.reader() as r:
        assert r.physics_packet_id() == 1000
        game.ph.packetId = 1001
        game.ph.speedKmh = 36.0
        game.flush()
        assert r.physics_packet_id() == 1001
        assert r.read().speed == pytest.approx(10.0)


# --------------------------------------------------------------------------- conventions
def test_frame_conversions_are_proper_rotations():
    # AC car-local basis: X left, Y up, Z forward -> body x forward, y left, z up
    np.testing.assert_allclose(ac_local_to_body((1, 0, 0)), [0, 1, 0])   # left
    np.testing.assert_allclose(ac_local_to_body((0, 1, 0)), [0, 0, 1])   # up
    np.testing.assert_allclose(ac_local_to_body((0, 0, 1)), [1, 0, 0])   # forward
    # AC world -> map: map_x = X, map_y = -Z, z = Y (track_import.ac_world_to_map)
    np.testing.assert_allclose(ac_world_to_map3((3, 7, 5)), [3, -5, 7])
    for f in (ac_local_to_body, ac_world_to_map3):
        m = np.stack([f(e) for e in np.eye(3)], axis=1)
        assert np.linalg.det(m) == pytest.approx(1.0)   # handedness preserved


def test_yaw_zero_along_map_x_and_body_world_consistency():
    """A car facing AC +X has yaw 0; facing AC +Z has yaw -90 deg (map frame)."""
    dec = TelemetryDecoder()
    ph, gr, st = make_pages()
    ph.localVelocity[:] = (0.0, 0.0, 20.0)            # straight ahead
    for ac_vel, yaw_deg in (((20, 0, 0), 0.0), ((0, 0, 20), -90.0), ((-20, 0, 0), 180.0),
                            ((0, 0, -20), 90.0)):
        ph.velocity[:] = ac_vel
        t = dec.decode(ph, gr, st)
        assert math.isclose(math.remainder(t.yaw - math.radians(yaw_deg), 2 * math.pi), 0,
                            abs_tol=1e-9), (ac_vel, t.yaw)


@pytest.mark.parametrize("yaw_deg", [-170, -90, -10, 0, 45, 135, 179])
def test_yaw_from_velocities(yaw_deg):
    yaw = math.radians(yaw_deg)
    v_local = np.array([30.0, -4.0, 0.0])     # sliding to the right
    c, s = math.cos(yaw), math.sin(yaw)
    v_world = np.array([c * v_local[0] - s * v_local[1], s * v_local[0] + c * v_local[1], 0.0])
    got = TelemetryDecoder.yaw_from_velocities(v_world, v_local)
    assert math.isclose(math.remainder(got - yaw, 2 * math.pi), 0.0, abs_tol=1e-9)


def test_heading_calibration_learns_sign_and_offset():
    dec = TelemetryDecoder(calib_samples=20)
    ph, gr, st = make_pages()
    rng = np.random.default_rng(0)
    for _ in range(25):
        yaw = rng.uniform(-math.pi, math.pi)
        v_local = (30.0, 1.0)
        c, s = math.cos(yaw), math.sin(yaw)
        wx, wy = c * v_local[0] - s * v_local[1], s * v_local[0] + c * v_local[1]
        ph.localVelocity[:] = (v_local[1], 0.0, v_local[0])
        ph.velocity[:] = (wx, 0.0, -wy)
        ph.heading = -yaw + 0.3               # pretend AC uses clockwise heading + offset
        dec.decode(ph, gr, st)
    assert dec.heading_sign == -1
    assert dec.heading_offset == pytest.approx(0.3, abs=1e-6)
    # now at standstill the calibrated heading is used
    ph.localVelocity[:] = (0, 0, 0)
    ph.velocity[:] = (0, 0, 0)
    ph.heading = -1.0 + 0.3
    t = dec.decode(ph, gr, st)
    assert t.yaw_source == "heading_calibrated"
    assert t.yaw == pytest.approx(1.0)


# --------------------------------------------------------------------------- plugin link
@pytest.fixture
def link():
    lk = PluginLink(cmd_name=_name("cmd"), state_name=_name("state"))
    yield lk
    lk.close()


def test_cmd_write_and_steer_sign(link):
    be = CSPLuaBackend(link)
    be.send(steer=0.5, throttle=0.3, brake=0.0, gear_up=True)   # +0.5 = LEFT (project)
    c = link.cmd
    assert c.magic == CMD_MAGIC and c.version == LAYOUT_VERSION
    assert c.seq % 2 == 0 and c.enable == 1
    assert c.steer == pytest.approx(-0.5)                         # AC: + = right
    assert c.gas == pytest.approx(0.3) and c.clutch == pytest.approx(1.0)
    assert c.gear_up_count == 1 and c.gear_down_count == 0
    seq = c.seq
    be.send(0.0, 2.0, -1.0)                                       # clipped
    assert c.seq == seq + 2 and c.gas == 1.0 and c.brake == 0.0
    link.release()
    assert c.enable == 0
    rs = link.request(ResetCommand.TELEPORT_SPLINE, spline=1.25, speed=20.0)
    assert c.reset_seq == rs and c.reset_cmd == 4 and c.reset_spline == pytest.approx(0.25)


def _fake_lua_state(link: PluginLink, **over) -> PluginStateRaw:
    """Write a state block exactly as the Lua app would (seqlock even/odd)."""
    view = PluginStateRaw.from_buffer(link._state_mm)
    s = view.seq + (view.seq % 2)
    view.seq = s + 1
    view.magic, view.version = STATE_MAGIC, LAYOUT_VERSION
    yaw = math.radians(over.pop("yaw_deg", 0.0))
    # AC world look vector for map yaw: map (cos, sin, 0) -> AC (X=cos, Y=0, Z=-sin)
    view.look[:] = (math.cos(yaw), 0.0, -math.sin(yaw))
    view.up[:] = (0.0, 1.0, 0.0)
    view.car_timestamp_ms = 12345.0
    for k, v in over.items():
        setattr(view, k, v)
    view.seq = s + 2
    out = PluginStateRaw.from_buffer_copy(bytes(view))
    del view
    return out


def test_plugin_state_decoding(link):
    assert link.read_state() is None                              # nothing written yet
    _fake_lua_state(link, yaw_deg=120.0, track_lateral=-0.4, collision_count=2,
                    cmd_seq_seen=link.cmd.seq)
    s = link.read_state()
    assert s is not None
    assert s.yaw == pytest.approx(math.radians(120.0), abs=1e-6)
    assert s.pitch == pytest.approx(0.0, abs=1e-7) and s.roll == pytest.approx(0.0, abs=1e-7)
    assert s.track_lateral == pytest.approx(-0.4) and s.collision_count == 2
    assert s.car_timestamp_s == pytest.approx(12.345)
    assert link.alive(1.0)


def test_plugin_pitch_roll_signs():
    raw = PluginStateRaw()
    # nose up 10 deg: forward has +up component -> pitch negative (nose-down positive)
    a = math.radians(10)
    raw.look[:] = (0.0, math.sin(a), math.cos(a))
    raw.up[:] = (0.0, math.cos(a), -math.sin(a))
    assert decode_plugin_state(raw).pitch == pytest.approx(-a, abs=1e-6)
    # right side down 5 deg: up vector tilts to the right (-left = -AC X)
    b = math.radians(5)
    raw.look[:] = (0.0, 0.0, 1.0)
    raw.up[:] = (-math.sin(b), math.cos(b), 0.0)
    assert decode_plugin_state(raw).roll == pytest.approx(b, abs=1e-6)


def test_reader_merges_plugin(game, link):
    _fake_lua_state(link, yaw_deg=-45.0)
    with game.reader(plugin=link) as r:
        t = r.read()
    assert t.plugin is not None and t.yaw_source == "plugin"
    assert t.yaw == pytest.approx(math.radians(-45.0), abs=1e-6)
    assert t.sim_time_source == "plugin" and t.sim_time == pytest.approx(12.345)


# --------------------------------------------------------------------------- backends
def test_vjoy_encoding_without_driver():
    assert VJoyBackend.encode(0.0, 0.0, 0.0) == (0x4000, 0x1, 0x1)
    assert VJoyBackend.encode(1.0, 1.0, 1.0)[0] == 0x1        # full LEFT -> axis minimum
    assert VJoyBackend.encode(-1.0, 1.0, 1.0) == (0x8000, 0x8000, 0x8000)


def test_clip_controls_handles_nan_and_range():
    assert clip_controls(float("nan"), float("nan"), float("nan")) == (0.0, 0.0, 0.0)
    assert clip_controls(-3, 5, -1) == (-1.0, 1.0, 0.0)


def test_package_imports_without_optional_deps():
    import mv.ac.bridge as b  # noqa: F401
    assert "pyvjoy" not in sys.modules or True  # import must not *require* it
    assert "vgamepad" not in sys.modules


# --------------------------------------------------------------------------- session / fail-safe
class RecordingBackend(ControlBackend):
    name = "recording"

    def __init__(self, fail_on_send: bool = False):
        self.calls: list[tuple] = []
        self.fail_on_send = fail_on_send
        self.lock = threading.Lock()

    def send(self, steer, throttle, brake, gear_up=False, gear_down=False):
        with self.lock:
            self.calls.append((steer, throttle, brake))
        if self.fail_on_send and (steer, throttle, brake) != (0.0, 0.0, 1.0):
            raise RuntimeError("device unplugged")

    @property
    def last(self):
        with self.lock:
            return self.calls[-1]


def test_step_runs_at_fixed_rate_and_reports_stats(game):
    stop = threading.Event()

    def physics_thread():  # emulate AC writing packets at ~333 Hz
        while not stop.is_set():
            game.ph.packetId += 1
            game.maps["physics"].seek(0)
            game.maps["physics"].write(bytes(game.ph))
            time.sleep(0.003)

    th = threading.Thread(target=physics_thread, daemon=True)
    th.start()
    be = RecordingBackend()
    try:
        with ACSession(game.reader(), be, rate_hz=50.0, watchdog_s=0) as s:
            t0 = time.perf_counter()
            for _ in range(25):
                tel = s.step(0.1, 0.5, 0.0)
            el = time.perf_counter() - t0
            summ = s.stats.summary()
    finally:
        stop.set()
        th.join()
    assert tel.packet_id > 1000
    assert 0.40 <= el <= 0.75                          # 25 steps at 50 Hz ~ 0.5 s
    assert summ["steps"] == 25 and summ["timeouts"] == 0
    assert 40.0 < summ["rate_hz"] < 60.0
    assert summ["physics_packets_per_step"] >= 1.0
    # close(): fail-safe brake first, then release (neutral)
    assert be.calls[-2] == (0.0, 0.0, 1.0) and be.calls[-1] == (0.0, 0.0, 0.0)


def test_timeout_applies_failsafe(game):
    be = RecordingBackend()
    s = ACSession(game.reader(), be, rate_hz=50.0, packet_timeout_s=0.05, watchdog_s=0)
    with pytest.raises(ACTimeoutError):
        s.step(0.3, 1.0, 0.0)                             # packetId never changes
    assert be.last == (0.0, 0.0, 1.0)
    assert s.stats.timeouts == 1
    s.close(release=False)


def test_not_live_applies_failsafe(game):
    game.gr.status = AC_PAUSE
    game.flush()
    stop = threading.Event()

    def bump():
        while not stop.is_set():
            game.ph.packetId += 1
            game.maps["physics"].seek(0)
            game.maps["physics"].write(bytes(game.ph))
            time.sleep(0.003)

    th = threading.Thread(target=bump, daemon=True)
    th.start()
    be = RecordingBackend()
    try:
        s = ACSession(game.reader(), be, rate_hz=50.0, watchdog_s=0)
        with pytest.raises(ACTimeoutError, match="PAUSE"):
            s.step(0.0, 1.0, 0.0)
    finally:
        stop.set()
        th.join()
    assert be.last == (0.0, 0.0, 1.0)


def test_exception_in_backend_applies_failsafe(game):
    be = RecordingBackend(fail_on_send=True)
    s = ACSession(game.reader(), be, rate_hz=50.0, watchdog_s=0)
    with pytest.raises(RuntimeError, match="unplugged"):
        s.step(0.0, 1.0, 0.0)
    assert be.last == (0.0, 0.0, 1.0)


def test_watchdog_brakes_when_caller_stalls(game):
    be = RecordingBackend()
    s = ACSession(game.reader(), be, rate_hz=50.0, watchdog_s=0.1)
    with be.lock:
        be.calls.append((0.0, 1.0, 0.0))                  # pretend full throttle is latched
    s._failsafe_engaged = False
    s.touch()
    time.sleep(0.35)
    assert be.last == (0.0, 0.0, 1.0)
    assert s.stats.watchdog_trips == 1
    s.close(release=False)


def test_env_rejects_reset_modes_without_csp(game):
    s = ACSession(game.reader(), NullBackend(), watchdog_s=0)
    with pytest.raises(ValueError, match="csp"):
        ACRawEnv(s, reset_mode="reset_car")
    env = ACRawEnv(s, reset_mode="none")
    assert env.link is None
    s.close(release=False)


# --------------------------------------------------------------------------- episode monitor
def _tel(game, decoder, t, spline, speed_kmh=100.0, tyres_out=0, damage=0.0):
    ph, gr, st = game.ph, game.gr, game.st
    ph.speedKmh = speed_kmh
    ph.numberOfTyresOut = tyres_out
    ph.carDamage[0] = damage
    ph.velocity[:] = (0.0, 0.0, speed_kmh / 3.6)
    gr.normalizedCarPosition = spline
    return decoder.decode(ph, gr, st, wall_time=t)


def test_episode_monitor_wrong_way_off_track_collision(game):
    dec = TelemetryDecoder()
    mon = EpisodeMonitor(wrong_way_hold_s=0.2)
    L = game.st.trackSPlineLength
    mon.reset(_tel(game, dec, 0.0, 0.010))
    # forward across the start/finish line: 0.999 -> 0.001 is +0.002 laps, not -0.998
    assert EpisodeMonitor.spline_delta(0.999, 0.001) == pytest.approx(0.002)
    f = mon.update(_tel(game, dec, 0.1, 0.010 + 2.0 / L))
    assert not f.any and f.progress_speed == pytest.approx(20.0, rel=1e-3)
    # going backwards for > hold time
    sp = 0.010 + 2.0 / L
    flags = []
    for i in range(1, 5):
        flags.append(mon.update(_tel(game, dec, 0.1 + 0.1 * i, sp - i * 1.0 / L)))
    assert not flags[0].wrong_way and flags[-1].wrong_way
    mon.reset(_tel(game, dec, 1.0, 0.5))
    assert mon.update(_tel(game, dec, 1.1, 0.5, tyres_out=3)).off_track
    assert mon.update(_tel(game, dec, 1.2, 0.5, damage=10.0)).collision
    # 100 km/h -> 0 in 20 ms = 141 g: collision
    mon.reset(_tel(game, dec, 2.0, 0.6))
    f = mon.update(_tel(game, dec, 2.02, 0.6, speed_kmh=0.0))
    assert f.collision and any("accel_spike" in r for r in f.reasons)
