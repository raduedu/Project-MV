"""Tests for the Assetto Corsa lap viewer (mv/ac/viewer_link.py + the MV Viewer Lua app, log WP32).

The game itself cannot run in tests; what can go wrong on our side is tested:
the shared-memory layouts must match the Lua file byte for byte, the lap must
arrive intact, and the lap preparation (ghost choice, resampling, heading)
must be right.
"""

from __future__ import annotations

import ctypes
import math
import re
import uuid
from pathlib import Path

import numpy as np

from mv.ac.viewer_link import (MAX_SAMPLES, VIEWER_STATE_MAGIC, VIEWER_VERSION, Command, ViewerData, ViewerLink,
                               ViewerState, find_ghost, lap_arrays, viewer_layout)

ROOT = Path(__file__).resolve().parents[1]
LUA = ROOT / "mv" / "ac" / "plugin" / "apps" / "lua" / "mv_viewer" / "mv_viewer.lua"


def _lua_block(name: str) -> str:
    m = re.search(rf"local {name} = \[\[\n(.*?)\]\]", LUA.read_text(encoding="utf-8"), re.S)
    assert m, name
    return m.group(1).strip()


def test_lua_layouts_match_python():
    assert _lua_block("DATA_LAYOUT") == viewer_layout(ViewerData)
    assert _lua_block("STATE_LAYOUT") == viewer_layout(ViewerState)
    text = LUA.read_text(encoding="utf-8")
    assert f"VIEWER_VERSION = {VIEWER_VERSION}" in text
    assert "0x3157564D" in text and "0x3153564D" in text
    for name in ("PLAY", "PAUSE", "STOP", "RESTART", "QUIT"):
        assert f"{name} = {getattr(Command, name)}" in text


def test_lua_uses_the_documented_frames():
    """map -> AC world: AC x = map_x, AC z = -map_y; steering negated once (ours: +1 = left)."""
    text = LUA.read_text(encoding="utf-8")
    assert "vec3(s.x, car.position.y, -s.y)" in text
    assert "vec3(math.cos(yaw), 0, -math.sin(yaw))" in text
    assert "controls.steer = math.max(-1, math.min(1, -s.steer))" in text
    # WP33: the recorded gears and engine speed are shown, the game's own gearbox is switched off meanwhile
    assert "physics.setCarAutoShifter(0, false)" in text and "physics.setEngineRPM(0, s.rpm)" in text
    assert "controls.requestedGearIndex = g" in text


def _payload(n=500, ghost=True):
    t = np.linspace(0, 50, n)
    car = {"t": t, "x": 100 * np.cos(t / 8), "y": 100 * np.sin(t / 8), "yaw": t / 8 + math.pi / 2,
           "speed": np.full(n, 12.5), "steer": np.full(n, 0.2), "thr": np.full(n, 0.7), "brk": np.zeros(n),
           "gear": np.full(n, 3.0), "rpm": np.full(n, 6000.0)}
    g = {k: car[k] for k in ("t", "x", "y", "yaw", "speed")}
    return {"meta": {"lap_time": 50.0}, "label": "test lap | ghost", "car": car,
            "ghost": g if ghost else None, "ghost_meta": {"lap_time": 49.0} if ghost else None}


def test_lap_round_trip_through_shared_memory():
    tag = "mv_viewer_test_" + uuid.uuid4().hex[:8]
    link = ViewerLink(tag, tag + "_state")
    try:
        p = _payload()
        lap_id = link.load(p, speed=0.5, loop=False)
        d = link.data
        assert d.seq % 2 == 0 and d.lap_id == lap_id and d.n_car == 500 and d.n_ghost == 500
        assert d.command == Command.PLAY and abs(d.speed - 0.5) < 1e-6 and d.loop == 0
        assert abs(d.car_x[123] - p["car"]["x"][123]) < 1e-3 and abs(d.ghost_yaw[499] - p["ghost"]["yaw"][499]) < 1e-5
        assert bytes(d.label).rstrip(b"\0").decode() == "test lap | ghost"
        seq0 = d.command_seq
        link.command(Command.PAUSE)
        assert d.command == Command.PAUSE and d.command_seq == seq0 + 1
        assert link.load(_payload(ghost=False)) == lap_id + 1 and d.n_ghost == 0
        assert link.state() is None                       # the game has not written yet
        # what the Lua side writes
        st = ViewerState.from_buffer(link._state_mm)
        st.magic, st.version, st.lap_id_loaded, st.playing, st.physics_allowed = VIEWER_STATE_MAGIC, 1, lap_id, 1, 1
        st.gap_s = 0.42
        s = link.state()
        assert s["lap_id_loaded"] == lap_id and s["playing"] and abs(s["gap_s"] - 0.42) < 1e-6
        del st, d
    finally:
        link.close()


def test_lap_preparation():
    n = MAX_SAMPLES + 3000
    t = np.linspace(0, 600, n)
    lap = {"samples": {"t": t.tolist(), "x": np.cos(t).tolist(), "y": np.sin(t).tolist(),
                       "speed": np.ones(n).tolist()}}
    a = lap_arrays(lap)
    assert len(a["t"]) == MAX_SAMPLES and a["t"][0] == 0 and a["t"][-1] == 600
    assert np.all(a["gear"] == 0)                         # missing fields become zeros
    # no recorded heading: derived from the path (CCW circle -> heading = angle + 90 deg)
    lap = {"samples": {"t": [0, 1, 2, 3], "x": [1, 0, -1, 0], "y": [0, 1, 0, -1], "speed": [1, 1, 1, 1]}}
    yaw = lap_arrays(lap)["yaw"]
    assert abs(math.cos(yaw[1]) - (-1)) < 1e-6

    metas = [{"kind": "reference", "car": "A", "track": "imola", "created": "2026-11-18", "key": "old"},
             {"kind": "reference", "car": "A", "track": "imola", "created": "2026-11-20", "key": "new"},
             {"kind": "reference", "car": "B", "track": "imola", "created": "2026-11-21", "key": "other car"},
             {"kind": "learned", "car": "A", "track": "imola", "created": "2026-11-22", "key": "not a ghost"}]
    assert find_ghost({"car": "A", "track": "imola"}, metas)["key"] == "new"
    assert find_ghost({"car": "A", "track": "monza"}, metas) is None


def test_structs_have_no_padding_surprises():
    """Lua's FFI lays the struct out like C with 4-byte alignment; so does ctypes with _pack_ = 4."""
    size = 4 * 12 + 96 + 4 * MAX_SAMPLES * 15
    assert ctypes.sizeof(ViewerData) == size
    assert ctypes.sizeof(ViewerState) == 4 * 12


# ---------------------------------------------------------------- engine speed (WP33)
def test_engine_rpm_follows_gear_ratios():
    from mv.ac.viewer_link import engine_rpm
    from mv.vehicle.params import find_vehicle, load_vehicle
    vp = load_vehicle(find_vehicle("porsche_911_gt3_rs_992"))
    gb = next(iter(vp.powertrain.gearboxes.values()))
    u = vp.powertrain.units[0]
    r = u.front_frac * vp.tire_front.radius_m + (1 - u.front_frac) * vp.tire_rear.radius_m
    v = np.array([30.0, 30.0, 200.0])
    rpm = engine_rpm(v, [3, 4, 1], vp)
    expect = 30.0 / r * gb.ratios[2] * gb.final_drive * 30 / math.pi
    assert abs(rpm[0] - expect) < 1e-6 and rpm[1] < rpm[0]           # higher gear, lower rpm
    assert rpm[2] == u.rpm_max                                         # never above the limiter


# ---------------------------------------------------------------- launcher (WP33) - the game is never started here
OWNER_INI = """[RACE]
TRACK=monza
CONFIG_TRACK=
MODEL=ks_porsche_918_spyder
CARS=6
AI_LEVEL=98
SKIN=old

[LIGHTING]
SUN_ANGLE=50.16
__CM_WEATHER_TYPE=15

[REMOTE]
ACTIVE=1

[SESSION_0]
NAME=Qualifying
TYPE=2

[SESSION_1]
NAME=Race
TYPE=3

[CAR_0]
MODEL=-
SKIN=old
DRIVER_NAME=Owner Name
NATIONALITY=Planet Earth

[CAR_3]
MODEL=some_ai_car
"""


def _ini(text):
    from mv.ac.launch import _parse_ini
    return {n: dict(k) for n, k in _parse_ini(text)}


def test_race_ini_keeps_personal_settings_and_sets_the_session():
    from mv.ac.launch import build_race_ini
    d = _ini(build_race_ini(OWNER_INI, "ks_porsche_911_gt3_rs", "00_lava", "ks_red_bull_ring", "layout_gp"))
    assert d["RACE"]["TRACK"] == "ks_red_bull_ring" and d["RACE"]["CONFIG_TRACK"] == "layout_gp"
    assert d["RACE"]["MODEL"] == "ks_porsche_911_gt3_rs" and d["RACE"]["CARS"] == "1" and d["RACE"]["SKIN"] == "00_lava"
    assert d["LIGHTING"]["SUN_ANGLE"] == "50.16" and d["LIGHTING"]["__CM_WEATHER_TYPE"] == "15"   # weather kept
    assert d["CAR_0"]["DRIVER_NAME"] == "Owner Name" and d["CAR_0"]["SKIN"] == "00_lava"
    assert d["SESSION_0"]["NAME"] == "Practice" and d["SESSION_0"]["TYPE"] == "1"
    assert "SESSION_1" not in d and "CAR_3" not in d and "CAR_1" not in d
    assert d["REMOTE"]["ACTIVE"] == "0"                                   # never online
    g = _ini(build_race_ini(OWNER_INI, "ks_porsche_911_gt3_rs", "00_lava", "imola", "", ghost_car=True,
                            ghost_skin="01_violet"))
    assert g["RACE"]["CARS"] == "2" and g["CAR_1"]["MODEL"] == "ks_porsche_911_gt3_rs"
    assert g["CAR_1"]["SKIN"] == "01_violet"


def test_open_lap_in_game_starts_restarts_or_plays(tmp_path, monkeypatch):
    """Three cases: game closed -> race.ini written (backup kept) + game started; running in the right
    session -> plays at once; running elsewhere -> asks first, closes it only when told to."""
    import mv.ac.launch as L
    ini = tmp_path / "race.ini"
    ini.write_text(OWNER_INI)
    ac_root = tmp_path / "ac"
    (ac_root / "content" / "cars" / "ks_porsche_911_gt3_rs" / "skins" / "00_lava").mkdir(parents=True)
    started, running, session = [], [False], [None]
    monkeypatch.setattr(L, "race_ini_path", lambda: ini)
    monkeypatch.setattr(L, "BACKUP_DIR", tmp_path / "backups")
    monkeypatch.setattr(L, "ac_ids_for_lap", lambda meta: ("ks_porsche_911_gt3_rs", "imola", ""))
    monkeypatch.setattr(L, "game_running", lambda: running[0])
    monkeypatch.setattr(L, "steam_running", lambda: True)
    monkeypatch.setattr(L, "running_session", lambda: session[0])
    monkeypatch.setattr(L, "start_game", lambda root: started.append(root))

    class Link:
        def __init__(self):
            self.loaded, self.cmds = 0, []

        def load(self, payload, **kw):
            self.loaded += 1

        def command(self, c, **kw):
            self.cmds.append(c)
            running[0] = False                        # the app closes the game

    link, payload = Link(), {"meta": {}, "label": "lap"}
    r = L.open_lap_in_game(payload, link, ac_root=ac_root)
    assert r["started"] and started == [ac_root] and link.loaded == 1
    assert _ini(ini.read_text())["RACE"]["TRACK"] == "imola"
    assert len(list((tmp_path / "backups").glob("race_*.ini"))) == 1       # the owner's file was kept

    running[0], session[0] = True, {"car": "ks_porsche_911_gt3_rs", "track": "imola", "layout": ""}
    r = L.open_lap_in_game(payload, link, ac_root=ac_root)
    assert "started" not in r and link.loaded == 2 and len(started) == 1   # same session: just play

    session[0] = {"car": "ks_porsche_918_spyder", "track": "monza", "layout": ""}
    r = L.open_lap_in_game(payload, link, ac_root=ac_root)
    assert r["needs_restart"] and not link.cmds and len(started) == 1      # asks, does nothing yet
    r = L.open_lap_in_game(payload, link, restart=True, ac_root=ac_root)
    assert link.cmds == [Command.QUIT] and r["started"] and len(started) == 2


def test_exit_watchdog_ends_only_a_game_that_hangs_after_a_clean_exit(tmp_path):
    """A stand-in process plays the game: untouched while no clean exit is logged, ended after one."""
    import subprocess
    import sys
    import threading
    import time as _t
    from mv.ac.launch import watch_game
    log = tmp_path / "log.txt"
    log.write_text("loading...\n")
    since = _t.time() - 1
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        # no "CLEAN EXIT" in the log: the watchdog must leave it alone
        assert watch_game(proc.pid, since, log, grace_s=0.3, poll_s=0.1, max_hours=1.5 / 3600) == "gave up"
        assert proc.poll() is None
        threading.Timer(0.5, lambda: log.write_text("...\nCLEAN EXIT\n")).start()
        assert watch_game(proc.pid, since, log, grace_s=0.5, poll_s=0.1, max_hours=0.01) == "ended after clean exit"
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
