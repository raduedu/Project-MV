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
    for name in ("PLAY", "PAUSE", "STOP", "RESTART"):
        assert f"{name} = {getattr(Command, name)}" in text


def test_lua_uses_the_documented_frames():
    """map -> AC world: AC x = map_x, AC z = -map_y; steering negated once (ours: +1 = left)."""
    text = LUA.read_text(encoding="utf-8")
    assert "vec3(s.x, car.position.y, -s.y)" in text
    assert "vec3(math.cos(yaw), 0, -math.sin(yaw))" in text
    assert "controls.steer = math.max(-1, math.min(1, -s.steer))" in text


def _payload(n=500, ghost=True):
    t = np.linspace(0, 50, n)
    car = {"t": t, "x": 100 * np.cos(t / 8), "y": 100 * np.sin(t / 8), "yaw": t / 8 + math.pi / 2,
           "speed": np.full(n, 12.5), "steer": np.full(n, 0.2), "thr": np.full(n, 0.7), "brk": np.zeros(n),
           "gear": np.full(n, 3.0)}
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
    size = 4 * 12 + 96 + 4 * MAX_SAMPLES * 14
    assert ctypes.sizeof(ViewerData) == size
    assert ctypes.sizeof(ViewerState) == 4 * 12
