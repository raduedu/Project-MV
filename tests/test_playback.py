"""Every lap played in Assetto Corsa is made safe for the game's physics engine (engineering log WP70).

The owner's game stopped with "ODE INTERNAL ERROR 1, !row1_norm_fault" on Verstappen's laps: positions
measured at ~4 Hz with uneven time stamps implied 637 m/s between samples, and the viewer placed the car
with that velocity. One stage (mv/ac/playback.py) now prepares every lap; the Lua apps guard every value
they hand to the physics engine. Hand-checkable numbers:

* a clean 25 Hz lap at 40 m/s on a 200 m circle comes back bit-identical;
* the same lap measured at ~4 Hz with jittered time stamps (+ a 15 m backwards glitch) is repaired to
  within 1.5 m of the circle, with path speed within 15 % (+2 m/s) of the recorded 40 m/s;
* a lap that leaves the circuit by 20 m ends no more than ~1.2 m past the edge.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from mv.ac.playback import LIMITS, PlaybackError, beyond_edge_m, prepare_playback, problems
from tests.test_core import circle_track

R, HALF, V = 200.0, 6.0, 40.0
LUA = Path(__file__).resolve().parents[1] / "mv" / "ac" / "plugin" / "apps" / "lua"


def _circle_lap(t: np.ndarray, radius: float = R) -> dict[str, np.ndarray]:
    """Anticlockwise at V m/s on a circle of ``radius``, sampled at times ``t``."""
    a = V * t / radius
    return {"t": t.copy(), "x": radius * np.cos(a), "y": radius * np.sin(a), "yaw": a + np.pi / 2,
            "speed": np.full_like(t, V), "steer": np.full_like(t, 0.1), "gear": np.full_like(t, 4.0)}


def _distance_to_circle(a, radius: float = R) -> np.ndarray:
    return np.abs(np.hypot(a["x"], a["y"]) - radius)


def test_a_clean_lap_comes_back_unchanged():
    lap = _circle_lap(np.arange(0, 30, 0.04))
    out, notes = prepare_playback(lap, track=circle_track(R, HALF))
    assert notes == []
    for k in lap:
        assert np.array_equal(out[k], lap[k]), k


def test_missing_values_and_repeated_time_stamps_are_dropped():
    lap = _circle_lap(np.arange(0, 30, 0.04))
    lap["x"][100] = np.nan
    lap["t"][200] = lap["t"][199]                       # a repeated time stamp
    out, notes = prepare_playback(lap, track=circle_track(R, HALF))
    assert problems(out) == []
    assert any("missing" in n for n in notes) and any("repeated" in n for n in notes)
    assert len(out["t"]) < len(lap["t"])


def test_a_lap_measured_at_4_hz_with_uneven_time_stamps_is_repaired_and_stays_on_its_line():
    rng = np.random.default_rng(3)
    t_meas = np.cumsum(rng.uniform(0.2, 0.32, 140))                 # ~4 Hz position fixes
    fix = _circle_lap(t_meas)
    t = np.sort(np.concatenate([np.arange(0, t_meas[-1], 0.04), t_meas + 0.008]))   # mixed 0.04 / 0.008 s steps
    idx = np.searchsorted(t_meas, t, side="right") - 1               # stair steps: last fix held
    idx = np.clip(idx, 0, len(t_meas) - 1)
    lap = {"t": t, "x": fix["x"][idx], "y": fix["y"][idx], "yaw": np.zeros_like(t), "speed": np.full_like(t, V)}
    k = len(t) // 2                                                  # a glitch: 15 m backwards for one sample
    back = np.array([-np.sin(V * t[k] / R), np.cos(V * t[k] / R)]) * -15.0
    lap["x"][k] += back[0]
    lap["y"][k] += back[1]
    lap["yaw"] = np.unwrap(np.arctan2(np.gradient(lap["y"]), np.gradient(lap["x"])))   # as viewer_link derives it
    assert problems(lap)                                             # broken as measured
    out, notes = prepare_playback(lap, track=circle_track(R, HALF))
    assert problems(out) == []
    assert notes and "recorded speed" in " ".join(notes)
    assert np.max(_distance_to_circle(out)) < 1.5
    path_v = np.hypot(np.diff(out["x"]), np.diff(out["y"])) / np.diff(out["t"])
    assert np.all(path_v <= LIMITS.speed_ratio * V + LIMITS.speed_slack)


def test_a_line_from_another_layout_is_kept_on_the_circuit():
    track = circle_track(R, HALF)
    lap = _circle_lap(np.arange(0, 30, 0.04))
    r = R + 20.0 * np.clip(np.sin(np.linspace(0, np.pi, len(lap["t"]))) * 3 - 2, 0, 1)   # bulges 20 m out
    a = np.arctan2(lap["y"], lap["x"])
    lap["x"], lap["y"] = r * np.cos(a), r * np.sin(a)
    assert beyond_edge_m(lap, track) > 10
    out, notes = prepare_playback(lap, track=track)
    assert problems(out) == []
    assert beyond_edge_m(out, track) < 1.25
    assert any("left this circuit" in n for n in notes)


def test_an_unusable_lap_is_refused_with_a_reason():
    lap = _circle_lap(np.arange(0, 1, 0.04))
    for k in lap:
        lap[k][2:] = np.nan
    with pytest.raises(PlaybackError, match="usable samples"):
        prepare_playback(lap)


def test_every_saved_lap_is_safe_and_clean_laps_are_untouched():
    """The owner's own laps (if any are on this computer): what the viewer would send is always safe, and a
    lap that was already safe is sent exactly as recorded."""
    from mv.ac.viewer_link import lap_arrays, viewer_payload
    from mv.laps import list_laps, load_lap
    laps = list_laps(True)
    if not laps:
        pytest.skip("no saved laps")
    chosen = [m for m in laps if m.get("kind") != "learned"] + [m for m in laps if m.get("kind") == "learned"][::25]
    played = 0
    for m in chosen:
        try:
            p = viewer_payload(m["key"])
        except PlaybackError:
            continue                                     # refused with a message: also safe
        except ValueError:
            continue                                     # not a game circuit
        played += 1
        assert problems(p["car"]) == [], m["key"]
        if p["ghost"] is not None:
            assert problems(p["ghost"]) == [], m["key"]
        if not p["repairs"]:
            raw = lap_arrays(load_lap(m["key"]))
            for k in ("t", "x", "y", "yaw", "speed"):
                assert np.array_equal(raw[k], p["car"][k]), (m["key"], k)
    if not played:
        pytest.skip("no laps on game circuits")


@pytest.mark.parametrize("app, guard, calls", [
    ("mv_viewer/mv_viewer.lua", "placeCarSafe", ("physics.setCarPosition", "physics.setCarVelocity")),
    ("mv_viewer/mv_follow.lua", "addForceSafe", ("physics.addForce",)),
    ("mv_bridge/mv_bridge.lua", "placeCarSafe", ("physics.setCarPosition", "physics.setCarVelocity")),
])
def test_lua_hands_positions_velocities_and_forces_to_the_physics_only_through_its_guard(app, guard, calls):
    src = (LUA / app).read_text(encoding="utf-8")
    start = src.index(f"local function {guard}(")
    end = src.index("\nend\n", start)
    for call in calls:
        where = [m.start() for m in re.finditer(re.escape(call) + r"\(", src)]
        assert where, f"{app}: {call} not found"
        assert all(start < w < end for w in where), f"{app}: {call} used outside {guard}"
    body = src[start:end]
    assert "finiteVec" in body and "length()" in body          # checks for NaN/inf and zero-length directions


def test_no_lua_app_calls_the_raw_physics_placement_or_forces_unguarded():
    for f in LUA.rglob("*.lua"):
        src = f.read_text(encoding="utf-8")
        for call in ("physics.setCarPosition(", "physics.setCarVelocity(", "physics.addForce("):
            for m in re.finditer(re.escape(call), src):
                before = src[:m.start()]
                fn = before.rfind("local function ")
                name = before[fn:].split("(")[0].split()[-1] if fn >= 0 else ""
                assert name.endswith("Safe"), f"{f.name}: {call} outside a *Safe guard (in {name or 'top level'})"
