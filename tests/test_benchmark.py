"""Real-lap benchmark (WP43) and track sets (WP42): aligning a foreign lap to our circuit, re-cutting it at our
start line, and the track sets that decide what may be trained on.

No internet and no game needed: the "real" lap is made up from one of our own circuits,
moved into another frame the way F1's position data is.
"""

from __future__ import annotations

import numpy as np
import pytest

from mv.benchmark.openf1 import F1Lap, align_path, to_lap_file
from mv.track.generator import random_tracks
from mv.track.sets import load_sets, trainable_names, validation_names


def _fake_f1_lap(track, start_m: float, scale: float = 10.0, angle_deg: float = 37.0, mirror: bool = False,
                 lateral_m: float = 2.0, rate_hz: float = 3.7, lap_time: float = 90.0) -> F1Lap:
    """A lap driven at a speed that varies with curvature, sampled like F1 data, in a rotated,
    scaled (decimetres) and possibly mirrored frame, starting ``start_m`` after our start line."""
    L = track.length
    s = np.arange(track.n) * track.ds
    v = 30.0 + 40.0 / (1.0 + 40.0 * np.abs(track.kappa))          # slow in corners
    t_of_s = np.concatenate([[0.0], np.cumsum(track.ds / v)[:-1]])
    t_of_s *= lap_time / (t_of_s[-1] + track.ds / v[-1])           # exactly lap_time per lap
    t = np.arange(-2.0, lap_time + 2.0, 1.0 / rate_hz)
    t0 = float(np.interp(start_m, s, t_of_s))
    tt = (t + t0) % lap_time
    ss = np.interp(tt, t_of_s, s)
    i = (ss / track.ds).astype(int) % track.n
    xy = track.xy[i] + track.normal[i] * lateral_m * np.sin(ss / 150.0)[:, None]
    ang = np.radians(angle_deg)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    p = (xy - 500.0) @ rot.T * scale
    if mirror:
        p[:, 1] = -p[:, 1]
    speed = np.interp(ss, s, v)
    return F1Lap(year=2023, circuit="Test", session="Qualifying", driver_number=1, lap_number=7, lap_time=lap_time,
                 date_start="2023-01-01T00:00:00+00:00", t_car=t, speed=speed, rpm=np.full_like(t, 11000.0),
                 gear=np.full_like(t, 6.0), throttle=np.ones_like(t), brake=np.zeros_like(t), drs=np.zeros_like(t),
                 t_pos=t, xy_raw=p, z_raw=np.zeros(len(t)))


@pytest.fixture(scope="module")
def track():
    return random_tracks(1, seed=11, cfg="mixed")[0]


@pytest.mark.parametrize("mirror", [False, True])
def test_alignment_recovers_the_frame(track, mirror):
    lap = _fake_f1_lap(track, start_m=0.0, mirror=mirror)
    al = align_path(lap.xy_raw, track, z_raw=lap.z_raw)
    assert np.median(al.residual_m) < 2.5            # only the 2 m sideways weave is left
    assert np.percentile(al.residual_m, 95) < 3.5


def test_lap_is_recut_at_our_start_line(track):
    """F1's timing line 180 m after ours (Imola-like): same lap time, distance from our line."""
    lap = _fake_f1_lap(track, start_m=180.0)
    out, al = to_lap_file(lap, track, "test_car")
    smp = out["samples"]
    s, t = np.asarray(smp["s"]), np.asarray(smp["t"])
    assert out["meta"]["lap_time"] == pytest.approx(lap.lap_time, abs=1e-3)
    assert out["meta"]["kind"] == "benchmark" and "steer" not in smp
    assert abs(out["meta"]["alignment"]["f1_line_at_m"] - 180.0) < 15.0
    assert s[0] < 20.0 and s[-1] > track.length - 20.0       # starts at OUR line, ends a lap later
    assert np.all(np.diff(t) > 0) and np.all(np.diff(s) >= -1e-6)
    # the recorded speed at each distance is the speed the fake car had there
    v_true = 30.0 + 40.0 / (1.0 + 40.0 * np.abs(track.kappa))
    v_at = np.interp(s, np.arange(track.n) * track.ds, v_true)
    assert np.median(np.abs(np.asarray(smp["speed"]) - v_at)) < 1.0


def test_track_sets_hold_circuits_out_of_training(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    for n in ("a", "b", "c", "d"):
        (local / f"{n}.npz").write_bytes(b"")
    sets_file = tmp_path / "sets.yaml"
    sets_file.write_text("validation: [a, b, zz]\nnever_train: [b]\nskip: [d]\n")
    sets = load_sets(sets_file)
    assert validation_names(local, sets) == ["a", "b"]           # only what is imported
    assert trainable_names("all", local, sets) == ["a", "c"]
    with pytest.raises(ValueError):
        trainable_names("a,b", local, sets)                      # b is held out: refuse, do not drop silently
    with pytest.raises(ValueError):
        trainable_names("a,x", local, sets)                      # not imported
    empty = load_sets(tmp_path / "missing.yaml")
    assert validation_names(local, empty) == ["a", "b", "c", "d"]


def test_track_margin_moves_both_edges_inwards(track):
    """WP48: imported circuits can be narrowed for training, so laps stay off the game's kerbs."""
    from mv.track.sets import narrowed
    n = narrowed(track, 0.6)
    assert n.name == track.name and n.meta["edge_margin_m"] == 0.6
    assert abs(n.length - track.length) < 0.01 * track.length          # same centre line
    wide = (track.w_left > 2.7) & (track.w_right > 2.7)
    assert np.allclose(n.w_left[wide], track.w_left[wide] - 0.6, atol=0.05)
    assert np.all(n.w_left >= 2.0 - 1e-9) and np.all(n.w_right >= 2.0 - 1e-9)   # never narrower than 4 m
    assert narrowed(track, 0.0) is track
