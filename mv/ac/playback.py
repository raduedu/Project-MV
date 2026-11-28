"""Every lap that is played in Assetto Corsa goes through here first: cleaned, checked, repaired or refused.

Why (engineering log WP70): the game's physics engine (ODE) stops the whole game with "ODE INTERNAL ERROR 1,
assertion !row1_norm_fault" when it is handed a degenerate state - a direction of zero length, a NaN, an
absurd speed. The viewer turns a lap into exactly such states: it places the car with the velocity
(next position - this position) / (time between samples), and a force follower pulls the car towards the
recorded position and heading 333 times per second. Laps from our simulator are smooth (25 Hz, path speed
within 8 % of the recorded speed, turning rate below 2 rad/s), but a lap measured in the real world is
not: Verstappen's laps (OpenF1 positions at ~4 Hz, uneven time stamps down to 0.007 s) imply speeds of up
to 637 m/s between samples and heading flips of 2.5 rad. The game was given a car moving at 637 m/s.

The fix is one stage, not one patch per lap: :func:`prepare_playback` takes any lap's sample arrays and

1. drops samples with non-finite values and makes time strictly increasing;
2. checks the lap against physical limits (:data:`LIMITS`) - a lap that passes is returned UNCHANGED, so
   playback stays an exact copy of what was driven (owner, 2026-11-27: "the AC viewer as accurate as
   possible");
3. otherwise repairs it - uniform time steps, the path smoothed just enough (the shortest smoothing
   window that satisfies the limits), the heading taken from the path where the recorded one is
   inconsistent - and reports what it did;
4. refuses (:class:`PlaybackError`) a lap it cannot make safe, so the control panel shows a message
   instead of the game crashing.

The Lua apps add a second, independent guard on every value they hand to the physics engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LIMITS", "Limits", "PlaybackError", "problems", "prepare_playback", "check_body",
           "beyond_edge_m", "CIRCUIT_SLACK_M"]


class PlaybackError(ValueError):
    """The lap cannot be played safely in the game (message says why)."""


@dataclass(frozen=True)
class Limits:
    """What a lap may ask of the game's physics. Set from all ~880 saved laps (WP70): clean laps reach
    at most 1.08 x the recorded speed along their path and 1.8 rad/s of turning; the broken ones
    9.8 x and 63 rad/s."""
    dt_min: float = 0.02            # s: shortest time step (the follower differentiates the path)
    dt_max: float = 0.25            # s: longest gap between samples
    speed_ratio: float = 1.15       # path speed may exceed the recorded speed by this factor ...
    speed_slack: float = 2.0        # ... plus this many m/s (slow corners, rounding)
    speed_max: float = 120.0        # m/s (432 km/h): nothing we drive is faster
    yaw_rate_max: float = 4.0       # rad/s: a car turning faster than this is spinning, not lapping
    min_samples: int = 3


LIMITS = Limits()
_SMOOTH_S = (0.12, 0.24, 0.48, 0.96, 1.92)     # s: smoothing windows tried, shortest first
_LINE_SMOOTH_M = (7.0, 15.0, 30.0)             # m: wiggles removed from a re-timed line, shortest first


def _yaw_from_path(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))


def problems(a: dict[str, np.ndarray], lim: Limits = LIMITS) -> list[str]:
    """Reasons why a lap's arrays are not safe to play (empty list = safe)."""
    t = np.asarray(a["t"], float)
    if len(t) < lim.min_samples:
        return [f"only {len(t)} samples"]
    out = []
    bad = [k for k, v in a.items() if not np.all(np.isfinite(np.asarray(v, float)))]
    if bad:
        return [f"non-finite values in {', '.join(sorted(bad))}"]
    dt = np.diff(t)
    if np.any(dt <= 0):
        return ["time not strictly increasing"]
    if dt.min() < lim.dt_min or dt.max() > lim.dt_max:
        out.append(f"time steps {dt.min():.3f}-{dt.max():.3f} s outside {lim.dt_min}-{lim.dt_max} s")
    x, y = np.asarray(a["x"], float), np.asarray(a["y"], float)
    path_v = np.hypot(np.diff(x), np.diff(y)) / dt
    if path_v.max() > lim.speed_max:
        out.append(f"path speed up to {path_v.max():.0f} m/s")
    if "speed" in a and np.any(np.asarray(a["speed"], float) > 0):      # all zeros = not recorded
        v = np.maximum(np.asarray(a["speed"], float)[:-1], np.asarray(a["speed"], float)[1:])
        excess = path_v - (lim.speed_ratio * v + lim.speed_slack)
        if excess.max() > 0:
            i = int(excess.argmax())
            out.append(f"path speed {path_v[i]:.0f} m/s where the car did {v[i]:.0f} m/s")
    if "yaw" in a:
        r = np.abs(np.diff(np.unwrap(np.asarray(a["yaw"], float)))) / dt
        if r.max() > lim.yaw_rate_max:
            out.append(f"heading turns at up to {r.max():.1f} rad/s")
    return out


def _clean(a: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Drop non-finite samples, sort by time, keep the first of equal time stamps."""
    notes = []
    n = len(a["t"])
    ok = np.ones(n, bool)
    for v in a.values():
        ok &= np.isfinite(np.asarray(v, float))
    if not ok.all():
        notes.append(f"dropped {int((~ok).sum())} samples with missing values")
    a = {k: np.asarray(v, float)[ok] for k, v in a.items()}
    order = np.argsort(a["t"], kind="stable")
    a = {k: v[order] for k, v in a.items()}
    keep = np.concatenate([[True], np.diff(a["t"]) > 0])
    if not keep.all():
        notes.append(f"dropped {int((~keep).sum())} repeated time stamps")
        a = {k: v[keep] for k, v in a.items()}
    return a, notes


def _resample(a: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    t = a["t"]
    tn = np.arange(t[0], t[-1] + 1e-9, dt)
    out = {}
    for k, v in a.items():
        if k == "t":
            out[k] = tn
        elif k == "yaw":
            out[k] = np.interp(tn, t, np.unwrap(v))
        elif k == "gear":
            out[k] = v[np.clip(np.searchsorted(t, tn, side="right") - 1, 0, len(t) - 1)]   # gears stay integers
        else:
            out[k] = np.interp(tn, t, v)
    return out


def _smooth(v: np.ndarray, k: int) -> np.ndarray:
    """Zero-phase moving average over 2k+1 samples (edges padded with the end values)."""
    w = np.ones(2 * k + 1) / (2 * k + 1)
    return np.convolve(np.pad(v, k, mode="edge"), w, "valid")


def _drop_backtracks(x: np.ndarray, y: np.ndarray, w: int = 5, rounds: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Remove measured points that step BACKWARDS against the local direction of travel (measuring
    glitches: a car on a lap never reverses). The local direction is the chord from w points before to
    w points after; a point whose step from its predecessor points against it is dropped."""
    for _ in range(rounds):
        n = len(x)
        if n < 2 * w + 3:
            break
        i = np.arange(1, n)
        lo, hi = np.clip(i - w, 0, n - 1), np.clip(i + w, 0, n - 1)
        cx, cy = x[hi] - x[lo], y[hi] - y[lo]
        sx, sy = x[i] - x[i - 1], y[i] - y[i - 1]
        back = (sx * cx + sy * cy) < 0
        if not back.any():
            break
        keep = np.concatenate([[True], ~back])
        x, y = x[keep], y[keep]
    return x, y


def _retime_along_path(a: dict[str, np.ndarray], smooth_m: float = 7.0) -> tuple[dict[str, np.ndarray], float]:
    """Keep the SHAPE of the driven line, but place the car along it by the recorded speed.

    A lap measured at a few Hz with uneven time stamps has its points on the right line but at the
    wrong moments (the car "stands", then "jumps"). Smoothing in time would cut the corners. Instead:
    the line as a curve (points 1 m apart along its length, wiggles shorter than ``smooth_m`` removed),
    and the distance travelled from the speed channel, integrated and scaled to the line's length.
    Returns the arrays and the scale factor (1.00 = the speed channel matches the line's length).
    """
    x, y, t, v = a["x"], a["y"], a["t"], a["speed"]
    x, y = _drop_backtracks(x, y)
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], np.diff(s) > 0.05])
    s, xs, ys = s[keep], x[keep], y[keep]
    sg = np.arange(0.0, s[-1], 1.0)
    xg, yg = np.interp(sg, s, xs), np.interp(sg, s, ys)
    k = max(1, int(round(smooth_m / 2)))
    xg, yg = _smooth(xg, k), _smooth(yg, k)
    sg = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xg), np.diff(yg)))])
    dist = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    scale = sg[-1] / max(dist[-1], 1e-6)
    dist *= scale
    b = dict(a, x=np.interp(dist, sg, xg), y=np.interp(dist, sg, yg))
    if "yaw" in a:                                   # heading of the smoothed line itself (1 m steps)
        b["yaw"] = np.interp(dist, sg, _yaw_from_path(xg, yg))
    return b, float(scale)


def _project_along(x: np.ndarray, y: np.ndarray, track, window_m: float = 150.0) -> tuple[np.ndarray, np.ndarray]:
    """Distance along the circuit (continuous over the start line) and sideways offset of each point,
    following the lap: each point is matched only to stations within ``window_m`` of where the previous
    one matched. (The globally nearest station can lie on another part of the circuit when the lap was
    driven on a different layout - the matched distance would leap.)"""
    from scipy.spatial import cKDTree
    m = len(track.xy)
    ds = float(track.length) / m
    w = max(5, int(window_m / ds))
    _, i0 = cKDTree(track.xy).query([x[0], y[0]])
    idx = np.empty(len(x), int)
    cur = int(i0)
    offs = np.arange(-w, w + 1)
    for j in range(len(x)):
        cand = (cur + offs) % m
        d2 = (track.xy[cand, 0] - x[j]) ** 2 + (track.xy[cand, 1] - y[j]) ** 2
        cur = int(cand[int(np.argmin(d2))])
        idx[j] = cur
    nrm = track.normal[idx]
    n = (x - track.xy[idx, 0]) * nrm[:, 0] + (y - track.xy[idx, 1]) * nrm[:, 1]
    L = float(track.length)
    s = np.unwrap(idx * ds * 2 * np.pi / L) * L / (2 * np.pi)
    return s, n


def _retime_on_track(a: dict[str, np.ndarray], track, smooth_m: float = 7.0,
                     beyond_m: float = 1.0) -> tuple[dict[str, np.ndarray], float]:
    """Like :func:`_retime_along_path`, but the line is described on the circuit: distance along the centre
    line (never going backwards - measuring noise that steps back is dropped) and the offset from it
    (wiggles shorter than ``smooth_m`` removed, never more than ``beyond_m`` past the edges - a lap driven
    on another layout of the circuit is kept on the game's one). Robust against positions that jitter."""
    L = float(track.length)
    ds = L / len(track.xy)
    sc, n = _project_along(*_drop_backtracks(a["x"], a["y"]), track)
    sc = np.maximum.accumulate(sc)
    keep = np.concatenate([[True], np.diff(sc) > 0.05])
    sc, n = sc[keep], n[keep]
    sg = np.arange(sc[0], sc[-1], 1.0)
    k = max(1, int(round(smooth_m / 2)))
    st_i = np.mod(np.round(sg / ds).astype(int), len(track.xy))
    lo, hi = -(track.w_right[st_i] + beyond_m), track.w_left[st_i] + beyond_m
    ng = _smooth(np.clip(np.interp(sg, sc, n), lo, hi), k)   # limit first, then smooth: no kinks at the limit
    st = np.arange(len(track.xy) + 1) * ds                                   # stations, closed
    xy_c = np.vstack([track.xy, track.xy[:1]])
    nr_c = np.vstack([track.normal, track.normal[:1]])
    sw = np.mod(sg, L)
    cx, cy = np.interp(sw, st, xy_c[:, 0]), np.interp(sw, st, xy_c[:, 1])
    nx, ny = np.interp(sw, st, nr_c[:, 0]), np.interp(sw, st, nr_c[:, 1])
    nl = np.maximum(np.hypot(nx, ny), 1e-9)
    xg, yg = cx + ng * nx / nl, cy + ng * ny / nl
    sa = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xg), np.diff(yg)))])
    t, v = a["t"], a["speed"]
    dist = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    scale = sa[-1] / max(dist[-1], 1e-6)
    dist *= scale
    b = dict(a, x=np.interp(dist, sa, xg), y=np.interp(dist, sa, yg))
    if "yaw" in a:                                   # heading of the smoothed line itself (1 m steps)
        b["yaw"] = np.interp(dist, sa, _yaw_from_path(xg, yg))
    return b, float(scale)


def _tame_heading(a: dict[str, np.ndarray], lim: Limits) -> dict[str, np.ndarray]:
    """If only the heading turns too fast (a rebuilt line folds for a few metres where it is pinned to the
    inside of a very tight bend), smooth the heading alone - progressively, only as much as needed. The
    heading steers the follower's turning couple and the placement direction; positions stay as they are."""
    if "yaw" not in a:
        return a
    dt = float(np.median(np.diff(a["t"])))
    for k in (1, 2, 4, 8, 16):
        r = np.abs(np.diff(np.unwrap(a["yaw"]))) / dt
        if r.max() <= lim.yaw_rate_max:
            return a
        a = dict(a, yaw=_smooth(np.unwrap(a["yaw"]), k))
    return a


def prepare_playback(a: dict[str, np.ndarray], lim: Limits = LIMITS, dt: float = 0.04,
                     what: str = "lap", track=None) -> tuple[dict[str, np.ndarray], list[str]]:
    """Make a lap's arrays safe for the game (see module text). Returns (arrays, notes about repairs).

    Arrays: ``t, x, y`` required; ``yaw, speed`` and any others (steer, thr, brk, gear, rpm) optional and
    carried along. A lap without problems is returned as it came (same values, no notes). ``track``
    (the circuit, optional) allows the best repair for badly timed positions.

    Repairs, in order of preference (each only if the previous one did not make the lap safe):
    even time steps; the line kept and the car placed along it by its recorded speed (on the circuit,
    then along the line itself); the path smoothed in time (cuts corners - last resort); refuse.
    """
    a = {k: np.asarray(v, float) for k, v in a.items()}
    off = lambda arr: track is not None and beyond_edge_m(arr, track) > CIRCUIT_SLACK_M  # noqa: E731
    if not problems(a, lim) and not off(a):
        return a, []
    a, notes = _clean(a)
    if len(a["t"]) < lim.min_samples:
        raise PlaybackError(f"{what}: only {len(a['t'])} usable samples")
    if not problems(a, lim) and not off(a):
        return a, notes
    was_off = beyond_edge_m(a, track) if off(a) else None
    dts = np.diff(a["t"])
    if dts.min() < lim.dt_min or dts.max() > lim.dt_max or np.ptp(dts) > 1e-3:
        a = _resample(a, dt)
        notes.append(f"resampled to even {dt:.2f} s steps (were {dts.min():.3f}-{dts.max():.3f} s)")
    step = float(np.median(np.diff(a["t"])))
    path_bad = lambda arr: [p for p in problems(arr, lim) if p.startswith(("path", "heading"))]  # noqa: E731
    yaw_given = "yaw" in a
    has_speed = "speed" in a and np.any(a["speed"] > 0)
    if (path_bad(a) or was_off is not None) and has_speed:
        retimers = ([(_retime_on_track, (track,), "its line on the circuit")] if track is not None else []) \
            + ([(_retime_along_path, (), "its own line")] if was_off is None else [])   # that one cannot keep it on
        for sm in _LINE_SMOOTH_M:                      # least smoothing first; the circuit before the bare line
            done = False
            for fn, extra, name in retimers:
                try:
                    b, scale = fn(a, *extra, smooth_m=sm)
                    b = _tame_heading(b, lim)
                except Exception:                      # a repair that cannot run is simply not used
                    continue
                if not path_bad(b) and not off(b) and 0.9 < scale < 1.1:
                    a, done = b, True
                    notes.append(f"placed along {name} (wiggles under {sm:.0f} m smoothed) by the recorded "
                                 f"speed (distance scale {scale:.3f}): the time stamps of the positions were not usable")
                    if was_off is not None:
                        notes.append(f"its line left this circuit by up to {was_off:.1f} m (driven on another "
                                     "layout?) and was kept within 1 m of the edges")
                    break
            if done:
                break
    if was_off is not None and off(a):
        raise PlaybackError(f"{what}: its line leaves this circuit by up to {was_off:.1f} m and could not be "
                            "brought onto it (driven on another layout?)")
    if path_bad(a):
        raw_x, raw_y = a["x"], a["y"]
        for win in _SMOOTH_S:
            k = max(1, int(round(win / step / 2)))
            b = dict(a, x=_smooth(raw_x, k), y=_smooth(raw_y, k))
            if yaw_given:
                # the heading must agree with the (smoothed) path unless the car was really sliding:
                # keep the recorded heading where it is plausible, the path's direction where not
                py = _yaw_from_path(b["x"], b["y"])
                ry = np.unwrap(a["yaw"])
                ry = ry + 2 * np.pi * np.round((py - ry) / (2 * np.pi))       # same turn count as the path
                turn = np.abs(np.gradient(ry)) / step
                b["yaw"] = np.where((np.abs(ry - py) < 0.8) & (turn < lim.yaw_rate_max), ry, py)
                b["yaw"] = _smooth(b["yaw"], k)
            if not path_bad(b):
                a = b
                notes.append(f"path smoothed over {2 * k + 1} samples ({(2 * k + 1) * step:.2f} s): it was not drivable as recorded")
                break
        else:
            raise PlaybackError(f"{what}: cannot be made safe to play ({'; '.join(path_bad(a))})")
    left = problems(a, lim) + ([f"line {beyond_edge_m(a, track):.1f} m beyond the circuit's edge"] if off(a) else [])
    if left:
        raise PlaybackError(f"{what}: cannot be made safe to play ({'; '.join(left)})")
    return a, notes


CIRCUIT_SLACK_M = 3.0      # m: how far the car's centre may run past the circuit's edge before the line is moved


def beyond_edge_m(a: dict[str, np.ndarray], track) -> float:
    """Furthest the car's centre goes past the circuit's edge (m; negative = always inside)."""
    from scipy.spatial import cKDTree
    _, i = cKDTree(track.xy).query(np.column_stack([a["x"], a["y"]]))
    n = (a["x"] - track.xy[i, 0]) * track.normal[i, 0] + (a["y"] - track.xy[i, 1]) * track.normal[i, 1]
    return float(np.max(np.maximum(n - track.w_left[i], -n - track.w_right[i])))


def check_body(mass_kg: float, yaw_inertia_kgm2: float) -> tuple[float, float]:
    """Mass and yaw inertia the follower pushes with: finite and within the range of real cars."""
    m = float(mass_kg) if np.isfinite(mass_kg) else 1400.0
    iz = float(yaw_inertia_kgm2) if np.isfinite(yaw_inertia_kgm2) else 2200.0
    return float(np.clip(m, 300.0, 5000.0)), float(np.clip(iz, 100.0, 20000.0))
