"""Procedural race-track generator.

Why this exists
---------------
A driver that only ever sees one circuit memorises it.  Project MV's driver has
to handle *any* circuit, so during training every simulated car gets its own
randomly generated track and the pool of tracks is continuously refreshed.  The
policy can therefore only succeed by reading the road ahead.

How a track is made
-------------------
Two styles are mixed at random:

``polygon`` (most tracks) - straights joined by corners, like a real circuit.
    1. Scatter control points around a circle with random angles and radii.
       Joining them with straight lines gives a closed polygon: its edges are
       the straights, its vertices the corners.  Very different neighbouring
       radii produce sharp, hairpin-like vertices.
    2. Some straights get a chicane: two extra points pushed left and right.
    3. Round the corners by smoothing the polygon.  The smoothing length (in
       metres) sets how tight the corners are, and it varies along the lap so
       one track has both slow and fast corners.

``spline`` - one continuous flowing curve with no real straights.

Then, for both styles: scale to a random lap length, mirror at random (so
clockwise and counter-clockwise laps are equally likely), give it a random
width that varies gently along the lap, and reject the track if a corner is
tighter than ``min_radius`` or two parts of the circuit come too close.

Every random choice is drawn from the ranges in :class:`TrackGenConfig`, so
"Formula Student style" versus "Grand Prix style" is only a config change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from mv.track.track import Track

__all__ = ["TrackGenConfig", "random_track", "random_tracks", "PRESETS", "max_length_m"]


@dataclass
class TrackGenConfig:
    length_m: tuple[float, float] = (1200.0, 5500.0)   # lap length range
    width_m: tuple[float, float] = (8.0, 14.0)         # total width range
    width_variation: float = 0.15                      # +- fraction the width wanders along the lap
    min_radius_m: tuple[float, float] = (10.0, 25.0)   # tightest allowed corner (sampled per track)
    control_points: tuple[int, int] = (6, 16)          # number of shape control points
    radius_spread: tuple[float, float] = (0.2, 0.65)   # how different the control radii are (0 = circle)
    angle_jitter: float = 0.7                          # 0 = evenly spaced control points
    polygon_prob: float = 0.75                         # share of tracks built in the "polygon" style
    corner_smooth_m: tuple[float, float] = (8.0, 45.0) # corner rounding length (small = tight corners)
    chicane_prob: float = 0.25                         # chance that a straight gets a chicane
    chicane_offset_m: tuple[float, float] = (8.0, 25.0)
    wiggle_prob: float = 0.5                           # spline style: chance of chicane-like ripples
    wiggle_amp: tuple[float, float] = (0.01, 0.05)
    wiggle_freq: tuple[int, int] = (10, 40)
    min_clearance_m: float = 6.0                       # min gap between separate parts of the circuit
    ds: float = 1.0                                    # sample spacing of the produced track
    max_tries: int = 300


PRESETS: dict[str, TrackGenConfig] = {
    # Wide, fast circuits - road and GT cars.
    "circuit": TrackGenConfig(),
    # Short, narrow, twisty - similar in scale to Formula Student autocross/trackdrive.
    "fs": TrackGenConfig(length_m=(200.0, 500.0), width_m=(3.0, 5.0), width_variation=0.05,
                         min_radius_m=(4.5, 7.0), control_points=(6, 16), corner_smooth_m=(2.0, 8.0),
                         chicane_offset_m=(2.0, 5.0), chicane_prob=0.35, min_clearance_m=3.0, ds=0.5),
    # Everything in between, for maximum variety.
    "mixed": TrackGenConfig(length_m=(700.0, 6000.0), width_m=(6.0, 15.0), min_radius_m=(8.0, 30.0),
                            control_points=(5, 18), corner_smooth_m=(6.0, 50.0)),
}


def _control_points(rng: np.random.Generator, cfg: TrackGenConfig) -> tuple[np.ndarray, np.ndarray]:
    n = int(rng.integers(cfg.control_points[0], cfg.control_points[1] + 1))
    spread = rng.uniform(*cfg.radius_spread)
    base = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    ang = base + rng.uniform(-0.5, 0.5, n) * cfg.angle_jitter * (2 * np.pi / n)
    rad = 1.0 + spread * rng.uniform(-1.0, 1.0, n)
    return ang, rad


def _finish(rng: np.random.Generator, xy: np.ndarray) -> np.ndarray:
    """Random stretch, rotation and mirroring."""
    stretch, rot = rng.uniform(0.55, 1.0), rng.uniform(0, 2 * np.pi)
    x, y = xy[:, 0], xy[:, 1] * stretch
    xy = np.stack([x * np.cos(rot) - y * np.sin(rot), x * np.sin(rot) + y * np.cos(rot)], axis=1)
    return xy[::-1].copy() if rng.random() < 0.5 else xy


def _perimeter(xy: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1).sum())


def _spline_shape(rng: np.random.Generator, cfg: TrackGenConfig, target: float) -> np.ndarray:
    ang, rad = _control_points(rng, cfg)
    a = np.concatenate([ang, [ang[0] + 2 * np.pi]])
    r = np.concatenate([rad, [rad[0]]])
    theta = np.linspace(0.0, 2 * np.pi, 4000, endpoint=False)
    rr = CubicSpline(a, r, bc_type="periodic")(ang[0] + theta)
    if rng.random() < cfg.wiggle_prob:
        for _ in range(int(rng.integers(1, 4))):
            k = int(rng.integers(cfg.wiggle_freq[0], cfg.wiggle_freq[1] + 1))
            centre, span = rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 1.5)   # only part of the lap
            d = np.angle(np.exp(1j * (theta - centre)))
            ripple = np.sin(k * theta + rng.uniform(0, 2 * np.pi)) * np.exp(-(d / span) ** 2)
            rr = rr + rng.uniform(*cfg.wiggle_amp) * ripple
    rr = np.clip(rr, 0.2, None)
    xy = _finish(rng, np.stack([rr * np.cos(theta), rr * np.sin(theta)], axis=1))
    return xy * (target / _perimeter(xy))


def _polygon_shape(rng: np.random.Generator, cfg: TrackGenConfig, target: float) -> np.ndarray:
    ang, rad = _control_points(rng, cfg)
    pts = _finish(rng, np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1))
    pts = pts * (target / _perimeter(pts))
    # chicanes: on some straights add two points pushed to opposite sides
    out = []
    for i in range(len(pts)):
        p, q = pts[i], pts[(i + 1) % len(pts)]
        out.append(p)
        edge = q - p
        length = float(np.linalg.norm(edge))
        off = rng.uniform(*cfg.chicane_offset_m)
        if length > 12.0 * off and rng.random() < cfg.chicane_prob:
            nrm = np.array([-edge[1], edge[0]]) / length
            mid, sign = rng.uniform(0.35, 0.65), rng.choice([-1.0, 1.0])
            gap = 2.0 * off / length
            out.append(p + edge * (mid - gap) + nrm * off * sign)
            out.append(p + edge * (mid + gap) - nrm * off * sign)
    pts = np.asarray(out)
    # dense polyline, about 1 m spacing
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    m = max(int(s[-1]), 64)
    step = s[-1] / m
    sq = np.arange(m) * step
    dense = np.stack([np.interp(sq, s, closed[:, 0]), np.interp(sq, s, closed[:, 1])], axis=1)
    # round the corners with two different smoothing lengths, blended along the lap
    lo = rng.uniform(cfg.corner_smooth_m[0], 0.5 * (cfg.corner_smooth_m[0] + cfg.corner_smooth_m[1]))
    hi = rng.uniform(lo, cfg.corner_smooth_m[1])
    tight = gaussian_filter1d(dense, lo / step, axis=0, mode="wrap")
    wide = gaussian_filter1d(dense, hi / step, axis=0, mode="wrap")
    phase = np.linspace(0, 2 * np.pi, m, endpoint=False)
    mix = 0.5 + 0.5 * np.sin(rng.integers(1, 4) * phase + rng.uniform(0, 2 * np.pi))
    mix = np.clip(mix * rng.uniform(0.6, 1.6), 0.0, 1.0)[:, None]
    return tight * (1 - mix) + wide * mix


def _candidate(rng: np.random.Generator, cfg: TrackGenConfig) -> tuple[np.ndarray, float, float, str]:
    target = rng.uniform(*cfg.length_m)
    style = "polygon" if rng.random() < cfg.polygon_prob else "spline"
    xy = _polygon_shape(rng, cfg, target) if style == "polygon" else _spline_shape(rng, cfg, target)
    return xy, rng.uniform(*cfg.width_m), rng.uniform(*cfg.min_radius_m), style


def random_track(rng: np.random.Generator | int | None = None, cfg: TrackGenConfig | str = "circuit",
                 name: str | None = None) -> Track:
    """Generate one valid random track."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    if isinstance(cfg, str):
        # "mixed+gp" = pick one of the listed presets at random for each track
        names = cfg.split("+")
        choice = names[int(rng.integers(len(names)))] if len(names) > 1 else names[0]
        from mv.track.segments import SEGMENT_PRESETS, segment_track
        if choice in SEGMENT_PRESETS:
            return segment_track(rng, choice, name=name)
        cfg = PRESETS[choice]
    else:
        from mv.track.segments import SegmentGenConfig, segment_track
        if isinstance(cfg, SegmentGenConfig):
            return segment_track(rng, cfg, name=name)
    last = None
    for _attempt in range(cfg.max_tries):
        xy, width, min_r, style = _candidate(rng, cfg)
        m = len(xy)
        phase = np.linspace(0, 2 * np.pi, m, endpoint=False)
        wobble = 1.0 + cfg.width_variation * np.sin(rng.integers(1, 4) * phase + rng.uniform(0, 2 * np.pi))
        half = 0.5 * width * wobble
        t = Track.from_centerline(xy, half, half, ds=cfg.ds, name=name or "random",
                                  meta={"generator": style, "min_radius_target": float(min_r)})
        last = t
        # a corner tighter than half the width would fold the inner edge over itself
        if t.min_radius() < max(min_r, 0.6 * float(half.max()) + 1.0):
            continue
        if t.self_clearance() < cfg.min_clearance_m:
            continue
        return t
    raise RuntimeError(f"no valid track after {cfg.max_tries} tries (last min radius {last.min_radius():.1f} m)")


def random_tracks(count: int, seed: int | None = None, cfg: TrackGenConfig | str = "circuit") -> list[Track]:
    rng = np.random.default_rng(seed)
    return [random_track(rng, cfg, name=f"random_{i:04d}") for i in range(count)]


def max_length_m(preset: str) -> float:
    """Longest lap a preset (or a "a+b" mix of presets) can produce, in metres."""
    from mv.track.segments import SEGMENT_PRESETS
    out = 0.0
    for name in preset.split("+"):
        if name in SEGMENT_PRESETS:
            out = max(out, SEGMENT_PRESETS[name].max_length_m * 1.02)
        else:
            out = max(out, PRESETS[name].length_m[1] * 1.15)
    return out
