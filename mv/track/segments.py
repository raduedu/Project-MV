"""Segment-based track generator: circuits built from straights and corners.

Why a second generator
----------------------
The first generator (``generator.py``) scatters points around a centre and
smooths them.  Its circuits look reasonable but have two blind spots that
matter for racing (engineering log 3.3 and WP13):

* corners at the end of long straights are almost always wide and fast,
  because the smoothing that rounds a corner grows with the straight;
* chicanes get smoothed into gentle kinks.

So a driver trained only on those circuits never practises the hardest thing
on a real circuit: braking from top speed into a tight corner or chicane
(Monza's first chicane, Imola's Rivazza).  The first long training run
failed at exactly those two places.

How a circuit is built
----------------------
1. **Draw a sequence of features**, like a track designer would: straights of
   random length (short links up to 1.2 km), single corners (radius 8-250 m,
   10-200 degrees), hairpins, chicanes (two tight opposite corners), and
   "heavy braking" features (a long straight that ends in a tight corner or
   chicane).
2. **Make the turning add up to one full circle** (360 degrees), which any
   closed loop needs: the corners turning in the main direction are scaled.
3. **Write the curvature along the lap** (1/radius in corners, 0 on
   straights) and soften the corner entries/exits so curvature builds up
   gradually (as real circuits do with transition curves).
4. **Close the loop exactly**: after integrating the curvature the end point
   misses the start by some vector.  Each straight can be lengthened or
   shortened along its own direction; the smallest set of length changes that
   cancels the miss is a small least-squares problem.  Corners are untouched.
5. Reject circuits that cross themselves or come too close to themselves,
   exactly like the other generator.

Everything is drawn from :class:`SegmentGenConfig`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from mv.track.track import Track

__all__ = ["SegmentGenConfig", "segment_track", "SEGMENT_PRESETS"]


@dataclass
class SegmentGenConfig:
    n_features: tuple[int, int] = (6, 14)              # corners/chicanes per lap
    width_m: tuple[float, float] = (9.0, 15.0)
    width_variation: float = 0.12
    straight_m: tuple[float, float] = (30.0, 450.0)    # ordinary straights (log-uniform)
    long_straight_m: tuple[float, float] = (400.0, 1000.0)
    corner_radius_m: tuple[float, float] = (12.0, 250.0)   # log-uniform
    corner_angle_deg: tuple[float, float] = (15.0, 140.0)
    hairpin_radius_m: tuple[float, float] = (9.0, 18.0)
    hairpin_angle_deg: tuple[float, float] = (150.0, 200.0)
    chicane_radius_m: tuple[float, float] = (9.0, 30.0)
    chicane_angle_deg: tuple[float, float] = (30.0, 80.0)
    chicane_gap_m: tuple[float, float] = (0.0, 25.0)
    p_hairpin: float = 0.12
    p_chicane: float = 0.18
    p_heavy_braking: float = 0.3                        # feature preceded by a long straight
    p_counter_turn: float = 0.3                         # share of corners turning against the main direction
    transition_m: tuple[float, float] = (4.0, 15.0)     # softening of corner entry/exit
    min_straight_m: float = 15.0
    min_radius_m: float = 8.0
    min_clearance_m: float = 6.0
    max_length_m: float = 7500.0                       # longer laps are rejected (0 = no limit)
    ds: float = 1.0
    max_tries: int = 400


SEGMENT_PRESETS: dict[str, SegmentGenConfig] = {
    # Grand-prix-like circuits with long straights and heavy braking zones.
    "gp": SegmentGenConfig(),
    # Formula Student scale (rules D 8: width >= 3 m, straights <= 80 m,
    # hairpins >= 9 m outside diameter, lap 200-500 m).
    "fs_segments": SegmentGenConfig(
        n_features=(8, 16), width_m=(3.0, 4.5), width_variation=0.05, straight_m=(8.0, 45.0),
        long_straight_m=(50.0, 78.0), corner_radius_m=(5.0, 30.0), corner_angle_deg=(20.0, 150.0),
        hairpin_radius_m=(4.6, 7.0), hairpin_angle_deg=(150.0, 190.0), chicane_radius_m=(4.6, 10.0),
        chicane_angle_deg=(25.0, 70.0), chicane_gap_m=(0.0, 6.0), transition_m=(1.0, 4.0),
        min_straight_m=4.0, min_radius_m=4.5, min_clearance_m=3.0, max_length_m=600.0, ds=0.5),
}


def _loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _draw_features(rng: np.random.Generator, cfg: SegmentGenConfig) -> list[tuple[str, float, float]]:
    """Return the lap as ('S', length, 0) and ('C', radius, signed angle in rad) items, straights between corners."""
    items: list[tuple[str, float, float]] = []
    n = int(rng.integers(cfg.n_features[0], cfg.n_features[1] + 1))
    for _ in range(n):
        heavy = rng.random() < cfg.p_heavy_braking
        length = rng.uniform(*cfg.long_straight_m) if heavy else _loguniform(rng, *cfg.straight_m)
        items.append(("S", length, 0.0))
        sign = -1.0 if rng.random() < cfg.p_counter_turn else 1.0
        r = rng.random()
        if r < cfg.p_chicane:
            rad = rng.uniform(*cfg.chicane_radius_m)
            ang = math.radians(rng.uniform(*cfg.chicane_angle_deg))
            items.append(("C", rad, sign * ang))
            gap = rng.uniform(*cfg.chicane_gap_m)
            if gap > 0.5:
                items.append(("S", gap, 0.0))
            items.append(("C", rng.uniform(*cfg.chicane_radius_m), -sign * ang * rng.uniform(0.8, 1.25)))
        elif r < cfg.p_chicane + cfg.p_hairpin:
            items.append(("C", rng.uniform(*cfg.hairpin_radius_m), sign * math.radians(rng.uniform(*cfg.hairpin_angle_deg))))
        else:
            rad = _loguniform(rng, *cfg.corner_radius_m)
            if heavy:                                         # a long straight should end in a slow corner
                rad = min(rad, rng.uniform(cfg.corner_radius_m[0], 3.0 * cfg.corner_radius_m[0]))
            items.append(("C", rad, sign * math.radians(rng.uniform(*cfg.corner_angle_deg))))
    return items


def _normalise_turning(items: list[tuple[str, float, float]]) -> list[tuple[str, float, float]] | None:
    """Scale corners so that the total turning is +2*pi (a counter-clockwise loop)."""
    pos = sum(a for k, _, a in items if k == "C" and a > 0)
    neg = sum(a for k, _, a in items if k == "C" and a < 0)
    if pos <= 0:
        return None
    k = (2 * math.pi - neg) / pos
    if not 0.5 < k < 2.0:
        return None
    out = []
    for kind, a, b in items:
        if kind == "C" and b > 0:
            out.append((kind, a, b * k))
        else:
            out.append((kind, a, b))
    return out


def _curvature(items, ds: float, transition_m: float) -> tuple[np.ndarray, list[slice]]:
    """Curvature samples along the lap and the index range of every straight."""
    kappa, straight_idx, pos = [], [], 0
    for kind, a, b in items:
        if kind == "S":
            n = max(int(round(a / ds)), 1)
            kappa.append(np.zeros(n))
            straight_idx.append(slice(pos, pos + n))
        else:
            n = max(int(round(a * abs(b) / ds)), 1)
            # exact heading change: n samples of curvature k give n * k * ds = angle
            kappa.append(np.full(n, b / (n * ds)))
        pos += n
    k = np.concatenate(kappa)
    if transition_m > 0:                     # smoothing keeps the total turning, spreads it over entry/exit
        k = gaussian_filter1d(k, transition_m / ds, mode="wrap")
    return k, straight_idx


def _integrate(kappa: np.ndarray, ds: float) -> np.ndarray:
    psi = np.concatenate([[0.0], np.cumsum(kappa[:-1] * ds)]) + 0.5 * kappa * ds
    return np.concatenate([[[0.0, 0.0]], np.cumsum(np.stack([np.cos(psi), np.sin(psi)], 1) * ds, 0)])[:-1], psi


def _closed_centerline(rng: np.random.Generator, cfg: SegmentGenConfig) -> np.ndarray | None:
    items = _normalise_turning(_draw_features(rng, cfg))
    if items is None:
        return None
    ds = cfg.ds
    transition = rng.uniform(*cfg.transition_m)
    for _ in range(8):                                      # converges in ~3-5 passes
        kappa, straights = _curvature(items, ds, transition)
        xy, psi = _integrate(kappa, ds)
        miss = xy[-1] + np.array([math.cos(psi[-1]), math.sin(psi[-1])]) * ds - xy[0]
        if np.linalg.norm(miss) < 2.0:
            # remove the last sub-metre gap by spreading it evenly along the lap
            frac = np.arange(len(xy))[:, None] / len(xy)
            xy = xy - frac * miss[None]
            if cfg.max_length_m and len(xy) * ds > cfg.max_length_m:
                return None
            return xy
        # Straight directions -> matrix D (2 x n); lengths change by delta with
        # D @ delta = -miss.  Straights point all the way round the compass (the lap
        # turns through 360 degrees), so the miss can always be cancelled while only
        # *lengthening* most of them.  Bounded least squares: closure weighted
        # heavily, plus a small penalty on the size of the changes; no straight may
        # become shorter than ``min_straight_m``.
        from scipy.optimize import lsq_linear
        # direction at the MIDDLE of each straight: the ends are still bent by the corner transitions
        mid = [(s.start + s.stop) // 2 for s in straights]
        dirs = np.array([[math.cos(psi[i]), math.sin(psi[i])] for i in mid]).T
        lengths = np.array([a for kind, a, _ in items if kind == "S"])
        w = 100.0
        A = np.vstack([w * dirs, 0.05 * np.eye(len(lengths))])
        rhs = np.concatenate([-w * miss, np.zeros(len(lengths))])
        sol = lsq_linear(A, rhs, bounds=(cfg.min_straight_m - lengths, np.inf))
        delta = sol.x
        if np.linalg.norm(dirs @ delta + miss) > 1.0:
            return None
        j = 0
        new = []
        for kind, a, b in items:
            if kind == "S":
                a = a + float(delta[j])
                j += 1
            new.append((kind, a, b))
        items = new
    return None


def segment_track(rng: np.random.Generator | int | None = None, cfg: SegmentGenConfig | str = "gp",
                  name: str | None = None) -> Track:
    """Generate one valid circuit from straights, corners, hairpins and chicanes."""
    # a track made from a plain seed + preset name can be made again exactly: remember the recipe
    recipe = ({"generator": "segment_track", "seed": int(rng), "preset": cfg}
              if isinstance(rng, (int, np.integer)) and isinstance(cfg, str) else None)
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    if isinstance(cfg, str):
        cfg = SEGMENT_PRESETS[cfg]
    for _ in range(cfg.max_tries):
        xy = _closed_centerline(rng, cfg)
        if xy is None:
            continue
        # random orientation and direction of travel
        rot = rng.uniform(0, 2 * np.pi)
        c, s = math.cos(rot), math.sin(rot)
        xy = xy @ np.array([[c, s], [-s, c]])
        if rng.random() < 0.5:
            xy = xy[::-1].copy()
        m = len(xy)
        width = rng.uniform(*cfg.width_m)
        phase = np.linspace(0, 2 * np.pi, m, endpoint=False)
        half = 0.5 * width * (1.0 + cfg.width_variation * np.sin(rng.integers(1, 4) * phase + rng.uniform(0, 2 * np.pi)))
        t = Track.from_centerline(xy, half, half, ds=cfg.ds, name=name or "segments", meta={"generator": "segments"})
        if recipe is not None:
            t.meta["recipe"] = recipe
        if t.min_radius() < max(cfg.min_radius_m, 0.6 * float(half.max()) + 0.5):
            continue
        if t.self_clearance() < cfg.min_clearance_m:
            continue
        return t
    raise RuntimeError(f"no valid segment track after {cfg.max_tries} tries")
