"""Racing-line optimisation: where on the track should the car drive?

A line is described by its sideways offset ``n(s)`` from the track's centre
line (positive = to the left of the direction of travel).  The car's centre
must stay at least ``margin`` away from both track edges, so

    -(w_right - margin)  <=  n  <=  +(w_left - margin).

Two optimisers are provided.

(a) Minimum curvature
---------------------
Corner speed is limited by ``v^2 * curvature <= ay_max``, so a line that bends
as little as possible can be driven faster.  We minimise the integral of
curvature squared.  Curvature depends on ``n`` in a complicated way, but if we
freeze the direction of the line (taken from the previous guess) it becomes a
*linear* function of ``n``, and "sum of squares of something linear, subject to
simple bounds" is a quadratic programme with one global optimum that can be
solved exactly.  We then update the frozen direction and solve again; a
handful of rounds is enough.

(b) Minimum time
----------------
Minimum curvature is not quite the fastest line: it treats every metre of the
lap alike, while the stopwatch cares most about the places where the car is
slow and on the grip limit, and about the length of the path.  So, starting
from the minimum-curvature line, we measure with the speed-profile solver how
much lap time one unit of curvature costs at each point, and solve the same
quadratic programme again with those costs as weights (plus a term for path
length).  Speeds and weights are re-measured on the new line and the process
repeats; a round is kept only if the real lap time drops.  See
:func:`min_time_offsets`.

Working grid
------------
Both optimisers work on a coarse grid (a point every ``step_m`` ~ 3 m); the
final line is a periodic cubic spline through the optimised points, sampled at
every station of the track.  That keeps long tracks (7 km = 7000 stations)
fast and makes the result smooth: its curvature is computed analytically from
the spline instead of by differencing noisy points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import splu

from mv.planning.ggv import GGV
from mv.planning.speed_profile import lap_times, solve_speed_profile
from mv.track.track import Track

INSIDE_FRAC = 0.7      # share of the centre-line radius the line may use on the inside of a corner

__all__ = ["Line", "CoarseGrid", "offset_limits", "make_grid", "centre_line", "line_from_offsets",
           "min_curvature_offsets", "min_time_offsets", "solve_box_qp", "discrete_curvature"]


@dataclass
class Line:
    """A closed path given at the M stations of a track."""
    n: np.ndarray           # (M,) offset from the centre line, m (+ = left)
    xy: np.ndarray          # (M, 2) coordinates
    kappa: np.ndarray       # (M,) curvature, 1/m (+ = left turn)
    ds: np.ndarray          # (M,) distance to the next point along the line, m
    meta: dict = field(default_factory=dict)

    @property
    def length(self) -> float:
        return float(self.ds.sum())


@dataclass
class CoarseGrid:
    """Every ``step``-th station of the track: the optimisation variables live here."""
    idx: np.ndarray         # (K,) station indices into the track
    p: np.ndarray           # (K, 2) centre-line points
    nrm: np.ndarray         # (K, 2) unit normals (pointing left)
    lo: np.ndarray          # (K,) lowest allowed offset (<= 0)
    hi: np.ndarray          # (K,) highest allowed offset (>= 0)
    m: int                  # number of stations of the full track

    @property
    def k(self) -> int:
        return len(self.idx)


# ==========================================================================
# Geometry helpers
# ==========================================================================
def offset_limits(track: Track, margin: float, inside_frac: float = INSIDE_FRAC) -> tuple[np.ndarray, np.ndarray]:
    """(lowest, highest) allowed offset at every station for a given edge margin.

    Two limits apply.  The obvious one: stay ``margin`` inside the track edge.
    The subtle one: on the INSIDE of a corner the offset must stay well below
    the centre line's radius of curvature.  At ``n = R`` all the normals of the
    corner meet in one point (the centre of the turn); beyond it the
    description "offset along the normal" folds over itself - and so does the
    track's own inner edge.  That only binds where a track is wider than its
    centre-line radius (e.g. La Source at Spa: radius 8 m, half-width 11 m);
    there the line may use at most ``inside_frac`` of the radius.
    """
    lo, hi = -np.maximum(track.w_right - margin, 0.0), np.maximum(track.w_left - margin, 0.0)
    # tightest curvature within +-5 m, so the limit does not flicker from station to station
    k = max(int(round(5.0 / track.ds)), 1)
    idx = (np.arange(track.n)[:, None] + np.arange(-k, k + 1)[None, :]) % track.n
    kappa = track.kappa[idx]
    room_left = inside_frac / np.maximum(kappa.max(1), 1e-9)       # left-hand corner: the inside is to the left
    room_right = inside_frac / np.maximum((-kappa).max(1), 1e-9)
    return np.maximum(lo, -room_right), np.minimum(hi, room_left)


def make_grid(track: Track, margin: float, step_m: float = 3.0, shrink: float = 0.03) -> CoarseGrid:
    """Coarse grid with conservative bounds.

    Between two coarse points the spline may bulge a little and the track may
    narrow, so each coarse bound is the tightest fine bound in its
    neighbourhood, pulled in by a further ``shrink`` metres.
    """
    m = track.n
    k = max(int(round(track.length / step_m)), 8)
    idx = np.unique(np.round(np.linspace(0.0, m, k, endpoint=False)).astype(int))
    lo, hi = offset_limits(track, margin)
    half = int(np.ceil(m / len(idx)))
    win = (idx[:, None] + np.arange(-half, half + 1)[None, :]) % m
    lo_c = np.minimum(lo[win].max(1) + shrink, 0.0)
    hi_c = np.maximum(hi[win].min(1) - shrink, 0.0)
    return CoarseGrid(idx=idx, p=track.xy[idx], nrm=track.normal[idx], lo=lo_c, hi=hi_c, m=m)


def discrete_curvature(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Curvature and segment length of closed polylines, shape (..., K, 2) -> (..., K) each.

    Three-point formula: with r' and r'' the first and second differences of
    the points, curvature = cross(r', r'') / |r'|^3.  It does not require the
    points to be evenly spaced.
    """
    nxt, prv = np.roll(points, -1, axis=-2), np.roll(points, 1, axis=-2)
    d1, d2 = 0.5 * (nxt - prv), nxt - 2.0 * points + prv
    speed = np.hypot(d1[..., 0], d1[..., 1])
    kappa = (d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]) / speed ** 3
    seg = nxt - points
    return kappa, np.hypot(seg[..., 0], seg[..., 1])


def _closed_spline(points: np.ndarray) -> tuple[CubicSpline, np.ndarray]:
    """Periodic cubic spline through closed points, parametrised by chord length."""
    closed = np.vstack([points, points[:1]])
    t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    return CubicSpline(t, closed, bc_type="periodic"), t


def _spline_curvature(spline: CubicSpline, t: np.ndarray) -> np.ndarray:
    d1, d2 = spline(t, 1), spline(t, 2)
    return (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.hypot(d1[:, 0], d1[:, 1]) ** 3


def centre_line(track: Track, grid: CoarseGrid) -> Line:
    """The centre line as a :class:`Line`.

    Its curvature comes from a spline through the coarse centre points, i.e. it
    is smoothed exactly like the optimised lines, so lap times are comparable.
    """
    spline, t = _closed_spline(grid.p)
    t_fine = np.interp(np.arange(track.n), np.append(grid.idx, track.n), t)
    return Line(n=np.zeros(track.n), xy=track.xy.copy(), kappa=_spline_curvature(spline, t_fine),
                ds=np.full(track.n, track.ds), meta={"kind": "centre"})


def line_from_offsets(track: Track, grid: CoarseGrid, alpha: np.ndarray, margin: float) -> Line:
    """Turn offsets on the coarse grid into a smooth line at every track station.

    1. Put a periodic cubic spline through the offset coarse points.
    2. For every station find where the spline crosses that station's normal
       (a few Newton steps) - that gives the offset ``n`` there.
    3. Curvature is evaluated analytically on the spline.

    Finally ``n`` is clipped to the allowed band as a hard guarantee (with the
    conservative coarse bounds this changes nothing or only millimetres; the
    largest correction is reported in ``meta['clip_max_m']``).
    """
    m = track.n
    spline, t = _closed_spline(grid.p + grid.nrm * alpha[:, None])
    period = t[-1]
    tangent = np.stack([np.cos(track.psi), np.sin(track.psi)], axis=1)
    normal = track.normal
    tt = np.interp(np.arange(m), np.append(grid.idx, m), t)             # first guess
    for _ in range(6):
        r, dr = spline(tt % period), spline(tt % period, 1)
        along = ((r - track.xy) * tangent).sum(1)                       # how far past the normal we are
        tt = tt - along / np.maximum((dr * tangent).sum(1), 0.2)
    ok = bool(np.all(np.diff(tt) > 0) and np.abs(along).max() < 1e-3)
    lo, hi = offset_limits(track, margin)
    if ok:
        n_raw = ((spline(tt % period) - track.xy) * normal).sum(1)
        kappa = _spline_curvature(spline, tt % period)
    else:
        # Fallback for pathological geometry (normals crossing inside a very tight corner):
        # interpolate the offsets along the centre line and difference the points.
        n_raw = CubicSpline(np.append(grid.idx, m), np.append(alpha, alpha[0]), bc_type="periodic")(np.arange(m))
        kappa = None
    n = np.clip(n_raw, lo, hi)
    xy = track.xy + normal * n[:, None]
    if kappa is None:
        kappa = discrete_curvature(xy)[0]
    ds = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    return Line(n=n, xy=xy, kappa=kappa, ds=ds,
                meta={"clip_max_m": float(np.abs(n - n_raw).max()), "projection_ok": ok})


# ==========================================================================
# (a) minimum curvature
# ==========================================================================
def solve_box_qp(h: sp.spmatrix, g: np.ndarray, lo: np.ndarray, hi: np.ndarray, x0: np.ndarray,
                 tol: float = 1e-9, max_iter: int = 200) -> tuple[np.ndarray, int]:
    """Minimise ``0.5 x'Hx + g'x`` subject to ``lo <= x <= hi`` (H sparse, positive definite).

    Projected Newton method: variables that sit on a bound and want to push
    through it are held; the others take an exact Newton step (one sparse
    solve); the step is shortened until the projected point really is better.
    Converges in a few dozen iterations even for thousands of variables,
    where a general dense solver would take minutes.
    """
    h = sp.csc_matrix(h)
    x = np.clip(x0, lo, hi)
    scale = max(float(np.abs(g).max()), 1e-12)

    def cost(z):
        return 0.5 * z @ (h @ z) + g @ z

    it = 0
    for it in range(max_iter):
        grad = h @ x + g
        held = ((x <= lo + 1e-10) & (grad > 0.0)) | ((x >= hi - 1e-10) & (grad < 0.0))
        free = np.nonzero(~held)[0]
        if len(free) == 0 or np.abs(grad[free]).max() < tol * scale:
            break
        step = np.zeros_like(x)
        step[free] = -splu(h[free][:, free].tocsc()).solve(grad[free])
        f0, t = cost(x), 1.0
        while True:
            x_new = np.clip(x + t * step, lo, hi)
            if cost(x_new) <= f0 + 1e-4 * grad @ (x_new - x) or t < 1e-8:
                break
            t *= 0.5
        if np.abs(x_new - x).max() < 1e-10:
            break
        x = x_new
    return x, it + 1


def _curvature_qp(grid: CoarseGrid, alpha: np.ndarray, weight: np.ndarray | None = None) -> tuple[sp.spmatrix, np.ndarray]:
    """Quadratic model of ``sum(weight * curvature^2 * ds)`` around the line ``alpha``.

    With the line's direction frozen, the curvature at point i is
    ``nh_i . (r_{i-1} - 2 r_i + r_{i+1}) / |r'_i|^2`` where ``nh`` is the unit
    normal of the current line and ``r = p + nrm * alpha`` - linear in alpha.
    Returns (H, g) of the cost ``0.5 a'Ha + g'a`` (constant dropped).
    """
    k = grid.k
    r = grid.p + grid.nrm * alpha[:, None]
    d1 = 0.5 * (np.roll(r, -1, axis=0) - np.roll(r, 1, axis=0))
    speed = np.hypot(d1[:, 0], d1[:, 1])
    nh = np.stack([-d1[:, 1], d1[:, 0]], axis=1) / speed[:, None]
    i = np.arange(k)
    d2 = sp.csr_matrix((np.concatenate([np.full(k, -2.0), np.ones(k), np.ones(k)]),
                        (np.concatenate([i, i, i]), np.concatenate([i, (i + 1) % k, (i - 1) % k]))), shape=(k, k))
    a = (sp.diags(nh[:, 0] / speed ** 2) @ d2 @ sp.diags(grid.nrm[:, 0])
         + sp.diags(nh[:, 1] / speed ** 2) @ d2 @ sp.diags(grid.nrm[:, 1]))
    c = (nh[:, 0] * (d2 @ grid.p[:, 0]) + nh[:, 1] * (d2 @ grid.p[:, 1])) / speed ** 2
    w = speed if weight is None else speed * weight        # arc length belonging to each point (x weight)
    h = (a.T @ sp.diags(w) @ a).tocsc()
    h = h + sp.identity(k, format="csc") * (1e-10 * h.diagonal().mean())
    return h, a.T @ (w * c)


def min_curvature_offsets(grid: CoarseGrid, iterations: int = 8, x0: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Offsets on the coarse grid that minimise the integral of curvature squared.

    Each round solves the quadratic programme built around the current line.
    Because the programme is only a model of the true (non-quadratic) cost,
    the step towards its solution is shortened if the true cost does not drop.
    """
    def true_cost(al):
        kappa, ds = discrete_curvature(grid.p + grid.nrm * al[:, None])
        return float((kappa ** 2 * ds).sum())

    alpha = np.clip(np.zeros(grid.k) if x0 is None else x0, grid.lo, grid.hi)
    cost = true_cost(alpha)
    history = [cost]
    for _ in range(iterations):
        h, g = _curvature_qp(grid, alpha)
        target, _ = solve_box_qp(h, g, grid.lo, grid.hi, alpha)
        for step in (1.0, 0.5, 0.25, 0.1):
            trial = alpha + step * (target - alpha)
            trial_cost = true_cost(trial)
            if trial_cost < cost:
                break
        else:
            break                                          # no shorter step helps: converged
        change = float(np.abs(trial - alpha).max())
        alpha, cost = trial, trial_cost
        history.append(cost)
        if change < 2e-3:
            break
    return alpha, {"cost_history": history}


# ==========================================================================
# (b) minimum time
# ==========================================================================
def _coarse_lap(grid: CoarseGrid, alpha: np.ndarray, ggv: GGV) -> np.ndarray:
    """Lap time of one or several lines given on the coarse grid (alpha: (K,) or (B, K))."""
    alpha = np.atleast_2d(alpha)
    kappa, ds = discrete_curvature(grid.p[None] + grid.nrm[None] * alpha[..., None])
    return lap_times(kappa, ds, ggv)


def _time_sensitivity(kappa: np.ndarray, ds: np.ndarray, ggv: GGV, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray, float]:
    """How much lap time each point's curvature costs: ``dT / d|kappa_i|`` (s per 1/m).

    Finite differences on the real speed-profile solver: make the path a
    little tighter at ONE point, recompute the lap, note the change.  All K
    perturbed laps are solved in one batch.  Returns the sensitivities, the
    unperturbed speed profile and lap time.
    """
    k = len(kappa)
    kap = np.abs(kappa)
    delta = 0.03 * kap + 3e-5
    base = solve_speed_profile(kappa, ds, ggv, order=1)     # first order is plenty for a slope and twice as fast
    sens = np.empty(k)
    for start in range(0, k, chunk):
        rows = np.arange(start, min(start + chunk, k))
        batch = np.repeat(kap[None, :], len(rows), axis=0)
        batch[np.arange(len(rows)), rows] += delta[rows]
        sens[rows] = (lap_times(batch, np.broadcast_to(ds, batch.shape), ggv, order=1) - base.lap_time) / delta[rows]
    return np.maximum(sens, 0.0), base.v, base.lap_time


def min_time_offsets(grid: CoarseGrid, ggv: GGV, alpha0: np.ndarray, iterations: int = 20,
                     kappa_floor: float = 2e-3, weight_floor: float = 0.03, patience: int = 3,
                     min_gain: float = 2e-4, keep: int = 3) -> tuple[list[np.ndarray], dict]:
    """Refine a line towards minimum lap time by re-weighted curvature minimisation.

    The idea in one paragraph: minimum curvature treats every metre of the lap
    alike.  But the stopwatch does not - a bit of extra curvature in a slow
    hairpin where the car is on the grip limit costs far more time than the
    same curvature in a braking zone where the car is not limited by cornering
    grip at all.  So we *measure* how much lap time one unit of curvature
    costs at every point (``_time_sensitivity``), and minimise

        sum_i  w_i * kappa_i^2 * ds_i   +   sum_i  ds_i / v_i

    with ``w_i = (dT/d|kappa_i|) / |kappa_i|``.  The first term has, at the
    current line, exactly the gradient of the lap time with respect to the
    curvatures; the second is the time needed to cover the path length at the
    current speeds, so shortening the path where the car is slow is rewarded.
    Both are (after freezing the line's direction) quadratic/linear in the
    offsets, so each round is the same kind of bounded quadratic programme as
    minimum curvature - smooth lines and exact track limits come for free.
    The speeds and weights are then re-measured on the new line and the
    process repeats.  A line that no longer changes satisfies the first-order
    conditions of the true minimum-time problem (within the model's
    linearisation).

    Returns the ``keep`` fastest lines met on the way (fastest first, the
    starting line included if it is among them) and a dict with the lap-time
    history.  Lap times are measured on the coarse grid here; the caller
    should re-rank the few returned lines at full resolution.
    """
    alpha = np.clip(alpha0, grid.lo, grid.hi)
    t_best = float(_coarse_lap(grid, alpha, ggv)[0])
    seen, history, weight_prev, stall = [(t_best, alpha)], [t_best], None, 0
    for _ in range(iterations):
        r = grid.p + grid.nrm * alpha[:, None]
        kappa, ds = discrete_curvature(r)
        sens, v, _ = _time_sensitivity(kappa, ds, ggv)
        weight = sens / np.maximum(np.abs(kappa), kappa_floor)
        # smooth along the lap (a single point's sensitivity is noisy) and keep a small
        # uniform share so that every part of the line stays smooth
        weight = np.convolve(np.concatenate([weight[-2:], weight, weight[:2]]), [0.1, 0.2, 0.4, 0.2, 0.1], "valid")
        weight = weight + weight_floor * weight.mean()
        if weight_prev is not None:
            weight = 0.5 * (weight + weight_prev)          # damping: weights and line chase each other
        weight_prev = weight
        # path-length term: d(sum ds_i / v_i) / d alpha
        chord = np.roll(r, -1, axis=0) - r
        chord /= np.linalg.norm(chord, axis=1, keepdims=True)
        pull = (chord * grid.nrm).sum(1) / v               # moving point i outwards lengthens segment i-1, shortens i ...
        pull_next = (chord * np.roll(grid.nrm, -1, axis=0)).sum(1) / v
        length_grad = np.roll(pull_next, 1) - pull
        h, g = _curvature_qp(grid, alpha, weight / ds)     # sensitivities are per point, not per metre
        target, _ = solve_box_qp(h, g + length_grad, grid.lo, grid.hi, alpha)
        # try the full step and shorter ones (all in one batch), keep the fastest
        steps = np.array([1.0, 0.6, 0.3, 0.1])
        cand = alpha[None, :] + steps[:, None] * (target - alpha)[None, :]
        t_cand = _coarse_lap(grid, cand, ggv)
        j = int(np.argmin(t_cand))
        alpha = cand[j]
        history.append(float(t_cand[j]))
        # stop when `patience` rounds in a row gained less than `min_gain` (relative) on the best lap
        stall = 0 if t_cand[j] < t_best * (1.0 - min_gain) else stall + 1
        t_best = min(t_best, float(t_cand[j]))
        seen.append((float(t_cand[j]), alpha))
        if stall >= patience:
            break
    seen.sort(key=lambda item: item[0])
    return [a for _, a in seen[:keep]], {"coarse_time_history": history}
