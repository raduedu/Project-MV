"""Track representation.

A track is a closed centre line sampled every ``ds`` metres, plus the distance
from the centre line to the left and right edge at every sample.  Everything
else (heading, curvature, edge coordinates) is derived from that.

Frames
------
World frame is 2-D, right-handed: looking at the map from above, x points right
and y points up, so a positive rotation is counter-clockwise.  "Left" of the
track is to the left of the direction of travel.  Curvature ``kappa`` is
positive in a left-hand corner; its unit is 1/m (kappa = 1/radius).

Two classes
-----------
:class:`Track`       one track, NumPy arrays, used for building/saving/plotting.
:class:`TrackBatch`  many tracks stacked into padded Torch tensors so that
                     thousands of simulated cars can each drive a different
                     track on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

__all__ = ["Track", "TrackBatch"]


def _periodic_smooth(a: np.ndarray, sigma_pts: float) -> np.ndarray:
    """Gaussian smoothing of a periodic signal (works on (M,) or (M, k))."""
    if sigma_pts <= 0:
        return a
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(a, sigma_pts, axis=0, mode="wrap")


def _resample_closed(xy: np.ndarray, extra: list[np.ndarray], ds: float) -> tuple[np.ndarray, list[np.ndarray], float]:
    """Resample a closed polyline to (almost exactly) ``ds`` spacing.

    Returns the new points, the resampled per-point ``extra`` arrays and the
    exact spacing used (total length / number of points).
    """
    closed = np.vstack([xy, xy[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    keep = np.concatenate([[True], seg[:-1] > 1e-6]) if len(seg) > 1 else np.array([True])
    xy, extra = xy[keep[: len(xy)]], [e[keep[: len(e)]] for e in extra]
    closed = np.vstack([xy, xy[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    m = max(int(round(total / ds)), 8)
    s_new = np.arange(m) * (total / m)
    out_xy = np.stack([np.interp(s_new, s, closed[:, 0]), np.interp(s_new, s, closed[:, 1])], axis=1)
    out_extra = [np.interp(s_new, s, np.concatenate([e, e[:1]])) for e in extra]
    return out_xy, out_extra, total / m


@dataclass
class Track:
    name: str
    xy: np.ndarray            # (M, 2) centre line, uniform spacing, closed (last point connects to first)
    w_left: np.ndarray        # (M,) distance from centre line to left edge
    w_right: np.ndarray       # (M,) distance from centre line to right edge
    ds: float                 # spacing between samples in metres
    psi: np.ndarray = field(default=None, repr=False)      # (M,) heading of the centre line
    kappa: np.ndarray = field(default=None, repr=False)    # (M,) curvature
    meta: dict = field(default_factory=dict)

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_centerline(cls, xy, w_left, w_right, ds: float = 1.0, name: str = "track",
                        smooth_m: float = 0.0, meta: dict | None = None) -> "Track":
        """Build a track from any closed polyline (points need not be evenly spaced).

        ``smooth_m`` applies a Gaussian filter of that many metres to the centre
        line, which is needed for measured/imported lines whose point-to-point
        noise would otherwise produce a wildly jagged curvature.
        """
        xy = np.asarray(xy, float)
        w_left = np.broadcast_to(np.asarray(w_left, float), (len(xy),)).copy()
        w_right = np.broadcast_to(np.asarray(w_right, float), (len(xy),)).copy()
        if np.linalg.norm(xy[0] - xy[-1]) < 1e-6:        # drop duplicated closing point
            xy, w_left, w_right = xy[:-1], w_left[:-1], w_right[:-1]
        xy, (w_left, w_right), ds_exact = _resample_closed(xy, [w_left, w_right], ds)
        if smooth_m > 0:
            xy = _periodic_smooth(xy, smooth_m / ds_exact)
            w_left = _periodic_smooth(w_left, smooth_m / ds_exact)
            w_right = _periodic_smooth(w_right, smooth_m / ds_exact)
            xy, (w_left, w_right), ds_exact = _resample_closed(xy, [w_left, w_right], ds)
        t = cls(name=name, xy=xy, w_left=w_left, w_right=w_right, ds=ds_exact, meta=meta or {})
        t._derive()
        return t

    def _derive(self) -> None:
        d = np.roll(self.xy, -1, axis=0) - np.roll(self.xy, 1, axis=0)     # central difference
        self.psi = np.arctan2(d[:, 1], d[:, 0])
        dpsi = np.roll(self.psi, -1) - np.roll(self.psi, 1)
        dpsi = (dpsi + np.pi) % (2 * np.pi) - np.pi                         # unwrap
        self.kappa = dpsi / (2.0 * self.ds)

    # ---- derived geometry ---------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.xy)

    @property
    def length(self) -> float:
        return self.n * self.ds

    @property
    def normal(self) -> np.ndarray:
        """Unit vector pointing to the LEFT of the direction of travel."""
        return np.stack([-np.sin(self.psi), np.cos(self.psi)], axis=1)

    @property
    def left_xy(self) -> np.ndarray:
        return self.xy + self.normal * self.w_left[:, None]

    @property
    def right_xy(self) -> np.ndarray:
        return self.xy - self.normal * self.w_right[:, None]

    def turning_direction(self) -> str:
        x, y = self.xy[:, 0], self.xy[:, 1]
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        return "counter-clockwise" if area > 0 else "clockwise"

    # ---- validity checks used by the random generator ---------------------------
    def min_radius(self) -> float:
        return float(1.0 / max(np.abs(self.kappa).max(), 1e-9))

    def self_clearance(self, stride_m: float = 4.0) -> float:
        """Smallest gap between two different parts of the track, edge to edge.

        Negative means the tarmac of one section overlaps another section.
        Points closer than 5 track-widths along the lap are not compared
        (neighbouring points are obviously close to each other).
        """
        step = max(int(round(stride_m / self.ds)), 1)
        p = self.xy[::step]
        half = np.maximum(self.w_left, self.w_right)[::step]
        m = len(p)
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2) - half[:, None] - half[None, :]
        idx = np.arange(m)
        sep = np.abs(idx[:, None] - idx[None, :])
        sep = np.minimum(sep, m - sep) * step * self.ds
        exclude = sep < max(5.0 * 2.0 * float(half.max()), 40.0)
        d[exclude] = np.inf
        return float(d.min())

    # ---- persistence ----------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, name=self.name, xy=self.xy, w_left=self.w_left, w_right=self.w_right,
                            ds=self.ds, meta=np.array([repr(self.meta)]))

    @classmethod
    def load(cls, path: str | Path) -> "Track":
        z = np.load(path, allow_pickle=False)
        import ast
        t = cls(name=str(z["name"]), xy=z["xy"], w_left=z["w_left"], w_right=z["w_right"], ds=float(z["ds"]),
                meta=ast.literal_eval(str(z["meta"][0])) if "meta" in z else {})
        t._derive()
        return t


class TrackBatch:
    """A set of tracks as padded tensors: ``tensor[track_id, point_index]``.

    Tracks have different lengths, so each tensor is padded to the longest
    track and ``self.n[track_id]`` holds the real number of points.  All
    indexing goes through :meth:`wrap`, which applies the per-track modulo, so
    padded entries are never read.

    All tracks in a batch are resampled to the same spacing ``ds`` so that
    "N metres ahead" is the same number of indices on every track.
    """

    def __init__(self, tracks: list[Track], device: torch.device | str = "cpu", ds: float | None = None,
                 min_points: int = 0):
        assert tracks, "need at least one track"
        ds = float(ds if ds is not None else tracks[0].ds)
        fixed = []
        for t in tracks:
            if abs(t.ds - ds) > 0.02 * ds:      # re-sample tracks that use a different spacing
                t = Track.from_centerline(t.xy, t.w_left, t.w_right, ds=ds, name=t.name, meta=t.meta)
            fixed.append(t)
        self.tracks = fixed
        self.device = torch.device(device)
        self.ds = ds
        self.num_tracks = len(fixed)
        # ``min_points`` reserves room so longer tracks can be swapped in later (see ``replace``)
        m_max = max(max(t.n for t in fixed), int(min_points))
        self.m_max = m_max

        def pad(get, width=None):
            shape = (len(fixed), m_max) if width is None else (len(fixed), m_max, width)
            out = np.zeros(shape, dtype=np.float32)
            for i, t in enumerate(fixed):
                out[i, : t.n] = get(t)
            return torch.as_tensor(out, device=self.device)

        self.n = torch.as_tensor([t.n for t in fixed], device=self.device, dtype=torch.long)
        self.ds_each = torch.as_tensor([t.ds for t in fixed], device=self.device, dtype=torch.float32)
        self.xy = pad(lambda t: t.xy, 2)
        self.psi = pad(lambda t: t.psi)
        self.kappa = pad(lambda t: t.kappa)
        self.w_left = pad(lambda t: t.w_left)
        self.w_right = pad(lambda t: t.w_right)
        self.left = pad(lambda t: t.left_xy, 2)
        self.right = pad(lambda t: t.right_xy, 2)
        self.length = self.n.float() * self.ds_each

    def replace(self, track_id: int, t: Track) -> None:
        """Swap one track for a new one in place (used to keep generating fresh tracks)."""
        if abs(t.ds - self.ds) > 0.02 * self.ds:
            t = Track.from_centerline(t.xy, t.w_left, t.w_right, ds=self.ds, name=t.name, meta=t.meta)
        if t.n > self.m_max:
            raise ValueError(f"track with {t.n} points does not fit in batch padded to {self.m_max}")
        self.tracks[track_id] = t
        n = t.n
        dev = self.device
        as_t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=dev)  # noqa: E731
        self.n[track_id] = n
        self.ds_each[track_id] = t.ds
        self.length[track_id] = n * t.ds
        self.xy[track_id, :n] = as_t(t.xy)
        self.psi[track_id, :n] = as_t(t.psi)
        self.kappa[track_id, :n] = as_t(t.kappa)
        self.w_left[track_id, :n] = as_t(t.w_left)
        self.w_right[track_id, :n] = as_t(t.w_right)
        self.left[track_id, :n] = as_t(t.left_xy)
        self.right[track_id, :n] = as_t(t.right_xy)

    def wrap(self, tid: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Wrap point indices around the end of each (closed) track.

        ``tid`` has shape (N,), ``idx`` has shape (N,) or (N, K).
        """
        n = self.n[tid]
        if idx.dim() == 2:
            n = n[:, None]
        return torch.remainder(idx, n)
