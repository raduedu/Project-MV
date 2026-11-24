"""Real Formula 1 laps from the OpenF1 API, as lap files on our imported circuits.

What this is for
----------------
The owner wants a real benchmark: race Max Verstappen *in equal machinery*.
Formula 1 publishes live timing; the open-source OpenF1 project
(https://openf1.org, free for historical data from 2023 on, no account) serves
it as JSON. For every car it has, about 3.7 times a second:

* ``car_data``: speed [km/h], engine rpm, gear, throttle [0-100 %], brake
  (on/off only - not the pressure), DRS state;
* ``location``: x, y, z position on the circuit [decimetres, in F1's own map
  frame].

What it does NOT have: steering, tyre slip, brake pressure, set-up. So it can
be a lap to *race against* and a source to *measure the car* from
(:mod:`mv.benchmark.car_fit`), not a set of inputs to copy.

Rights (read before using)
--------------------------
The OpenF1 software is open source, but the data itself comes from Formula 1's
live timing and belongs to Formula One Management - neither project grants a
licence for it. We therefore treat it exactly like the Assetto Corsa data:
fetched on demand into ``.cache/openf1`` and lap files in ``runs/laps`` -
both git-ignored - for the owner's private, non-commercial research; never
committed, published or redistributed. (Engineering log WP43, decision D-038.)

How a lap is turned into ours
-----------------------------
1. Find the session and his fastest valid lap (``laps`` endpoint).
2. Fetch ``car_data`` and ``location`` for that lap's time window.
3. *Align* F1's map frame to the circuit imported from Assetto Corsa: find the
   rotation, scale (decimetres -> metres, ~0.1), shift and possible mirror that
   put his path on our circuit (:func:`align_path`). The remaining distance
   between his path and our circuit is reported per stretch: large values mean
   the two circuits differ there (e.g. a chicane added or removed since the
   game's version) and that stretch is not comparable.
4. Express his lap in our frame: distance along our centre line ``s`` and
   sideways offset, interpolated to 25 samples per second like our laps.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

from mv.track.track import Track

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "openf1"
API = "https://api.openf1.org/v1"

__all__ = ["fetch", "find_session", "fastest_lap", "F1Lap", "align_path", "Alignment", "to_lap_file"]


# ----------------------------------------------------------------------------------------------
# fetching (cached: the historical data never changes, and the API asks for few requests)
# ----------------------------------------------------------------------------------------------
def fetch(endpoint: str, cache: Path | None = None, **params) -> list[dict]:
    """GET ``/v1/<endpoint>?<params>``; answers are kept in ``.cache/openf1`` (git-ignored).

    Comparison filters such as ``date>=...`` are passed as keys ending in
    ``__ge`` / ``__le`` (``date__ge="2023-..."``).
    """
    parts = []
    for k, v in params.items():
        op = "="
        if k.endswith("__ge"):
            k, op = k[:-4], ">="
        elif k.endswith("__le"):
            k, op = k[:-4], "<="
        parts.append(f"{urllib.parse.quote(k)}{op}{urllib.parse.quote(str(v), safe=':+-.T')}")
    url = f"{API}/{endpoint}?" + "&".join(parts)
    cache = Path(cache) if cache else CACHE
    f = cache / f"{endpoint}_{hashlib.sha1(url.encode()).hexdigest()[:16]}.json"
    if f.is_file():
        return json.loads(f.read_text(encoding="utf-8"))
    for attempt in range(4):
        try:
            data = json.loads(_get(url))
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2.0 * (attempt + 1))
    if isinstance(data, dict):          # an error message, e.g. {"detail": "..."} - do not cache it
        raise RuntimeError(f"OpenF1 {endpoint}: {data}")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(0.4)                     # be gentle with a free service
    return data


def _get(url: str) -> str:
    """Download a URL as text, with certificate checking always on.

    Python's own certificate store can reject a server that Windows accepts (seen
    with this API: "certificate has expired" on an old cross-signed root). Then
    Windows' own ``curl.exe``, which uses the Windows certificate store, is tried -
    never an unverified connection.
    """
    import ssl
    import subprocess
    req = urllib.request.Request(url, headers={"User-Agent": "ProjectMV-research/1.0", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read().decode("utf-8")
    except urllib.error.URLError as e:
        if not isinstance(getattr(e, "reason", None), ssl.SSLError):
            raise
    out = subprocess.run(["curl.exe", "-sSf", "--max-time", "120", "-H", "Accept: application/json", url],
                         capture_output=True, check=True)
    return out.stdout.decode("utf-8")


def _t(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def find_session(year: int, circuit: str, session_name: str = "Qualifying", cache: Path | None = None) -> dict:
    """The session of that year at that circuit (``circuit`` = OpenF1's circuit_short_name)."""
    rows = fetch("sessions", cache, year=year, session_name=session_name)
    hits = [r for r in rows if r.get("circuit_short_name", "").lower() == circuit.lower()]
    if not hits:
        names = sorted({r.get("circuit_short_name", "") for r in rows})
        raise LookupError(f"no {session_name} at {circuit!r} in {year}; circuits: {', '.join(names)}")
    return hits[0]


@dataclass
class F1Lap:
    """One real lap: times in seconds from the lap's start, SI units."""
    year: int
    circuit: str
    session: str
    driver_number: int
    lap_number: int
    lap_time: float
    date_start: str
    t_car: np.ndarray              # [s] car_data sample times
    speed: np.ndarray              # [m/s]
    rpm: np.ndarray
    gear: np.ndarray
    throttle: np.ndarray           # [0..1]
    brake: np.ndarray              # [0/1] (on/off only)
    drs: np.ndarray                # raw DRS code (10, 12, 14 = flap open)
    t_pos: np.ndarray              # [s] location sample times
    xy_raw: np.ndarray             # [N, 2] F1 map frame, decimetres
    z_raw: np.ndarray              # [N] height in F1's frame, decimetres
    rainfall: bool = False
    meta: dict = field(default_factory=dict)


def fastest_lap(year: int, circuit: str, driver_number: int = 1, session_name: str = "Qualifying",
                cache: Path | None = None, lap_number: int | None = None) -> F1Lap:
    """His fastest timed lap of a session (or a given lap), with telemetry and positions."""
    ses = find_session(year, circuit, session_name, cache)
    laps = [l for l in fetch("laps", cache, session_key=ses["session_key"], driver_number=driver_number)
            if l.get("lap_duration") and not l.get("is_pit_out_lap") and l.get("date_start")]
    if lap_number is not None:
        laps = [l for l in laps if l["lap_number"] == lap_number]
    if not laps:
        raise LookupError(f"no timed lap for car {driver_number} in {year} {circuit} {session_name}")
    lap = min(laps, key=lambda l: l["lap_duration"])
    t0 = _t(lap["date_start"])
    t1 = t0 + timedelta(seconds=float(lap["lap_duration"]))
    win = dict(session_key=ses["session_key"], driver_number=driver_number,
               date__ge=(t0 - timedelta(seconds=3)).isoformat(), date__le=(t1 + timedelta(seconds=3)).isoformat())
    car = fetch("car_data", cache, **win)
    loc = fetch("location", cache, **win)
    if len(car) < 20 or len(loc) < 20:
        raise LookupError(f"too little telemetry for that lap ({len(car)} car, {len(loc)} location samples)")
    tc = np.array([(_t(r["date"]) - t0).total_seconds() for r in car])
    tp = np.array([(_t(r["date"]) - t0).total_seconds() for r in loc])
    weather = fetch("weather", cache, session_key=ses["session_key"])
    near = [w for w in weather if abs((_t(w["date"]) - t0).total_seconds()) < 900]
    return F1Lap(year=year, circuit=circuit, session=session_name, driver_number=driver_number,
                 lap_number=int(lap["lap_number"]), lap_time=float(lap["lap_duration"]), date_start=lap["date_start"],
                 t_car=tc, speed=np.array([r["speed"] for r in car], float) / 3.6,
                 rpm=np.array([r["rpm"] for r in car], float), gear=np.array([r["n_gear"] for r in car], float),
                 throttle=np.clip(np.array([r["throttle"] for r in car], float) / 100.0, 0, 1),
                 brake=(np.array([r["brake"] for r in car], float) > 0).astype(float),
                 drs=np.array([r.get("drs") or 0 for r in car], float),
                 t_pos=tp, xy_raw=np.array([[r["x"], r["y"]] for r in loc], float),
                 z_raw=np.array([r.get("z") or 0.0 for r in loc], float),
                 rainfall=any(bool(w.get("rainfall")) for w in near),
                 meta={"session_key": ses["session_key"], "meeting_key": ses.get("meeting_key"),
                       "sectors": [lap.get("duration_sector_1"), lap.get("duration_sector_2"), lap.get("duration_sector_3")],
                       "air_temp_c": near[0].get("air_temperature") if near else None,
                       "track_temp_c": near[0].get("track_temperature") if near else None})


# ----------------------------------------------------------------------------------------------
# putting his path on our circuit
# ----------------------------------------------------------------------------------------------
@dataclass
class Alignment:
    """F1 map frame -> our map frame.

    Step 1 (shape match): ``p = scale * R @ (M @ p_f1) + shift`` (M = optional mirror).
    Step 2 (refinement, ``lin3``): ``p = A @ [x, y, z]_f1 + b``, a linear map from
    F1's 3-D positions to our flat map. Why: F1's frame is not exactly level - on a
    hilly circuit (Spa, Red Bull Ring) a slightly tilted frame squeezes the plan
    view by ~2 % and mixes height into x/y; using the height undoes exactly that.
    """
    scale: float
    rot: np.ndarray                 # 2x2 rotation
    shift: np.ndarray               # [2] metres
    mirror: bool
    residual_m: np.ndarray          # per F1 position sample: distance from our circuit's centre line [m]
    s_m: np.ndarray                 # per F1 position sample: distance along our centre line [m]
    offset_m: np.ndarray            # per F1 position sample: sideways offset from our centre line [m], + = left
    lin3: np.ndarray | None = None  # [2, 4]: A | b of step 2 (None = step 1 only)

    def apply(self, xy_raw: np.ndarray, z_raw: np.ndarray | None = None) -> np.ndarray:
        p = np.asarray(xy_raw, float).copy()
        if self.lin3 is not None and z_raw is not None:
            q = np.column_stack([p, np.asarray(z_raw, float), np.ones(len(p))])
            return q @ self.lin3.T
        if self.mirror:
            p[:, 1] = -p[:, 1]
        return self.scale * p @ self.rot.T + self.shift

    def summary(self, half_width_m: np.ndarray | None = None) -> dict:
        r = self.residual_m
        return {"scale": round(self.scale, 5), "tilt_fix": self.lin3 is not None, "mirror": self.mirror, "rot_deg": round(float(np.degrees(np.arctan2(self.rot[1, 0], self.rot[0, 0]))), 2),
                "median_m": round(float(np.median(r)), 2), "p95_m": round(float(np.percentile(r, 95)), 2),
                "max_m": round(float(np.max(r)), 2)}


def _umeyama(src: np.ndarray, dst: np.ndarray, scale_range: tuple[float, float]) -> tuple[float, np.ndarray, np.ndarray]:
    """Best similarity transform (scale, rotation, shift) taking src onto dst in the least-squares sense.

    The scale is kept inside ``scale_range``: left free, closest-point matching can
    "succeed" by shrinking the whole path onto one point of the circuit.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(b.T @ a / len(src))
    d = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    rot = u @ d @ vt
    scale = float(np.trace(np.diag(sig) @ d) / max((a ** 2).sum() / len(src), 1e-12))
    scale = float(np.clip(scale, *scale_range))
    return scale, rot, mu_d - scale * rot @ mu_s


def align_path(xy_raw: np.ndarray, track: Track, iterations: int = 30, z_raw: np.ndarray | None = None) -> Alignment:
    """Find the transform that lays his path onto our circuit (rotation search, then ICP).

    His path is a racing line, not the centre line, so even a perfect alignment
    leaves a few metres (up to about half the track width). A stretch where the
    two circuits really differ shows up as tens of metres. The scale starts from
    the ratio of the two shapes' sizes and may only change by +-5 % (F1's frame
    is in decimetres, so it should come out near 0.1).
    """
    centre = np.asarray(track.xy, float)
    tree = cKDTree(centre)
    src0 = np.asarray(xy_raw, float)
    c_cent, c_rms = centre.mean(0), np.sqrt(((centre - centre.mean(0)) ** 2).sum(1).mean())
    best = None
    for mirror in (False, True):
        src = src0.copy()
        if mirror:
            src[:, 1] = -src[:, 1]
        s_cent, s_rms = src.mean(0), np.sqrt(((src - src.mean(0)) ** 2).sum(1).mean())
        scale0 = c_rms / max(s_rms, 1e-9)
        rng = (0.95 * scale0, 1.05 * scale0)
        for ang in np.radians(np.arange(0, 360, 4)):
            scale = scale0
            rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            shift = c_cent - scale * rot @ s_cent
            for _ in range(iterations):                      # iterative closest point
                p = scale * src @ rot.T + shift
                d, idx = tree.query(p)
                keep = d <= np.percentile(d, 90)             # ignore the worst 10 % (layout differences)
                scale, rot, shift = _umeyama(src[keep], centre[idx[keep]], rng)
            d, _ = tree.query(scale * src @ rot.T + shift)
            score = float(np.median(d))
            if best is None or score < best[0]:
                best = (score, scale, rot, shift, mirror)
    _, scale, rot, shift, mirror = best
    al = Alignment(scale, rot, shift, mirror, np.zeros(0), np.zeros(0), np.zeros(0))
    if z_raw is not None and len(z_raw) == len(src0):
        q = np.column_stack([src0, np.asarray(z_raw, float), np.ones(len(src0))])
        p = al.apply(src0)
        for _ in range(iterations):                          # ICP with the 3-D -> 2-D linear map
            d, idx = tree.query(p)
            keep = d <= np.percentile(d, 85)
            sol, *_ = np.linalg.lstsq(q[keep], centre[idx[keep]], rcond=None)
            p = q @ sol
        d_lin, _ = tree.query(p)
        d_rig, _ = tree.query(al.apply(src0))
        if np.median(d_lin) < np.median(d_rig):
            al.lin3 = sol.T
    p = al.apply(src0, z_raw)
    d, idx = tree.query(p)
    nrm = track.normal
    off = np.einsum("ij,ij->i", p - centre[idx], nrm[idx])
    al.residual_m, al.offset_m = d, off
    al.s_m = idx * track.ds
    return al


def _unwrap_s(s: np.ndarray, length: float) -> np.ndarray:
    """Distance along the lap that keeps growing across the start line (no jump back to 0)."""
    out = np.asarray(s, float).copy()
    for i in range(1, len(out)):
        while out[i] < out[i - 1] - length / 2:
            out[i:] += length
        while out[i] > out[i - 1] + length / 2:
            out[i:] -= length
    return out


def to_lap_file(lap: F1Lap, track: Track, car_id: str, driver: str = "Max Verstappen",
                notes: str = "") -> tuple[dict, Alignment]:
    """His lap as one of our lap files (kind "benchmark"), plus the alignment used.

    Distance along the lap comes from his positions (projected on our centre
    line, interpolated smoothly between the 3.7-per-second samples); speed,
    throttle, brake and gear from the car data. No steering is recorded by F1,
    so the lap file has no steering channel (the viewer says so).
    """
    from mv.laps import make_lap

    al = align_path(lap.xy_raw, track, z_raw=lap.z_raw)
    L = track.length
    t, s = lap.t_pos, _unwrap_s(al.s_m, L)
    # s at the lap's start (t = 0) is where F1's timing line is on our circuit
    s0 = float(np.interp(0.0, t, s))
    line_offset = s0 % L
    s = s - s0
    order = np.argsort(t)
    t, s, off = t[order], s[order], al.offset_m[order]
    keep = np.concatenate([[True], np.diff(s) > 0.05])  # PCHIP needs strictly increasing distance in time
    t, s, off = t[keep], np.maximum.accumulate(s[keep]), off[keep]
    keep = np.concatenate([[True], np.diff(s) > 1e-3])
    t, s, off = t[keep], s[keep], off[keep]
    ts = np.arange(0.0, lap.lap_time, 0.04)
    s_rel = np.maximum.accumulate(np.clip(PchipInterpolator(t, s, extrapolate=True)(ts), 0.0, L))
    ch = {"off": np.interp(ts, t, off), "speed": np.interp(ts, lap.t_car, lap.speed),
          "thr": np.interp(ts, lap.t_car, lap.throttle), "brk": np.interp(ts, lap.t_car, lap.brake),
          "gear": lap.gear[np.clip(np.searchsorted(lap.t_car, ts), 0, len(lap.gear) - 1)]}
    # Our laps start at the game's start line, F1's at its timing line - up to ~180 m apart (Imola).
    # A flying lap is (almost) periodic, so his lap is re-cut at our line: the part after our
    # line first, then the part before it. Lap time unchanged; distances now match our laps.
    t_new, s_new = ts, (s_rel + line_offset) % L
    if line_offset > track.ds:
        t_cross = float(np.interp(L - line_offset, s_rel, ts))
        k = int(np.searchsorted(ts, t_cross))
        t_new = np.concatenate([ts[k:] - t_cross, ts[:k] + lap.lap_time - t_cross])
        s_new = np.concatenate([s_rel[k:] + line_offset - L, s_rel[:k] + line_offset])
        ch = {key: np.concatenate([v[k:], v[:k]]) for key, v in ch.items()}
    s_t = np.clip(s_new, 0.0, L)
    i = (s_t / track.ds).astype(int) % track.n
    f = (s_t / track.ds) - np.floor(s_t / track.ds)
    j = (i + 1) % track.n
    cxy = track.xy[i] * (1 - f)[:, None] + track.xy[j] * f[:, None]
    nrm = track.normal[i]
    xy = cxy + nrm * ch["off"][:, None]
    d = np.gradient(xy, axis=0)
    yaw = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    samples = {"t": t_new, "s": s_t, "x": xy[:, 0], "y": xy[:, 1], "yaw": yaw, "speed": ch["speed"],
               "thr": ch["thr"], "brk": ch["brk"], "gear": ch["gear"]}
    summ = al.summary()
    src = (f"OpenF1 (F1 live timing): {lap.year} {lap.circuit} {lap.session}, car {lap.driver_number}, "
           f"lap {lap.lap_number}; aligned to the Assetto Corsa circuit: median {summ['median_m']} m, "
           f"95 % {summ['p95_m']} m from our centre line; F1's timing line is {line_offset:.0f} m after our start "
           f"line, so the lap is re-cut at our line (same lap time)")
    note = (notes + " " if notes else "") + ("Real lap: no steering recorded; brake is on/off only. "
                                              + ("Rain reported in the session. " if lap.rainfall else "")
                                              + "Private research use only - not to be published (D-038).")
    out = make_lap(car_id, track, driver, "benchmark", lap.lap_time, samples, src, notes=note)
    out["meta"]["alignment"] = {**summ, "f1_line_at_m": round(line_offset, 1)}
    out["meta"]["f1"] = {"year": lap.year, "circuit": lap.circuit, "session": lap.session, "lap": lap.lap_number,
                         "sectors": lap.meta.get("sectors"), "rain": lap.rainfall,
                         "air_temp_c": lap.meta.get("air_temp_c"), "track_temp_c": lap.meta.get("track_temp_c")}
    return out, al
