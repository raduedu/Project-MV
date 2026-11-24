"""Which imported real circuits are used for validation, which may be trained on.

Real circuits come from the owner's Assetto Corsa install (``tracks/local``, never
committed).  ``tracks/sets.yaml`` (committed: names only) says what each one is for:

* ``validation`` - driven at every training check and by strict evaluations;
* ``never_train`` - kept out of training forever, so that "the driver can race a
  real circuit it has never seen" stays something we *measure*, not assume;
* ``skip`` - imported layouts that are not closed racing circuits.

Every other imported circuit may be added to the training pool
(``scripts/train.py --real-tracks``).  Why train on real circuits at all?  The
generator makes realistic circuits (WP31), but real ones contain combinations it
never produces (Eau Rouge, the Nordschleife's crests and kinks). Holding a few
out keeps the test honest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "tracks" / "local"
SETS_FILE = ROOT / "tracks" / "sets.yaml"

__all__ = ["load_sets", "imported_tracks", "validation_names", "trainable_names", "narrowed"]


def narrowed(track, margin_m, min_half_width_m: float = 2.0):
    """The same circuit with both edges moved ``margin_m`` metres inwards (WP48).

    Why: an imported circuit's edges sit about at the white lines; just beyond them the
    game has kerbs and, at some chicanes, raised "sausage" bumps. Our simulator has a
    flat road and no kerbs, so a driver trained to use every centimetre rides those
    bumps when its lap is played in the game, and the car is thrown in the air. A margin
    keeps the wheels off them. Each half-width stays at least ``min_half_width_m``.
    ``margin_m`` may also be one value per track point (more room where it is needed, WP57).
    """
    m = np.broadcast_to(np.asarray(margin_m, float), (track.n,))
    if np.all(m <= 0):
        return track
    from mv.track.track import Track
    wl = np.maximum(np.asarray(track.w_left, float) - m, min_half_width_m)
    wr = np.maximum(np.asarray(track.w_right, float) - m, min_half_width_m)
    meta = dict(track.meta or {})
    meta["edge_margin_m"] = float(m.max()) if np.ptp(m) > 0 else float(m[0])
    return Track.from_centerline(track.xy, wl, wr, ds=track.ds, name=track.name, meta=meta)


def load_sets(path: Path | None = None) -> dict[str, list[str]]:
    """The three lists from ``tracks/sets.yaml`` (empty lists if the file is missing)."""
    path = Path(path) if path else SETS_FILE
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data = data or {}
    return {k: [str(x) for x in (data.get(k) or [])] for k in ("validation", "never_train", "skip")}


def imported_tracks(local: Path | None = None) -> list[str]:
    """Names of all imported circuits (files ``tracks/local/<name>.npz``)."""
    local = Path(local) if local else LOCAL
    return sorted(p.stem for p in local.glob("*.npz") if not p.stem.startswith("_")) if local.is_dir() else []


def validation_names(local: Path | None = None, sets: dict | None = None) -> list[str]:
    """Imported circuits used for validation.

    The listed ones that are actually imported; if the file lists none, every
    imported circuit (the behaviour before the sets file existed).
    """
    have = imported_tracks(local)
    sets = sets or load_sets()
    listed = [n for n in sets["validation"] if n in have]
    return listed if sets["validation"] else have


def trainable_names(wanted: str = "all", local: Path | None = None, sets: dict | None = None) -> list[str]:
    """Imported circuits that may be trained on.

    ``wanted`` is ``"all"`` or a comma-separated list of names.  Circuits in
    ``never_train`` or ``skip`` are always removed - asking for one is an error,
    because silently training on the held-out test circuit would make every
    later "unseen circuit" result meaningless.
    """
    have = imported_tracks(local)
    sets = sets or load_sets()
    banned = set(sets["never_train"]) | set(sets["skip"])
    if wanted.strip().lower() == "all":
        return [n for n in have if n not in banned]
    names = [n.strip() for n in wanted.split(",") if n.strip()]
    missing = [n for n in names if n not in have]
    if missing:
        raise ValueError(f"not imported (scripts/import_ac_track.py): {', '.join(missing)}")
    held = [n for n in names if n in banned]
    if held:
        raise ValueError(f"held out of training in tracks/sets.yaml: {', '.join(held)}")
    return names
