"""System identification: recordings, replay and the confidence check (fast checks only;
the full recovery test is `scripts/sysid_fit.py --synthetic`, ~7 minutes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mv.sysid.replay import FitResult, Recording, apply_factors, replay_error
from mv.sysid.synthetic import record_in_simulator
from mv.track.generator import random_tracks
from mv.vehicle.params import load_vehicle

ROOT = Path(__file__).resolve().parents[1]
CAR = ROOT / "vehicles" / "library" / "porsche_911_gt3_rs_992.yaml"


@pytest.fixture(scope="module")
def setup():
    vp = load_vehicle(CAR)
    altered = apply_factors(vp, {"grip": 0.93, "brakes": 0.88})
    rec = record_in_simulator(altered, random_tracks(1, seed=12, cfg="mixed")[0], seconds=30)
    return vp, altered, rec


def test_apply_factors_scales_the_right_numbers():
    vp = load_vehicle(CAR)
    v = apply_factors(vp, {"grip": 0.9, "grip_balance": 1.21, "downforce": 1.1, "drag": 0.8, "power": 1.05, "brakes": 0.7})
    assert v.tire_front.mu_lat == pytest.approx(vp.tire_front.mu_lat * 0.9 * 1.1)
    assert v.tire_rear.mu_lat == pytest.approx(vp.tire_rear.mu_lat * 0.9 / 1.1)
    assert v.aero.cla_rear_m2 == pytest.approx(vp.aero.cla_rear_m2 * 1.1)
    assert v.aero.cda_m2 == pytest.approx(vp.aero.cda_m2 * 0.8)
    assert v.powertrain.units[0].torque_nm[3] == pytest.approx(vp.powertrain.units[0].torque_nm[3] * 1.05)
    assert v.brakes.max_torque_nm == pytest.approx(vp.brakes.max_torque_nm * 0.7)
    assert vp.brakes.max_torque_nm != v.brakes.max_torque_nm          # original untouched


def test_recording_roundtrip(setup, tmp_path):
    _, _, rec = setup
    rec.save(tmp_path / "r.npz")
    back = Recording.load(tmp_path / "r.npz")
    assert len(back) == len(rec) > 500
    assert np.allclose(back.vx, rec.vx) and back.meta["source"] == "simulator"


def test_replay_prefers_the_true_parameters(setup):
    vp, altered, rec = setup
    e_true, per_true = replay_error(altered, rec)
    e_nominal, _ = replay_error(vp, rec)
    assert e_true < 0.6 * e_nominal
    assert per_true["vx"] < 0.1 and per_true["r"] < 0.03     # the true car replays its own laps closely


def test_confidence_check():
    r = FitResult({"a": 1.1, "b": 1.0}, 1.0, 0.5, {}, {}, 10, sensitivity={"a": 0.05, "b": 0.001})
    assert r.well_determined() == {"a": True, "b": False}
