"""Tests for the car catalogue (mv/vehicle/catalog.py) and the batch import.

Run from the project root:  .venv\\Scripts\\python.exe -m pytest tests/test_catalog.py -q

Most tests use made-up ``ui_car.json`` contents and a small made-up car written
in Assetto Corsa's file format, so they run without the game installed.  The
few tests that read the real game are skipped when it is not found.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mv.ac.car_import import parse_lut
from mv.vehicle import catalog as C

ROOT = Path(__file__).resolve().parents[1]


def _ac_cars_dir() -> Path | None:
    try:
        from mv.ac.paths import find_ac_root
        return find_ac_root() / "content" / "cars"
    except Exception:
        return None


AC_CARS = _ac_cars_dir()
needs_ac = pytest.mark.skipif(AC_CARS is None, reason="Assetto Corsa is not installed here")


# --------------------------------------------------------------------------
# categories
# --------------------------------------------------------------------------
@pytest.mark.parametrize("car_id, ui, expected", [
    ("a_gt3", {"name": "Brand X GT3", "class": "race", "tags": ["#GTE-GT3", "rwd", "race", "gt", "gt3"]}, "gt3"),
    ("a_f1", {"name": "Brand F1 2017", "class": "race", "tags": ["#F1", "race", "gp", "singleseater"]}, "formula"),
    ("an_indy", {"name": "Dallara IR18 Indy", "class": "race", "tags": ["race", "open-wheeler"]}, "formula"),
    # carries both "lmp3" and "singleseater": a prototype (rule order)
    ("a_lmp3", {"name": "Praga-like", "class": "race", "tags": ["#lmp3", "race", "singleseater"]}, "prototype"),
    ("a_group_c", {"name": "Old Prototype", "class": "race", "tags": ["#Proto C", "prototype C", "vintage"]}, "prototype"),
    ("a_gte", {"name": "Brand RSR", "class": "race", "tags": ["#GTE-GT3", "race", "gt2"]}, "gt2_gte"),
    ("a_gt1", {"name": "Brand MC GT1", "class": "race", "tags": ["#Vintage GT", "race", "vintage"]}, "gt2_gte"),
    ("a_gt4", {"name": "Brand GT4 Clubsport", "class": "race", "tags": ["#GT4", "race"]}, "gt4"),
    # class "street" but a race tag and no street tag (like the Lotus Evora GX): a race car
    ("a_gx", {"name": "Brand GX", "class": "street", "tags": ["#GT4", "race"]}, "gt4"),
    # a road car the game also tags gt4 (like the KTM X-Bow R): stays road
    ("a_road_gt4", {"name": "Brand R", "class": "street", "tags": ["street", "race", "trackday", "gt4"]}, "road"),
    # "GT3" in the name, but a one-make cup car
    ("a_cup", {"name": "Brand 911 GT3 Cup", "class": "race", "tags": ["#Porsche Cup", "race"]}, "touring_cup"),
    ("a_touring", {"name": "Brand Gr.A", "class": "race", "tags": ["#90s touring", "race", "touring", "vintage"]}, "touring_cup"),
    ("a_track_toy", {"name": "Brand Track Special", "class": "race", "tags": ["race"]}, "touring_cup"),
    # "GTE" in the name of a street car (Lotus Evora GTE): a road car
    ("a_street_gte", {"name": "Brand GTE", "class": "street", "tags": ["street"]}, "road"),
    ("a_vintage", {"name": "Old Racer", "class": "race", "tags": ["#Vintage GT", "race", "vintage"]}, "vintage"),
    ("a_drift", {"name": "Brand Drift", "class": "street", "tags": ["#drift", "street"]}, "drift"),
    ("no_ui", {}, "road"),
])
def test_classify(car_id, ui, expected):
    assert C.classify(car_id, ui)[0] == expected


def test_classify_flags_and_overrides():
    cat, flags = C.classify("brand_car_s2", {"name": "Brand Car Step 2", "class": "street", "tags": ["street"]})
    assert cat == "road" and "tuned" in flags
    cat, flags = C.classify("ks_lamborghini_huracan_st", {"class": "race", "tags": ["#GTE-GT3", "race"]})
    assert cat == "touring_cup" and "override" in flags           # from the override table
    assert C.classify("fs_generic_ev", {})[0] == "formula"
    assert set(C.RACING) <= set(C.CATEGORIES)


def test_parse_specs():
    s = C.parse_specs({"bhp": "500bhp", "weight": "1,495kg", "topspeed": "250+km/h", "acceleration": "3.1s 0-100"})
    assert s["power_kw"] == pytest.approx(500 * 0.7457)
    assert s["mass_kg"] == 1495 and s["top_kmh"] == 250 and s["top_at_least"] and s["t_0_100_s"] == 3.1
    s = C.parse_specs({"bhp": "109 whp", "weight": "--kg", "topspeed": "--km/h", "acceleration": "<3.0s 0-100"})
    assert s["power_at_wheels"] and "mass_kg" not in s and "top_kmh" not in s and "t_0_100_s" not in s
    assert C.parse_specs(None) == {}


def test_read_ui_car_is_lenient(tmp_path):
    ui = tmp_path / "car" / "ui"
    ui.mkdir(parents=True)
    # Windows-1252 trademark sign and a raw tab inside a string: both seen in the game's files
    (ui / "ui_car.json").write_bytes(b'{"name": "Brand P1\x99\tGTR", "class": "race", "tags": ["race"]}')
    d = C.read_ui_car(tmp_path / "car")
    assert d["class"] == "race" and d["name"].startswith("Brand P1")
    assert C.read_ui_car(tmp_path / "missing") == {}


def test_parse_lut_keeps_pairs_with_a_typo():
    # regression: "0.0.91" made x one entry longer than y and sorting crashed (IndexError)
    x, y = parse_lut("-10|-0.05\n0|0.112\n8|0.0.91\n4|0.14\nbad|1\n")
    assert len(x) == len(y) == 4
    assert list(x) == [-10, 0, 4, 8] and y[-1] == 0.0       # read like the game (C strtod): "0.0"


# --------------------------------------------------------------------------
# favourites
# --------------------------------------------------------------------------
def test_favourites_round_trip(tmp_path):
    f = tmp_path / "favourites.yaml"
    assert C.load_favourites(f) == []
    C.save_favourites(["car_a", "car_b", "car_a"], f)
    assert C.load_favourites(f) == ["car_a", "car_b"]
    assert C.set_favourite("car_c", True, f) == ["car_a", "car_b", "car_c"]
    assert C.set_favourite("car_a", False, f) == ["car_b", "car_c"]
    assert C.set_favourite("car_b", True, f) == ["car_b", "car_c"]          # no duplicates
    assert f.read_text(encoding="utf-8").startswith("#")                    # explanatory header kept
    with pytest.raises(ValueError):
        C.save_favourites(["../evil path"], f)


def test_committed_favourites_file():
    ids = C.load_favourites()
    for car in ("ks_porsche_911_gt3_rs", "ks_porsche_918_spyder", "porsche_taycan_turbo_gt_weissach", "fs_generic_ev"):
        assert car in ids
    text = (ROOT / "vehicles" / "favourites.yaml").read_text(encoding="utf-8")
    assert "bhp" not in text and "kg" not in text                           # ids only, no game data


# --------------------------------------------------------------------------
# a small made-up car in Assetto Corsa's format (unpacked "data" folder)
# --------------------------------------------------------------------------
def _write_fake_car(car_dir: Path, ui: dict | None = None) -> None:
    d = car_dir / "data"
    d.mkdir(parents=True)
    files = {
        "car.ini": "[INFO]\nSCREEN_NAME=Test Car\n[BASIC]\nTOTALMASS=1000\nINERTIA=1.8,1.2,4.2\n"
                   "[FUEL]\nFUEL=20\n[CONTROLS]\nSTEER_LOCK=270\nSTEER_RATIO=12\n",
        "suspensions.ini": "[BASIC]\nWHEELBASE=2.5\nCG_LOCATION=0.45\n[FRONT]\nBASEY=-0.05\nTRACK=1.55\n"
                           "SPRING_RATE=80000\n[REAR]\nBASEY=-0.05\nTRACK=1.55\nSPRING_RATE=90000\n"
                           "[ARB]\nFRONT=20000\nREAR=10000\n",
        "tyres.ini": "[COMPOUND_DEFAULT]\nINDEX=0\n" + "".join(
            f"[{ax}]\nNAME=Semislick\nRADIUS=0.32\nWIDTH=0.25\nDY_REF=1.3\nLS_EXPY=0.85\nDX_REF=1.35\n"
            "LS_EXPX=0.85\nFZ0=3000\nFRICTION_LIMIT_ANGLE=8\nFALLOFF_LEVEL=0.85\n" for ax in ("FRONT", "REAR")),
        "engine.ini": "[HEADER]\nPOWER_CURVE=power.lut\n[ENGINE_DATA]\nLIMITER=7500\nMINIMUM=900\n"
                      "[COAST_REF]\nRPM=7000\nTORQUE=60\n",
        "power.lut": "1000|250\n4000|320\n7500|300\n",
        "drivetrain.ini": "[TRACTION]\nTYPE=RWD\n[GEARS]\nCOUNT=6\nGEAR_1=3.2\nGEAR_2=2.2\nGEAR_3=1.7\n"
                          "GEAR_4=1.35\nGEAR_5=1.1\nGEAR_6=0.9\nFINAL=3.6\n[GEARBOX]\nCHANGE_UP_TIME=80\n"
                          "CHANGE_DN_TIME=120\n",
        "brakes.ini": "[DATA]\nMAX_TORQUE=2000\nFRONT_SHARE=0.65\n",
        "aero.ini": "[WING_0]\nNAME=BODY\nCHORD=1\nSPAN=1\nPOSITION=0,0,0\nLUT_AOA_CL=cl.lut\n"
                    "LUT_AOA_CD=cd.lut\nANGLE=0\n",
        "cl.lut": "-10|0\n0|0.2\n8|0.0.91\n",          # the typo found in two real cars
        "cd.lut": "-10|0.3\n0|0.35\n10|0.4\n",
    }
    for name, text in files.items():
        (d / name).write_text(text, encoding="utf-8")
    if ui is not None:
        (car_dir / "ui").mkdir()
        (car_dir / "ui" / "ui_car.json").write_text(json.dumps(ui), encoding="utf-8")


def test_batch_import_skips_unavailable_and_survives_failures(tmp_path):
    cars, out = tmp_path / "cars", tmp_path / "vehicles" / "local"
    (cars / "dlc_not_owned").mkdir(parents=True)                       # empty folder, like an unowned DLC
    broken = cars / "broken_car" / "data"
    broken.mkdir(parents=True)
    (broken / "car.ini").write_text("[INFO]\nSCREEN_NAME=Broken\n", encoding="utf-8")
    _write_fake_car(cars / "test_gt3", {"name": "Test GT3", "class": "race", "tags": ["gt3", "race"],
                                        "specs": {"bhp": "315bhp", "weight": "1000kg", "topspeed": "250km/h"}})
    log: list[str] = []
    res = C.import_cars(["dlc_not_owned", "broken_car", "test_gt3", "no_such_folder"], cars, out,
                        simulate=True, log=log.append)
    by = {r["id"]: r for r in res}
    assert by["dlc_not_owned"]["status"] == "unavailable"
    assert by["no_such_folder"]["status"] == "unavailable"
    assert by["broken_car"]["status"] == "failed" and by["broken_car"]["reasons"]
    good = by["test_gt3"]
    assert good["status"] == "ok", good["reasons"]
    assert good["category"] == "gt3"
    assert good["ours"]["finite"] and 2.0 < good["ours"]["t_0_100_s"] < 8.0
    assert (out / "test_gt3.yaml").is_file() and not (out / "broken_car.yaml").exists()
    assert len(log) == 4
    report = json.loads((out / "_catalog.json").read_text(encoding="utf-8"))
    assert set(report) == {"dlc_not_owned", "broken_car", "test_gt3", "no_such_folder"}

    # the merged list: favourites first, status taken from the report, library-less test tree
    C.save_favourites(["test_gt3"], tmp_path / "vehicles" / "favourites.yaml")
    entries = C.list_cars(ac_root=False, vehicles_dir=tmp_path / "vehicles")
    assert [e.id for e in entries] == ["test_gt3"]
    e = entries[0]
    assert e.favourite and e.imported and e.where == "local" and e.status == "ok" and e.category == "gt3"
    assert C.racing_cars(entries) == entries
    assert C.select_ids("racing", entries) == ["test_gt3"] and C.select_ids("road", entries) == []
    with pytest.raises(ValueError):
        C.select_ids("trucks", entries)


def test_plausibility_flags_wrong_numbers(tmp_path):
    from mv.ac.car_import import import_ac_car

    _write_fake_car(tmp_path / "car")
    vp = import_ac_car(tmp_path / "car")
    status, reasons, ours = C.plausibility(vp, {"power_kw": 470.0, "mass_kg": 1000.0, "top_kmh": 400.0},
                                           simulate=False)
    assert status == "suspicious"
    assert any(r.startswith("power") for r in reasons) and any(r.startswith("top speed") for r in reasons)
    assert not any(r.startswith("mass") for r in reasons)
    # a lower-bound figure ("250+ km/h") is only checked from below
    status, reasons, _ = C.plausibility(vp, {"top_kmh": 200.0, "top_at_least": True}, simulate=False)
    assert status == "ok", reasons


def test_turbo_controllers_replace_the_wastegate(tmp_path):
    """Like the Ferrari 488 GTB: one turbo per gear, enabled by its controller."""
    import numpy as np

    from mv.ac.car_import import import_ac_car

    _write_fake_car(tmp_path / "car")
    d = tmp_path / "car" / "data"
    turbo = "MAX_BOOST=1.4\nWASTEGATE=0.69\nREFERENCE_RPM=1500\nGAMMA=2.5\n"
    with open(d / "engine.ini", "a", encoding="utf-8") as f:
        f.write("".join(f"[TURBO_{i}]\n{turbo}" for i in range(6)))
    for i in range(6):                     # turbo i boosts only in gear i+1; gear 6 = top gear
        gear_lut = "|".join(f"{g}={1 if g == i + 1 else 0}" for g in range(7))
        (d / f"ctrl_turbo{i}.ini").write_text(
            "[CONTROLLER_0]\nINPUT=RPMS\nCOMBINATOR=ADD\nLUT=(|0=0|2000=0.5|8000=0.5|)\nUP_LIMIT=10\nDOWN_LIMIT=0\n"
            f"[CONTROLLER_1]\nINPUT=GEAR\nCOMBINATOR=MULT\nLUT=(|{gear_lut}|)\nUP_LIMIT=10\nDOWN_LIMIT=0\n"
            "[CONTROLLER_2]\nINPUT=GAS\nCOMBINATOR=MULT\nLUT=(|0=0.5|1=1|)\nUP_LIMIT=0.6\nDOWN_LIMIT=0\n",
            encoding="utf-8")
    eng = import_ac_car(tmp_path / "car").powertrain.units[0]
    tq = float(np.interp(4000, eng.rpm, eng.torque_nm))
    # only the top-gear turbo counts: 320 Nm * (1 + 0.5); summing all six wastegates would give x5.1
    assert tq == pytest.approx(320 * 1.5, rel=1e-3)


def test_kers_becomes_a_hybrid_motor(tmp_path):
    from mv.ac.car_import import import_ac_car

    _write_fake_car(tmp_path / "car")
    d = tmp_path / "car" / "data"
    (d / "kers.ini").write_text("[KERS]\nTORQUE_CURVE=kers_torque.lut\nATTACH=ENGINE\nDISCHARGE_TIME=28800\n",
                                encoding="utf-8")
    (d / "kers_torque.lut").write_text("0|200\n6000|150\n7500|100\n7600|0\n", encoding="utf-8")
    vp = import_ac_car(tmp_path / "car")
    assert vp.powertrain.type == "hybrid"
    motor = [u for u in vp.powertrain.units if u.kind == "motor"]
    assert len(motor) == 1 and motor[0].gearbox == "main"
    peak_kw = 150 * 6000 * 3.141592653589793 / 30 / 1000          # ~94 kW at 6000 rpm
    b = vp.powertrain.battery
    assert b.max_discharge_kw == pytest.approx(peak_kw, abs=0.1)
    assert b.capacity_kwh == pytest.approx(peak_kw * 28.8 / 3600 / 0.8, rel=1e-2)


# --------------------------------------------------------------------------
# against the real game (skipped without it)
# --------------------------------------------------------------------------
@needs_ac
def test_real_cars_classify_and_former_failures_import():
    from mv.ac.car_import import import_ac_car

    expect = {"ks_ferrari_sf70h": "formula", "ks_porsche_919_hybrid_2016": "prototype",
              "ks_porsche_911_gt3_r_2016": "gt3", "ks_porsche_911_rsr_2017": "gt2_gte",
              "ks_porsche_cayman_gt4_clubsport": "gt4", "ks_porsche_911_gt3_cup_2017": "touring_cup",
              "ks_porsche_911_gt3_rs": "road"}
    for car, cat in expect.items():
        if (AC_CARS / car).is_dir():
            assert C.classify(car, C.read_ui_car(AC_CARS / car))[0] == cat, car
    for car in ("ks_porsche_cayman_gt4_std", "ks_praga_r1"):             # IndexError before the parse_lut fix
        if (AC_CARS / car).is_dir():
            import_ac_car(AC_CARS / car).validate()
