"""Race bot stage 5 (engineering log WP65): the game link and the bot's inputs from live data.

No game needed: the Lua app's state block is filled in by the test the way the app fills it in.
* the C layouts in the Lua file equal the Python ones (the two sides cannot drift apart);
* game axes -> project axes (yaw, body velocity, yaw rate) for the bot and the other cars;
* steering goes out in the game's sign (+1 = right);
* the bot's observation built from game data equals the training environment's observation of the
  same situation - track position, preview, lidar, horizon and the opponents block;
* the race session written for the owner: 2 cars, same model, a standing-start race.
"""

from __future__ import annotations

import ctypes
import math
import mmap
import re
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from mv.ac.driver import ActuatorModel
from mv.ac.launch import build_race_ini
from mv.ac.racebot import RaceBotAdapter
from mv.ac.racebot_link import (CMD_MAGIC, LAYOUT_VERSION, STATE_MAGIC, RaceBotCmd, RaceBotLink, RaceBotStateRaw,
                                c_layout, decode)
from mv.envs.racing_env import EnvConfig, RacingEnv
from mv.rl.evaluate import eval_config
from mv.track.track import TrackBatch
from tests.test_core import car, circle_track  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
LUA = ROOT / "mv" / "ac" / "plugin" / "apps" / "lua" / "mv_racebot" / "mv_racebot.lua"
R = 300.0


def _lua_block(name: str) -> str:
    m = re.search(rf"local {name} = \[\[\n(.*?)\]\]", LUA.read_text(encoding="utf-8"), re.S)
    assert m, name
    return m.group(1).strip()


def test_lua_layouts_match_python():
    assert _lua_block("CMD_LAYOUT") == c_layout(RaceBotCmd)
    assert _lua_block("STATE_LAYOUT") == c_layout(RaceBotStateRaw)
    text = LUA.read_text(encoding="utf-8")
    assert f"0x{CMD_MAGIC:08X}" in text and f"0x{STATE_MAGIC:08X}" in text
    assert "MAX_OTHERS = 8" in text


def _raw_on_circle(s_bot: float, v_bot: float, others: list[tuple[float, float, float]]) -> RaceBotStateRaw:
    """A state block as the app writes it: the bot s_bot metres along the anticlockwise circle at v_bot,
    other cars at (s, lateral left, speed).  AC world: X = map x, Z = -map y."""
    raw = RaceBotStateRaw()
    raw.magic, raw.version, raw.seq, raw.bot_index, raw.cars_count = STATE_MAGIC, LAYOUT_VERSION, 2, 1, 1 + len(others)
    raw.session_started, raw.game_time_s = 1, 10.0

    def pose(s, lat):
        a = s / R
        rr = R - lat
        x, y = rr * math.cos(a), rr * math.sin(a)
        fx, fy = -math.sin(a), math.cos(a)                  # anticlockwise: heading = a + 90 deg
        return (x, 0.0, -y), (fx, 0.0, -fy)

    p, lk = pose(s_bot, 0.0)
    raw.position[:], raw.look[:] = p, lk
    raw.velocity[:] = (lk[0] * v_bot, 0.0, lk[2] * v_bot)
    raw.local_velocity[:] = (0.0, 0.0, v_bot)              # AC car-local: X left, Y up, Z forward
    raw.local_angular_velocity[:] = (0.0, v_bot / R, 0.0)   # turning left at v / R
    raw.gear, raw.rpm = 3, 6000.0
    for k, (s, lat, v) in enumerate(others):
        p, lk = pose(s, lat)
        raw.other_index[k], raw.other_active[k] = (0 if k == 0 else k + 1), 1
        raw.other_position[3 * k:3 * k + 3] = p
        raw.other_velocity[3 * k:3 * k + 3] = (lk[0] * v, 0.0, lk[2] * v)
        raw.other_look[3 * k:3 * k + 3] = lk
    return raw


def test_game_axes_become_project_axes():
    st = decode(_raw_on_circle(0.0, 30.0, [(40.0, 2.0, 25.0)]))
    assert st.bot.x == pytest.approx(R) and st.bot.y == pytest.approx(0.0, abs=1e-6)
    assert st.bot.yaw == pytest.approx(math.pi / 2)           # facing +map y
    assert st.bot.vx == pytest.approx(30.0) and st.bot.vy == pytest.approx(0.0)
    assert st.yaw_rate == pytest.approx(30.0 / R)              # + = turning left
    o = st.others[0]
    assert o.index == 0 and o.vx == pytest.approx(25.0, rel=1e-5) and abs(o.vy) < 1e-4
    assert o.yaw == pytest.approx(math.pi / 2 + 40.0 / R, abs=1e-5)
    assert len(st.others) == 1


def test_link_sends_game_steering_sign_and_reads_state():
    tag = uuid.uuid4().hex[:8]
    link = RaceBotLink(bot_index=1, cmd_name=f"Local\\mvt_cmd_{tag}", state_name=f"Local\\mvt_st_{tag}")
    try:
        assert link.read() is None                             # the app has not written yet
        link.drive(0.3, 0.5, 0.0)                              # project: +0.3 = left
        c = link._cmd
        assert c.enable == 1 and c.bot_index == 1 and c.steer == pytest.approx(-0.3) and c.gas == pytest.approx(0.5)
        assert c.seq % 2 == 0
        del c                                                  # (a live view would keep the mapping open)
        raw = _raw_on_circle(0.0, 20.0, [])
        mm = mmap.mmap(-1, ctypes.sizeof(RaceBotStateRaw), tagname=f"Local\\mvt_st_{tag}")
        mm[:] = bytes(raw)
        st = link.read()
        assert st is not None and st.bot.vx == pytest.approx(20.0) and st.session_started
        mm.close()
        link.release()
        assert link._cmd.enable == 0
    finally:
        link.close()


def test_bot_observation_equals_training_observation(car):  # noqa: F811
    """The same situation seen by the training environment and built from game data: identical inputs."""
    cfg = replace(EnvConfig(), history_steps=8, lap_memory=True, self_recovery=True, own_map=True, race_group=4,
                  opponents=3)
    track = circle_track(R, 7.0)
    others = [(40.0, 2.0, 25.0), (-20.0, -3.0, 35.0)]          # the owner 40 m ahead, another car 20 m behind
    ad = RaceBotAdapter(car, track, cfg, view_m=120.0)
    obs = ad.update(decode(_raw_on_circle(0.0, 30.0, others)), ActuatorModel(car, cfg.control_dt))[0]

    ref_cfg = replace(eval_config(cfg, strict=False), race_group=4)
    env = RacingEnv(car, TrackBatch([track], ds=track.ds), 4, ref_cfg, device="cpu")
    env.fixed_start = (0, 0.0)
    env.reset(torch.arange(4))
    sim = env.sim
    for i, (s, lat, v) in enumerate([(0.0, 0.0, 30.0)] + others):
        a = s / R
        sim.reset(torch.tensor([i]), torch.tensor([(R - lat) * math.cos(a)]), torch.tensor([(R - lat) * math.sin(a)]),
                  torch.tensor([a + math.pi / 2]), torch.tensor([v]))
        env.idx[i], env.along[i], env.tid[i] = int(round(s % (2 * math.pi * R))) % int(env.tracks.n[0]), 0.0, 0
    for i in range(4 - 1 - len(others)):                        # empty slot: far away on the same circuit, unseen
        j = 1 + len(others) + i
        env.idx[j] = int(env.tracks.n[0]) // 2
        p = env.tracks.xy[0, env.idx[j]]
        sim.x[j], sim.y[j] = p[0], p[1]
    env._locate()
    env.set_view(120.0)
    ref = env._observe(torch.arange(1))[0]
    lay = env.obs_layout
    for block in ("position", "kappa", "width", "horizon", "lidar", "car_desc", "opponents"):
        assert torch.allclose(obs[lay[block]], ref[lay[block]], atol=2e-3), block
    opp = obs[lay["opponents"]].view(3, 5)
    assert opp[0, 0] == 1 and opp[0, 1] == pytest.approx(-20.0 / 50, abs=0.01)   # nearest first: 20 m behind
    assert opp[1, 0] == 1 and opp[1, 1] == pytest.approx(40.0 / 50, abs=0.01)
    assert opp[2, 0] == 0


class _FakeGame:
    """Stands in for the game + MV Race Bot app: car 1 is simulated by our physics, controlled through
    exactly what RaceBotLink.drive() writes (AC steering sign), and reported through decode()."""

    def __init__(self, car, track, clock):
        from mv.sim.vehicle_sim import VehicleSim
        self.sim, self.track, self.clock = VehicleSim(car, 1, "cpu", dt=0.008), track, clock
        p, psi = track.xy[0], float(track.psi[0])
        self.sim.reset(torch.tensor([0]), torch.tensor([float(p[0])]), torch.tensor([float(p[1])]),
                       torch.tensor([psi]), torch.tensor([15.0]))
        self.cmd = (0.0, 0.0, 1.0)
        self.takeover, self.bot_index, self.t = 1, 1, 0.0

    # --- RaceBotLink interface ---
    def drive(self, steer_left, gas, brake):
        steer_ac = -float(steer_left)                 # what RaceBotLink writes (+1 = right) ...
        self.cmd = (-steer_ac, gas, brake)            # ... and what the game does with it (project: + = left)

    def release(self):
        self.cmd = (0.0, 0.0, 1.0)

    def read(self):
        while self.clock.now - self.t > 1e-9:         # the game runs in real time: catch up with the clock
            for _ in range(5):
                s, g, b = self.cmd
                self.sim.step(torch.tensor([s]), torch.tensor([g]), torch.tensor([b]))
            self.t += 0.04
        sim = self.sim
        x, y, yaw = float(sim.x[0]), float(sim.y[0]), float(sim.yaw[0])
        vx, vy = float(sim.vx[0]), float(sim.vy[0])
        c, s = math.cos(yaw), math.sin(yaw)
        raw = RaceBotStateRaw()
        raw.magic, raw.version, raw.bot_index, raw.cars_count, raw.session_started = STATE_MAGIC, LAYOUT_VERSION, 1, 1, 1
        raw.game_time_s, raw.gear, raw.rpm = self.t, int(sim.gear[0, 0]) + 1, 5000.0
        raw.position[:] = (x, 0.0, -y)
        raw.look[:] = (c, 0.0, -s)
        raw.velocity[:] = (vx * c - vy * s, 0.0, -(vx * s + vy * c))
        raw.local_velocity[:] = (vy, 0.0, vx)
        raw.local_angular_velocity[:] = (0.0, float(sim.r[0]), 0.0)
        raw.gas, raw.brake, raw.control_ok = float(sim.thr[0]), float(sim.brk[0]), 1
        return decode(raw)


class _Clock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def sleep(self, s):
        self.now += max(s, 0.0)


def test_closed_loop_drives_the_game_car_around(car, monkeypatch):  # noqa: F811
    """The real race() loop drives a stand-in game car for 40 s with the rule-based driver: it must follow
    the road (a wrong steering sign or axis would put it off within a second)."""
    import mv.ac.racebot as rb
    from mv.control.scripted import ScriptedDriver
    clock = _Clock()
    monkeypatch.setattr(rb.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(rb.time, "sleep", clock.sleep)
    track = circle_track(120.0, 6.0)
    cfg = EnvConfig()
    ad = RaceBotAdapter(car, track, cfg, view_m=120.0)
    game = _FakeGame(car, track, clock)
    worst = [0.0]
    printed = []

    class Driver(ScriptedDriver):
        def act(self, obs, deterministic=True):
            worst[0] = max(worst[0], abs(ad.lateral_offset))
            return super().act(obs, deterministic)

    reason = rb.race(game, Driver(ad.env), ad, car, max_time_s=40.0, printer=printed.append)
    assert reason == "time"
    assert worst[0] < 6.0 - 0.5 * car.chassis.width_m          # stayed on the road all the time
    assert float(game.sim.vx[0]) > 15.0                         # and is driving, not parked
    assert ad.env.s_total[0] > 400.0                            # ~ a lap of the 754 m circle and more


def test_race_session_for_the_owner():
    text = build_race_ini("[CAR_0]\nDRIVER_NAME=Owner\n", "ks_porsche_911_gt3_rs", "white", "ks_nordschleife",
                          "nordschleife", ghost_car=True, ghost_skin="red", race_laps=3, ghost_name="MV Race Bot")
    assert "TYPE=3" in text and "LAPS=3" in text and "SPAWN_SET=START" in text and "RACE_LAPS=3" in text
    assert "CARS=2" in text and "DRIVER_NAME=MV Race Bot" in text and "DRIVER_NAME=Owner" in text
    car1 = text.split("[CAR_1]")[1]
    assert "MODEL=ks_porsche_911_gt3_rs" in car1                # same machinery
    practice = build_race_ini("", "c", "s", "t", "", ghost_car=False)
    assert "TYPE=1" in practice and "SPAWN_SET=PIT" in practice
