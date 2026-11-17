"""Fixed-rate real-time loop around AC: send controls, wait, observe.

Timing model
============
``step(action)`` at control rate ``f`` (25–50 Hz):

1. send ``action`` immediately (it was computed from the previous observation);
2. sleep until the next tick of a drift-free schedule ``t0 + k/f``;
3. make sure at least one *new* physics packet exists since the send, waiting
   up to ``packet_timeout_s``;
4. read and return :class:`Telemetry`.

So the action is held for one control period (zero-order hold, like the
training simulator) and every observation is at most one physics step old.
If the loop falls behind by more than one period (e.g. a GC pause), the
schedule re-synchronises instead of bursting to catch up, and the event is
counted as an overrun.

Fail-safe
=========
* Timeout (no new physics packet: game paused, alt-tabbed with pause-on-
  unfocus, crashed, session ended) or any exception inside ``step`` ->
  ``backend.failsafe()`` (wheel straight, zero throttle, full brake) and
  :class:`ACTimeoutError` / the original exception is raised.
* A watchdog thread applies the same fail-safe if ``step`` has not been
  called for ``watchdog_s`` (the policy process hung or a debugger stopped
  it) — virtual joysticks would otherwise hold the last throttle forever.
* The CSP Lua app has its own independent fail-safe (cmd sequence stale for
  ``stale_timeout_s`` -> brake), which also covers Python being killed.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .backends import ControlBackend, CSPLuaBackend
from .reader import ACReader
from .shm_layout import AC_LIVE, AC_STATUS_NAMES
from .telemetry import Telemetry

__all__ = ["ACSession", "ACTimeoutError", "LoopStats"]


class ACTimeoutError(RuntimeError):
    """No new physics packet within the timeout (fail-safe has been applied)."""


def _q(a: np.ndarray, p: float) -> float:
    return float(np.percentile(a, p)) if a.size else float("nan")


@dataclass
class LoopStats:
    """Rolling timing statistics (last ``window`` steps), all in seconds."""

    window: int = 2000
    steps: int = 0
    timeouts: int = 0
    overruns: int = 0
    watchdog_trips: int = 0
    periods: deque = field(default_factory=deque)          # between step returns
    obs_latency: deque = field(default_factory=deque)      # send -> first new physics packet
    ack_latency: deque = field(default_factory=deque)      # send -> Lua app saw the cmd (csp)
    packets_per_step: deque = field(default_factory=deque)

    def _push(self, dq: deque, v: float) -> None:
        dq.append(v)
        while len(dq) > self.window:
            dq.popleft()

    def summary(self) -> dict:
        per = np.asarray(self.periods, dtype=float)
        lat = np.asarray(self.obs_latency, dtype=float)
        ack = np.asarray(self.ack_latency, dtype=float)
        pps = np.asarray(self.packets_per_step, dtype=float)
        return {
            "steps": self.steps, "timeouts": self.timeouts, "overruns": self.overruns,
            "watchdog_trips": self.watchdog_trips,
            "rate_hz": float(1.0 / per.mean()) if per.size else float("nan"),
            "period_mean_ms": float(per.mean() * 1e3) if per.size else float("nan"),
            "jitter_std_ms": float(per.std() * 1e3) if per.size else float("nan"),
            "period_p99_ms": _q(per, 99) * 1e3,
            "obs_latency_p50_ms": _q(lat, 50) * 1e3,
            "obs_latency_p95_ms": _q(lat, 95) * 1e3,
            "ack_latency_p50_ms": _q(ack, 50) * 1e3,
            "ack_latency_p95_ms": _q(ack, 95) * 1e3,
            "physics_packets_per_step": float(pps.mean()) if pps.size else float("nan"),
        }

    def format(self) -> str:
        s = self.summary()
        return (f"steps={s['steps']} rate={s['rate_hz']:.1f}Hz period={s['period_mean_ms']:.2f}ms "
                f"jitter(std)={s['jitter_std_ms']:.2f}ms p99={s['period_p99_ms']:.2f}ms "
                f"obs_lat p50/p95={s['obs_latency_p50_ms']:.2f}/{s['obs_latency_p95_ms']:.2f}ms "
                f"ack p50/p95={s['ack_latency_p50_ms']:.2f}/{s['ack_latency_p95_ms']:.2f}ms "
                f"pkts/step={s['physics_packets_per_step']:.1f} timeouts={s['timeouts']} "
                f"overruns={s['overruns']} watchdog={s['watchdog_trips']}")


def _sleep_until(deadline: float) -> None:
    """Sleep with ~0.1 ms accuracy: coarse sleep, then a short yield-spin.

    Python >= 3.11 on Windows uses a high-resolution waitable timer for
    ``time.sleep`` (~0.5 ms granularity); the final millisecond is spun with
    ``sleep(0)`` (yields the core) to remove the residual jitter.
    """
    while True:
        rem = deadline - time.perf_counter()
        if rem <= 0:
            return
        if rem > 0.0015:
            time.sleep(rem - 0.001)
        else:
            time.sleep(0)


class ACSession:
    """Real-time control loop with latency/jitter stats and fail-safe."""

    def __init__(self, reader: ACReader, backend: ControlBackend, rate_hz: float = 50.0,
                 packet_timeout_s: float = 0.25, watchdog_s: float = 0.3,
                 require_live: bool = True, poll_s: float = 0.0005):
        if not 1.0 <= rate_hz <= 200.0:
            raise ValueError("rate_hz must be within [1, 200]")
        self.reader = reader
        self.backend = backend
        self.period = 1.0 / rate_hz
        self.packet_timeout_s = packet_timeout_s
        self.require_live = require_live
        self.poll_s = poll_s
        self.stats = LoopStats()
        self._lock = threading.RLock()
        self._next_deadline: float | None = None
        self._last_return: float | None = None
        self._last_activity = time.perf_counter()
        self._watchdog_s = watchdog_s
        self._closed = False
        self._failsafe_engaged = False
        self._wd_thread = None
        if watchdog_s and watchdog_s > 0:
            self._wd_thread = threading.Thread(target=self._watchdog, name="ac-watchdog",
                                               daemon=True)
            self._wd_thread.start()

    # ---------------------------------------------------------------- safety
    def failsafe(self) -> None:
        """Wheel straight, zero throttle, full brake (never raises)."""
        with self._lock:
            try:
                self.backend.failsafe()
                self._failsafe_engaged = True
            except Exception:  # pragma: no cover - last line of defence
                pass

    def _watchdog(self) -> None:
        while not self._closed:
            time.sleep(min(0.05, self._watchdog_s / 3))
            if self._closed:
                return
            idle = time.perf_counter() - self._last_activity
            if idle > self._watchdog_s and not self._failsafe_engaged:
                self.stats.watchdog_trips += 1
                self.failsafe()

    def touch(self) -> None:
        """Tell the watchdog the caller is alive (e.g. during a long reset)."""
        self._last_activity = time.perf_counter()

    # --------------------------------------------------------------- observe
    def wait_new_packet(self, after_id: int, timeout_s: float | None = None) -> tuple[int, float]:
        """Wait until physics ``packetId != after_id``; returns (id, perf_counter)."""
        timeout_s = self.packet_timeout_s if timeout_s is None else timeout_s
        deadline = time.perf_counter() + timeout_s
        while True:
            pid = self.reader.physics_packet_id()
            now = time.perf_counter()
            if pid != after_id:
                return pid, now
            if now >= deadline:
                raise ACTimeoutError(self._timeout_message(timeout_s))
            time.sleep(self.poll_s)

    def _timeout_message(self, timeout_s: float) -> str:
        try:
            status = AC_STATUS_NAMES.get(self.reader.graphics.snapshot().status, "?")
        except Exception:
            status = "unreadable"
        return (f"no new AC physics packet for {timeout_s * 1e3:.0f} ms (game status {status}); "
                "fail-safe applied")

    def observe(self, timeout_s: float | None = None) -> Telemetry:
        """Wait for a fresh packet and return telemetry (no control output)."""
        self.touch()
        pid = self.reader.physics_packet_id()
        try:
            self.wait_new_packet(pid, timeout_s)
        except ACTimeoutError:
            self.stats.timeouts += 1
            self.failsafe()
            raise
        return self.reader.read()

    # ------------------------------------------------------------------ step
    def step(self, steer: float, throttle: float, brake: float,
             gear_up: bool = False, gear_down: bool = False) -> Telemetry:
        """Send controls, hold them for one period, return the new telemetry."""
        if self._closed:
            raise RuntimeError("session is closed")
        try:
            t_send = time.perf_counter()
            pid0 = self.reader.physics_packet_id()
            with self._lock:
                self.backend.send(steer, throttle, brake, gear_up, gear_down)
                self._failsafe_engaged = False
            self._last_activity = t_send

            # drift-free schedule with re-sync on overrun
            if self._next_deadline is None:
                self._next_deadline = t_send + self.period
            else:
                self._next_deadline += self.period
                if t_send > self._next_deadline:
                    self.stats.overruns += 1
                    self._next_deadline = t_send + self.period

            # latency probes while waiting for the tick
            first_pkt_t = None
            ack_t = None
            link = self.backend.link if isinstance(self.backend, CSPLuaBackend) else None
            want_seq = getattr(self.backend, "last_seq", 0)
            while True:
                now = time.perf_counter()
                if first_pkt_t is None and self.reader.physics_packet_id() != pid0:
                    first_pkt_t = now
                if link is not None and ack_t is None:
                    raw = link.read_raw(retries=2)
                    if raw is not None and _seq_reached(raw.cmd_seq_seen, want_seq):
                        ack_t = now
                if now >= self._next_deadline:
                    break
                time.sleep(min(self.poll_s * 4, max(0.0, self._next_deadline - now)))

            if first_pkt_t is None:
                _, first_pkt_t = self.wait_new_packet(pid0)
            tel = self.reader.read()
            if self.require_live and tel.status != AC_LIVE:
                raise ACTimeoutError(f"AC status is {AC_STATUS_NAMES.get(tel.status, tel.status)}"
                                     ", not LIVE; fail-safe applied")
        except ACTimeoutError:
            self.stats.timeouts += 1
            self.failsafe()
            raise
        except BaseException:
            self.failsafe()
            raise

        end = time.perf_counter()
        st = self.stats
        st.steps += 1
        if self._last_return is not None:
            st._push(st.periods, end - self._last_return)
        self._last_return = end
        st._push(st.obs_latency, first_pkt_t - t_send)
        if ack_t is not None:
            st._push(st.ack_latency, ack_t - t_send)
        st._push(st.packets_per_step, float(tel.packet_id - pid0))
        self._last_activity = end
        return tel

    def resync(self) -> None:
        """Restart the tick schedule (call after resets or other long pauses)."""
        self._next_deadline = None
        self._last_return = None
        self.touch()

    # ----------------------------------------------------------------- close
    def close(self, release: bool = True) -> None:
        """Brake, then (by default) hand control back to the user."""
        if self._closed:
            return
        self.failsafe()
        self._closed = True
        try:
            if release:
                with self._lock:
                    self.backend.release()
        finally:
            if self._wd_thread is not None:
                self._wd_thread.join(timeout=0.5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _seq_reached(seen: int, want: int) -> bool:
    """uint32 sequence comparison with wrap-around."""
    return ((seen - want) & 0xFFFFFFFF) < 0x80000000
