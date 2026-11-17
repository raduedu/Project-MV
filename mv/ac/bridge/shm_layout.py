"""Exact ``ctypes`` layouts of Assetto Corsa's shared-memory pages.

AC (the original game, v1.16.x — *not* ACC or EVO) publishes three named
file mappings while a session is loaded:

=========================  =====================  ===========================
mapping name               struct                 update rate
=========================  =====================  ===========================
``Local\\acpmf_physics``    ``SPageFilePhysics``    every physics step (~333 Hz
                                                   nominal; measured by
                                                   ``scripts/ac_bridge_check``)
``Local\\acpmf_graphics``   ``SPageFileGraphic``    every rendered frame
``Local\\acpmf_static``     ``SPageFileStatic``     once per session
=========================  =====================  ===========================

Layout provenance (cross-checked, see docs/07-assetto-corsa-bridge.md §9)
-----------------------------------------------------------------------
* ``mdjarv/assettocorsasharedmemory`` (C#, ``Pack = 4``, ``CharSet.Unicode``)
  — physics up to ``localVelocity``, graphics up to ``mandatoryPitDone``,
  static up to ``pitWindowEnd``.
* ``Spacefreak18/simapi`` ``acdata.h`` (``#pragma pack(4)``, ``char16_t``) and
  the Kunos AC-EVO shared-memory reference (which inherits the AC1 physics
  header verbatim) — physics fields after ``localVelocity``
  (``P2PActivations`` … ``tyreTemp``).  These were added in AC 1.14 together
  with Push-To-Pass (see AC ``changelog.txt``, 1.14).
* The game's own ``changelog.txt`` (local install) for the version history,
  in particular ``wchar_t trackConfiguration[33]`` (1.7.2).  The C# library
  declares ``[15]`` for that field; we trust the Kunos changelog.  Only
  static fields *after* ``trackConfiguration`` depend on this and none of
  them is used for control.

Why ``_pack_ = 4``: the Kunos header is compiled with ``#pragma pack(4)``.
All members are 4-byte except the ``wchar_t`` strings; with pack 4 a ``float``
after a ``wchar_t[33]`` (66 bytes) is aligned to the next multiple of 4.  It
makes no difference anywhere else, but getting it wrong would shift every
field after ``tyreCompound`` by 2 bytes, so it is pinned and tested.

Why ``c_wchar``: on Windows ``wchar_t`` is UTF-16 (2 bytes), which is what
AC writes.  ``ctypes.c_wchar`` is 2 bytes on Windows (and 4 on Linux; the
tests therefore only check sizes on Windows).

Units in these raw structs are AC's (km/h, g, ms, AC axes).  Never consume
them directly — use :mod:`mv.ac.bridge.telemetry`, which converts to SI and
to the project's axis convention.
"""

from __future__ import annotations

import ctypes
from ctypes import c_float, c_int32, c_wchar

__all__ = [
    "SPageFilePhysics", "SPageFileGraphic", "SPageFileStatic",
    "PHYSICS_MAP_NAME", "GRAPHICS_MAP_NAME", "STATIC_MAP_NAME",
    "PHYSICS_SIZE", "GRAPHICS_SIZE", "STATIC_SIZE",
    "AC_OFF", "AC_REPLAY", "AC_LIVE", "AC_PAUSE", "AC_STATUS_NAMES",
    "AC_SESSION_NAMES",
]

PHYSICS_MAP_NAME = "Local\\acpmf_physics"
GRAPHICS_MAP_NAME = "Local\\acpmf_graphics"
STATIC_MAP_NAME = "Local\\acpmf_static"

# AC_STATUS (graphics.status)
AC_OFF, AC_REPLAY, AC_LIVE, AC_PAUSE = 0, 1, 2, 3
AC_STATUS_NAMES = {AC_OFF: "OFF", AC_REPLAY: "REPLAY", AC_LIVE: "LIVE", AC_PAUSE: "PAUSE"}

# AC_SESSION_TYPE (graphics.session) — AC1 values (ACC/EVO differ!)
AC_SESSION_NAMES = {
    -1: "UNKNOWN", 0: "PRACTICE", 1: "QUALIFY", 2: "RACE", 3: "HOTLAP",
    4: "TIME_ATTACK", 5: "DRIFT", 6: "DRAG",
}

_f4 = c_float * 4
_f3 = c_float * 3


class _Packed(ctypes.Structure):
    _pack_ = 4
    _layout_ = "ms"  # Python >= 3.14 wants this with _pack_; ignored before.


class SPageFilePhysics(_Packed):
    """``SPageFilePhysics`` as of AC 1.16 (shared-memory version 1.7).

    Wheel arrays are ordered FL, FR, RL, RR.  Vectors are in AC axes:
    car-local X = left, Y = up, Z = forward; world frame has the same
    handedness with Y up.  (Verified from the install: every left-hand-drive
    car has ``DRIVEREYES`` x > 0, every right-hand-drive car x < 0.)
    """

    _fields_ = [
        ("packetId", c_int32),
        ("gas", c_float),
        ("brake", c_float),
        ("fuel", c_float),            # litres
        ("gear", c_int32),            # 0 = R, 1 = N, 2 = 1st ...
        ("rpms", c_int32),
        ("steerAngle", c_float),      # normalised, see telemetry.py
        ("speedKmh", c_float),
        ("velocity", _f3),            # world, m/s, AC axes
        ("accG", _f3),                # car-local, g: [0]=X lat, [1]=Y vert, [2]=Z long
        ("wheelSlip", _f4),
        ("wheelLoad", _f4),           # N
        ("wheelsPressure", _f4),      # psi
        ("wheelAngularSpeed", _f4),   # rad/s
        ("tyreWear", _f4),
        ("tyreDirtyLevel", _f4),
        ("tyreCoreTemperature", _f4),  # deg C
        ("camberRAD", _f4),
        ("suspensionTravel", _f4),    # m
        ("drs", c_float),
        ("tc", c_float),              # TC intervention (0 = none)
        ("heading", c_float),         # rad
        ("pitch", c_float),           # rad
        ("roll", c_float),            # rad
        ("cgHeight", c_float),
        ("carDamage", c_float * 5),
        ("numberOfTyresOut", c_int32),
        ("pitLimiterOn", c_int32),
        ("abs", c_float),             # ABS intervention (0 = none)
        ("kersCharge", c_float),      # 0..1
        ("kersInput", c_float),       # 0..1
        ("autoShifterOn", c_int32),
        ("rideHeight", c_float * 2),
        # since 1.5
        ("turboBoost", c_float),
        ("ballast", c_float),
        ("airDensity", c_float),
        # since 1.6
        ("airTemp", c_float),
        ("roadTemp", c_float),
        ("localAngularVel", _f3),     # car-local rad/s, AC axes
        ("finalFF", c_float),
        # since 1.7
        ("performanceMeter", c_float),
        ("engineBrake", c_int32),
        ("ersRecoveryLevel", c_int32),
        ("ersPowerLevel", c_int32),
        ("ersHeatCharging", c_int32),
        ("ersIsCharging", c_int32),
        ("kersCurrentKJ", c_float),
        ("drsAvailable", c_int32),
        ("drsEnabled", c_int32),
        ("brakeTemp", _f4),
        # since 1.10
        ("clutch", c_float),
        ("tyreTempI", _f4),
        ("tyreTempM", _f4),
        ("tyreTempO", _f4),
        # since 1.10.2
        ("isAIControlled", c_int32),
        # since 1.11
        ("tyreContactPoint", _f3 * 4),
        ("tyreContactNormal", _f3 * 4),
        ("tyreContactHeading", _f3 * 4),
        ("brakeBias", c_float),
        # since 1.12
        ("localVelocity", _f3),       # car-local m/s, AC axes
        # since 1.14
        ("P2PActivations", c_int32),
        ("P2PStatus", c_int32),
        ("currentMaxRpm", c_int32),
        ("mz", _f4),
        ("fx", _f4),
        ("fy", _f4),
        ("slipRatio", _f4),
        ("slipAngle", _f4),           # rad (per reference doc; unverified in-game)
        ("tcinAction", c_int32),
        ("absInAction", c_int32),
        ("suspensionDamage", _f4),
        ("tyreTemp", _f4),            # tyre surface temperature, deg C
    ]


class SPageFileGraphic(_Packed):
    """``SPageFileGraphic`` as of AC 1.13+ (ends with ``mandatoryPitDone``)."""

    _fields_ = [
        ("packetId", c_int32),
        ("status", c_int32),          # AC_STATUS
        ("session", c_int32),         # AC_SESSION_TYPE
        ("currentTime", c_wchar * 15),
        ("lastTime", c_wchar * 15),
        ("bestTime", c_wchar * 15),
        ("split", c_wchar * 15),
        ("completedLaps", c_int32),
        ("position", c_int32),
        ("iCurrentTime", c_int32),    # ms
        ("iLastTime", c_int32),       # ms
        ("iBestTime", c_int32),       # ms
        ("sessionTimeLeft", c_float),
        ("distanceTraveled", c_float),
        ("isInPit", c_int32),
        ("currentSectorIndex", c_int32),
        ("lastSectorTime", c_int32),
        ("numberOfLaps", c_int32),
        ("tyreCompound", c_wchar * 33),
        ("replayTimeMultiplier", c_float),
        ("normalizedCarPosition", c_float),  # spline position 0..1
        ("carCoordinates", _f3),             # world, m, AC axes
        ("penaltyTime", c_float),
        ("flag", c_int32),
        ("idealLineOn", c_int32),
        # since 1.5
        ("isInPitLane", c_int32),
        ("surfaceGrip", c_float),
        # since 1.13
        ("mandatoryPitDone", c_int32),
    ]


class SPageFileStatic(_Packed):
    """``SPageFileStatic`` as of AC 1.13+ (ends with ``pitWindowEnd``)."""

    _fields_ = [
        ("smVersion", c_wchar * 15),
        ("acVersion", c_wchar * 15),
        ("numberOfSessions", c_int32),
        ("numCars", c_int32),
        ("carModel", c_wchar * 33),
        ("track", c_wchar * 33),
        ("playerName", c_wchar * 33),
        ("playerSurname", c_wchar * 33),
        ("playerNick", c_wchar * 33),
        ("sectorCount", c_int32),
        ("maxTorque", c_float),
        ("maxPower", c_float),
        ("maxRpm", c_int32),
        ("maxFuel", c_float),
        ("suspensionMaxTravel", _f4),
        ("tyreRadius", _f4),
        # since 1.5
        ("maxTurboBoost", c_float),
        ("deprecated_1", c_float),
        ("deprecated_2", c_float),
        ("penaltiesEnabled", c_int32),
        ("aidFuelRate", c_float),
        ("aidTireRate", c_float),
        ("aidMechanicalDamage", c_float),
        ("aidAllowTyreBlankets", c_int32),
        ("aidStability", c_float),
        ("aidAutoClutch", c_int32),
        ("aidAutoBlip", c_int32),
        # since 1.7.1
        ("hasDRS", c_int32),
        ("hasERS", c_int32),
        ("hasKERS", c_int32),
        ("kersMaxJ", c_float),
        ("engineBrakeSettingsCount", c_int32),
        ("ersPowerControllerCount", c_int32),
        # since 1.7.2
        ("trackSPlineLength", c_float),       # m
        ("trackConfiguration", c_wchar * 33),
        # since 1.10.2
        ("ersMaxJ", c_float),
        # since 1.13
        ("isTimedRace", c_int32),
        ("hasExtraLap", c_int32),
        ("carSkin", c_wchar * 33),
        ("reversedGridPositions", c_int32),
        ("pitWindowStart", c_int32),
        ("pitWindowEnd", c_int32),
    ]


# Sizes computed by hand from the field lists above (see tests).  AC allocates
# at least this much (mappings are page-rounded to 4 KiB).
PHYSICS_SIZE = 712
GRAPHICS_SIZE = 288
STATIC_SIZE = 684
