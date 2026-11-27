# 09 — The race bot: a driver to race against in the game

*Owner's request, 2026-11-27.* "A model I could race against on any track, in the same machinery, that
can correct itself instead of being restarted, learns to overtake, based on our training so far. It must
be able to learn a track from nothing - I want to race it on the long Nordschleife without it knowing
anything about the track (realistic lidar range), against myself."

This chapter is the plan. Progress and measured results are in the engineering log (WP60 onwards).

## What "from nothing" means here

A real autonomous car has no map of an unknown circuit: it sees the road only as far as its sensors
reach. Our drivers normally get a *preview* of the track's shape up to 340 m ahead - a map, in effect.
The race bot instead:

* **sees only a sensor range** - track preview and lidar both cut to *R* metres. Automotive lidars are
  specified at about 200-265 m for dark (10 %-reflective) targets; an independent benchmark of eight of
  them measured 25-120 m (50 m on average). Training draws *R* from 50-200 m per episode; the test uses
  **120 m** (the best value measured independently);
* **has never been trained on the test circuit** - both Nordschleife layouts are in `never_train`
  (`tracks/sets.yaml`). Note: `multi_v7_racing` (started before this rule) did train on them, so the race
  bot descends from `multi_v5_zigzag`, which never saw any real circuit in training;
* **learns lap by lap** - the lap memory (WP21) remembers, per 5 m of track, what happened there on the
  previous lap (minimum speed, how close to the limit, offs). Next step: remember the track *shape* it has
  already driven, so on lap 2 it can "see" beyond its sensors where it has been before (its own map).

## Stages

| Stage | What | Status |
|---|---|---|
| 1. Explorer | sensor range 50-200 m, memory + lap memory, real circuits except the held-out ones | training `race_v1_sensor` (8 h) |
| 2. Self-recovery | leaving the road no longer ends the episode or teleports the car: grass/gravel with less grip, the driver must rejoin by itself; only stuck/backwards for long ends it | to build |
| 3. Own map | remembers the shape of track already driven (beyond the sensor range) | to build |
| 4. Racing others | several cars per circuit in the simulator (self-play: copies of the driver), opponents within sensor range as inputs, simple contact physics, reward for places gained, penalty for contact; learns to defend and overtake | to build - the biggest part |
| 5. Live in the game | the bot drives the second car of an offline session through the MV Bridge app (`ac.overrideCarControls(1)` - CSP allows it for AI cars), sees only its sensor range of the imported circuit and the owner's car; the owner drives car 0 in the same car model | to build |

## Measured so far (2026-11-27)

`multi_v5_zigzag/best` (911 GT3 RS), Nordschleife, never trained there, standing-start first lap:
full preview **7:58.4**, then 7:56.0 and 7:56.5 (3 clean laps); sensors 120 m **8:35.3**, then 8:32.7 and 8:33.5 - no lap-by-lap learning yet; 60 m **9:19.7** (for scale: the real 991 GT3 RS did
about 7:20). Validation circuits with a 125 m view before any sensor training: 55 % completed
(full view: ~100 %).
