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
  previous lap (minimum speed, how close to the limit, offs). Plus the own map (stage 3): the track *shape* it has
  already driven is shown as preview, so on lap 2 it "sees" beyond its sensors where it has been before.

## Stages

| Stage | What | Status |
|---|---|---|
| 1. Explorer | sensor range 50-200 m, memory + lap memory, real circuits except the held-out ones | `race_v1_sensor` stopped after 1.8 h: the put-back rule made it go off ~1/km and it got worse (9/11 -> 3/11 circuits); continued as `race_v2_recover` (WP61) |
| 2. Self-recovery | leaving the road no longer ends the episode or teleports the car: grass/gravel with less grip per wheel and ploughing drag, a barrier 2-20 m past the edge, the driver must rejoin by itself; only the barrier, backwards for 8 s or stuck for 10 s end it; 10 % of episodes start off the road or spun (`train.py --self-recovery`, WP61, D-045) | done: `race_v2_recover` completes 11/11 circuits at 120 m, rejoins after its rare excursions (WP66) |
| 3. Own map | remembers the shape of track already driven (beyond the sensor range), shown as preview - no new inputs (`--own-map`, WP62, D-046) | done: lap 2 at 120 m = full-preview pace without retraining (Nordschleife 7:56.5) |
| 4. Racing others | several cars per circuit in the simulator (self-play: copies of the driver), opponents within sensor range as inputs, simple contact physics, penalty for contact (doubled for the car behind), no bonus for places - being stuck behind already costs progress | built (`--race 4`, WP64, D-048); `race_v3_racing` learned to lean on cars instead of passing (contact was nearly free) - fixed with a per-second contact cost (D-050), training `race_v4_contact` (WP66) |
| 5. Live in the game | the bot drives the second car of an offline session through its own app, MV Race Bot (AI silenced with `physics.setAINoInput`, then `ac.overrideCarControls(1)`), sees only its sensor range of the imported circuit and the owner's car; the owner drives car 0 in the same car model | built (app MV Race Bot, `scripts/race_bot.py`, WP65, D-049); tested against a stand-in game, not yet in the game |

## Measured so far (2026-11-27)

`multi_v5_zigzag/best` (911 GT3 RS), Nordschleife, never trained there, standing-start first lap:
full preview **7:58.4**, then 7:56.0 and 7:56.5 (3 clean laps); sensors 120 m **8:35.3**, then 8:32.7 and 8:33.5 - no lap-by-lap learning yet; 60 m **9:19.7** (for scale: the real 991 GT3 RS did
about 7:20). Validation circuits with a 125 m view before any sensor training: 55 % completed
(full view: ~100 %).

## Measured on 2026-11-28 (WP61-WP65)

* Own map, base driver, 120 m sensors, no retraining - lap 1 / lap 2: Nordschleife 8:35.1 / **7:56.5**
  (full preview 7:56.0), Spa 2:53.5 / 2:43.6, Monza 2:11.9 / 2:02.5.
* Race in our simulator, base driver (cannot see cars), starting last behind 3 slower copies: Spa P1 but
  22 contact steps, Monza a pile-up (all out), Brands Hatch P1 by pushing through. This is what stage 4
  has to fix.
* Training speed with CUDA graphs: 60k decisions/s for the race bot (45k before), results identical.

## Measured on 2026-11-28 after training (WP66)

* Nordschleife (never trained there), 120 m sensors, strict: lap 1 / lap 2 - `race_v3_racing/latest`
  8:22.7 / **7:49.0**, `race_v2_recover/best` 8:26.5 / 7:52.7 (base driver with its own map: 8:35.1 /
  7:56.5; with the whole circuit known in advance: 7:56.0).
* Unseen GP circuits 1.5-4 s per lap faster than the base driver; 11/11 circuits completed (base: 9/11).
* Races: `race_v3_racing/latest` gains 2 places per race with 14-67 contact steps (`best`: up to 666).
* Use `latest.pt` of these runs: `best.pt` (picked by a validation score that is 94-100 % everywhere)
  went off at Spa.

## How to race it (owner)

1. Once: `powershell -ExecutionPolicy Bypass -File mv\ac\plugin\install_plugin.ps1`; in the game enable
   the *MV Race Bot* app.
2. `python scripts\race_bot.py --policy runs\race_v3_racing\latest.pt --track ks_nordschleife__nordschleife
   --vehicle ks_porsche_911_gt3_rs --start-game --laps 3 --dry-run` - first a dry run: the game's AI drives
   car 1 and the bot only prints what it would do (positions, speed, its steering/pedal).
3. The same without `--dry-run`: the bot waits on the grid, then races. The app window must say "pedals
   follow the bot: true"; if the game will not hand them over, the bot stops and says so.