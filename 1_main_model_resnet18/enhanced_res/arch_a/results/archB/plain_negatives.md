# Tried and left out: plain-surface negatives for B's AoN (28 Oct)

> Historical experiment: measured before the OOD change, using OOD α 0.08 where applicable (the threshold sweep lists its own values). Current B uses OOD α 0.05 and CP α 0.10. These recorded numbers are not current-deployment results; see the regenerated [Architecture B report](README.md).

> B's AoN was refitted with 3,880 extra negative crops (10 Places365 plain scenes, 47 DTD textures; about half held out by category), because held-out runway crops were shown as a defect 10.5% of the time. It did not fix runway (11.5%) and made the original held-out categories worse (helmet 1.8% -> 4.5%, hand 2.2% -> 6.8%, hangar outdoor 6.5% -> 8.0%) and the visor spoke more on clean crops (26.9% -> 29.0%); only wallpapers (17.2% -> 12.0% shown) and static (3.1% -> 2.5%) improved. So B keeps the original negatives; `calibrate.py --plain-negatives` turns them back on. Below: the full B report and choices as they came out with them.

## Architecture B: gc7 + the calibrated layer, quality gate off

Model enhanced_res gc7 (7 classes, with no-defect), crops d1_gc7_fix (D1 with the 27 Oct box corrections), T 1.056, OOD α 0.08 (kNN + AoN), CP α 0.1 (LAC, fitted on gate-passed cal-half crops), miss guard none. Architecture A (v2, 6 classes, quality gate on) is shown next to it where the set allows. A re-aim on undamaged skin counts as the visor speaking.

## Test half (87 families; report only)

| | value (95% interval over families) |
|---|---:|
| **silent miss**: defect crops shown as nothing | **0.3%** (0.1%–0.6%) |
| **visor speaks on no-defect crops** (false alarm) | **29.0%** (23.1%–35.5%) |
| … healed / clean-cropper no-defect | 33.1% / 26.2% |
| CP coverage (target 90%) | 90.4% (84.7%–94.5%) |
| mean set size / single class | 1.48 / 70.9% |
| single wrong defect class shown without warning | 6.7% |
| "POSSIBLE …: check" / re-aim | 8.7% / 3.5% |

Per class: coverage, and what the visor does.

| class | crops | coverage | shown nothing | POSSIBLE | re-aim |
|---|---:|---:|---:|---:|---:|
| corrosion | 257 | 77.8% | 0.0% | 5.4% | 7.0% |
| crack | 636 | 86.9% | 0.2% | 9.9% | 5.0% |
| dent | 1129 | 94.7% | 0.6% | 5.9% | 1.9% |
| missing-head | 132 | 97.0% | 0.0% | 0.0% | 0.0% |
| paint-peel-off | 142 | 85.2% | 0.0% | 6.3% | 2.8% |
| scratch | 121 | 76.0% | 0.0% | 16.5% | 0.8% |
| no-defect | 335 | 96.7% | 71.0% | 19.7% | 6.3% |

**Coverage depends on the class mix.** The test half is 12% no-defect; a walkaround is mostly undamaged skin. Reweighted (defects keep their own mix):

| share of undamaged crops | coverage | visor speaks, per crop |
|---:|---:|---:|
| 12% | 90.4% | 91.2% |
| 50% | 93.1% | 64.3% |
| 90% | 96.0% | 36.0% |

Global LAC does not bound P(defect → shown as nothing) directly, and with 88 calibration families no rate below about 1/88 can be certified; the silent-miss figure above is a measurement, not a guarantee.

## extra_eval: 760 no-defect crops never trained on

| | B | of which healed / clean |
|---|---:|---:|
| shown nothing | 71.7% | 66.3% / 75.3% |
| visor speaks (false alarm) | 28.3% | |

## FGVC / D6: undamaged aircraft (headline test; never used for fitting)

| set | | A: nothing / re-aim / re-take / shown a defect | B: nothing / re-aim / shown a defect |
|---|---|---:|---:|
| aircraft skin patch (no damage) | per crop | 0.0% / 75.7% / 0.1% / 24.2% | 2.6% / 82.0% / 15.4% |
| aircraft skin patch (no damage) | per photo | visor speaks on 100.0% of 1000 photos | visor speaks on 97.4% of 1000 photos |
| whole aircraft (no damage) | per crop | 0.0% / 98.6% / 0.0% / 1.4% | 0.0% / 98.8% / 1.2% |
| whole aircraft (no damage) | per photo | visor speaks on 100.0% of 500 photos | visor speaks on 100.0% of 500 photos |

## Off-target photos (each photo: whole-frame square + box patch)

Held out = AoN refitted without that category (fit categories only); shipped = the deployed AoN, which saw every category below. Shown = a defect or POSSIBLE on the visor (the false accept that matters); nothing = silent.

| category | A shown | B shipped: re-aim / nothing / shown | B held out: re-aim / nothing / shown |
|---|---:|---:|---:|
| Car (fit) | 0.0% | 99.0% / 0.0% / 1.0% | - |
| Human face (fit) | 0.0% | 100.0% / 0.0% / 0.0% | - |
| Ladder (fit) | 0.0% | 100.0% / 0.0% / 0.0% | - |
| Person (fit) | 1.0% | 98.2% / 0.0% / 1.8% | - |
| Tool (fit) | 1.0% | 96.7% / 0.0% / 3.3% | - |
| Truck (fit) | 0.0% | 99.5% / 0.0% / 0.5% | - |
| airfield (fit) | 5.0% | 98.5% / 0.0% / 1.5% | - |
| auto_factory (fit) | 0.0% | 98.0% / 0.0% / 2.0% | - |
| desert/sand (fit) | 27.5% | 84.0% / 1.0% / 15.0% | - |
| driveway (fit) | 14.5% | 95.0% / 0.0% / 5.0% | - |
| engine_room (fit) | 1.0% | 95.0% / 0.0% / 5.0% | - |
| garage/indoor (fit) | 3.5% | 94.5% / 0.5% / 5.0% | - |
| highway (fit) | 14.0% | 96.5% / 1.5% / 2.0% | - |
| industrial_area (fit) | 1.0% | 96.5% / 1.5% / 2.0% | - |
| landing_deck (fit) | 14.5% | 91.0% / 1.5% / 7.5% | - |
| loading_dock (fit) | 3.5% | 96.0% / 1.5% / 2.5% | - |
| parking_garage/indoor (fit) | 6.0% | 95.5% / 1.0% / 3.5% | - |
| parking_lot (fit) | 8.0% | 96.0% / 1.5% / 2.5% | - |
| sky (fit) | 22.5% | 89.0% / 1.0% / 10.0% | - |
| texture: banded (fit) | 12.5% | 100.0% / 0.0% / 0.0% | - |
| texture: braided (fit) | 7.5% | 100.0% / 0.0% / 0.0% | - |
| texture: bumpy (fit) | 5.0% | 97.5% / 0.0% / 2.5% | - |
| texture: cobwebbed (fit) | 0.0% | 100.0% / 0.0% / 0.0% | - |
| texture: crosshatched (fit) | 5.0% | 97.5% / 2.5% / 0.0% | - |
| texture: dotted (fit) | 2.5% | 100.0% / 0.0% / 0.0% | - |
| texture: flecked (fit) | 5.0% | 97.5% / 0.0% / 2.5% | - |
| texture: frilly (fit) | 12.5% | 100.0% / 0.0% / 0.0% | - |
| texture: grid (fit) | 2.5% | 100.0% / 0.0% / 0.0% | - |
| texture: honeycombed (fit) | 2.5% | 100.0% / 0.0% / 0.0% | - |
| texture: knitted (fit) | 0.0% | 100.0% / 0.0% / 0.0% | - |
| texture: lined (fit) | 2.5% | 100.0% / 0.0% / 0.0% | - |
| texture: matted (fit) | 2.5% | 100.0% / 0.0% / 0.0% | - |
| texture: paisley (fit) | 0.0% | 100.0% / 0.0% / 0.0% | - |
| texture: pitted (fit) | 12.5% | 92.5% / 5.0% / 2.5% | - |
| texture: polka-dotted (fit) | 12.5% | 92.5% / 2.5% / 5.0% | - |
| texture: potholed (fit) | 45.0% | 82.5% / 5.0% / 12.5% | - |
| texture: smeared (fit) | 20.0% | 100.0% / 0.0% / 0.0% | - |
| texture: sprinkled (fit) | 5.0% | 95.0% / 0.0% / 5.0% | - |
| texture: stratified (fit) | 2.5% | 100.0% / 0.0% / 0.0% | - |
| texture: studded (fit) | 22.5% | 97.5% / 0.0% / 2.5% | - |
| texture: veined (fit) | 2.5% | 100.0% / 0.0% / 0.0% | - |
| texture: woven (fit) | 0.0% | 100.0% / 0.0% / 0.0% | - |
| texture: zigzagged (fit) | 0.0% | 100.0% / 0.0% / 0.0% | - |
| Glove (test) | 0.6% | 100.0% / 0.0% / 0.0% | 97.6% / 0.0% / 2.4% |
| Helmet (test) | 0.2% | 99.5% / 0.0% / 0.5% | 95.5% / 0.0% / 4.5% |
| Human hand (test) | 0.8% | 98.8% / 0.0% / 1.2% | 93.2% / 0.0% / 6.8% |
| airport_terminal (test) | 0.0% | 98.0% / 0.0% / 2.0% | 97.0% / 0.0% / 3.0% |
| construction_site (test) | 9.0% | 97.0% / 0.5% / 2.5% | 94.5% / 0.5% / 5.0% |
| hangar/indoor (test) | 1.5% | 97.5% / 0.5% / 2.0% | 95.0% / 0.5% / 4.5% |
| hangar/outdoor (test) | 2.0% | 95.0% / 0.5% / 4.5% | 91.5% / 0.5% / 8.0% |
| helmet photo (user) (test) | 0.0% | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% |
| repair_shop (test) | 1.5% | 96.0% / 0.5% / 3.5% | 96.0% / 0.5% / 3.5% |
| runway (test) | 2.5% | 92.0% / 1.0% / 7.0% | 87.5% / 1.0% / 11.5% |
| storage_room (test) | 0.0% | 99.5% / 0.0% / 0.5% | 99.0% / 0.0% / 1.0% |
| street (test) | 3.5% | 97.5% / 0.0% / 2.5% | 95.0% / 0.5% / 4.5% |
| texture: blotchy (test) | 10.0% | 95.0% / 0.0% / 5.0% | 90.0% / 2.5% / 7.5% |
| texture: bubbly (test) | 7.5% | 97.5% / 2.5% / 0.0% | 97.5% / 2.5% / 0.0% |
| texture: chequered (test) | 0.0% | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% |
| texture: cracked (test) | 15.0% | 100.0% / 0.0% / 0.0% | 97.5% / 0.0% / 2.5% |
| texture: crystalline (test) | 22.5% | 95.0% / 0.0% / 5.0% | 77.5% / 7.5% / 15.0% |
| texture: fibrous (test) | 0.0% | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% |
| texture: freckled (test) | 25.0% | 87.5% / 0.0% / 12.5% | 72.5% / 10.0% / 17.5% |
| texture: gauzy (test) | 52.5% | 75.0% / 5.0% / 20.0% | 55.0% / 5.0% / 40.0% |
| texture: grooved (test) | 7.5% | 97.5% / 0.0% / 2.5% | 95.0% / 0.0% / 5.0% |
| texture: interlaced (test) | 7.5% | 100.0% / 0.0% / 0.0% | 97.5% / 0.0% / 2.5% |
| texture: lacelike (test) | 0.0% | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% |
| texture: marbled (test) | 12.5% | 92.5% / 5.0% / 2.5% | 75.0% / 10.0% / 15.0% |
| texture: meshed (test) | 2.5% | 97.5% / 0.0% / 2.5% | 97.5% / 0.0% / 2.5% |
| texture: perforated (test) | 12.5% | 97.5% / 0.0% / 2.5% | 97.5% / 0.0% / 2.5% |
| texture: pleated (test) | 22.5% | 97.5% / 2.5% / 0.0% | 90.0% / 5.0% / 5.0% |
| texture: porous (test) | 12.5% | 92.5% / 0.0% / 7.5% | 80.0% / 7.5% / 12.5% |
| texture: scaly (test) | 7.5% | 97.5% / 2.5% / 0.0% | 97.5% / 2.5% / 0.0% |
| texture: spiralled (test) | 5.0% | 97.5% / 0.0% / 2.5% | 97.5% / 0.0% / 2.5% |
| texture: stained (test) | 32.5% | 92.5% / 0.0% / 7.5% | 77.5% / 7.5% / 15.0% |
| texture: striped (test) | 12.5% | 92.5% / 2.5% / 5.0% | 90.0% / 5.0% / 5.0% |
| texture: swirly (test) | 0.0% | 97.5% / 0.0% / 2.5% | 97.5% / 0.0% / 2.5% |
| texture: waffled (test) | 7.5% | 100.0% / 0.0% / 0.0% | 97.5% / 0.0% / 2.5% |
| texture: wrinkled (test) | 27.5% | 95.0% / 0.0% / 5.0% | 87.5% / 0.0% / 12.5% |
| **all held-out categories** | 4.8% | 2.5% shown | 5.7% shown |
| wallpaper patches | 15.2% | 86.8% / 1.2% / 12.0% | |
| static / colours / gradients | 0.0% | 95.3% / 2.2% / 2.5% | |

The user's hard-hat photo: A re-aim, re-aim; B re-aim, re-aim.

## D5: other photos of dents and missing heads

| | A | B |
|---|---:|---:|
| CP coverage | 95.0% | 91.6% |
| stopped by a gate | 6.6% | 0.6% |
| silent miss (shown nothing) | - | 2.2% |

Defects that never become a crop are silent misses no decision layer can see; how frames become crops on the helmet is still open.

## Architecture B: choices made inside the cal half (test half not used)

Model gc7, crops d1_gc7_fix, quality gate off. 200 random halvings of the 88 cal-half families (fit on 44, measure on the other 44); mean (5th–95th percentile).

## 1. OOD α

| OOD α | defect crops stopped (per family) | no-defect crops stopped (per family) | off-target photos (AoN trained on these) passing | wallpaper patches passing | static passing |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 1.1% (0.0–3.1%) | 0.8% (0.0–2.0%) | 7.3% | 20.8% | 9.4% |
| 0.08 | 4.5% (1.1–10.0%) | 5.0% (1.2–8.8%) | 3.4% | 13.2% | 4.7% |
| 0.10 | 6.3% (1.8–12.3%) | 6.3% (2.2–11.5%) | 2.3% | 9.0% | 3.1% |
| 0.13 | 9.0% (3.1–16.1%) | 8.1% (2.7–13.3%) | 1.7% | 6.2% | 1.6% |
| 0.15 | 10.9% (3.8–17.9%) | 9.3% (3.8–15.6%) | 1.3% | 5.5% | 1.6% |

## 2. Miss guard (OOD α 0.08, CP α 0.1)

Silent miss = a defect crop shown as nothing. A lone no-defect with the guard failing is shown as "POSSIBLE …: check" instead. With 44 calibration families, α_miss below about 1/45 can't be certified.

| miss guard | silent miss | no-defect shown nothing |
|---|---:|---:|
| none (plain LAC) | 0.5% (0.1–1.0%) | 70.8% (61.7–78.9%) |
| α_miss 0.02 | 0.0% (0.0–0.0%) | 0.0% (0.0–0.0%) |
| α_miss 0.05 | 0.5% (0.1–1.0%) | 70.8% (61.7–78.8%) |
| α_miss 0.10 | 0.5% (0.1–1.0%) | 70.8% (61.7–78.9%) |

## 3. Damaged images on the silent path (deployed B, no quality gate)

428 cal-half defect crops (≤ 6 per family) and 14 synthetic corruptions of them: share shown as nothing (a silent miss).

| set | shown nothing |
|---|---:|
| clean | 1.2% |
| blur s1 | 3.5% |
| blur s2 | 3.3% |
| motion blur s1 | 2.3% |
| motion blur s2 | 3.3% |
| dark s1 | 4.2% |
| dark s2 | 0.9% |
| overexposed s1 | 1.4% |
| overexposed s2 | 0.2% |
| glare s1 | 2.1% |
| glare s2 | 1.6% |
| noise s1 | 0.0% |
| noise s2 | 0.0% |
| jpeg s1 | 4.9% |
| jpeg s2 | 4.9% |
