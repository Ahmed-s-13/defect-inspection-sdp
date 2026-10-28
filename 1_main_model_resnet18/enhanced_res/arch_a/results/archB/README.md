# Architecture B: gc7 + the calibrated layer, quality gate off

Model enhanced_res gc7 (7 classes, with no-defect), crops d1_gc7_fix (D1 with the 27 Oct box corrections), T 1.056, OOD α 0.08 (kNN + AoN), CP α 0.1 (LAC, fitted on gate-passed cal-half crops), miss guard none. Architecture A (v2, 6 classes, quality gate on) is shown next to it where the set allows. A re-aim on undamaged skin counts as the visor speaking.

## Test half (87 families; report only)

| | value (95% interval over families) |
|---|---:|
| **silent miss**: defect crops shown as nothing | **0.3%** (0.1%–0.6%) |
| **visor speaks on no-defect crops** (false alarm) | **26.9%** (21.4%–33.1%) |
| … healed / clean-cropper no-defect | 30.1% / 24.8% |
| CP coverage (target 90%) | 90.4% (84.7%–94.5%) |
| mean set size / single class | 1.48 / 70.9% |
| single wrong defect class shown without warning | 6.7% |
| "POSSIBLE …: check" / re-aim | 8.8% / 2.9% |

Per class: coverage, and what the visor does.

| class | crops | coverage | shown nothing | POSSIBLE | re-aim |
|---|---:|---:|---:|---:|---:|
| corrosion | 257 | 77.8% | 0.0% | 5.4% | 7.0% |
| crack | 636 | 86.9% | 0.2% | 9.9% | 4.9% |
| dent | 1129 | 94.7% | 0.6% | 6.1% | 1.2% |
| missing-head | 132 | 97.0% | 0.0% | 0.0% | 0.0% |
| paint-peel-off | 142 | 85.2% | 0.0% | 6.3% | 1.4% |
| scratch | 121 | 76.0% | 0.0% | 16.5% | 0.8% |
| no-defect | 335 | 96.7% | 73.1% | 19.7% | 4.2% |

**Coverage depends on the class mix.** The test half is 12% no-defect; a walkaround is mostly undamaged skin. Reweighted (defects keep their own mix):

| share of undamaged crops | coverage | visor speaks, per crop |
|---:|---:|---:|
| 12% | 90.4% | 90.9% |
| 50% | 93.1% | 63.3% |
| 90% | 96.0% | 34.1% |

Global LAC does not bound P(defect → shown as nothing) directly, and with 88 calibration families no rate below about 1/88 can be certified; the silent-miss figure above is a measurement, not a guarantee.

## extra_eval: 760 no-defect crops never trained on

| | B | of which healed / clean |
|---|---:|---:|
| shown nothing | 72.8% | 67.6% / 76.2% |
| visor speaks (false alarm) | 27.2% | |

## FGVC / D6: undamaged aircraft (headline test; never used for fitting)

| set | | A: nothing / re-aim / re-take / shown a defect | B: nothing / re-aim / shown a defect |
|---|---|---:|---:|
| aircraft skin patch (no damage) | per crop | 0.0% / 75.7% / 0.1% / 24.2% | 3.9% / 80.1% / 16.0% |
| aircraft skin patch (no damage) | per photo | visor speaks on 100.0% of 1000 photos | visor speaks on 96.1% of 1000 photos |
| whole aircraft (no damage) | per crop | 0.0% / 98.6% / 0.0% / 1.4% | 0.0% / 99.2% / 0.8% |
| whole aircraft (no damage) | per photo | visor speaks on 100.0% of 500 photos | visor speaks on 100.0% of 500 photos |

## Off-target photos (each photo: whole-frame square + box patch)

Held out = AoN refitted without that category (fit categories only); shipped = the deployed AoN, which saw every category below. Shown = a defect or POSSIBLE on the visor (the false accept that matters); nothing = silent.

| category | A shown | B shipped: re-aim / nothing / shown | B held out: re-aim / nothing / shown |
|---|---:|---:|---:|
| Car (fit) | 0.0% | 100.0% / 0.0% / 0.0% | - |
| Human face (fit) | 0.0% | 100.0% / 0.0% / 0.0% | - |
| Ladder (fit) | 0.0% | 100.0% / 0.0% / 0.0% | - |
| Person (fit) | 1.0% | 99.8% / 0.0% / 0.2% | - |
| Tool (fit) | 1.0% | 100.0% / 0.0% / 0.0% | - |
| Truck (fit) | 0.0% | 100.0% / 0.0% / 0.0% | - |
| auto_factory (fit) | 0.0% | 100.0% / 0.0% / 0.0% | - |
| engine_room (fit) | 1.0% | 100.0% / 0.0% / 0.0% | - |
| garage/indoor (fit) | 3.5% | 100.0% / 0.0% / 0.0% | - |
| industrial_area (fit) | 1.0% | 99.0% / 1.0% / 0.0% | - |
| loading_dock (fit) | 3.5% | 99.0% / 0.5% / 0.5% | - |
| Glove (test) | 0.6% | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% |
| Helmet (test) | 0.2% | 100.0% / 0.0% / 0.0% | 98.2% / 0.0% / 1.8% |
| Human hand (test) | 0.8% | 100.0% / 0.0% / 0.0% | 97.8% / 0.0% / 2.2% |
| airport_terminal (test) | 0.0% | 100.0% / 0.0% / 0.0% | 96.5% / 0.0% / 3.5% |
| hangar/indoor (test) | 1.5% | 100.0% / 0.0% / 0.0% | 96.0% / 0.5% / 3.5% |
| hangar/outdoor (test) | 2.0% | 99.5% / 0.5% / 0.0% | 93.0% / 0.5% / 6.5% |
| helmet photo (user) (test) | 0.0% | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% |
| repair_shop (test) | 1.5% | 100.0% / 0.0% / 0.0% | 97.0% / 0.5% / 2.5% |
| runway (test) | 2.5% | 100.0% / 0.0% / 0.0% | 88.5% / 1.0% / 10.5% |
| storage_room (test) | 0.0% | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% |
| **all held-out categories** | 0.9% | 0.0% shown | 3.2% shown |
| wallpaper patches | 15.2% | 80.5% / 2.2% / 17.2% | |
| static / colours / gradients | 0.0% | 87.5% / 9.4% / 3.1% | |

The user's hard-hat photo: A re-aim, re-aim; B re-aim, re-aim.

## D5: other photos of dents and missing heads

| | A | B |
|---|---:|---:|
| CP coverage | 95.0% | 91.6% |
| stopped by a gate | 6.6% | 0.3% |
| silent miss (shown nothing) | - | 2.2% |

Defects that never become a crop are silent misses no decision layer can see; how frames become crops on the helmet is still open.
