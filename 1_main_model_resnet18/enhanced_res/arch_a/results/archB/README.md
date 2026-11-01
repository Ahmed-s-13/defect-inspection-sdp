# Architecture B: gc7 + the calibrated layer, quality gate off

Model enhanced_res gc7 (7 classes, with no-defect), crops d1_gc7_fix (D1 with the 27 Oct box corrections), family-weighted validation T 1.089562, OOD α 0.05 (kNN + AoN), CP α 0.1 (LAC, fitted on gate-passed cal-half crops), miss guard none. Architecture A (v2, 6 classes, quality gate on) is shown next to it where the set allows. A re-aim on undamaged skin counts as the visor speaking.

## Test half (87 families; report only)

| | value (95% interval over families) |
|---|---:|
| **silent miss**: defect crops shown as nothing | **0.3%** (0.1%–0.6%) |
| **visor speaks on no-defect crops** (false alarm) | **28.1%** (22.3%–34.4%) |
| … healed / clean-cropper no-defect | 30.8% / 26.2% |
| CP coverage (target 90%) | 91.0% (85.4%–94.9%) |
| mean set size / single class | 1.54 / 68.8% |
| single wrong defect class shown without warning | 6.1% |
| "POSSIBLE …: check" / re-aim | 10.4% / 1.5% |

Per class: coverage, and what the visor does.

| class | crops | coverage | shown nothing | POSSIBLE | re-aim |
|---|---:|---:|---:|---:|---:|
| corrosion | 257 | 80.2% | 0.0% | 7.0% | 4.7% |
| crack | 636 | 87.3% | 0.2% | 12.6% | 2.7% |
| dent | 1129 | 95.3% | 0.6% | 6.6% | 0.3% |
| missing-head | 132 | 97.0% | 0.0% | 0.0% | 0.0% |
| paint-peel-off | 142 | 85.9% | 0.0% | 9.9% | 0.7% |
| scratch | 121 | 76.0% | 0.0% | 18.2% | 0.0% |
| no-defect | 335 | 97.3% | 71.9% | 23.0% | 2.4% |

**Coverage depends on the class mix.** The test half is 12% no-defect; a walkaround is mostly undamaged skin. Reweighted (defects keep their own mix):

| share of undamaged crops | coverage | visor speaks, per crop |
|---:|---:|---:|
| 12% | 91.0% | 91.1% |
| 50% | 93.7% | 63.9% |
| 90% | 96.6% | 35.2% |

Global LAC does not bound P(defect → shown as nothing) directly, and with 88 calibration families no rate below about 1/88 can be certified; the silent-miss figure above is a measurement, not a guarantee.

## extra_eval: 760 no-defect crops never trained on

| | B | of which healed / clean |
|---|---:|---:|
| shown nothing | 72.4% | 66.3% / 76.4% |
| visor speaks (false alarm) | 27.6% | |

## FGVC / D6: undamaged aircraft (headline test; never used for fitting)

| set | | A: nothing / re-aim / re-take / shown a defect | B: nothing / re-aim / shown a defect |
|---|---|---:|---:|
| aircraft skin patch (no damage) | per crop | 0.0% / 75.7% / 0.1% / 24.2% | 4.7% / 71.9% / 23.4% |
| aircraft skin patch (no damage) | per photo | visor speaks on 100.0% of 1000 photos | visor speaks on 95.3% of 1000 photos |
| whole aircraft (no damage) | per crop | 0.0% / 98.6% / 0.0% / 1.4% | 0.0% / 97.6% / 2.4% |
| whole aircraft (no damage) | per photo | visor speaks on 100.0% of 500 photos | visor speaks on 100.0% of 500 photos |

## Off-target photos (each photo: whole-frame square + box patch)

Held out = AoN refitted without that category (fit categories only); shipped = the deployed AoN, which saw every category below. Shown = a defect or POSSIBLE on the visor (the false accept that matters); nothing = silent.

| category | A shown | B shipped: re-aim / nothing / shown | B held out: re-aim / nothing / shown |
|---|---:|---:|---:|
| Car (fit) | 0.0% | 100.0% / 0.0% / 0.0% | - |
| Human face (fit) | 0.0% | 100.0% / 0.0% / 0.0% | - |
| Ladder (fit) | 0.0% | 100.0% / 0.0% / 0.0% | - |
| Person (fit) | 1.0% | 99.2% / 0.0% / 0.8% | - |
| Tool (fit) | 1.0% | 98.0% / 0.0% / 2.0% | - |
| Truck (fit) | 0.0% | 100.0% / 0.0% / 0.0% | - |
| auto_factory (fit) | 0.0% | 99.0% / 0.0% / 1.0% | - |
| engine_room (fit) | 1.0% | 96.5% / 0.0% / 3.5% | - |
| garage/indoor (fit) | 3.5% | 95.5% / 1.0% / 3.5% | - |
| industrial_area (fit) | 1.0% | 96.5% / 1.5% / 2.0% | - |
| loading_dock (fit) | 3.5% | 96.0% / 1.5% / 2.5% | - |
| Glove (test) | 0.6% | 99.4% / 0.0% / 0.6% | 98.2% / 0.0% / 1.8% |
| Helmet (test) | 0.2% | 100.0% / 0.0% / 0.0% | 96.5% / 0.0% / 3.5% |
| Human hand (test) | 0.8% | 99.5% / 0.0% / 0.5% | 95.8% / 0.0% / 4.2% |
| airport_terminal (test) | 0.0% | 98.5% / 0.0% / 1.5% | 95.5% / 0.0% / 4.5% |
| hangar/indoor (test) | 1.5% | 99.0% / 0.5% / 0.5% | 93.5% / 0.5% / 6.0% |
| hangar/outdoor (test) | 2.0% | 96.5% / 0.5% / 3.0% | 86.0% / 1.0% / 13.0% |
| helmet photo (user) (test) | 0.0% | 100.0% / 0.0% / 0.0% | 50.0% / 0.0% / 50.0% |
| repair_shop (test) | 1.5% | 96.5% / 0.5% / 3.0% | 95.5% / 0.5% / 4.0% |
| runway (test) | 2.5% | 95.0% / 1.0% / 4.0% | 77.0% / 1.0% / 22.0% |
| storage_room (test) | 0.0% | 100.0% / 0.0% / 0.0% | 99.0% / 0.0% / 1.0% |
| **all held-out categories** | 0.9% | 1.2% shown | 6.3% shown |
| wallpaper patches | 15.2% | 70.2% / 2.8% / 27.0% | |
| static / colours / gradients | 0.0% | 78.8% / 15.6% / 5.6% | |

The user's hard-hat photo: A re-aim, re-aim; B re-aim, re-aim.

## D5: other photos of dents and missing heads

| | A | B |
|---|---:|---:|
| CP coverage | 95.0% | 92.0% |
| stopped by a gate | 6.6% | 0.3% |
| silent miss (shown nothing) | - | 2.2% |

Defects that never become a crop are silent misses no decision layer can see; how frames become crops on the helmet is still open.
