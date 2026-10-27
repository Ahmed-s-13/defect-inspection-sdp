# gc7: test report

enhanced_res with 7 classes (the six defects + no-defect), trained on [d1_gc7](../../../d1_gc7/README.md).
Written by `../../build_head_gc7.py`, which checks the numbers in `summary.json`. Every choice was made on val; test
was only read here. One training run (seed 0).

| file | what |
|---|---|
| `summary.json` | every number below, plus the no-defect cut for 1 / 2 / 5% val misses ("dial") |
| `confusion_gc7.png` | the test confusion matrix (7 classes, top answer) |
| `conformal_study.json` | architecture A with 7 classes: the old v2 calibration kept vs recalibrated, 100 re-splits, both gates (see `../../arch_a/README.md`) |

## d1_gc7 test (4,255 defect crops + 615 no-defect crops, 175 source groups)

| | v2 (6 classes) | **gc7** |
|---|---|---|
| 7-class accuracy / macro-F1 (95% group interval) | n/a | 0.798 / **0.747** (0.705–0.781) |
| defects: accuracy / macro-F1, top answer | 0.804 / 0.738 | 0.781 / 0.738 |
| defects: accuracy / macro-F1, best defect class | 0.804 / 0.738 | **0.805 / 0.742** |
| defects called no-defect | n/a | 3.9% (corrosion 1.9, crack 5.8, dent 2.9, missing-head 3.2, paint-peel-off 0.8, scratch 8.5) |
| no-defect crops called no-defect | 0% (dent 54%, scratch 24%, crack 7%, ...) | **91.1%** (healed 88.0%, clean 93.2%) |
| the 760 trimmed no-defect crops (`d1_gc7/extra_eval.csv`) | 0% | 89.9% |

**The dial:** say no-defect only when p(no-defect) is above a cut chosen on val.

| val defects missed (sets the cut) | test defects missed | test defect accuracy | no-defect caught (healed / clean) |
|---|---|---|---|
| 1% | **0.9%** | 0.802 | 80.2% (73.6% / 84.7%) |
| 2% | 2.4% | 0.791 | 88.0% (84.4% / 90.4%) |
| 5% | 5.1% | 0.774 | 92.9% (89.6% / 95.1%) |
| top answer (no cut) | 3.9% | 0.781 | 91.1% (88.0% / 93.2%) |

**LaMa check.** 301 val/test defect crops with a clean patch LaMa-filled (`2_lama_healing/no_defect/decoys.py`):

| | decoy | the same crops untouched |
|---|---|---|
| called no-defect | 4.7% | 3.0% |
| mean p(no-defect) | 0.062 | 0.046 |
| right class | 75.4% | 76.7% |

That is a small effect on a small sample. Together with healed crops being recognised *less* often than clean ones,
there is no sign that gc7 learned "LaMa fill = no defect".

**Training:** backbone best epoch 5 of 10. Val 7-class macro-F1 was 0.680 at epoch 3, then flat within ±0.003
while the train loss kept falling. Head: kNN weight 0.25, prior correction 0.5, both chosen on val.
