# Architecture A: false accept and false reject (enhanced_res v2)

Final gates: quality α 0.05, OOD α 0.08 (kNN + AoN, Bonferroni), in parallel. False accept = an off-target crop that passes both gates and is shown as a defect. Each photo gives a whole-frame centre square and a box-sized patch; rates average the two.

## False reject (normal defect crops wrongly stopped)

| set | stopped | re-take | re-aim |
|---|---:|---:|---:|
| D1 test half, per photo family (report only) | **6.7%** | | |
| D1 test half, per crop (report only) | 6.6% | 4.3% | 2.4% |
| D5 (other photos of normal defects) | 6.6% | 5.2% | 1.5% |

## False accept (off-target photos shown as a defect)

**Held out** = AoN never trained on that category (refitted on the fit categories only). **Deployed** = the shipped AoN, which trained on every category listed (in-sample, optimistic).

| group | category | photos | held out | deployed |
|---|---|---:|---:|---:|
| people | Person (fit) | 200 | - | 1.0% |
| people | Human face (fit) | 150 | - | 0.0% |
| people | Human hand (test) | 200 | 4.2% | 0.8% |
| people | Glove (test) | 85 | 2.9% | 0.6% |
| people | Helmet (test) | 200 | 2.8% | 0.2% |
| objects and vehicles | Tool (fit) | 150 | - | 1.0% |
| objects and vehicles | Ladder (fit) | 83 | - | 0.0% |
| objects and vehicles | Car (fit) | 100 | - | 0.0% |
| objects and vehicles | Truck (fit) | 100 | - | 0.0% |
| hangar / airport / industrial scenes | hangar/indoor (test) | 100 | 6.5% | 1.5% |
| hangar / airport / industrial scenes | hangar/outdoor (test) | 100 | 11.5% | 2.0% |
| hangar / airport / industrial scenes | airport_terminal (test) | 100 | 3.0% | 0.0% |
| hangar / airport / industrial scenes | runway (test) | 100 | 15.5% | 2.5% |
| hangar / airport / industrial scenes | repair_shop (test) | 100 | 5.0% | 1.5% |
| hangar / airport / industrial scenes | storage_room (test) | 100 | 0.5% | 0.0% |
| hangar / airport / industrial scenes | auto_factory (fit) | 100 | - | 0.0% |
| hangar / airport / industrial scenes | engine_room (fit) | 100 | - | 1.0% |
| hangar / airport / industrial scenes | garage/indoor (fit) | 100 | - | 3.5% |
| hangar / airport / industrial scenes | industrial_area (fit) | 100 | - | 1.0% |
| hangar / airport / industrial scenes | loading_dock (fit) | 100 | - | 3.5% |

### Averages

| | held out | deployed |
|---|---:|---:|
| people | 3.3% (3 held-out categories) | 0.5% |
| objects and vehicles | - (0 held-out categories) | 0.2% |
| hangar / airport / industrial scenes | 7.0% (6 held-out categories) | 1.5% |
| humans (person, face, hand) | 4.2% | 0.6% |
| helmets | 2.8% | 0.2% |
| hands and gloves | 3.6% | 0.7% |
| **all held-out categories** | **5.8%** | 1.0% |

Person and Human face trained the AoN in both versions, so they have no held-out number; Human hand, Glove and Helmet are the held-out human categories. The user's hard-hat photo: stopped (re-aim).

## False accept: bad images and other sets

From [`../quality_alpha/`](../quality_alpha/README.md): synthetic damage of cal-half crops, gates recalibrated on half the cal-half families and measured on the other half (200 times; the test half is not used). Deployed: quality α 0.05.

| set | quality α 0.10 (before) | quality α 0.05 (deployed) |
|---|---:|---:|
| blur s1 | 78.7% | 87.0% |
| blur s2 | 72.8% | 83.9% |
| motion blur s1 | 83.6% | 89.0% |
| motion blur s2 | 77.4% | 83.6% |
| dark s1 | 38.1% | 74.4% |
| dark s2 | 0.0% | 0.3% |
| overexposed s1 | 21.4% | 40.6% |
| overexposed s2 | 2.7% | 8.0% |
| glare s1 | 22.7% | 77.3% |
| glare s2 | 3.9% | 66.4% |
| noise s1 | 0.0% | 4.8% |
| noise s2 | 0.0% | 0.0% |
| jpeg s1 | 55.7% | 57.5% |
| jpeg s2 | 12.2% | 12.8% |
| **all corruptions (mean)** | **33.5%** | **49.0%** |
| off-target photos | 0.9% | 0.9% |
| wallpapers | 10.8% | 15.5% |
| static | 0.0% | 0.0% |
