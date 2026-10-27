# Dropping the quality gate (enhanced_res v2, test half not used)

Inside the cal half: 428 crops (≤ 6 per family, 88 families) and their 14 synthetic corruptions; 200 random halvings (gates recalibrated on 44 families, measured on the other 44). D5 and the off-target sets use gates calibrated on the whole cal half.

## False reject (good photos stopped)

| set | deployed (quality 0.05 + OOD 0.08) | OOD only, α 0.08 | OOD only, α 0.13 |
|---|---:|---:|---:|
| normal cal-half families | 7.4% | 4.5% | 8.1% |
| D5, other photos of normal defects | 6.6% | 1.5% | 4.2% |

## False accept (bad images that get through)

| set | deployed (quality 0.05 + OOD 0.08) | OOD only, α 0.08 | OOD only, α 0.13 |
|---|---:|---:|---:|
| blur s1 | 87.0% | 90.0% | 81.6% |
| blur s2 | 83.9% | 86.8% | 78.5% |
| motion blur s1 | 89.0% | 91.2% | 82.3% |
| motion blur s2 | 83.6% | 86.1% | 72.6% |
| dark s1 | 74.4% | 84.1% | 74.0% |
| dark s2 | 0.3% | 24.5% | 16.1% |
| overexposed s1 | 40.6% | 79.6% | 70.0% |
| overexposed s2 | 8.0% | 49.3% | 39.9% |
| glare s1 | 77.3% | 87.7% | 76.8% |
| glare s2 | 66.4% | 87.4% | 75.7% |
| noise s1 | 4.8% | 72.0% | 59.6% |
| noise s2 | 0.0% | 15.2% | 8.0% |
| jpeg s1 | 57.5% | 58.1% | 48.9% |
| jpeg s2 | 12.8% | 13.2% | 9.1% |
| **all corruptions (mean)** | **49.0%** | **66.1%** | **56.6%** |
| off-target photos | 0.9% | 1.0% | 0.0% |
| wallpapers | 15.5% | 18.0% | 8.5% |
| static | 0.0% | 0.3% | 0.0% |

## Does it matter? What the visor shows for bad images that get through

Coverage = the CP set holds the true class (target 90%). Silent wrong = a single wrong class shown with no warning, as a share of all corrupted crops.

| set | deployed (quality 0.05 + OOD 0.08): coverage / silent wrong | OOD only, α 0.08: coverage / silent wrong | OOD only, α 0.13: coverage / silent wrong |
|---|---:|---:|---:|
| clean | 89.3% / 6.8% | 89.5% / 6.8% | 89.8% / 6.1% |
| blur s1 | 89.0% / 5.1% | 89.3% / 5.1% | 90.1% / 4.2% |
| blur s2 | 82.4% / 8.8% | 82.9% / 8.8% | 83.4% / 8.3% |
| motion blur s1 | 89.3% / 5.5% | 89.5% / 5.5% | 90.2% / 4.7% |
| motion blur s2 | 79.6% / 10.4% | 80.2% / 10.4% | 80.8% / 8.7% |
| dark s1 | 89.2% / 4.9% | 88.7% / 5.8% | 88.9% / 5.3% |
| dark s2 | 100.0% / 0.0% | 78.1% / 2.9% | 77.0% / 2.6% |
| overexposed s1 | 89.3% / 1.4% | 87.8% / 3.6% | 87.9% / 3.1% |
| overexposed s2 | 98.6% / 0.1% | 77.2% / 5.1% | 76.1% / 4.6% |
| glare s1 | 70.8% / 16.0% | 69.8% / 18.6% | 69.7% / 16.6% |
| glare s2 | 65.1% / 16.6% | 63.4% / 22.7% | 63.1% / 19.9% |
| noise s1 | 85.3% / 0.1% | 79.2% / 5.1% | 78.5% / 4.4% |
| noise s2 | – (none pass) | 75.3% / 1.8% | 80.7% / 1.0% |
| jpeg s1 | 89.3% / 3.5% | 89.4% / 3.5% | 89.8% / 3.2% |
| jpeg s2 | 85.6% / 0.4% | 86.1% / 0.4% | 92.3% / 0.0% |
| **all corruptions (mean)** | **85.7% / 5.2%** | **81.2% / 7.1%** | **82.0% / 6.2%** |
