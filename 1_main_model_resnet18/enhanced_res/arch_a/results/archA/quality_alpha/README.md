# Quality gate α 0.10 vs 0.05 (enhanced_res v2, test half not used)

OOD gate unchanged (α 0.08). Inside the cal half: 428 crops (≤ 6 per family, 88 families) and their 14 synthetic corruptions; 200 random halvings, gates recalibrated on 44 families and measured on the other 44 (mean, 5th–95th percentile).

## False reject (good photos stopped)

| set | α 0.10 | α 0.05 |
|---|---:|---:|
| normal cal-half families (halvings) | 12.3% (4.5%–20.5%) | 7.4% (1.5%–14.5%) |
| D5, other photos of normal defects | 14.4% | 6.6% |

## False accept (bad or off-target images that get through)

| set | α 0.10 | α 0.05 |
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
