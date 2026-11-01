# Gate variants for architecture B (local experiment, cal half only)

> Historical experiment: measured before the OOD change, using OOD α 0.08 where applicable (the threshold sweep lists its own values). Current B uses OOD α 0.05 and CP α 0.10. These recorded numbers are not current-deployment results; see the regenerated [Architecture B report](../README.md).

gc7 on d1_gc7_fix. OOD α 0.08, Q2 = Laplacian sharpness + brightness (both sides), parallel at α 0.05 or folded into the OOD test. False reject: 200 halvings of the 88 cal-half families, mean (5th–95th).

| variant | false reject, defect | false reject, no-defect | off-target held out shown | FGVC skin shown | wallpaper shown | static shown | corrupted defects: silent miss | corrupted: stopped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B: kNN + AoN, Bonferroni | 5.0% (0.9–10.9) | 3.6% (0.0–7.5) | 3.2% | 16.0% | 17.2% | 3.1% | 2.6% | 39.3% |
| kNN + AoN, Simes | 5.6% (1.0–11.4) | 4.2% (0.0–7.9) | 2.8% | 13.7% | 14.0% | 2.2% | 2.5% | 41.2% |
| AoN only | 6.1% (2.3–11.5) | 7.2% (1.3–14.0) | 1.9% | 8.7% | 14.2% | 21.9% | 3.7% | 28.4% |
| B + Q2 parallel | 7.0% (2.4–12.8) | 7.6% (2.8–13.1) | 3.0% | 15.2% | 11.0% | 0.0% | 2.2% | 55.1% |
| Simes + Q2 parallel | 7.5% (2.7–13.3) | 8.2% (2.8–14.1) | 2.7% | 13.0% | 9.5% | 0.0% | 2.1% | 56.0% |
| AoN only + Q2 parallel | 8.2% (3.8–13.1) | 10.9% (3.6–19.0) | 1.8% | 8.3% | 10.0% | 0.0% | 3.3% | 48.2% |
| Simes(kNN, AoN, Q2) | 2.7% (0.4–5.5) | 3.3% (0.5–6.6) | 5.2% | 19.9% | 14.2% | 0.6% | 2.5% | 48.0% |
| AoN only, Simes(AoN, Q2) | 4.2% (0.9–8.2) | 6.0% (1.6–11.5) | 3.4% | 17.0% | 23.2% | 0.9% | 3.7% | 41.8% |

Shown = a defect or POSSIBLE on the visor. Silent miss = a (corrupted) defect crop shown as nothing.

## Corruptions per kind: silent miss / stopped

| corruption | B: kNN + AoN, Bonferroni | kNN + AoN, Simes | AoN only | B + Q2 parallel | Simes + Q2 parallel | AoN only + Q2 parallel | Simes(kNN, AoN, Q2) | AoN only, Simes(AoN, Q2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.9% / 7.2% | 0.9% / 7.7% | 0.9% / 7.2% | 0.7% / 10.5% | 0.7% / 11.0% | 0.7% / 10.7% | 1.2% / 5.4% | 0.7% / 6.3% |
| blur s1 | 4.0% / 11.7% | 4.0% / 13.6% | 4.0% / 10.7% | 3.7% / 18.5% | 3.7% / 20.3% | 3.7% / 17.8% | 4.0% / 10.3% | 3.7% / 9.6% |
| blur s2 | 3.5% / 13.3% | 3.5% / 15.0% | 3.5% / 5.8% | 2.6% / 21.7% | 2.6% / 23.1% | 2.6% / 14.7% | 3.0% / 11.0% | 2.6% / 8.4% |
| motion blur s1 | 4.2% / 10.7% | 4.2% / 12.4% | 4.2% / 9.6% | 4.0% / 14.7% | 4.0% / 16.4% | 4.0% / 13.8% | 4.0% / 9.6% | 4.0% / 7.2% |
| motion blur s2 | 3.3% / 15.4% | 3.3% / 17.3% | 3.0% / 11.9% | 2.8% / 20.6% | 2.8% / 22.4% | 2.6% / 17.3% | 3.3% / 13.6% | 3.0% / 9.1% |
| dark s1 | 5.1% / 15.2% | 5.1% / 15.9% | 4.7% / 9.3% | 3.7% / 58.6% | 3.7% / 58.6% | 3.5% / 57.0% | 4.0% / 38.6% | 3.7% / 46.0% |
| dark s2 | 1.9% / 71.7% | 1.6% / 76.4% | 1.2% / 36.9% | 0.0% / 100.0% | 0.0% / 100.0% | 0.0% / 100.0% | 0.0% / 100.0% | 0.0% / 100.0% |
| overexposed s1 | 0.5% / 27.3% | 0.5% / 28.5% | 0.5% / 23.1% | 0.2% / 68.0% | 0.2% / 68.2% | 0.2% / 68.7% | 0.9% / 54.9% | 0.2% / 62.6% |
| overexposed s2 | 0.0% / 61.0% | 0.0% / 64.0% | 0.0% / 52.1% | 0.0% / 91.6% | 0.0% / 91.8% | 0.0% / 92.1% | 0.5% / 87.6% | 0.0% / 89.5% |
| glare s1 | 0.9% / 15.9% | 0.9% / 16.6% | 1.9% / 12.9% | 0.9% / 25.7% | 0.9% / 25.7% | 1.9% / 22.2% | 1.6% / 15.0% | 2.1% / 14.3% |
| glare s2 | 1.9% / 21.3% | 1.9% / 23.8% | 4.9% / 21.0% | 1.6% / 35.3% | 1.6% / 37.4% | 4.7% / 33.2% | 2.6% / 26.4% | 4.7% / 20.6% |
| noise s1 | 0.0% / 69.6% | 0.0% / 72.9% | 0.0% / 67.5% | 0.0% / 100.0% | 0.0% / 100.0% | 0.0% / 100.0% | 0.0% / 100.0% | 0.0% / 100.0% |
| noise s2 | 0.0% / 100.0% | 0.0% / 100.0% | 0.0% / 99.3% | 0.0% / 100.0% | 0.0% / 100.0% | 0.0% / 100.0% | 0.0% / 100.0% | 0.0% / 100.0% |
| jpeg s1 | 5.6% / 35.7% | 5.4% / 37.1% | 7.9% / 11.9% | 5.6% / 36.2% | 5.4% / 37.4% | 7.9% / 12.1% | 5.8% / 28.7% | 8.6% / 5.8% |
| jpeg s2 | 4.9% / 80.6% | 4.2% / 82.9% | 15.7% / 25.0% | 4.9% / 80.6% | 4.2% / 82.9% | 15.7% / 25.7% | 5.4% / 76.4% | 18.9% / 11.9% |
