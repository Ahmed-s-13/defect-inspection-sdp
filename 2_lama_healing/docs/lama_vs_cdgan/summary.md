| # | class | p(true) original | CD-GAN | OpenCV | LaMa (now predicted) |
|---|---|---|---|---|---|
| 1 | crack | 0.09 | 0.01 | 0.01 | 0.02 (dent) |
| 2 | dent | 0.97 | 0.95 | 0.95 | 0.06 (corrosion) |
| 3 | scratch | 0.58 | 0.39 | 0.01 | 0.11 (dent) |
| 4 | dent | 0.97 | 0.67 | 0.97 | 0.96 (dent) |
| 5 | crack | 0.91 | 0.92 | 0.01 | 0.56 (crack) |
| 6 | missing-head | 0.97 | 0.01 | 0.01 | 0.03 (dent) |
| 7 | corrosion | 0.97 | 0.01 | 0.01 | 0.01 (dent) |
| 8 | corrosion | 0.01 | 0.01 | 0.01 | 0.01 (dent) |
| 9 | paint-peel-off | 0.85 | 0.32 | 0.01 | 0.01 (dent) |
| 10 | paint-peel-off | 0.97 | 0.87 | 0.01 | 0.01 (dent) |

LaMa time per photo on the GPU: median 960 ms (first call 3008 ms incl. warm-up).