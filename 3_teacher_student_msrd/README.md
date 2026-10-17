# 3. Teacher–student anomaly model (MSRD)

`msrd_anomaly.py` implements Multi-Scale Reverse Distillation (Liu & Chung, *Applied
Soft Computing* 2025), built on Reverse Distillation (Deng & Li, CVPR 2022). It is
trained on defect-free images only and outputs a pixel anomaly map plus an image score.

It is kept here because **`2_healing_experiment` imports it** (`ts.py`, `healer.py`).
The architecture is used unchanged there.

## Standalone use (MVTec AD)

It was first tested on MVTec AD `metal_nut`. That data and its 290 MB checkpoint are
no longer used and are not in the repo. The metrics of that run are kept in
[`metal_nut_metrics.json`](metal_nut_metrics.json) (image AUROC 1.000, pixel AUROC 0.979).

```bash
python msrd_anomaly.py --data_root ./mvtec_anomaly_detection --category metal_nut
python msrd_anomaly.py --help
```
