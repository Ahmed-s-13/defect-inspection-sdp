# enhanced_res (whole model): PyTorch vs TensorRT on NVIDIA GeForce RTX 2050

4255 grouped D1-test crops (175 source groups), same trained model. torch 2.14.0+cu130, CUDA 13.0, TensorRT 11.4.0.106. Every row runs the whole model (backbone + 8-view TTA + MLP heads + 5-NN); the TensorRT rows need no PyTorch for the model. Latency = mean of 300 runs after 30 warm-up runs.

| backend | accuracy | macro-F1 (95% group CI) | balanced acc | same class as FP32 | McNemar vs FP32 (b, c, p) | latency b1 (ms) | p95 | end-to-end b1 (ms) | throughput b4 (crops/s) | whole test set (s) | speed-up | size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.8045 | 0.7376 (0.692-0.776) | 0.7367 | 100.00% | 0, 0, p=1.000 | 48.47 | 53.02 | 54.41 | 24.1 | 179.9 | 1.00x | 136.3 |
| PyTorch FP16 | 0.8045 | 0.7376 (0.692-0.776) | 0.7367 | 100.00% | 0, 0, p=1.000 | 34.06 | 34.72 | 34.52 | 35.6 | 126.1 | 1.42x | 113.9 |
| TensorRT FP32 | 0.8045 | 0.7376 (0.692-0.776) | 0.7367 | 100.00% | 0, 0, p=1.000 | 39.45 | 39.98 | 39.97 | 27.9 | 164.5 | 1.23x | 149.7 |
| TensorRT FP16 | 0.8045 | 0.7376 (0.692-0.776) | 0.7367 | 100.00% | 0, 0, p=1.000 | 14.49 | 14.81 | 15.14 | 78.8 | 54.7 | 3.34x | 117.9 |

Per-class recall: PyTorch FP32: corrosion 0.587, crack 0.801, dent 0.876, missing-head 0.945, paint-peel-off 0.675, scratch 0.535; PyTorch FP16: corrosion 0.587, crack 0.801, dent 0.876, missing-head 0.945, paint-peel-off 0.675, scratch 0.535; TensorRT FP32: corrosion 0.587, crack 0.801, dent 0.876, missing-head 0.945, paint-peel-off 0.675, scratch 0.535; TensorRT FP16: corrosion 0.587, crack 0.801, dent 0.876, missing-head 0.945, paint-peel-off 0.675, scratch 0.535

PyTorch FP32 vs the repo's evaluate.py (0.8045 / 0.7376 / 0.7367): MATCH (0.8045 / 0.7376 / 0.7367).

Whole benchmark run: 662 s. One run, one seed; RTX 2050 is not the Jetson Orin NX (relative speed-ups carry over, absolute ms do not).