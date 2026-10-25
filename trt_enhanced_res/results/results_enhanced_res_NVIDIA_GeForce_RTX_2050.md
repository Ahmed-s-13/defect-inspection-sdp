# enhanced_res: PyTorch vs TensorRT on NVIDIA GeForce RTX 2050

4255 grouped D1-test crops (175 source groups), same trained model. torch 2.14.0+cu130, CUDA 13.0, TensorRT 11.4.0.106. One crop = 8 TTA views in the backbone; the MLP head + 5-NN runs on PyTorch CUDA in every row. Latency = mean of 300 runs after 30 warm-up runs.

| backend | accuracy | macro-F1 (95% group CI) | balanced acc | fc-only acc / F1 | same class as FP32 | McNemar vs FP32 (b, c, p) | latency b1 backbone (ms) | p95 | b1 backbone+head (ms) | end-to-end b1 (ms) | throughput b4 (crops/s) | whole test set (s) | speed-up (backbone) | speed-up (+head) | size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.8045 | 0.7376 (0.692-0.776) | 0.7367 | 0.8042 / 0.7405 | 100.00% | 0, 0, p=1.000 | 48.58 | 49.24 | 51.77 | 52.24 | 20.5 | 216.0 | 1.00x | 1.00x | 44.8 |
| PyTorch FP16 | 0.8045 | 0.7376 (0.692-0.776) | 0.7367 | 0.8042 / 0.7403 | 100.00% | 0, 0, p=1.000 | 67.72 | 162.80 | 56.10 | 73.62 | 21.9 | 141.0 | 0.72x | 0.92x | 22.4 |
| TensorRT FP32 | 0.8045 | 0.7376 (0.692-0.776) | 0.7367 | 0.8042 / 0.7405 | 100.00% | 0, 0, p=1.000 | 37.24 | 39.17 | 41.39 | 42.59 | 26.8 | 188.4 | 1.30x | 1.25x | 81.2 |
| TensorRT FP16 | 0.8045 | 0.7376 (0.692-0.776) | 0.7367 | 0.8042 / 0.7405 | 100.00% | 0, 0, p=1.000 | 12.25 | 13.02 | 22.62 | 15.67 | 88.8 | 49.1 | 3.96x | 2.29x | 25.6 |

Per-class recall: PyTorch FP32: corrosion 0.587, crack 0.801, dent 0.876, missing-head 0.945, paint-peel-off 0.675, scratch 0.535; PyTorch FP16: corrosion 0.587, crack 0.801, dent 0.876, missing-head 0.945, paint-peel-off 0.675, scratch 0.535; TensorRT FP32: corrosion 0.587, crack 0.801, dent 0.876, missing-head 0.945, paint-peel-off 0.675, scratch 0.535; TensorRT FP16: corrosion 0.587, crack 0.801, dent 0.876, missing-head 0.945, paint-peel-off 0.675, scratch 0.535

PyTorch FP32 vs the repo's evaluate.py (0.8045 / 0.7376 / 0.7367): MATCH (0.8045 / 0.7376 / 0.7367).

Whole benchmark run: 828 s. One run, one seed; RTX 2050 is not the Jetson Orin NX (relative speed-ups carry over, absolute ms do not).