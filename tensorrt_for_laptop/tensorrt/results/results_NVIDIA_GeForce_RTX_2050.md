# ResNet-18: PyTorch vs TensorRT on NVIDIA GeForce RTX 2050

4293 D1-test crops, same trained model (model_best.pt). torch 2.14.0+cu130, CUDA 13.0, TensorRT 11.4.0.106. Latency = mean of 1000 runs after 50 warm-up runs.

| backend | accuracy | macro-F1 | + MLP head acc | same class as PyTorch FP32 | latency b1 mean (ms) | p50 | p95 | end-to-end b1 (ms) | throughput b32 (crops/s) | whole test set (s) | speed-up (latency) | model size (MB) | compressed by | total run time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.8796 | 0.8125 | 0.9010 | 100.00% | 4.11 | 3.60 | 6.54 | 4.53 | 628 | 10.15 | 1.00x | 44.8 | 1.00x (0% smaller) | 31.5 |
| PyTorch FP16 | 0.8803 | 0.8134 | 0.9010 | 99.93% | 4.87 | 4.24 | 8.21 | 5.04 | 1053 | 6.67 | 0.84x | 22.4 | 2.00x (50% smaller) | 24.6 |
| TensorRT FP32 | 0.8796 | 0.8122 | 0.9010 | 99.95% | 2.01 | 1.98 | 2.13 | 2.36 | 976 | 4.75 | 2.05x | 90.6 | 0.49x (-102% smaller) | 18.3 |
| TensorRT FP16 | 0.8800 | 0.8129 | 0.9010 | 99.95% | 1.47 | 1.39 | 2.38 | 1.69 | 2110 | 3.00 | 2.80x | 23.5 | 1.90x (47% smaller) | 9.9 |

Whole benchmark run (loading the crops + every backend): 100.0 s.

Expected accuracy (Mac, PyTorch FP32): fc 0.8796, macro-F1 0.8125, + MLP head 0.9010.