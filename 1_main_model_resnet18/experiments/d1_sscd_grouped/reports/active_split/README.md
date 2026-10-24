# Active grouped D1 dataset

Prepared 22 October 2026. This is the active, training-ready **conservative grouped split**. Earlier drafts under `drafts/` are not active.

## Split sizes

| Split | Source images | Usable defect crops | Split blocks |
|---|---:|---:|---:|
| train | 10,330 | 20,046 | 503 |
| val | 2,216 | 4,295 | 108 |
| test | 2,214 | 4,296 | 109 |

Total: **14,760 source images, 28,637 crops, 720 split blocks**. The algorithm found 1,701 components linked by stricter matching rules. Broader blocks may contain more than one independent scene: their count is not an estimate of original-photo count.

Whole blocks are assigned to one split. Class balance aims for 70/15/15 but is approximate because blocks cannot be divided. Allocation uses labels/counts only, never classifier predictions.

| Class | Train crops | Val crops | Test crops |
|---|---:|---:|---:|
| corrosion | 1766 | 378 | 378 |
| crack | 6109 | 1310 | 1310 |
| dent | 8191 | 1755 | 1755 |
| missing-head | 1830 | 392 | 392 |
| paint-peel-off | 1211 | 259 | 260 |
| scratch | 939 | 201 | 201 |

## How to use it

From the workspace root:

```sh
.venv/bin/python scripts/check_ready.py
.venv/bin/python main_model/train.py --workers 0
.venv/bin/python main_model/embed.py --workers 0
.venv/bin/python main_model/train_mlp.py
```

The first command checks readiness. The second trains ResNet-18 and its linear head; the last two build the optional MLP ensemble/kNN head. Full training has not been run during preparation. The smoke check trains only a disposable batch and saves no production checkpoint.

## Files

- `images/train`, `images/val`, `images/test`: raw photos, as relative links into `../D1`.
- `labels/train`, `labels/val`, `labels/test`: corresponding bounding-box labels.
- `data.yaml`: YOLO-format dataset description, provided for compatibility. The main classifier is ResNet-18, not YOLO.
- `split_manifest.csv`: source path, original split, new split, source group, split block, SHA256 for every image.
- `work/d1_split.csv`: frozen crop manifest consumed by ResNet-18. Here `group_id` means split block; `source_group` is the stricter matching component.
- `work/crops/D1`: 224×224 JPEG crops. `work/crops/D1.csv` is their complete inventory.
- `train_crops.csv`, `val_crops.csv`, `test_crops.csv`: convenient per-split tables.
- `groups.csv`, `pair_evidence.csv`: initial source-group hypotheses and checked matching evidence.
- `review_colocate.csv`: explicit conservative constraints from assistant visual review; not independent human identity labels.
- `refined_links.csv`: lower-score candidates with geometric alignment and pixel agreement.
- `split_summary.json`, `readiness_check.json`: audit counts and model/data integration checks.

## Audit and limits

- No accepted matching edge crosses splits. All 55 recorded visual-review constraints and 16 refinement links are co-located.
- All-pairs original-view SSCD comparison: **zero cross-split pairs at cosine ≥ 0.45**. Maximum remaining cross-split score: 0.449006. This separation is partly by construction, not an independent measure of duplicate recall.
- Perceptual hashes checked all eight flips/quarter turns. Full SSCD embeddings used one view per photo; this is not an exhaustive eight-view SSCD audit.
- The broad similarity rule intentionally keeps uncertain pairs together. This reduces leakage risk but can join unrelated scenes in the same split block.
- Group connections use automatic rules plus assistant visual review. They are not a fully human-labelled ground truth. Undetected transformed/partial copies can remain; do not call this a certified leak-free benchmark.
- Raw source images remain untouched. The original crop rule omits 1,848 boxes with width or height below 4 pixels, leaving 28,637 crops; 145 source images have no retained crop.
- Copies are not deleted within splits. Crop-level accuracy still weights heavily augmented scenes more. Also report block-level metrics/uncertainty for final research results.
- D5 was not supplied and has not been evaluated.

Readiness result: **PASS**. Checked path coverage, labels, image/crop assignments, source/block separation, ResNet training step, feature extraction, MLP training step, kNN outputs, checkpoint loading and stale-checkpoint rejection.

## Frozen version

Seed: `20261021`. Crop manifest SHA256: `2f75881baa61be766243b309e483c2e0f26051fa757e02a649d1a01de84ab71b`.

Do not overwrite this split to compare models. Keep it fixed and write a separately versioned dataset for future grouping changes. The `FROZEN` marker protects the grouping scripts. The main training script only reads the saved split.

The raw folders use relative symbolic links; keep `data/D1` with them. The crop manifest and YAML contain absolute workspace paths and must be deliberately rewritten if moved to another machine. Rewriting changes the checksum, so do it before training, not midway through an experiment.

See `../../README.md` for the lay explanation of SSCD, and `../../main_model/README.md` for architecture and prediction defaults.
