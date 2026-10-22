> Historical pilot report. The active grouped dataset is now documented in `../data/d1_sscd_grouped/README.md`. This report describes the initial shipped-split audit, not the final split.

# D1 SSCD pilot — 21 October 2026

## Conclusion

SSCD is useful for retrieving D1 augmentation families. Visual inspection finds clear same-source pairs across the shipped splits, including colour changes, reflections and rotations. A single high cosine threshold is unsuitable for producing the final grouped dataset: verified-looking copies also occur at low scores, and low-feature images need special care.

This pilot does not produce a leak-free split, estimate all duplicate families, reproduce the previously claimed 861 groups, or train a defect classifier.

## Inputs and execution

- All 14,760 images from the supplied `1_main_model_resnet18.zip`.
- Original splits: 11,808 train, 1,476 validation, 1,476 test.
- Annotation inventory: 14,760 YOLO label files, 30,485 boxes. Six classes; no malformed rows found in the basic parser check.
- No identical file SHA256 hashes. Visually identical/re-encoded images can still have different hashes.
- Official SSCD DISC mixup ResNet-50 TorchScript model, direct 320×320 RGB resize and ImageNet normalization, 512-dimensional L2-normalized descriptors.
- One view per image in the full retrieval pass; Apple MPS. Embedding pass approximately 400 seconds.
- Checked descriptors: shape (14760, 512), all finite, norms approximately 1.
- Exact top-20 cosine neighbours per image, excluding itself: 295,200 directed candidate records. Full best-cross-split scores also computed without the top-20 limitation.
- These are the dataset's shipped splits. The old classifier's regenerated split manifest was not supplied, so these figures do not quantify contamination of its exact evaluation split.

## Cross-split evidence

Similarity is not a calibrated probability of duplication. Counts below are candidate counts, not confirmed duplicates.

| Query set | Images | Best training-image cosine ≥0.95 | ≥0.90 |
|---|---:|---:|---:|
| Validation | 1,476 | 1,009 (68.4%) | 1,134 (76.8%) |
| Test | 1,476 | 959 (65.0%) | 1,094 (74.1%) |

Across all images, 5,673 have a candidate in another shipped split with cosine ≥0.95. This broader figure includes validation–test matches.

## Pair inspection

Seed 42 selected 12 unique cross-split candidate pairs from each of five cosine bands: [0.95,1], [0.90,0.95), [0.80,0.90), [0.70,0.80), [0.50,0.70). All six resulting contact sheets were visually inspected by the assistant. Many show clearly corresponding scene details with changes in exposure, colour, reflection, rotation or framing. This is exploratory model-assisted review, not independent human ground truth or a precision estimate.

Examples (image indices refer to images.json):

- 241 / 3487: same A467 aircraft marking and surrounding panel, cosine 0.998.
- 1214 / 13594: same damaged riveted panel and writing, horizontally mirrored, cosine 0.918.
- 195 / 9273: same torn panel, grayscale versus colour, cosine 0.912.
- 10714 / 14708: same yellow/red panel after rotation, cosine 0.573.
- 9048 / 13566: same A467 scene with altered framing/appearance, cosine 0.580.

Reflection-aware SIFT diagnostics use mutual Lowe-ratio matches (0.75), homography RANSAC (4 px at maximum 800 px image size), and record inlier count, ratio, spatial coverage, and median reprojection error. They test the second image both normally and horizontally reflected.

Using an **exploratory diagnostic rule**, at least 20 inliers, 50% inlier ratio, and 5% convex-hull coverage in both images:

| Cosine band | Reviewed pairs | Pass diagnostic rule |
|---|---:|---:|
| ≥0.95 | 12 | 11 |
| 0.90–0.95 | 12 | 9 |
| 0.80–0.90 | 12 | 11 |
| 0.70–0.80 | 12 | 12 |
| 0.50–0.70 | 12 | 7 |

This rule has not been calibrated against independently labelled positive and negative pairs. Passing it is supporting evidence, not automatic acceptance. Failure is not proof of independence: blur and smooth surfaces can provide too few keypoints. No groups were formed from these thresholds.

## Controlled augmentation probe

32 source images chosen with seed 123, each transformed nine ways: horizontal flip, vertical flip, 90/180/270-degree rotation, two diagonal reflections, central 80%-width/height crop, and reduced brightness/saturation. Each query was searched against all original D1 descriptors.

The exact source was in the top 20 for **288/288 queries**. It ranked first (allowing numeric ties) for 195/288 queries; sibling augmentations can rank ahead of the exact source. This is a small synthetic positive-only probe, not recall on real D1 duplicate families.

| Transformation | Median source cosine | Minimum |
|---|---:|---:|
| Horizontal flip | 0.958 | 0.911 |
| 90-degree rotation | 0.795 | 0.627 |
| 180-degree rotation | 0.811 | 0.726 |
| 270-degree rotation | 0.778 | 0.652 |
| Central 80% crop | 0.836 | 0.706 |
| Brightness/saturation | 0.989 | 0.979 |

A threshold such as 0.90 would miss known transformed copies. The next grouping pass should union candidates across flip/rotation views, then geometrically verify them and audit ambiguous links. Do not average away the orientation evidence or assume SSCD is exactly transformation-invariant.

## Next step toward d1_sscd_grouped

1. Expand candidate retrieval using eight flip/rotation views and an adaptive neighbour count for large families.
2. Calibrate geometric acceptance and rejection using independently reviewed positive pairs and difficult negatives. Preserve uncertain cases for review rather than declaring them unrelated.
3. Review suspicious bridge links and large connected components before splitting whole source groups.
4. Audit cross-split pairs again, including partial crops and texture-poor images, then freeze the manifest.
5. Retrain the classifier from ImageNet weights and recompute all training embeddings and kNN references. Keep a separate external evaluation dataset.

## Artifacts

- `images.json`: image-index/path/split mapping.
- `sscd_embeddings.npy`, `neighbors.npz`, `neighbors.csv`: reusable retrieval outputs.
- `best_cross_split.csv`, `cross_split_summary.json`, `heldout_to_train.json`: overlap candidates.
- `review_01.jpg` through `review_06.jpg`: 60 sampled pairs; captions show initial plain-SIFT diagnostics.
- `review_geometry.csv`: updated reflection-aware diagnostics; these supersede the plain-SIFT captions.
- `transform_probe.json`: all probe results and sampled IDs.
- `../scripts/`: reproducible audit, review verification, summary, and transformation-probe scripts.

Model documentation: https://github.com/facebookresearch/sscd-copy-detection
