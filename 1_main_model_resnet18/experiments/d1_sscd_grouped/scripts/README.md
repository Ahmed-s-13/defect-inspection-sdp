# What each Python file does

Run commands from the workspace root using `.venv/bin/python scripts/NAME.py`.

| File | Purpose | Main output |
|---|---|---|
| `sscd_audit.py` | Run pretrained SSCD on all D1 photos, search top-20 neighbours, make initial review sheets | `reports/sscd_embeddings.npy`, `neighbors.csv`, `review_*.jpg` |
| `sscd_transform_probe.py` | Test known flips, rotations, crops and colour changes on a 32-image sample | `reports/transform_probe.json` |
| `summarize_candidates.py` | Find the strongest candidate in another original split without a top-20 cap | `reports/best_cross_split.csv` |
| `verify_review.py` | Check the initial review sample using normal and reflected SIFT keypoints | `reports/review_geometry.csv` |
| `build_groups.py` | Add eight-orientation perceptual hashes, verify candidate links, form connected groups | `data/d1_sscd_grouped/groups.csv`, `pair_evidence.csv` |
| `refine_links.py` | Check selected lower-score candidates using geometric alignment followed by pixel correlation | `data/d1_sscd_grouped/refined_links.csv` |
| `prepare_grouped.py` | Keep accepted/potential copies together, balance classes, save the split and raw links | `data/d1_sscd_grouped/split_manifest.csv`, `work/d1_split.csv` |
| `review_split.py` | Make visual sheets for the largest source groups and strongest residual cross-split matches | `reports/final_split_review/` |
| `check_ready.py` | Check manifests/files and execute a disposable ResNet/MLP training and checkpoint smoke test | `data/d1_sscd_grouped/readiness_check.json` |

## How grouping works

SSCD's 512-number fingerprint is compared using cosine similarity. It retrieves candidates; it does not itself prove two photos share an origin.

The primary grouping pass retrieves original-view SSCD top-20 candidates scoring at least 0.5 and supplements them with every all-pairs perceptual-hash match within 10 differing bits across eight flips/quarter turns. It accepts a link if either:

- the hash difference is at most 6 and the best flip/quarter-turn grayscale pixel correlation is at least 0.97; or
- mutual SIFT feature matches support a homography with at least 20 inliers, at least 55% inlier ratio, and at least 8% spatial coverage in both images.

The geometry check uses both the original and reflected second image, maximum 640-pixel image edge, and 3-pixel RANSAC tolerance. Once two images are already in the same connected group, redundant links can be skipped. The evidence file records actual checks, not every pair inside the final groups.

The refinement pass considers up to five outside-block candidates per image from its top 50 neighbours, with cosine at least 0.35. It requires at least six geometric inliers, ratio 0.6, spatial coverage 0.02, at least 20% valid warp overlap, and aligned grayscale correlation at least 0.9. These are conservative co-location constraints, not independently calibrated identity labels.

For the final split, `prepare_grouped.py` combines accepted groups, broad SSCD similarity constraints, refinement links, and explicit visual-review constraints from `review_colocate.csv`. It keeps each resulting **split block** wholly in one split. Current broad-threshold settings are recorded in `split_summary.json`. Different source scenes may share a split block; this is intentional caution and must not be interpreted as proving they are copies.

Allocation uses class-crop counts, image counts, and block counts; it tries 24 seeded greedy allocations, improving them with moves and swaps, then a 30-second mixed-integer class-balancing solve. A very large block is kept in training so it cannot dominate validation/test. The solver returned a feasible allocation at the time limit, not a proof of global optimality. This is data balancing, not model/hyperparameter selection. No accuracy metric is used.

## Reproducibility and changes

The active manifest is frozen. `prepare_grouped.py` refuses to overwrite it. Do not rerun grouping or crop preparation over a live experiment just to train another model. Use a new version directory for regrouping.

The initial SSCD embedding cache assumes the image manifest and model are unchanged. Scripts save relative raw-image references, while the classifier's crop manifest uses absolute paths. Moving the workspace requires a deliberate path update and a new checksum.

The broad similarity threshold and geometric rules were explored using pair appearance, not classification test performance. They have not been independently calibrated on a fully labelled same-source/different-source benchmark. Residual missed copies and conservative over-grouping remain possible.

The frozen manifest is the authoritative reproducible artifact. A time-limited allocation solve can select a different feasible solution on another machine, even with the same seed. Preserve the supplied manifest/checksum for comparable experiments.
