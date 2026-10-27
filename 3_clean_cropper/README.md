# 3. Clean cropper: intact-surface crops from the unboxed parts of D1 photos

The clean cropper finds crops of **intact aircraft surface** in D1 photos: no defect, not background. It is the second
half of the 7th class of [`d1_gc7`](../1_main_model_resnet18/d1_gc7/README.md), which trains the main model
(enhanced_res gc7). The first half is the healed crops in
[`2_lama_healing/no_defect`](../2_lama_healing/no_defect/README.md). Its result, `no_defect_clean.csv`, has
**3,000 crops**.

The unboxed parts of a D1 photo are not automatically clean. A hand-labelled random sample of 300 found:
- 52% clean surface;
- 18% background: sky, grass, ground, buildings;
- 18% damage nobody boxed, because D1's labels are incomplete;
- 11% other: blurry, dark, text, or unclear.

So the candidates are scored, a small classifier learns "clean" from the hand labels, and a person reviews the
picks. enhanced_res is not used to choose crops, because they are meant to train it.

## Steps (from this folder)

| step | command | output | time |
|---|---|---|---|
| 1. candidates | `python candidates.py` | `work/candidates/<split>/`, `candidates.csv`: 25,413 squares from 9,328 photos, 695 groups | 15 s |
| 2. scores | `python score.py` | `scores.csv` (+ `work/cache/`): p_skin, p_damage (and optional yolo_conf) per candidate | ~15 min (MPS) |
| 3. audit | `python sample_audit.py`, then `python review_server.py` → http://127.0.0.1:8792, tab "audit" | `review/audit.csv`, `review/labels_audit.json` | you |
| 4. pick | `python pick.py` | `review/final.csv` (3,000 picks), `p_clean.csv`, `pick.json` | 1 min |
| 5. review | review page, tab "final" | `review/labels_final.json` | you |
| 6. final list | `python finalize.py` | `no_defect_clean.csv`, crops in `1_main_model_resnet18/data/work/crops/no_defect/clean/` | 2 s |

The tables, the audit labels and the picks are in git. The images, CLIP features and thumbnails in `work/` are not.
To rebuild them you need `D1_raw.zip` and `main_model_data_work.zip`. CLIP (`open_clip_torch`, in
`requirements.txt`) downloads its ViT-B/32 LAION-2B weights the first time. **To train or test the model you don't
need any of this.** The final crops are `d1_gc7_no_defect_crops.zip` on the team Drive.

Every crop keeps its photo's **d1_grouped** group and split, so no test photo reaches training.

## 1. Candidates (`candidates.py`)

Up to 3 random squares per labelled photo. Each square:
- stays clear of every labelled box grown by 50% per side, including boxes too small to become defect crops, and of
  the other squares;
- takes its side from the photo's own defect crop sizes (box + 25% margin), so candidates match the defect crops'
  scale;
- is resized to 224 px exactly like the defect crops;
- is skipped if it has black corners (from D1's rotated copies).

## 2. Scores (`score.py`)

- `p_skin`: a zero-shot CLIP score (ViT-B/32), aircraft-surface prompts against background prompts.
  - A high score is reliable. But it calls many close-ups of grey or white skin "concrete" or "a person", so a low
    score doesn't mean background. Audit AUC for background: 0.66.
- `p_damage`: how much a candidate looks like a labelled defect.
  - It is a logistic regression on CLIP features: D1 defect crops vs candidates. Each candidate is scored by a model
    trained without its group (5-fold GroupKFold). Out-of-fold AUC is 0.94.
  - High scores are mostly real unboxed cracks, dents, scratches and peeling paint (`sheets/high_damage.jpg`). Audit
    AUC for clean vs damage: 0.76.
- `yolo_conf`: optional, from a YOLO detector trained on d1_grouped train, placed at `work/yolo_d1_grouped.pt`.
  - The detector is not in this repo. It fired on only 2.7% of candidates and its audit AUC was about 0.5, so
    `pick.py` doesn't use it.

## 3. Audit (`sample_audit.py`, review page)

300 random candidates (at most 2 per group, 212 groups), labelled clean / damage / background / other. Each card
shows the crop next to its whole photo, with the square in red and the labelled defects in yellow. The 298 labels
are in `review/labels_audit.json`. No single score is good enough, so these labels train one.

## 4. Pick (`pick.py`)

- **`p_clean`.** A logistic regression on the audit labels (clean vs everything else), over the CLIP features, CLIP's
  prompt scores, `p_damage` and `p_skin`. Cross-validated over groups on the audit, its AUC is 0.79. The clean share
  among its top-ranked candidates:

  | top 10% | top 20% | top 30% | at random |
  |---|---|---|---|
  | 90% | 85% | 80% | 52% |

- **Busyness matched.** Ranking by `p_clean` alone picked mostly blank panels: median gradient 8, against 12 for the
  crops labelled clean (damage is 23). That would teach "featureless = no defect". So the pool is split into 5
  busyness bins at the audit-clean quantiles, and each bin gets 600 picks, taken by `p_clean`. The picks then match
  the clean labels (median 10.7 vs 11.5) and include rivets, screws, seams and panel edges
  (`sheets/final_random_32.jpg`).
- **Spread:** at most 1 crop per photo and 10 per group. Audit crops labelled clean are always in; other audit
  labels never are.
- **Picked:** 3,000 crops from 625 groups (train 1,812, val 621, test 567).

## 5–6. Review and final list

In the "final" tab, a click rejects a card and undecided cards count as kept. Nothing was rejected. `finalize.py`
copies the kept crops to `1_main_model_resnet18/data/work/crops/no_defect/clean/<split>/` and writes
`no_defect_clean.csv`. It has the D1 crop-list columns (`crop_path, cls, image, key, src_group, split`) plus the
square (`left, top, side`).

## Files

| file | what |
|---|---|
| `candidates.py`, `score.py`, `sample_audit.py`, `pick.py`, `finalize.py` | steps 1–6 |
| `review_server.py`, `review/index.html`, `thumbs.py` | the review page (Python standard library + pandas) |
| `candidates.csv`, `scores.csv`, `p_clean.csv`, `pick.json` | candidates, scores, the pick |
| `review/` | audit sample and labels, the final picks |
| `sheets/` | spot-check sheets |
| `work/` | candidate crops, CLIP features, thumbnails (not in git) |
