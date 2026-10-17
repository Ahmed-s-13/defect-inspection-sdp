## Findings (written by the team; numbers from the run of 2026-10-16)

All numbers below are symmetric KL at PCA k = 32, one random crop per image, mean of 5 seeds. Deduplicated version: n = 505 images per dataset. All-images version: n = 759.

**1. The public datasets overlap far more than their sizes suggest.**
- 98% of D2's images are also in D1, and 98% of D3's are in D2. We dropped D3 from the KL analysis: it exists only as a 3× augmented export and is essentially D2.
- 42% of D5 is in D1.
- D4 (wind turbines) shares almost nothing with the aircraft sets.
- D1 itself holds about 5,600 distinct photos among its 14,760 images; most of the rest are brightness, grayscale or exposure variants.
- D1's shipped split leaks: 77% of its shipped test images have a near-copy in shipped train. Any accuracy reported on that split is optimistic. We re-split by photo group instead.

**2. D2's shift cannot be measured honestly.**
After removing images found in D1, only 24 D2 images are left, too few for a KL value. With all images, D2 vs D1 is 5.3 (Gaussian) against a noise floor of 0.9, even though nearly every D2 photo is in D1. The likely cause is preprocessing: D2 is stored at about 3000 px, while D1's copies are 640 px and often stretched. Resolution and resizing alone move the embeddings noticeably.

**3. After deduplication, D5 and D4 are both far from D1.**

| comparison | Gaussian | k-NN |
|---|---|---|
| noise floor (two halves of D1-held-out) | 1.4 | ≈ 0 |
| D1-train vs D1-held-out | 1.2 | ≈ 0 |
| D1 vs D5 (aircraft) | 16.8 | 6.4 |
| D1 vs D4 (wind-turbine blades) | 22.6 | 8.3 |

- **Scale.** D5 is about as far from D1 as D1 is from itself after severity-2 Gaussian noise or JPEG compression (KL 14.9 and 16.3). D4 sits between severity 2 and 3.
- **Order.** Both estimators, and every k from 8 to 64, agree on it: D5 is closer to D1 than D4 is.
- **Material barely matters to this model.** The gap between D5 (aircraft) and D4 (blades) is small. Close-up surface patches look alike to it whatever the material.

**4. With all images, D5 looks much closer to D1 (4.4) than it really is (16.8).**
The 42% of D5 that is already in D1 hides most of the shift. This is the effect the task spec (`docs/TASK_SPEC.md`) warned about, measured.

**5. The shift is not only a different mix of classes.**
D4 crack vs D1 crack gives 12.9, against a class noise floor of 0.5. The other per-class comparisons have too few images on the D1 side to give a value.

**6. The direction of KL tells us coverage.**
KL(D1‖D5) = 23.8 is much larger than KL(D5‖D1) = 9.9, and the D4 pair shows the same pattern. D1, assembled from 24 sources, spreads over a wider region than D4 or D5, and each of them covers only part of it. The PCA plot shows D5 filling one side of D1's cloud.

**7. The no-damage datasets (D6 to D9) are far from everything in D1, and easy to tell apart from it.**
D6 is FGVC-Aircraft, D7 and D8 are Roboflow aircraft sets, D9 is the Open Images "Fixed-wing aircraft" subset. All four are photos of whole planes taken from a distance. We measured them two ways: as whole-plane crops, and as skin patches cut from the middle of each plane. We also cut D1nd, patches of D1's own held-out photos away from every defect box, as a "no damage from the same cameras" reference.

| set | KL vs D1 defects | KL vs D1nd | classifier tells it from D1nd |
|---|---|---|---|
| D1nd (D1 no-damage patches) | 11.1 | | |
| D6 to D9, whole planes | 43 to 77 | 31 to 54 | 97% to 99% |
| D6p to D9p, skin patches | 23 to 35 | 17 to 28 | 91% to 95% |

The KL values are Gaussian, symmetric, deduplicated, at n = 505 and k = 32. The noise floor is 1.4, and a linear classifier tells two halves of D1 apart 53% of the time, which is chance.

- Even the skin patches are farther from D1's no-damage patches (17 to 28) than D1's own defect crops are (11).
- A simple classifier separates any D6 to D9 set from D1nd more easily (91% to 99%) than it separates D1's defects from D1's no-damage patches (83%).

So if these sets became the "no damage" class next to D1 to D5 defects, the model could get the class right just by recognising the kind of photo. It would learn "distant airport photo = no damage" instead of what damage looks like. This is the shortcut-learning risk, measured.

The four no-damage sets are much closer to each other (2.3 to 9.5) than to D1. D6 vs D8, both airport-spotter photo collections, is only 2.3, not far above the noise floor of 1.4. Pooling them with each other is reasonable; pooling them with the defect sets is the problem.

**8. Where a no-damage class should come from.**
Undamaged areas of the same kind of close-up photos are a better source: D1nd-style patches, or close-ups of intact skin from GAES or our own photos. Even D1nd is not clean. Many of its patches are background (ground, grass, shadow) rather than aircraft skin, and D1's labels may miss some defects. So it should be filtered before being used as a training class.

**9. D10 (fuselage mock-up) is the farthest dataset from D1 we have measured.**
D10 is a Kaggle anomaly-detection set: 6,415 video frames (1280×720) from inside a fuselage mock-up, two panels filmed at 15, 25 and 35 cm. It has no boxes, only "normal" or "anomaly" per frame. The anomalies are objects placed on the panel (tape or paper pieces, thin wire-like bits) rather than cracks, dents or corrosion. Each frame gave two overlapping 720×720 squares. Consecutive video frames are near-identical, so we kept only frames that differ from every frame already kept: 528 distinct normal views and 550 anomaly views. Full tables are in `results/d10/summary.md`.

| comparison | Gaussian | k-NN | classifier tells it from D1 |
|---|---|---|---|
| noise floor (two halves of D1-held-out) | 1.2 | ≈ 0 | 53% |
| D1 vs D5 (for scale) | 16.8 | 6.5 | 83% |
| D1 vs D10 normal frames | 213 | 21.3 | 99.5% |
| D1 vs D10 anomaly frames | 153 | 18.9 | 99.5% |
| D5 vs D10 normal / anomaly | 192 / 162 | 24.6 / 21.6 | |
| D10 normal vs D10 anomaly | 8.3 | 5.3 | 78% (67% on an unseen panel and distance) |

- D10 is about ten times farther from D1 than D5 is, and farther than the whole-plane photos (D6 to D9, at most 77). It is equally far from D5. Lab mock-up, primer colour, wide frames, video camera and placed objects all differ from real defect photos at once.
- Inside D10, normal and anomaly frames differ (8.3 against a floor of 1.2). A classifier separates them 78% of the time on random frames, but only 67% when tested on a panel and distance it did not train on. So part of the difference comes from the separate recording sessions, not from the anomalies.
- D10 cannot be added to the D1/D5 training data without the model learning "mock-up look". It is better suited as a separate test of an anomaly-detection approach, keeping in mind that its anomalies are placed objects, not real damage.

**10. The Cropper: a standard way to cut "no damage" patches (tested on D1).**
`klshift/cropper.py`, run by `run_cropper.py` with settings from `cropper.yaml`, sends every photo through four steps:
1. Propose random squares sized like that photo's own defect crops.
2. Drop any square touching a defect box (enlarged by 25%).
3. Keep squares that CLIP (a general image-and-text model, used without training) scores as aircraft skin rather than sky, grass or ground.
4. Record a CLIP "looks damaged" score.

It keeps at most 2 squares per photo. On D1 the run gave 19,224 patches (set `D1nc`) from 10,083 photos. 4,532 photos got none, because their defect boxes fill the frame.

- **The surface threshold has to be calibrated.** A fixed threshold of 0.5 rejected 37% of squares, but it would also have rejected 31% of real defect crops, which are always on aircraft skin. Most of what it removed was plain metal that CLIP mistook for concrete or ground. The Cropper now sets the threshold so that 90% of the dataset's own defect crops pass (0.066 for D1). Then it rejects 14% of squares, mostly grass, trees, sky and ground.
- **Cleaning removes obvious background but barely changes the overall picture.** D1nc and the earlier rule-based D1nd are about as different as two halves of D1 (KL 1.3, noise floor 1.3; a classifier separates them 57% of the time). D1 defects vs D1nc is 9.3, against 10.5 for D1nd. Only 9% to 14% of D1nd falls below the calibrated threshold. My earlier remark that "many D1nd patches are background" was based on one sample sheet and overstated it.
- **Missed labels are the bigger problem.** In a visual check of 50 random accepted patches (by Claude, not yet by the team), about 85% were clearly aircraft skin. But about 6 to 8 showed marks that could be unlabelled damage: scraped or chipped paint, a scratch, a hole, rust. CLIP's damage score is a weak guide (it ranks a real defect crop above an accepted patch only 66% of the time, where 50% is chance), so it is recorded but not used as a filter.
- **Next steps.**
  - Hand-check the 200 patches in `results/cropper/D1nc/check_list.csv` to measure precision and the missed-label rate properly.
  - Run the Cropper on D5 too, so each source supplies both classes.
  - Only then decide on a damage filter.

**What this means for the helmet model.**
- **No-damage data.** Don't take a "no damage" class from D6 to D9 as they are. Use undamaged areas of close-up photos taken like the defect photos.
- **Test sets.** A model trained on D1 and tested on D1's own split (or on D2) will look better than it is. Test on deduplicated, out-of-source data.
- **D5 as the real benchmark.** Of the public sets, D5 minus its D1 overlap is the most useful aircraft test of generalisation. D4 is a useful "different material" test.
- **Image preprocessing.** Resolution and resizing are a shift source in their own right, so the helmet's camera pipeline should match the training data's.
