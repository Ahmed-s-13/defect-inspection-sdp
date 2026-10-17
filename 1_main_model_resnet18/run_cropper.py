"""Run the Cropper (klshift/cropper.py) on a defect dataset to cut "no damage" patches.

  python run_cropper.py --dataset D1 --name D1nc --config cropper.yaml

Outputs:
  <work>/crops/<name>/*.jpg, <work>/crops/<name>.csv   accepted patches (same columns as prepare.py, plus scores)
  <results>/cropper/<name>/candidates.csv               every proposed square, its scores and why it was kept or dropped
  <results>/cropper/<name>/report.md                    counts per step, score checks, sample sheets
  <results>/cropper/<name>/check_list.csv + check_sheet_*.png   random accepted patches for a hand check
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from klshift.common import CROP_SIZE, pick_device, seed_everything
from klshift.cropper import Cropper, CropperConfig


class PhotoCandidates(Dataset):
    """One item per photo: its proposed squares, cut and resized to CROP_SIZE (same as prepare.py)."""

    def __init__(self, photos, cropper, seed):
        self.photos, self.cropper, self.seed = photos, cropper, seed

    def __len__(self):
        return len(self.photos)

    def __getitem__(self, i):
        p = self.photos[i]
        rng = np.random.default_rng(zlib.crc32(p["image"].encode()) + self.seed)
        cands = self.cropper.propose(p["img_w"], p["img_h"], p["boxes"], rng)
        if not cands:
            return i, [], np.zeros((0, CROP_SIZE, CROP_SIZE, 3), np.uint8)
        img = ImageOps.exif_transpose(Image.open(p["path"])).convert("RGB")
        arr = np.stack([np.asarray(img.crop((l, t, l + s, t + s)).resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC))
                        for l, t, s in cands])
        return i, cands, arr


def collate_list(batch):
    return batch


def sheet(paths, titles, path, cols=8, title=""):
    if not len(paths):
        return
    rows = int(np.ceil(len(paths) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.1), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for k, (f, t) in enumerate(zip(paths, titles)):
        a = ax[k // cols, k % cols]
        a.imshow(Image.open(f))
        a.set_title(t, fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="D1")
    ap.add_argument("--name", default="D1nc", help="name of the new crop set")
    ap.add_argument("--config", default="cropper.yaml")
    ap.add_argument("--work", default="data/work")
    ap.add_argument("--results", default="results")
    ap.add_argument("--batch-photos", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check-n", type=int, default=200, help="accepted patches drawn for the hand check")
    a = ap.parse_args()
    cfg = CropperConfig.from_yaml(a.config) if Path(a.config).exists() else CropperConfig()
    seed_everything(cfg.seed)
    work, rep = Path(a.work), Path(a.results) / "cropper" / a.name
    out_dir = work / "crops" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    cropper = Cropper(cfg, device)

    defects = pd.read_csv(work / "crops" / f"{a.dataset}.csv")
    images = pd.read_csv(work / "images" / f"{a.dataset}.csv")
    boxes = defects.groupby("image")[["x1", "y1", "x2", "y2"]].apply(lambda g: [tuple(r) for r in g.to_numpy()])
    photos = [dict(image=r.image, path=r.path, img_w=r.img_w, img_h=r.img_h, name=r.name,
                   orig_stem=r.orig_stem, photo_stem=r.photo_stem, shipped_split=r.shipped_split,
                   boxes=boxes.get(r.image, [])) for r in images.itertuples()]
    print(f"{a.dataset}: {len(photos)} photos, {int(sum(1 for p in photos if p['boxes']))} with defect boxes; "
          f"CLIP on {device}", flush=True)

    # ---------------------------------------------------------------- sanity check on real defect crops
    # D1 defect crops show aircraft skin WITH damage: the surface score should be high on them and the
    # damage score should be higher on them than on the accepted no-damage patches.
    rng = np.random.default_rng(cfg.seed)
    ref = defects.sample(min(2000, len(defects)), random_state=cfg.seed)
    ref_s, ref_d = [], []
    for i in range(0, len(ref), 128):
        arr = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in ref.crop_path.iloc[i:i + 128]])
        s, d = cropper.score(arr)
        ref_s.append(s); ref_d.append(d)
    ref_s, ref_d = np.concatenate(ref_s), np.concatenate(ref_d)
    if cfg.calibrate_keep is not None:
        cfg.surface_threshold = float(np.quantile(ref_s, 1 - cfg.calibrate_keep))
        print(f"surface threshold calibrated to {cfg.surface_threshold:.4f} "
              f"(keeps {cfg.calibrate_keep:.0%} of {len(ref_s)} {a.dataset} defect crops)", flush=True)

    # workers only propose squares, so they get a Cropper without the CLIP model
    dl = DataLoader(PhotoCandidates(photos, Cropper(cfg), cfg.seed), batch_size=a.batch_photos,
                    num_workers=a.workers, collate_fn=collate_list)
    cand_rows, keep_rows = [], []
    for bi, batch in enumerate(dl):
        arrs = [arr for _, _, arr in batch if len(arr)]
        if not arrs:
            continue
        surf, dmg = cropper.score(np.concatenate(arrs))
        k = 0
        for i, cands, arr in batch:
            p, n = photos[i], len(cands)
            if not n:
                continue
            s, d = surf[k:k + n], dmg[k:k + n]
            k += n
            keep, reason = cropper.select(s, d)
            for j, (l, t, side) in enumerate(cands):
                cand_rows.append(dict(image=p["image"], cand=j, crop_left=l, crop_top=t, crop_side=side,
                                      surface_prob=round(float(s[j]), 4), damage_prob=round(float(d[j]), 4),
                                      decision=reason[j]))
            for j in keep:
                l, t, side = cands[j]
                f = out_dir / f"{Path(p['image']).with_suffix('').as_posix().replace('/', '__')}__nc{j}.jpg"
                Image.fromarray(arr[j]).save(f, quality=95)
                keep_rows.append(dict(crop_path=str(f), dataset=a.name, source=a.dataset, cls="no-damage",
                                      image=p["image"], src_name=p["name"], orig_stem=p["orig_stem"],
                                      group=f"{a.dataset}:{p['photo_stem']}",
                                      x1=l, y1=t, x2=l + side, y2=t + side, crop_left=l, crop_top=t, crop_side=side,
                                      img_w=p["img_w"], img_h=p["img_h"], shipped_split=p["shipped_split"],
                                      surface_prob=round(float(s[j]), 4), damage_prob=round(float(d[j]), 4)))
        if bi % 100 == 0:
            print(f"  {min((bi + 1) * a.batch_photos, len(photos))}/{len(photos)} photos, "
                  f"{len(keep_rows)} patches kept", flush=True)
    C, K = pd.DataFrame(cand_rows), pd.DataFrame(keep_rows)
    C.to_csv(rep / "candidates.csv", index=False)
    K.to_csv(work / "crops" / f"{a.name}.csv", index=False)

    def auc(pos, neg):  # P(score of a random positive > random negative)
        allv = np.concatenate([pos, neg])
        ranks = allv.argsort().argsort() + 1
        return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    damage_auc = auc(ref_d, K.damage_prob.to_numpy()) if len(K) else np.nan

    # ---------------------------------------------------------------- figures and sheets
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = np.linspace(0, 1, 41)
    ax[0].hist(C.surface_prob, bins, color="#52514e", alpha=0.6, label="all proposed squares")
    ax[0].hist(ref_s, bins, color="#2a78d6", alpha=0.6, label="D1 defect crops (reference)")
    ax[0].axvline(cfg.surface_threshold, color="#e34948", ls="--", label="threshold")
    ax[0].set(title="Step 3: CLIP surface score", xlabel="P(aircraft skin)", ylabel="count")
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].hist(K.damage_prob, bins, color="#1baf7a", alpha=0.6, density=True, label="accepted no-damage patches")
    ax[1].hist(ref_d, bins, color="#2a78d6", alpha=0.6, density=True, label="D1 defect crops (reference)")
    ax[1].set(title=f"Step 4: CLIP damage score (AUC {damage_auc:.2f})", xlabel="P(damaged)", ylabel="density")
    ax[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(rep / "scores.png", dpi=150)
    plt.close(fig)

    def cut(rows):  # re-cut squares from candidates.csv for the sheets
        out = []
        for r in rows.itertuples():
            pth = images.set_index("image").path[r.image]
            img = ImageOps.exif_transpose(Image.open(pth)).convert("RGB")
            f = rep / "tmp" / f"{abs(hash((r.image, r.cand)))}.jpg"
            f.parent.mkdir(exist_ok=True)
            img.crop((r.crop_left, r.crop_top, r.crop_left + r.crop_side, r.crop_top + r.crop_side)).resize(
                (CROP_SIZE, CROP_SIZE)).save(f)
            out.append(f)
        return out

    acc = K.sample(min(48, len(K)), random_state=1)
    sheet(acc.crop_path, [f"skin {s:.2f} dmg {d:.2f}" for s, d in zip(acc.surface_prob, acc.damage_prob)],
          rep / "accepted_sample.png", title="Accepted no-damage patches (random)")
    rej = C[C.decision == "not_surface"]
    rej = rej.sample(min(48, len(rej)), random_state=1).sort_values("surface_prob")
    sheet(cut(rej), [f"skin {s:.2f}" for s in rej.surface_prob], rep / "rejected_sample.png",
          title="Rejected by step 3 (random, lowest score first)")
    near = C[(C.surface_prob - cfg.surface_threshold).abs() < 0.1].sample(
        min(32, int(((C.surface_prob - cfg.surface_threshold).abs() < 0.1).sum())), random_state=1).sort_values("surface_prob")
    sheet(cut(near), [f"skin {s:.2f} {'kept' if s >= cfg.surface_threshold else 'drop'}" for s in near.surface_prob],
          rep / "borderline_sample.png", title=f"Borderline: surface score within 0.1 of the threshold {cfg.surface_threshold:.3f}")
    top = K.nlargest(min(32, len(K)), "damage_prob")
    sheet(top.crop_path, [f"dmg {d:.2f}" for d in top.damage_prob], rep / "highest_damage_accepted.png",
          title="Accepted patches with the highest damage score (check for missed defects)")

    chk = K.sample(min(a.check_n, len(K)), random_state=2).reset_index(drop=True)
    chk.insert(0, "check_id", range(1, len(chk) + 1))
    chk["is_skin"] = ""
    chk["has_damage"] = ""
    chk[["check_id", "crop_path", "image", "surface_prob", "damage_prob", "is_skin", "has_damage"]].to_csv(
        rep / "check_list.csv", index=False)
    for s in range(0, len(chk), 50):
        part = chk.iloc[s:s + 50]
        sheet(part.crop_path, [f"#{i}" for i in part.check_id], rep / f"check_sheet_{s // 50 + 1}.png", cols=10,
              title=f"Hand check: is each patch aircraft skin, and is it free of damage? (#{s + 1}-#{s + len(part)})")

    # ---------------------------------------------------------------- report
    n_req = len(photos) * cfg.candidates_per_image
    with_boxes = sum(1 for p in photos if p["boxes"])
    counts = C.decision.value_counts().to_dict()
    st = {
        "photos": len(photos), "photos_with_defect_boxes": with_boxes,
        "squares_requested": with_boxes * cfg.candidates_per_image,
        "step1_2_proposed_clear_of_defects": len(C),
        "step3_rejected_not_surface": counts.get("not_surface", 0),
        "step4_rejected_maybe_damage": counts.get("maybe_damage", 0),
        "passed_but_over_per_photo_limit": counts.get("over_limit", 0),
        "accepted": len(K), "photos_with_a_patch": int(K.image.nunique()) if len(K) else 0,
        "surface_threshold": round(cfg.surface_threshold, 4),
        "surface_score_median_on_defect_crops": round(float(np.median(ref_s)), 3),
        "defect_crops_passing_surface_threshold": round(float((ref_s >= cfg.surface_threshold).mean()), 3),
        "damage_score_auc_defect_vs_accepted": round(float(damage_auc), 3),
    }
    (rep / "stats.json").write_text(json.dumps(st, indent=2))
    md = [f"# Cropper report: {a.name} (from {a.dataset})\n",
          "| step | count |", "|---|---|",
          f"| photos | {st['photos']} ({with_boxes} with defect boxes) |",
          f"| 1+2. squares proposed clear of every defect box (of {st['squares_requested']} requested) | {st['step1_2_proposed_clear_of_defects']} |",
          f"| 3. dropped: not aircraft skin (CLIP surface score < {cfg.surface_threshold:.3f}"
          f"{f', calibrated to keep {cfg.calibrate_keep:.0%} of defect crops' if cfg.calibrate_keep is not None else ''}) | {st['step3_rejected_not_surface']} |",
          f"| 4. dropped: looks damaged | {st['step4_rejected_maybe_damage']} {'(filter off, score recorded only)' if cfg.damage_threshold is None else ''} |",
          f"| passed, but over the {cfg.keep_per_image}-per-photo limit | {st['passed_but_over_per_photo_limit']} |",
          f"| **accepted patches** | **{st['accepted']}** from {st['photos_with_a_patch']} photos |", "",
          "## Checks without hand labels\n",
          f"- D1 defect crops (always on aircraft skin) pass the surface threshold {st['defect_crops_passing_surface_threshold']:.0%} "
          f"of the time (median score {st['surface_score_median_on_defect_crops']}). This is a rough recall estimate for step 3.",
          f"- The damage score ranks a real defect crop above an accepted patch with probability {st['damage_score_auc_defect_vs_accepted']:.2f} "
          "(0.5 = no better than chance). It is also affected by how the patches differ in other ways, so treat it as a rough guide.\n",
          "![scores](scores.png)\n", "![accepted](accepted_sample.png)\n", "![rejected](rejected_sample.png)\n",
          "![borderline](borderline_sample.png)\n", "![damage](highest_damage_accepted.png)\n",
          "## Hand check\n",
          f"`check_list.csv` lists {len(chk)} random accepted patches, shown in `check_sheet_*.png` by number. "
          "Fill in `is_skin` and `has_damage` (y/n) to measure the Cropper's precision and its missed-label rate.\n",
          "## Settings\n", "```yaml", yaml_dump(cfg.to_dict()), "```"]
    (rep / "report.md").write_text("\n".join(md))
    for f in (rep / "tmp").glob("*.jpg"):
        f.unlink()
    if (rep / "tmp").exists():
        (rep / "tmp").rmdir()
    print(json.dumps(st, indent=2))


def yaml_dump(d):
    import yaml
    return yaml.safe_dump(d, sort_keys=False).strip()


if __name__ == "__main__":
    main()
