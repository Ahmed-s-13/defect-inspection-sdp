"""enhanced_res, step 2: build the MLP head on each backbone, pick the final model on val, test it once.

Backbones (train_backbone.py):
  r0  the plain retrain on the grouped split (--sampler base --recipe base: ../train.py's settings)
  r1  + group- and class-balanced sampling
  r2  + sampling and the enhanced training recipe
Head steps, added one at a time (the head is an ensemble of 5 MLPs plus a 5-NN vote, as in train_mlp.py):
  h0  original head: last-layer (layer4) features, MLP trained on the train crops' features
  h1  + layer3 features concatenated to layer4 (mid-level texture, 256 + 512 = 768-d)
  h2  + test-time augmentation: features averaged over the 8 flips/90-degree rotations of every crop
  h3  + augmented views: the MLP also trains on 3 randomly augmented versions of every train crop
  h4  + class-prior correction: p(c) / prior(c)^tau, tau picked on val
kNN weight and tau are picked on val; MLP epochs on val. Test (and the D5 outside test) only feed the report.
The final model (enhanced_res) is the backbone with the best val macro-F1 under h4. It is saved as
data/work/enhanced_res/enhanced_res_backbone.pt + enhanced_res_head.pt; the report goes to results/.

Needs the three runs from train_backbone.py and results/d5/ext_d5.csv from external_d5.py. From 1_main_model_resnet18/:
  python enhanced_res/build_head.py                  # ~1 h: features of 3 backbones + 15 heads
  python enhanced_res/build_head.py --report-only    # reuse results/ablation.csv, refit only baseline + final
"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from model import (BACKBONE_FILE, EWORK, HEAD_FILE, MAIN, RESULTS, Head, dihedral_t, feats, group_bootstrap_ci,
                   load_backbone, load_split)
from klshift.common import CropDataset, pick_device, seed_everything, train_transform
from klshift.mlp import knn_proba, l2n
from train_mlp import plot_confusion

BACKBONES = {"r0": EWORK / "runs/r0_plain/backbone.pt",
             "r1": EWORK / "runs/r1_sampler/backbone.pt",
             "r2": EWORK / "runs/r2_sampler_recipe/backbone.pt"}
LABEL = {"r0": "plain retrain", "r1": "+ balanced sampling", "r2": "+ balanced sampling + training recipe"}
HEADS = {"h0": "original MLP head (layer4)", "h1": "+ layer3 features", "h2": "+ test-time augmentation",
         "h3": "+ augmented training views", "h4": "+ class-prior correction"}
AUG_VIEWS = 3


@torch.no_grad()
def extract(name, paths, device, aug_paths):
    """Cache: plain features, 8-view TTA mean, and AUG_VIEWS augmented views of the train crops."""
    f = EWORK / f"cache/feats_{name}.npz"
    if f.exists():
        z = np.load(f)
        return z["plain"], z["tta"], z["aug"]
    f.parent.mkdir(parents=True, exist_ok=True)
    m, _ = load_backbone(BACKBONES[name], device)
    t0 = time.time()
    plain, tta = [], []
    for x, _ in DataLoader(CropDataset(paths), batch_size=64, num_workers=4):
        x = x.to(device)
        v = [feats(m, dihedral_t(x, k)) for k in range(8)]
        plain.append(v[0].cpu().numpy())
        tta.append(torch.stack(v).mean(0).cpu().numpy())
    aug = []
    for r in range(AUG_VIEWS):
        seed_everything(100 + r)
        aug.append(np.concatenate([feats(m, x.to(device)).cpu().numpy() for x, _ in
                                   DataLoader(CropDataset(aug_paths, transform=train_transform()), batch_size=64,
                                              num_workers=4)]))
    plain, tta, aug = np.concatenate(plain), np.concatenate(tta), np.stack(aug)
    np.savez(f, plain=plain.astype(np.float32), tta=tta.astype(np.float32), aug=aug.astype(np.float32))
    print(f"  {name}: features of {len(paths)} crops (+{AUG_VIEWS} aug views of {len(aug_paths)}) in "
          f"{time.time() - t0:.0f}s", flush=True)
    return plain, tta, aug


def train_head(Xtr, ytr, Xva, yva, n_classes, seed, device, epochs=60, batch=512, lr=1e-3, wd=1e-2, noise=0.5,
               smoothing=0.1):
    """train_mlp.py's train_one: AdamW + cosine, Gaussian feature noise, label smoothing, best epoch on val."""
    torch.manual_seed(seed)
    head = Head(Xtr.shape[1], n_classes)
    head.mu.copy_(Xtr.mean(0).cpu())
    head.sd.copy_(Xtr.std(0).cpu() + 1e-6)
    head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing)
    g = torch.Generator(device=device).manual_seed(seed)
    ns = noise * head.sd
    best, state, yv = -1.0, None, yva.cpu().numpy()
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Xtr), device=device, generator=g)
        for i in range(0, len(Xtr), batch):
            b = perm[i:i + batch]
            loss = loss_fn(head(Xtr[b] + ns * torch.randn(len(b), Xtr.shape[1], device=device, generator=g)), ytr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            f = f1_score(yv, head(Xva).argmax(1).cpu().numpy(), average="macro")
        if f > best:
            best, state = f, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(state)
    return head.eval()


def metrics(y, p, classes):
    lab = list(range(len(classes)))
    return dict(accuracy=accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro", labels=lab),
                balanced_accuracy=balanced_accuracy_score(y, p),
                recall=dict(zip(classes, recall_score(y, p, labels=lab, average=None, zero_division=0).round(4).tolist())))


def run_head(step, F, y, sp, ext, classes, device, seeds=5):
    """F: dict of feature arrays for this backbone. Returns val/test/ext probabilities and the fitted pieces."""
    cols = slice(256, 768) if step == "h0" else slice(0, 768)
    use_tta = step in ("h2", "h3", "h4")
    X = (F["tta"] if use_tta else F["plain"])[:, cols]
    Xe = (F["ext_tta"] if use_tta else F["ext_plain"])[:, cols]
    tr, va, te = sp == "train", sp == "val", sp == "test"
    Xtr, ytr = X[tr], y[tr]
    if step in ("h3", "h4"):
        Xtr = np.concatenate([Xtr, *F["aug"][:, :, cols]])
        ytr = np.tile(y[tr], 1 + F["aug"].shape[0])
    T = lambda v, dt=torch.float32: torch.as_tensor(v, dtype=dt, device=device)  # noqa: E731
    heads = [train_head(T(Xtr), T(ytr, torch.long), T(X[va]), T(y[va], torch.long), len(classes), s, device)
             for s in range(seeds)]
    with torch.no_grad():
        mlp = {k: torch.stack([h(T(v)).softmax(1) for h in heads]).mean(0)
               for k, v in [("val", X[va]), ("test", X[te]), ("ext", Xe)]}
    bank, bank_y = l2n(T(X[tr])), T(y[tr], torch.long)
    knn = {k: knn_proba(bank, bank_y, l2n(T(v)), 5, len(classes)) for k, v in [("val", X[va]), ("test", X[te]), ("ext", Xe)]}
    mf = lambda yy, pp: f1_score(yy, pp.argmax(1).cpu().numpy(), average="macro")  # noqa: E731
    alpha = max([0.0, 0.25, 0.5, 0.75], key=lambda w: mf(y[va], (1 - w) * mlp["val"] + w * knn["val"]))
    prob = {k: (1 - alpha) * mlp[k] + alpha * knn[k] for k in mlp}
    prior = T(np.bincount(y[tr], minlength=len(classes)) / tr.sum())
    tau = 0.0
    if step == "h4":
        adj = lambda p, t: p / prior ** t  # noqa: E731
        tau = max([0.0, 0.25, 0.5, 0.75, 1.0], key=lambda t: mf(y[va], adj(prob["val"], t)))
        prob = {k: adj(v, tau) / adj(v, tau).sum(1, keepdim=True) for k, v in prob.items()}
    pieces = dict(heads=[h.state_dict() for h in heads], d_in=X.shape[1], alpha=alpha, tau=tau,
                  prior=prior.cpu(), cols=(cols.start, cols.stop), tta=use_tta,
                  knn_bank=torch.as_tensor(X[tr]), knn_labels=torch.as_tensor(y[tr]))
    return {k: v.cpu().numpy() for k, v in prob.items()}, pieces


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-only", nargs="*", help="only cache the features of these backbones, then stop")
    ap.add_argument("--report-only", action="store_true",
                    help="reuse ablation.csv; refit only the baseline and the final model for the report")
    args = ap.parse_args()
    device = pick_device()
    RESULTS.mkdir(parents=True, exist_ok=True)
    crops = load_split()
    classes = sorted(crops.cls.unique())
    y = crops.cls.map({c: i for i, c in enumerate(classes)}).values
    sp = crops.split.values
    ext = pd.read_csv(RESULTS / "d5/ext_d5.csv")
    ext["crop_path"] = [str(MAIN / p) for p in ext.crop_path]
    ye = ext.cls.map({c: i for i, c in enumerate(classes)}).values
    paths = list(crops.crop_path) + list(ext.crop_path)
    train_paths = list(crops.crop_path[sp == "train"])
    res, probs, pieces = [], {}, {}
    if args.features_only:
        for bb in args.features_only:
            extract(bb, paths, device, train_paths)
        return

    def record(bb, step, prob, pc):
        pv, pt, pe = (prob[k].argmax(1) for k in ("val", "test", "ext"))
        mv = metrics(y[sp == "val"], pv, classes)
        mt = metrics(y[sp == "test"], pt, classes)
        lab_e = sorted(set(ye))
        res.append(dict(backbone=bb, head=step, val_macro_f1=mv["macro_f1"], val_acc=mv["accuracy"],
                        test_macro_f1=mt["macro_f1"], test_acc=mt["accuracy"], test_bal_acc=mt["balanced_accuracy"],
                        d5_acc=accuracy_score(ye, pe),
                        d5_mean_recall=np.mean(recall_score(ye, pe, labels=lab_e, average=None, zero_division=0)),
                        knn_weight=pc["alpha"], tau=pc["tau"], **{f"test_recall_{c}": mt["recall"][c] for c in classes},
                        **{f"d5_recall_{classes[c]}": r for c, r in
                           zip(lab_e, recall_score(ye, pe, labels=lab_e, average=None, zero_division=0))}))
        probs[(bb, step)], pieces[(bb, step)] = prob, pc
        r = res[-1]
        print(f"  {bb} {step}: val F1 {r['val_macro_f1']:.4f} | test F1 {r['test_macro_f1']:.4f} acc {r['test_acc']:.4f}"
              f" | D5 acc {r['d5_acc']:.4f}  (kNN w {pc['alpha']}, tau {pc['tau']})", flush=True)

    def feats_of(bb):
        plain, tta, aug = extract(bb, paths, device, train_paths)
        n = len(crops)
        return dict(plain=plain[:n], tta=tta[:n], ext_plain=plain[n:], ext_tta=tta[n:], aug=aug)

    if args.report_only:
        R = pd.read_csv(RESULTS / "ablation.csv")
    else:
        for bb in BACKBONES:
            F = feats_of(bb)
            for step in HEADS:
                prob, pc = run_head(step, F, y, sp, ext, classes, device)
                record(bb, step, prob, pc)
        R = pd.DataFrame(res)
        R.to_csv(RESULTS / "ablation.csv", index=False)

    # final model: best val macro-F1 among the enhanced backbones with the full head
    full = R[(R["head"] == "h4") & R.backbone.isin(["r1", "r2"])]
    final_bb = full.loc[full.val_macro_f1.idxmax(), "backbone"]
    final = (final_bb, "h4")
    base = ("r0", "h0")
    if args.report_only:                      # refit these two (same seeds) to get their predictions back
        for bb, step in [base, final]:
            prob, pc = run_head(step, feats_of(bb), y, sp, ext, classes, device)
            res.clear()
            record(bb, step, prob, pc)
            new = pd.DataFrame(res).iloc[0]
            old = R[(R.backbone == bb) & (R["head"] == step)].iloc[0]
            print(f"  refit {bb} {step}: val F1 {old.val_macro_f1:.4f} -> {new.val_macro_f1:.4f}, "
                  f"test F1 {old.test_macro_f1:.4f} -> {new.test_macro_f1:.4f}")
            R.loc[(R.backbone == bb) & (R["head"] == step), new.index] = new.values
        R.to_csv(RESULTS / "ablation.csv", index=False)
    shutil.copy(BACKBONES[final_bb], EWORK / BACKBONE_FILE)
    pc = pieces[final]
    torch.save(dict(classes=classes, backbone=BACKBONE_FILE, **pc,
                    note="features = pooled layer3+layer4 of the backbone, averaged over the 8 flips/rotations "
                         "(see evaluate.feats / dihedral_t); prob = ((1-alpha)*mean softmax of heads + alpha*5NN) "
                         "/ prior**tau"), EWORK / HEAD_FILE)

    # report pieces
    te, va = sp == "test", sp == "val"
    groups = crops.src_group.values[te]
    pb, pf = probs[base]["test"].argmax(1), probs[final]["test"].argmax(1)
    ci_b, ci_f = group_bootstrap_ci(y[te], pb, groups), group_bootstrap_ci(y[te], pf, groups)
    rng = np.random.default_rng(0)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    diffs = []
    for _ in range(2000):
        ii = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        diffs.append(f1_score(y[te][ii], pf[ii], average="macro") - f1_score(y[te][ii], pb[ii], average="macro"))
    ok_f, ok_b = pf == y[te], pb == y[te]
    nb, nc = int((ok_f & ~ok_b).sum()), int((~ok_f & ok_b).sum())
    mt_b, mt_f = metrics(y[te], pb, classes), metrics(y[te], pf, classes)
    plot_confusion(confusion_matrix(y[te], pf, labels=list(range(len(classes)))), classes, "enhanced_res: grouped D1-test", RESULTS / "confusion_enhanced_res.png")
    rb, rf = R.set_index(["backbone", "head"]).loc[base], R.set_index(["backbone", "head"]).loc[final]
    summary = dict(final_backbone=final_bb, base=base, final=final,
                   test=dict(base=mt_b, final=mt_f, base_ci=ci_b.tolist(), final_ci=ci_f.tolist(),
                             diff_ci=np.quantile(diffs, [0.025, 0.975]).tolist(),
                             mcnemar=dict(only_final_right=nb, only_base_right=nc,
                                          p=binomtest(nb, nb + nc, 0.5).pvalue if nb + nc else 1.0)),
                   d5=dict(base_acc=rb.d5_acc, final_acc=rf.d5_acc, n=len(ext), counts=ext.cls.value_counts().to_dict()))
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    md = ["# enhanced_res: ResNet-18 + MLP head, improved, on the leak-free D1 split\n",
          f"Final model: backbone **{final_bb}** ({LABEL[final_bb]}) with the full head (h4). Picked on val; test and "
          f"D5 were read only for this report.\n",
          "## Headline (grouped D1-test, 4,255 crops / 175 source groups)\n",
          "| model | accuracy | macro-F1 (95% group CI) | balanced accuracy |", "|---|---|---|---|",
          f"| baseline (plain retrain + original head) | {fmt(mt_b['accuracy'])} | {fmt(mt_b['macro_f1'])} "
          f"({fmt(ci_b[0])}-{fmt(ci_b[1])}) | {fmt(mt_b['balanced_accuracy'])} |",
          f"| **enhanced_res** | {fmt(mt_f['accuracy'])} | {fmt(mt_f['macro_f1'])} ({fmt(ci_f[0])}-{fmt(ci_f[1])}) | "
          f"{fmt(mt_f['balanced_accuracy'])} |",
          f"\nmacro-F1 gain {mt_f['macro_f1'] - mt_b['macro_f1']:+.3f}, 95% group-bootstrap CI of the gain "
          f"{summary['test']['diff_ci'][0]:+.3f} to {summary['test']['diff_ci'][1]:+.3f}. Same test crops: enhanced_res "
          f"right where the baseline is wrong {nb} times, the reverse {nc} times (McNemar p = "
          f"{summary['test']['mcnemar']['p']:.2g}).\n",
          "## Per-class recall on test\n", "| class | baseline | enhanced_res |", "|---|---|---|"]
    md += [f"| {c} | {fmt(mt_b['recall'][c])} | {fmt(mt_f['recall'][c])} |" for c in classes]
    md += [f"\n## Outside test: D5 photos not in D1 ({len(ext)} crops: "
           + ", ".join(f"{v} {k}" for k, v in ext.cls.value_counts().items()) + ")\n",
           "| model | accuracy | dent recall | missing-head recall |", "|---|---|---|---|"]
    for name, r in [("baseline", rb), ("enhanced_res", rf)]:
        md.append(f"| {name} | {fmt(r.d5_acc)} | {fmt(r['d5_recall_dent'])} | {fmt(r['d5_recall_missing-head'])} |")
    md += ["\n## What each change did (val picks, test shown for reference)\n",
           "| backbone | head | val macro-F1 | test macro-F1 | test accuracy | D5 accuracy |", "|---|---|---|---|---|---|"]
    for r in R.itertuples():
        md.append(f"| {r.backbone}: {LABEL[r.backbone]} | {r.head}: {HEADS[r.head]} | {fmt(r.val_macro_f1)} | "
                  f"{fmt(r.test_macro_f1)} | {fmt(r.test_acc)} | {fmt(r.d5_acc)} |")
    md += ["\n![confusion](confusion_enhanced_res.png)\n"]
    (RESULTS / "report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
