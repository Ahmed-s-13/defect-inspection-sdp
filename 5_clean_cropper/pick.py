"""Step 4 of clean_cropper: learn "clean" from the audit labels, score every candidate, pick the final set.

  python pick.py                   -> review/final.csv (+ thumbnails), pick.json

p_clean: logistic regression trained on the hand-labelled audit sample (clean vs damage / background / other) over
the CLIP features, CLIP's zero-shot prompt scores and score.py's p_damage / p_skin. Checked by 5-fold cross-validation
over groups on the audit sample (pick.json).

Ranking by p_clean alone favours blank, featureless panels (busyness = mean gradient magnitude: picks median 8,
audit-clean 12, damage 23), which would teach "featureless = no defect". So the candidates are split into N_BINS
busyness bins at the quantiles of the audit-clean crops (rivets, seams and screws included) and each bin gets the same
share of N_PICK; inside a bin they are taken by p_clean, round-robin across bins, with at most PER_PHOTO per photo and
PER_GROUP per group (a group is one source photo and its augmented copies). Audit crops labelled clean are always in;
audit crops labelled anything else are never in. The picks go to the review page's "final" tab for keep / reject.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import score
import thumbs

HERE = Path(__file__).resolve().parent
N_PICK, PER_PHOTO, PER_GROUP, C, N_BINS = 3000, 1, 10, 1.0, 5


def busyness(s):
    f = HERE / "work/cache/busy.csv"
    if f.exists():
        return pd.read_csv(f).set_index("id").busy.loc[s.id].values

    def one(p):
        g = cv2.imread(str(HERE / p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        return float(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean())
    with ThreadPoolExecutor(8) as ex:
        b = list(ex.map(one, s.crop))
    pd.DataFrame(dict(id=s.id, busy=b)).to_csv(f, index=False)
    return np.array(b)


def features(s):
    F = np.load(HERE / "work/cache/clip_candidates.npy")
    T, scale = score.clip_text(score.SKIN + score.BACKGROUND)
    return np.c_[F, scale * F @ T.T / 10, s.p_damage.values, s.p_skin.values]


def main():
    s = pd.read_csv(HERE / "scores.csv")
    X = features(s)
    lab = json.loads((HERE / "review/labels_audit.json").read_text())
    s["audit"] = s.id.map(lab)
    a = s.index[s.audit.notna()].values
    y = (s.audit[a] == "clean").astype(int).values

    oof = np.zeros(len(a))
    for tr, te in GroupKFold(5).split(a, y, s.group.values[a]):
        m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X[a][tr], y[tr])
        oof[te] = m.predict_proba(X[a][te])[:, 1]
    order = np.argsort(-oof)
    cv = dict(audit=len(a), clean_share=round(float(y.mean()), 3), auc=round(float(roc_auc_score(y, oof)), 3),
              clean_share_in_top={f"{q:.0%}": round(float(y[order[:int(q * len(a))]].mean()), 3) for q in (0.1, 0.2, 0.3, 0.5)})
    print("cross-validated on the audit:", cv)

    m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X[a], y)
    s["p_clean"] = m.predict_proba(X)[:, 1].round(3)

    s["busy"] = busyness(s).round(2)
    edges = np.quantile(s.busy[s.audit == "clean"], np.linspace(0, 1, N_BINS + 1)[1:-1])
    s["bin"] = np.searchsorted(edges, s.busy)
    s = s.sort_values("p_clean", ascending=False).set_index("id", drop=False)
    pick = s[s.audit == "clean"]
    n_photo = pick.key.value_counts().to_dict()
    n_group = pick.group.value_counts().to_dict()
    n_bin = pick.bin.value_counts().to_dict()
    chosen = list(pick.id)
    queues = {b: iter(g.itertuples()) for b, g in s[s.audit.isna()].groupby("bin")}
    quota = N_PICK // N_BINS
    while len(chosen) < N_PICK and queues:
        for b in list(queues):
            if n_bin.get(b, 0) >= quota and len(queues) > 1:
                del queues[b]
                continue
            for r in queues[b]:
                if n_photo.get(r.key, 0) < PER_PHOTO and n_group.get(r.group, 0) < PER_GROUP:
                    chosen.append(r.id)
                    for d, k in ((n_photo, r.key), (n_group, r.group), (n_bin, b)):
                        d[k] = d.get(k, 0) + 1
                    break
            else:
                del queues[b]
            if len(chosen) >= N_PICK:
                break
    f = s.loc[chosen].reset_index(drop=True)
    s[["id", "p_clean", "busy", "bin"]].to_csv(HERE / "p_clean.csv", index=False)
    f[["id"]].to_csv(HERE / "review/final.csv", index=False)
    thumbs.make(f)

    # expected clean share: each pick's rank in the pool, read off the cross-validated "clean share in the top x%"
    rank = s.p_clean.rank(ascending=False, pct=True)
    curve = list(cv["clean_share_in_top"].values())
    est = float(np.mean([1.0 if a_ == "clean" else np.interp(r, [0.1, 0.2, 0.3, 0.5], curve)
                         for a_, r in zip(f.audit, rank.loc[f.id].values)]))
    out = dict(cv=cv, picked=len(f), deepest_rank_in_pool=round(float(rank.loc[f.id].max()), 3),
               by_split=f.split.value_counts().to_dict(), groups=int(f.group.nunique()), photos=int(f.key.nunique()),
               est_clean_share=round(est, 2), busy_bin_edges=[round(float(e), 1) for e in edges],
               by_bin=f.bin.value_counts().sort_index().to_dict(),
               busy_quantiles=dict(picks=f.busy.quantile([.1, .25, .5, .75, .9]).round(1).tolist(),
                                   audit_clean=s.busy[s.audit == "clean"].quantile([.1, .25, .5, .75, .9]).round(1).tolist()))
    (HERE / "pick.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
