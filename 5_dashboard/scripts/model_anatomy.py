"""Follow one test crop through enhanced_res (the served model, gc7), stage by stage, for the "How it works" section
of the Model page.
Writes web/src/data/anatomy.json and web/public/anatomy/ (layer activation maps, the 5 nearest training crops).

    python scripts/model_anatomy.py
"""
import json
import shutil
import sys
from pathlib import Path

DASH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DASH))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from server.paths import REPO_MAIN  # noqa: E402
from server.preprocess import to_batch  # noqa: E402
from server.registry import registry  # noqa: E402
from server.states import describe  # noqa: E402
from enhanced_res.model import dihedral_t, feats, load_split  # noqa: E402
from klshift.mlp import knn_proba, l2n  # noqa: E402

CROP = "test/aircraft_000969.jpg"            # the Data page's cropping example (a crack)
OUT_PUB, OUT_JSON = DASH / "web/public/anatomy", DASH / "web/src/data/anatomy.json"
ABLATION = REPO_MAIN / "enhanced_res/results/ablation.csv"
LADDER = [("r0", "h0", "Plain ResNet-18 + MLP head"), ("r2", "h0", "Balanced sampling, gentler training"),
          ("r2", "h1", "+ layer3 features"), ("r2", "h2", "+ 8 views averaged"), ("r2", "h4", "+ augmented views, kNN vote")]
INK, TEAL = np.array([242, 241, 238]), np.array([14, 124, 134])  # --ground to --control


def heat(a, size=168):
    """Channel-mean activation of one layer as a ground-to-teal map."""
    m = a.abs().mean(0).cpu().numpy()
    m = (m - m.min()) / (m.max() - m.min() + 1e-9)
    rgb = (INK[None, None] * (1 - m[..., None]) + TEAL[None, None] * m[..., None]).astype(np.uint8)
    return Image.fromarray(rgb).resize((size, size), Image.NEAREST)


@torch.no_grad()
def main():
    OUT_PUB.mkdir(parents=True, exist_ok=True)
    a = registry.active()
    m, bb = a.m, a.m.backbone
    split = load_split("d1_gc7")    # the model's own training split: its kNN bank is these train crops, in order
    row = split[(split.key == CROP) & (split.cls != "no-defect")].iloc[0]
    crop = Image.open(row.crop_path).convert("RGB")
    x = to_batch([crop]).to(m.device)

    # the backbone, stage by stage, on the crop as it is
    h = bb.maxpool(bb.relu(bb.bn1(bb.conv1(x))))
    layers = []
    for i, layer in enumerate([bb.layer1, bb.layer2, bb.layer3, bb.layer4], 1):
        h = layer(h)
        heat(h[0]).save(OUT_PUB / f"layer{i}.png")
        layers.append({"name": f"layer{i}", "channels": h.shape[1], "size": h.shape[2]})

    f = m.features(x)                                     # 768: pooled layer3 + layer4, mean of 8 views
    views = torch.stack([feats(bb, dihedral_t(x, k)) for k in range(8)])
    assert torch.allclose(views.mean(0), f, atol=1e-4)
    heads = [hd(f).softmax(1)[0].cpu().numpy() for hd in m.heads]
    mlp = np.mean(heads, 0)
    sim, idx = (l2n(f) @ m.bank.T).topk(5, dim=1)
    knn = knn_proba(m.bank, m.bank_y, l2n(f), 5, len(m.classes))[0].cpu().numpy()
    blend = (1 - m.alpha) * mlp + m.alpha * knn
    prior = m.prior.cpu().numpy()
    corrected = blend / prior ** m.tau if m.tau > 0 else blend   # the class-prior correction, picked on val
    corrected = corrected / corrected.sum()
    probs, decisions = a.predict(x, [crop])
    assert np.allclose(probs[0], corrected, atol=1e-5)
    r = describe(a.classes, probs[0].tolist(), decisions[0], a.arch)

    train = split[split.split == "train"].reset_index(drop=True)
    assert len(train) == len(m.bank), "the kNN bank is not the train crops in split order"
    nbrs = []
    for j, (s, i) in enumerate(zip(sim[0].tolist(), idx[0].tolist())):
        t = train.iloc[i]
        shutil.copy(t.crop_path, OUT_PUB / f"nn{j}.jpg")
        nbrs.append({"cls": t.cls, "similarity": round(s, 3), "group": int(t.src_group), "key": t.key})
    crop.save(OUT_PUB / "crop.jpg", quality=92)

    ab = pd.read_csv(ABLATION)
    ladder = [{"label": lab, "val": round(float(q.val_macro_f1), 3), "test": round(float(q.test_macro_f1), 3)}
              for bk, hk, lab in LADDER for q in [ab[(ab.backbone == bk) & (ab["head"] == hk)].iloc[0]]]
    summ = registry.entry().summary
    if summ and "no-defect" in m.classes:  # the served model: the same recipe with the 7th class, over 7 classes
        g = json.loads(summ.read_text())
        ladder.append({"label": "+ no-defect class (7 classes)", "val": round(g["val"]["macro_f1"], 3),
                       "test": round(g["test"]["macro_f1"], 3), "seven": True})
    st = m.heads[0].state_dict()
    hidden = [st[k].shape[0] for k in st if k.endswith(".weight") and st[k].dim() == 2][:-1]
    ck = torch.load(Path(registry.entry().weights) / "enhanced_res_backbone.pt", map_location="cpu")
    out = {
        "crop": {"key": CROP, "cls": row.cls, "group": int(row.src_group)},
        "classes": list(m.classes), "layers": layers, "d_in": int(f.shape[1]),
        "features": [round(v, 4) for v in f[0].cpu().tolist()],
        "heads": [[round(float(v), 4) for v in p] for p in heads], "hidden": hidden,
        "mlp": [round(float(v), 4) for v in mlp], "knn": [round(float(v), 4) for v in knn],
        "alpha": float(m.alpha), "k": 5, "bank": len(m.bank), "neighbours": nbrs,
        "blend": [round(float(v), 4) for v in blend], "tau": float(m.tau), "prior": [round(float(v), 4) for v in prior],
        "corrected": [round(float(v), 4) for v in corrected], "tempered": [round(v, 4) for v in r["probs"]],
        "state": r["state"], "answer": r["answer"], "temperature": round(float(a.arch.T), 4),
        "train": {"epoch": int(ck["epoch"]), **{k: ck["args"][k] for k in ("lr", "head_lr", "smoothing", "ema", "batch")}},
        "ladder": ladder,
    }
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"{CROP}: MLP {m.classes[mlp.argmax()]} {mlp.max():.3f}, kNN {m.classes[knn.argmax()]} {knn.max():.3f}, "
          f"blend {m.classes[blend.argmax()]} {blend.max():.3f}, corrected {corrected.max():.3f}, "
          f"tempered {max(r['probs']):.3f} -> {r['state']} {r['answer']}")
    print("neighbours:", [(n["cls"], n["similarity"]) for n in nbrs])


if __name__ == "__main__":
    main()
