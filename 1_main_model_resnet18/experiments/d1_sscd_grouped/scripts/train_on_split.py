"""Train the baseline ResNet-18 on any of the four splits, detached, awake and on the GPU.

  python scripts/train_on_split.py {original,sscd,d1_grouped,union,siamese} [train.py args, e.g. --epochs 30]

- Builds data/splits/<name>/work/ (crop-level d1_split.csv; crops are shared, not copied).
- Refuses to start unless the Apple GPU (MPS) is available: no silent CPU fallback.
- Runs main_model/train.py in its own session under `caffeinate -ims`, so closing the terminal,
  the app or turning the screen off does not stop it. Logs and status go to reports/runs/<name>/.
Lid-close sleep, power loss and shutdown cannot be prevented by software: keep the Mac on AC power.
"""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPLITS = {"original": None, "sscd": None, "d1_grouped": ("dg_split", "dg_group"), "union": ("union_split", "union_group"),
          "siamese": ("siamese_split", "siamese_group")}


def build(name):
    work = ROOT / "data" / "splits" / name / "work"
    (work / "crops").mkdir(parents=True, exist_ok=True)
    src = ROOT / "data" / "d1_sscd_grouped" / "work"
    crops = pd.read_csv(src / "crops" / "D1.csv")
    sscd = pd.read_csv(src / "d1_split.csv")
    out = sscd.copy()
    if name != "sscd":
        out["key"] = out["image"].str.extract(r"images/(.+)$")[0]
        if name == "original":
            m = pd.read_csv(ROOT / "manifests" / "union_split.csv")[["key", "original_split"]]
            out = out.merge(m, on="key", validate="m:1")
            out["split"], out["group_id"] = out["original_split"], out["image_id"]   # shipped split, no groups
        else:
            col_split, col_group = SPLITS[name]
            m = pd.read_csv(ROOT / "manifests" / ("siamese_split.csv" if name == "siamese" else "union_split.csv"))[["key", col_split, col_group]] \
                if name != "d1_grouped" else None
            if name == "d1_grouped":      # dg_* columns are not in union_split.csv, read them from compare loader
                sys.path.insert(0, str(HERE)); import compare_d1_grouped as c
                m = c.load()[["key", col_split, col_group]]
            out = out.merge(m, on="key", validate="m:1")
            out["split"], out["group_id"] = out[col_split], out[col_group]
        out = out[sscd.columns]
    out.to_csv(work / "d1_split.csv", index=False)
    crops[crops.crop_path.isin(out.crop_path)].to_csv(work / "crops" / "D1.csv", index=False)
    assert len(out) == len(crops) == 28637, (len(out), len(crops))
    return work, out


def require_gpu():
    if not torch.backends.mps.is_available():
        raise SystemExit("Apple GPU (MPS) is not available; refusing to train on CPU.")
    x = torch.randn(256, 256, device="mps")
    assert (x @ x).device.type == "mps"


def main():
    name = sys.argv[1]
    extra = sys.argv[2:]
    if name not in SPLITS:
        raise SystemExit(f"split must be one of {list(SPLITS)}")
    require_gpu()
    work, df = build(name)
    out = ROOT / "reports" / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    if (work / "model_best.pt").exists():
        raise SystemExit(f"{work / 'model_best.pt'} exists; move it before retraining.")
    env = dict(os.environ, PYTHONUNBUFFERED="1", MPLCONFIGDIR=str(ROOT / ".cache" / "matplotlib"))
    cmd = ["caffeinate", "-ims", sys.executable, "main_model/train.py", "--workers", "4",
           "--work", str(work), "--results", str(out), *extra]
    log = (out / "train.log").open("w")
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (out / "status.json").write_text(json.dumps({
        "split": name, "pid": p.pid, "device": "mps", "command": cmd,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "split_sha256": hashlib.sha256((work / "d1_split.csv").read_bytes()).hexdigest(),
        "crops": df.split.value_counts().to_dict()}, indent=2))
    print(f"started {name}: pid {p.pid}\n  log  {out / 'train.log'}\n  weights -> {work / 'model_best.pt'}")


if __name__ == "__main__":
    main()
