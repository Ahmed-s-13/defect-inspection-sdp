"""Step 0: download the datasets in YOLO format into <raw>/<DS>/.

D1-D5 are defect datasets; D6-D9 are no-damage aircraft datasets (one "aircraft" box per plane).

Credentials come from the environment only (never written to files):
  ROBOFLOW_API_KEY                    for D2-D5
  KAGGLE_USERNAME + KAGGLE_KEY        for D1 (or ~/.kaggle/kaggle.json / KAGGLE_API_TOKEN)

  python download.py --list                 # show Roboflow versions, image counts, augmentation
  python download.py --get D2:22 D3:1 ...   # download chosen versions
  python download.py --get D1               # Kaggle
  python download.py --get D6 D9            # FGVC-Aircraft / Open Images, converted to YOLO (no key needed)
  python download.py --get D10              # fuselage mock-up (Kaggle), two 720 px squares per frame
"""
import argparse
import os
import shutil
import ssl
import subprocess
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

ROBOFLOW = {
    "D2": ("lemi-debele", "aircraft-surface-damage"),
    "D3": ("lemi-debele", "aircraft-m3c6e"),
    "D4": ("123-hmdgs", "-6-classes-wind-turbine-dataset"),
    "D5": ("university-of-technology-sydney-21uto", "aircraft-defect-detection"),
    "D7": ("aircraft-recognition", "aircraft-detection-model-citoe"),   # no-damage, whole aircraft
    "D8": ("tans-workspace-01wim", "aircraft-kdg2w"),                    # no-damage, whole aircraft
}
KAGGLE_D1 = "sakettiw11/aircraft-dataset"
KAGGLE_D10 = "orvile/anomaly-detection-dataset-fuselage"
FGVC_URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
OI_CSV = "https://storage.googleapis.com/openimages/{}"
OI_FILES = {"validation": "v5/validation-annotations-bbox.csv", "test": "v5/test-annotations-bbox.csv"}
OI_IMG = "https://s3.amazonaws.com/open-images-dataset/{}/{}.jpg"
OI_PLANE = "/m/0cmf2"  # "Fixed-wing aircraft"


def fetch(url, dest):
    if not dest.exists():
        import certifi  # python.org builds on macOS ship without system CA certificates
        dest.parent.mkdir(parents=True, exist_ok=True)
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=ctx, timeout=60) as r:
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(r.read())
            tmp.rename(dest)
    return dest


def write_yolo(out, split, name, img, boxes):
    """boxes: list of (x1, y1, x2, y2) in pixels of img; one class (0 = aircraft)."""
    (out / split / "images").mkdir(parents=True, exist_ok=True)
    (out / split / "labels").mkdir(parents=True, exist_ok=True)
    img.save(out / split / "images" / f"{name}.jpg", quality=95)
    w, h = img.size
    lines = [f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
             for x1, y1, x2, y2 in boxes]
    (out / split / "labels" / f"{name}.txt").write_text("\n".join(lines))


def get_fgvc(out, src):
    """FGVC-Aircraft 2013b: one box per image. The bottom 20 px copyright banner is cut off."""
    # The official server is slow (~200 KB/s). A copy of the same archive unpacked under <src>
    # (e.g. Kaggle seryouxblaster764/fgvc-aircraft) is used when present.
    box = next(src.rglob("data/images_box.txt"), None)
    if box is None:
        with tarfile.open(fetch(FGVC_URL, src / "fgvc-aircraft-2013b.tar.gz")) as t:
            t.extractall(src, filter="data")
        box = next(src.rglob("data/images_box.txt"))
    data = box.parent
    n_img = len(list((data / "images").glob("*.jpg")))
    print(f"  FGVC data at {data}: {n_img} images")
    assert n_img == 10000, "incomplete FGVC copy"
    boxes = {l.split()[0]: list(map(float, l.split()[1:])) for l in (data / "images_box.txt").read_text().splitlines()}
    split = {}
    for sp, f in [("train", "images_train.txt"), ("val", "images_val.txt"), ("test", "images_test.txt")]:
        split.update({l.strip(): sp for l in (data / f).read_text().splitlines() if l.strip()})
    for i, (img_id, (x1, y1, x2, y2)) in enumerate(boxes.items()):
        img = Image.open(data / "images" / f"{img_id}.jpg").convert("RGB")
        w, h = img.size
        img = img.crop((0, 0, w, h - 20))  # remove banner
        b = (max(x1 - 1, 0), max(y1 - 1, 0), min(x2, w), min(y2, h - 20))
        write_yolo(out, split.get(img_id, "train"), img_id, img, [b])
        if i % 2000 == 0:
            print(f"  FGVC {i}/{len(boxes)}", flush=True)
    (out / "data.yaml").write_text("names: [aircraft]\nnc: 1\n")


def get_fuselage(out, src):
    """D10: fuselage mock-up video frames (1280x720), label per frame from its folder:
    Panel*_Normal -> normal, Panel*_Anomaly/big|small -> anomaly-big|anomaly-small.
    No boxes, so each frame gets two 720x720 'boxes' (left and right square) covering it."""
    if not any(src.rglob("*.png")):
        kaggle = shutil.which("kaggle") or str(Path(__file__).resolve().parent / ".venv/bin/kaggle")
        subprocess.run([kaggle, "datasets", "download", "-d", KAGGLE_D10, "-p", str(src), "--unzip"], check=True)
    names = ["normal", "anomaly-big", "anomaly-small"]
    files = sorted(src.rglob("*.png"))
    for i, f in enumerate(files):
        parts = f.relative_to(src).parts
        panel = next(x for x in parts if x.startswith("Panel"))
        cls = 0 if panel.endswith("Normal") else (1 if "big" in parts else 2)
        dist = next((x for x in parts if x.endswith("cm")), "")
        img = Image.open(f).convert("RGB")
        w, h = img.size
        side = min(w, h)
        boxes = [(0, 0, side, side), (w - side, 0, w, side)]
        name = f"{panel}_{'big_' if cls == 1 else 'small_' if cls == 2 else ''}{dist}_{f.stem}"
        (out / "all" / "images").mkdir(parents=True, exist_ok=True)
        (out / "all" / "labels").mkdir(parents=True, exist_ok=True)
        img.save(out / "all" / "images" / f"{name}.jpg", quality=95)
        (out / "all" / "labels" / f"{name}.txt").write_text("\n".join(
            f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            for x1, y1, x2, y2 in boxes))
        if i % 1000 == 0:
            print(f"  D10 {i}/{len(files)}", flush=True)
    (out / "data.yaml").write_text("names: [" + ", ".join(names) + "]\nnc: 3\n")


def get_openimages(out, src, workers=16):
    """Open Images V7 validation+test images with 'Fixed-wing aircraft' boxes
    (group-of and depiction boxes dropped)."""
    def one(job):
        split, img_id, bx = job
        f = fetch(OI_IMG.format(split, img_id), src / "images" / split / f"{img_id}.jpg")
        img = Image.open(f).convert("RGB")
        w, h = img.size
        write_yolo(out, "val" if split == "validation" else "test", img_id, img,
                   [(r.XMin * w, r.YMin * h, r.XMax * w, r.YMax * h) for r in bx.itertuples()])
    jobs = []
    for split, csv in OI_FILES.items():
        b = pd.read_csv(fetch(OI_CSV.format(csv), src / Path(csv).name))
        b = b[(b.LabelName == OI_PLANE) & (b.IsGroupOf == 0) & (b.IsDepiction == 0)]
        jobs += [(split, i, g) for i, g in b.groupby("ImageID")]
    print(f"  Open Images: {len(jobs)} images")
    with ThreadPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(one, jobs)):
            if k % 500 == 0:
                print(f"  Open Images {k}/{len(jobs)}", flush=True)
    (out / "data.yaml").write_text("names: [aircraft]\nnc: 1\n")


def rf_project(ds):
    from roboflow import Roboflow
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY in the environment")
    ws, proj = ROBOFLOW[ds]
    return Roboflow(api_key=key).workspace(ws).project(proj)


def list_versions(dss):
    for ds in dss:
        p = rf_project(ds)
        print(f"\n[{ds}] {ROBOFLOW[ds][0]}/{ROBOFLOW[ds][1]}")
        for v in p.versions():
            aug = {k: x for k, x in v.augmentation.items() if x} if isinstance(v.augmentation, dict) else v.augmentation
            print(f"  version {v.version}: {v.images} images, splits {v.splits}")
            print(f"      augmentation: {aug or 'none'}")
            print(f"      preprocessing: {v.preprocessing}")


def get(spec, raw, src_root="data/src"):
    ds, _, ver = spec.partition(":")
    out = Path(raw) / ds
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if ds == "D6":
        get_fgvc(out, Path(src_root) / ds)
    elif ds == "D9":
        get_openimages(out, Path(src_root) / ds)
    elif ds == "D10":
        get_fuselage(out, Path(src_root) / ds)
    elif ds == "D1":
        cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_D1, "-p", str(out), "--unzip"]
        kaggle = shutil.which("kaggle") or str(Path(__file__).resolve().parent / ".venv/bin/kaggle")
        subprocess.run([kaggle] + cmd[1:], check=True)
    else:
        if not ver:
            raise SystemExit(f"Give a version for {ds}, e.g. {ds}:1 (see --list)")
        rf_project(ds).version(int(ver)).download("yolov8", location=str(out), overwrite=True)
    n = sum(1 for p in out.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    (out / "SOURCE.txt").write_text(f"{spec}\n")
    print(f"[{ds}] {n} images in {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--get", nargs="+", default=[], help="D1, or D2:<version> ...")
    a = ap.parse_args()
    if a.list:
        list_versions(list(ROBOFLOW))
    for spec in a.get:
        get(spec, a.raw)


if __name__ == "__main__":
    main()
