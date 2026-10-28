"""Realistic off-target benchmark for architecture A's unfamiliar-input gate: people, faces, helmets, hands, gloves,
tools, vehicles, hangar / warehouse / airport scenes, first-person hands, plus undamaged aircraft (D6 / D9).

From 1_main_model_resnet18/:
  python enhanced_res/arch_a/experiments/offtarget_data.py           # downloads (no keys), crops, manifest

Sources (third-party images: local only, never in git; licences in <OUT>/SOURCES.md):
  Open Images V7   validation + test boxes (Person, Human face, Helmet, Human hand, Glove, Tool, Ladder, Car, Truck);
                   images that also contain any aircraft box are dropped. Reuses download.py's fetch / bbox CSVs.
  Places365        validation set, 100 images per scene class
  EgoHands         dropped: its download address is gone (redirects / 404 as of 2026-10-27); hands come from Open
                   Images "Human hand" instead
  D6 / D9          FGVC-Aircraft and Open Images planes (download.py --get D6 D9; prepare.py makes D6p / D9p
                   undamaged-skin patches): aircraft without damage, expected to look familiar
  user photo       the helmet photo from 2026-10-27, if present at <OUT>/user/

Two crops per source image, as the helmet could produce them:
  centre   the whole frame's centre square (predict.py without --box)
  box      a box-sized patch: the object's box + 25% margin (square_crop_box, like training) where a box exists,
           otherwise a random square of 25-60% of the short side
Roles: fit (may train a learned AoN check), test (held-out categories only), nodamage, sanity.
An image never appears in two roles. Writes <OUT>/crops/ and <OUT>/manifest.csv.
"""
import hashlib
import io
import sys
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[2]
sys.path.insert(0, str(MAIN))

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from download import OI_CSV, OI_FILES, OI_IMG, fetch
from klshift.common import CROP_SIZE, square_crop_box

SRC = MAIN / "data/src/offtarget"                        # raw downloads (git-ignored, data/)
OUT = MAIN / "data/work/enhanced_res/offtarget"          # crops + manifest (git-ignored)
SEED = 0
OI_CLASSES = {  # name: (role, images)
    "Helmet": ("test", 200), "Human hand": ("test", 200), "Glove": ("test", 150),
    "Person": ("fit", 200), "Human face": ("fit", 150), "Tool": ("fit", 150), "Ladder": ("fit", 100),
    "Car": ("fit", 100), "Truck": ("fit", 100),
}
PLACES = {  # scene: role (100 validation images each)
    "/h/hangar/indoor": "test", "/h/hangar/outdoor": "test", "/s/storage_room": "test",   # Places365 has no warehouse
    "/r/repair_shop": "test", "/a/airport_terminal": "test", "/r/runway": "test",
    "/g/garage/indoor": "fit", "/i/industrial_area": "fit", "/l/loading_dock": "fit",
    "/a/auto_factory": "fit", "/e/engine_room": "fit",
}
# Added 28 Oct for architecture B (python offtarget_data.py --add-plain): plain surfaces, which a model with a
# no-defect class may confuse with clean skin. B's AoN uses them; A's calibration.json leaves them out (unchanged).
PLACES_PLAIN = {
    "/p/parking_lot": "fit", "/h/highway": "fit", "/a/airfield": "fit", "/p/parking_garage/indoor": "fit",
    "/d/driveway": "fit", "/s/sky": "fit", "/d/desert/sand": "fit", "/l/landing_deck": "fit",
    "/s/street": "test", "/c/construction_site": "test",
}
DTD_URL = "https://thor.robots.ox.ac.uk/dtd/dtd-r1.0.1.tar.gz"     # Describable Textures Dataset (Cimpoi et al. 2014)
N_DTD = 20                                                          # images per texture category
B_ONLY_SOURCES = ("places365-plain", "dtd")
PLACES_VAL = "http://data.csail.mit.edu/places/places365/val_256.tar"
PLACES_LIST = "http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar"
EGOHANDS = "http://vision.soic.indiana.edu/egohands_files/egohands_data.zip"
N_EGO = 300
SOURCES_MD = """# Off-target benchmark sources (local only, not for git)

| source | licence / terms | used for |
|---|---|---|
| Open Images V7 (Google), validation + test images | images: CC BY 2.0 (per-image authors on Flickr); annotations: CC BY 4.0 | people, faces, helmets, hands, gloves, tools, ladders, vehicles |
| Places365 (MIT CSAIL), validation set | research / non-commercial use | hangar, warehouse, airport, runway, garage, industrial scenes |
| FGVC-Aircraft 2013b (Oxford VGG) | research / non-commercial use | whole aircraft, undamaged skin (D6) |
| Open Images "Fixed-wing aircraft" | CC BY 2.0 / CC BY 4.0 | whole aircraft, undamaged skin (D9) |
| Places365 (MIT CSAIL), validation set, plain scenes (added 28 Oct) | research / non-commercial use | parking lots, highways, airfields, sky, sand, streets, construction sites (architecture B only) |
| DTD, Describable Textures Dataset (Oxford VGG, Cimpoi et al. 2014) | research use | 47 texture categories (architecture B only) |
"""


def sha1(b):
    return hashlib.sha1(b).hexdigest()


def centre(img):
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC)


def box_crop(img, box=None, rng=None):
    w, h = img.size
    if box is not None:
        left, top, side = square_crop_box(*box, w, h)
    else:
        side = int(min(w, h) * rng.uniform(0.25, 0.6))
        left, top = int(rng.integers(0, w - side + 1)), int(rng.integers(0, h - side + 1))
    return img.crop((left, top, left + side, top + side)).resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC)


def save_pair(img, key, source, category, role, rows, box=None, rng=None, digest=""):
    for mode, c in (("centre", centre(img)), ("box", box_crop(img, box, rng))):
        p = OUT / "crops" / mode / source / f"{key}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        c.save(p, quality=95)
        rows.append(dict(crop_path=str(p.relative_to(MAIN)), source=source, category=category, role=role,
                         mode=mode, image=key, sha1=digest))


# ------------------------------------------------------------------ Open Images
def openimages(rows, rng):
    src = MAIN / "data/src/D9"                           # download.py keeps its bbox CSVs here
    desc = pd.read_csv(fetch(OI_CSV.format("v5/class-descriptions-boxable.csv"), SRC / "oi/class-descriptions-boxable.csv"),
                       header=None, names=["mid", "name"])
    mid = dict(zip(desc.name, desc.mid))
    air = set(desc[desc.name.str.contains("aircraft|airplane|helicopter|aeroplane", case=False)].mid)
    b = pd.concat([pd.read_csv(fetch(OI_CSV.format(csv), src / Path(csv).name)).assign(split=split)
                   for split, csv in OI_FILES.items()])
    has_air = set(b[b.LabelName.isin(air)].ImageID)
    b = b[(b.IsGroupOf == 0) & (b.IsDepiction == 0) & ~b.ImageID.isin(has_air)]
    b = b[(b.XMax - b.XMin) * (b.YMax - b.YMin) >= 0.03]          # object visible: >= 3% of the image
    used, jobs = set(), []
    for name, (role, n) in sorted(OI_CLASSES.items(), key=lambda kv: kv[1][0] != "test"):   # test classes first
        c = b[(b.LabelName == mid[name]) & ~b.ImageID.isin(used)]
        ids = rng.permutation(c.ImageID.unique())[:n]
        used.update(ids)
        first = c[c.ImageID.isin(ids)].groupby("ImageID").head(1)
        jobs += [(name, role, r.split, r.ImageID, (r.XMin, r.YMin, r.XMax, r.YMax)) for r in first.itertuples()]
    print(f"Open Images: {len(jobs)} images", flush=True)

    def get(job):
        name, role, split, iid, bx = job
        try:
            f = fetch(OI_IMG.format(split, iid), SRC / "oi/images" / f"{iid}.jpg")
            return job, f.read_bytes()
        except Exception as e:                          # a few Open Images URLs are gone
            return job, None
    with ThreadPoolExecutor(16) as ex:
        for (name, role, split, iid, bx), data in ex.map(get, jobs):
            if data is None:
                continue
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            w, h = img.size
            save_pair(img, iid, "openimages", name, role, rows,
                      box=(bx[0] * w, bx[1] * h, bx[2] * w, bx[3] * h), digest=sha1(data))


# ------------------------------------------------------------------ Places365
def places(rows, rng, scenes=None, source="places365"):
    PLACES_ = scenes or PLACES
    lst = fetch(PLACES_LIST, SRC / "places/filelist_places365-standard.tar")
    with tarfile.open(lst) as t:
        cats = t.extractfile(next(m for m in t.getmembers() if m.name.endswith("categories_places365.txt"))).read().decode()
        val = t.extractfile(next(m for m in t.getmembers() if m.name.endswith("places365_val.txt"))).read().decode()
    idx = {line.split()[0]: int(line.split()[1]) for line in cats.splitlines()}
    missing = [s for s in PLACES_ if s not in idx]
    assert not missing, f"Places365 has no class {missing}"
    want = {}
    for line in val.splitlines():
        f, k = line.split()
        for scene, role in PLACES_.items():
            if int(k) == idx[scene]:
                want[f] = (scene, role)
    print(f"Places365: {len(want)} validation images in {len(PLACES_)} scenes (downloading the ~500 MB set) ...", flush=True)
    tar = fetch(PLACES_VAL, SRC / "places/val_256.tar")
    with tarfile.open(tar) as t:
        for m in t:
            name = Path(m.name).name
            if m.isfile() and name in want:             # read members by name only; nothing is extracted to disk
                data = t.extractfile(m).read()
                scene, role = want[name]
                img = Image.open(io.BytesIO(data)).convert("RGB")
                save_pair(img, Path(name).stem, source, scene.split("/", 2)[2], role, rows, rng=rng, digest=sha1(data))


# ------------------------------------------------------------------ EgoHands
def egohands(rows, rng):
    print("EgoHands: downloading ~1.3 GB ...", flush=True)
    z = fetch(EGOHANDS, SRC / "egohands/egohands_data.zip")
    with zipfile.ZipFile(z) as zf:
        frames = sorted(n for n in zf.namelist() if n.lower().endswith(".jpg") and "_LABELLED_SAMPLES/" in n)
        vids = sorted({n.split("/")[-2] for n in frames})
        per = max(1, N_EGO // len(vids))
        pick = []
        for v in vids:
            fv = [n for n in frames if n.split("/")[-2] == v]
            pick += list(rng.permutation(fv)[:per])
        for n in pick[:N_EGO]:
            data = zf.read(n)
            img = Image.open(io.BytesIO(data)).convert("RGB")
            key = n.split("/")[-2] + "_" + Path(n).stem
            save_pair(img, key, "egohands", "first-person hands", "test", rows, rng=rng, digest=sha1(data))


# ------------------------------------------------------------------ undamaged aircraft + user photo
def aircraft(rows, rng):
    for ds, src in (("D6", "fgvc-aircraft"), ("D9", "openimages-planes")):
        raw = MAIN / "data/raw" / ds
        patches = MAIN / f"data/work/crops/{ds}p.csv"
        if not raw.exists() or not patches.exists():
            print(f"{ds}: missing (run download.py --get {ds} and prepare.py --datasets {ds} {ds}p), skipped")
            continue
        imgs = sorted(p for p in raw.rglob("*.jpg") if "images" in p.parts)
        for p in rng.permutation(imgs)[:500]:          # whole-plane photos, centre square only
            img = Image.open(p).convert("RGB")
            c = OUT / "crops/centre" / src / f"{p.stem}.jpg"
            c.parent.mkdir(parents=True, exist_ok=True)
            centre(img).save(c, quality=95)
            rows.append(dict(crop_path=str(c.relative_to(MAIN)), source=src, category="whole aircraft (no damage)",
                             role="nodamage", mode="centre", image=p.stem, sha1=sha1(p.read_bytes())))
        pc = pd.read_csv(patches).sample(frac=1, random_state=SEED).head(1000)
        for r in pc.itertuples():                       # undamaged skin patches (prepare.py patch mode)
            rows.append(dict(crop_path=r.crop_path, source=src, category="aircraft skin patch (no damage)",
                             role="nodamage", mode="box", image=Path(r.crop_path).stem, sha1=""))
    user = OUT / "user"
    for p in sorted(user.glob("*.jpg")) if user.exists() else []:
        data = p.read_bytes()
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        save_pair(img, p.stem, "user", "helmet photo (user)", "test", rows, rng=rng, digest=sha1(data))


def dtd(rows, rng):
    """DTD: 47 texture categories, alternately fit / held-out test (sorted by name), N_DTD images each."""
    tar = fetch(DTD_URL, SRC / "dtd/dtd-r1.0.1.tar.gz")
    with tarfile.open(tar) as t:
        members = [m for m in t.getmembers() if m.isfile() and "/images/" in m.name and m.name.endswith(".jpg")]
        cats = sorted({Path(m.name).parent.name for m in members})
        role = {c: ("fit" if i % 2 == 0 else "test") for i, c in enumerate(cats)}
        for c in cats:
            ms = [m for m in members if Path(m.name).parent.name == c]
            for k in rng.choice(len(ms), min(N_DTD, len(ms)), replace=False):
                data = t.extractfile(ms[k]).read()
                img = Image.open(io.BytesIO(data)).convert("RGB")
                save_pair(img, Path(ms[k].name).stem, "dtd", f"texture: {c}", role[c], rows, rng=rng, digest=sha1(data))
    print(f"DTD: {len(cats)} texture categories, {N_DTD} images each", flush=True)


def add_plain():
    """Append the plain-surface negatives (Places365 plain scenes + DTD) to the existing manifest; nothing else changes."""
    man = pd.read_csv(OUT / "manifest.csv", keep_default_na=False)
    man = man[~man.source.isin(B_ONLY_SOURCES)]                       # re-running replaces only these rows
    rng = np.random.default_rng(SEED + 1)
    rows = []
    places(rows, rng, PLACES_PLAIN, "places365-plain")
    dtd(rows, rng)
    m = pd.concat([man, pd.DataFrame(rows)], ignore_index=True)
    h = m[m.sha1 != ""].groupby("sha1").role.nunique()
    assert (h <= 1).all(), f"{(h > 1).sum()} images appear in two roles"
    m.to_csv(OUT / "manifest.csv", index=False)
    print(pd.DataFrame(rows).groupby(["role", "source"]).size().to_string())


def main():
    if "--add-plain" in sys.argv:
        return add_plain()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SOURCES.md").write_text(SOURCES_MD)
    rng = np.random.default_rng(SEED)
    rows = []
    for step in (openimages, places, aircraft):
        try:
            step(rows, rng)
        except Exception as e:
            print(f"!! {step.__name__} failed: {type(e).__name__}: {e}", flush=True)
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "manifest.csv", index=False)
    # no source image in two roles
    h = m[m.sha1 != ""].groupby("sha1").role.nunique()
    assert (h <= 1).all(), f"{(h > 1).sum()} images appear in two roles"
    print(m.groupby(["role", "source", "category", "mode"]).size().to_string())
    print(f"\n{len(m)} crops, {m[m.sha1 != ''].sha1.nunique()} distinct source images; no image in two roles")


if __name__ == "__main__":
    main()
