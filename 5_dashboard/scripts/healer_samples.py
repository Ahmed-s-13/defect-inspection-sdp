"""Before/after pairs and counts from the healed "no defect" set, for the healer section of the Data page.
Reads 2_lama_healing/no_defect (its tables are in git; the healed photos in work/healed/ are made by its
heal_groups.py) and the D1 photos; writes web/public/healer/ and web/src/data/healer.json.

    python scripts/healer_samples.py
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

DASH = Path(__file__).resolve().parents[1]
HEALED = DASH.parent / "2_lama_healing/no_defect"
RAW = DASH.parent / "1_main_model_resnet18/data/raw/D1/content/Aircraft_dataset/images"
OUT_PUB, OUT_JSON = DASH / "web/public/healer", DASH / "web/src/data/healer.json"
SAMPLES = [604, 550, 728]  # groups: dent + scratches, scratches on a riveted panel, a missing screw head
ROUTE = {"lama": "painted over", "head": "screw head put back", "lama only": "painted over, no head put back"}


def main():
    OUT_PUB.mkdir(parents=True, exist_ok=True)
    man, boxes = pd.read_csv(HEALED / "manifest.csv"), pd.read_csv(HEALED / "boxes.csv")
    kept, crops = pd.read_csv(HEALED / "no_defect_healed.csv"), pd.read_csv(HEALED / "crops.csv")
    samples = []
    for g in SAMPLES:
        r = man[man.group == g].iloc[0]
        for name, src in (("before", RAW / r.key), ("after", HEALED / r.file)):
            Image.open(src).convert("RGB").save(OUT_PUB / f"g{g}_{name}.jpg", quality=88)
        b = boxes[boxes.group == g]
        routes = Counter(ROUTE[x.split(":")[0].split(" (")[0]] for x in b.route)
        samples.append({"group": g, "split": r.split, "key": r.key, "w": int(r.w), "h": int(r.h),
                        "classes": sorted(set(b.cls)), "boxes": [json.loads(x) for x in b.box],
                        "routes": dict(routes), "healed_frac": float(r.healed_frac)})
    route = Counter(x.split(":")[0].split(" (")[0] for x in boxes.route)
    out = {"photos": len(man), "boxes": len(boxes), "routes": dict(route), "crops": len(crops), "auto": dict(Counter(crops.auto)), "kept": len(kept),
           "kept_split": dict(Counter(kept.split)), "samples": samples}
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "samples"})


if __name__ == "__main__":
    main()
