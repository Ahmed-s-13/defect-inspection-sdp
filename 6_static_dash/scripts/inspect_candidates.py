"""Contact sheets of candidate Inspect photos: d1_gc7_fix test crops (never trained on), the largest one per
photo group for each class, so a person can pick the photos that show each defect most clearly. Writes
scripts/candidates/<class>.jpg (numbered tiles) + <class>.csv (tile number -> crop). Read-only on the repo.

    ../../Experiments/.venv/bin/python scripts/inspect_candidates.py      (from 6_static_dash/)
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO_MAIN = HERE.parents[1] / "1_main_model_resnet18"
sys.path.insert(0, str(REPO_MAIN))
from enhanced_res.model import load_split  # noqa: E402

OUT = HERE / "candidates"
N, T = 48, 180  # tiles per sheet, tile side


def main():
    OUT.mkdir(exist_ok=True)
    s = load_split("d1_gc7_fix")
    s = s[s.split == "test"].copy()
    s["px"] = [min(Image.open(p).size) for p in s.crop_path]
    for cls, g in s.groupby("cls"):
        g = g[g.px >= 160].sort_values("px", ascending=False).drop_duplicates("src_group").head(N)
        sheet = Image.new("RGB", (8 * T, ((len(g) + 7) // 8) * T), "white")
        d = ImageDraw.Draw(sheet)
        for k, p in enumerate(g.crop_path):
            im = Image.open(p).convert("RGB")
            side = min(im.size); l, t = (im.width - side) // 2, (im.height - side) // 2   # Inspect's centre square
            im = im.crop((l, t, l + side, t + side)).resize((T - 4, T - 4))
            x, y = (k % 8) * T, (k // 8) * T
            sheet.paste(im, (x + 2, y + 2))
            d.rectangle((x + 2, y + 2, x + 30, y + 20), fill="black"); d.text((x + 5, y + 5), str(k), fill="white")
        sheet.save(OUT / f"{cls}.jpg", quality=85)
        g[["crop_path", "src_group", "px"]].reset_index(drop=True).to_csv(OUT / f"{cls}.csv", index_label="tile")
        print(cls, len(g), "of", s[s.cls == cls].src_group.nunique(), "groups")


if __name__ == "__main__":
    main()
