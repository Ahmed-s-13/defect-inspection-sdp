"""Apply (or undo) the D1 box corrections in corrections.csv to the raw YOLO label files.

From 1_main_model_resnet18/:
  python d1_grouped/label_corrections/apply_corrections.py            # apply
  python d1_grouped/label_corrections/apply_corrections.py --check    # only report the state of each line
  python d1_grouped/label_corrections/apply_corrections.py --revert   # put the original lines back

Each row names a label file under data/raw/D1/content/Aircraft_dataset/, the 0-based line number of the box, the
action (move: the box gets new coordinates; delete: the line is blanked, which prepare.py skips), the original line
(`old`) and the corrected one (`new`, empty for delete). A line is only changed when it currently equals `old` (or
`new`, when reverting), so running it twice is safe and a line that differs is reported, not touched. Blanking instead
of removing keeps every line number stable, so --revert always works.
Only the YOLO .txt labels change (they are what prepare.py reads); D1's Aircraft_*.json files keep the old boxes.
After applying, re-cut the D1 crops (prepare.py) and re-run anything built on them.
"""
import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "data/raw/D1/content/Aircraft_dataset"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    c = pd.read_csv(HERE / "corrections.csv", keep_default_na=False)
    frm, to = ("new", "old") if a.revert else ("old", "new")
    counts = {"changed": 0, "already done": 0, "differs": 0, "missing": 0}
    for f, d in c.groupby("label_file"):
        p = ROOT / f
        if not p.exists():
            counts["missing"] += len(d)
            print(f"missing: {p}")
            continue
        lines = p.read_text().split("\n")
        dirty = False
        for _, r in d.iterrows():
            cur = lines[r.line].strip() if r.line < len(lines) else None
            if cur == r[to]:
                counts["already done"] += 1
            elif cur == r[frm]:
                if not a.check:
                    lines[r.line] = r[to]
                    dirty = True
                counts["changed"] += 1
            else:
                counts["differs"] += 1
                print(f"differs, not touched: {f} line {r.line}: {cur!r}")
        if dirty:
            p.write_text("\n".join(lines))
    print(f"{'revert' if a.revert else 'apply'}: {'would change' if a.check else 'changed'} {counts['changed']}, "
          f"already done {counts['already done']}, differs (skipped) {counts['differs']}, missing {counts['missing']}"
          f"  (of {len(c)})")


if __name__ == "__main__":
    main()
