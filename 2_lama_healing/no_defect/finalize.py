"""Apply the review decisions: crops.csv + review/decisions.json -> no_defect_healed.csv (the crops to train on).

  python finalize.py                   refuses while flagged crops are still undecided
  python finalize.py --pending drop    treat undecided flagged crops as rejected (or: keep)

A crop is used if a person kept it, or if nobody decided and make_crops.py said keep. Rejected and dropped crops
are left out. The kept crops are copied to 1_main_model_resnet18/data/work/crops/no_defect/healed/<split>/, where
d1_gc7 reads them, and listed in no_defect_healed.csv with paths relative to 1_main_model_resnet18 (like
data/work/crops/D1.csv), cls = "no-defect", the photo's d1_grouped group and split, and the source defect crop.
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[1] / "1_main_model_resnet18"
DEST = Path("data/work/crops/no_defect/healed")          # relative to MAIN


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pending", choices=["keep", "drop"], help="what to do with flagged crops nobody decided")
    a = ap.parse_args()

    df = pd.read_csv(HERE / "crops.csv")
    f = HERE / "review/decisions.json"
    dec = json.loads(f.read_text()) if f.exists() else {}
    df["decision"] = df.id.map(dec)
    pending = df[(df.auto == "review") & df.decision.isna()]
    if len(pending) and not a.pending:
        raise SystemExit(f"{len(pending)} flagged crops are still undecided: finish the review page "
                         "(python review_server.py) or pass --pending keep|drop")
    auto = df.auto.map({"keep": "keep", "drop": "reject", "review": a.pending and {"keep": "keep", "drop": "reject"}[a.pending]})
    df["final"] = df.decision.fillna(auto)

    out = df[df.final == "keep"]
    dst = [DEST / s / Path(c).name for s, c in zip(out.split, out.crop)]
    for c, d in zip(out.crop, dst):
        (MAIN / d).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / c, MAIN / d)
    out = pd.DataFrame(dict(crop_path=[str(d) for d in dst], cls="no-defect", image=out.key, key=out.key,
                            src_group=out.group, split=out.split, healed_from=out.cls, source_crop=out.source_crop))
    out.to_csv(HERE / "no_defect_healed.csv", index=False)
    print(f"{len(out)} of {len(df)} crops kept ({(df.decision == 'reject').sum()} rejected on review, "
          f"{(df.auto == 'drop').sum()} dropped by rules, {(df.decision == 'keep').sum()} kept on review)")
    print(pd.crosstab(out.healed_from, out.split, margins=True).to_string())


if __name__ == "__main__":
    main()
