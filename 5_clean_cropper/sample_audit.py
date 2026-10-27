"""Pick the audit sample: N random candidates, at most 2 per group -> review/audit.csv + thumbnails.

  python sample_audit.py

Labelled by hand on the review page (clean / damage / background / other), it measures how well each score in
scores.csv finds background and hidden damage, and sets pick.py's cuts. Random, so the labelled shares estimate
the shares in the whole pool (per group, as the final pick is capped per group too).
"""
from pathlib import Path

import pandas as pd

import thumbs

HERE = Path(__file__).resolve().parent
N, PER_GROUP = 300, 2

if __name__ == "__main__":
    c = pd.read_csv(HERE / "candidates.csv").sample(frac=1, random_state=0)
    s = c.groupby("group").head(PER_GROUP).head(N)
    (HERE / "review").mkdir(exist_ok=True)
    s[["id"]].to_csv(HERE / "review/audit.csv", index=False)
    thumbs.make(s)
    print(f"audit sample: {len(s)} candidates from {s.group.nunique()} groups; splits {s.split.value_counts().to_dict()}")
