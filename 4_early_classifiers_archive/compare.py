"""Put several evaluated runs side by side.

    python compare.py model_1_plain_cnn/results/aircraft model_2_supcon_resnet18/results/aircraft
"""
import json
import sys
from pathlib import Path

ROWS = [
    ("Test macro-F1", lambda r: f"{r['test']['macro_f1']:.3f} ({r['test']['macro_f1_ci95'][0]:.3f} to {r['test']['macro_f1_ci95'][1]:.3f})"),
    ("Test balanced accuracy", lambda r: f"{r['test']['balanced_accuracy']:.3f}"),
    ("Worst class recall", lambda r: "{1:.3f} ({0})".format(*min(((c, v["recall"]) for c, v in r["test"]["per_class"].items() if v["support"]), key=lambda t: t[1]))),
    ("Overfitting: train minus test macro-F1", lambda r: f"{r['overfit_gap_macro_f1']:+.3f}"),
    ("Shift: mean macro-F1 under corruption", lambda r: f"{r['corruption_mean_macro_f1']:.3f} ({r['corruption_retained']:.0%} retained)"),
    ("Shift: never-trained-on data macro-F1", lambda r: "; ".join(f"{v['macro_f1']:.3f} [{k}]" for k, v in r.get("shift_sets", {}).items()) or "not run"),
    ("Shortcut: context reliance (lower is better)", lambda r: f"{r['shortcut']['context_reliance']:.0%}"),
    ("Shortcut: accuracy with context hidden", lambda r: f"{r['shortcut']['context_hidden']:.3f}"),
    ("Shortcut: Grad-CAM focus on defect (above 1 is good)", lambda r: f"{r['shortcut'].get('gradcam_focus_ratio', float('nan')):.2f}x"),
    ("Calibration error (lower is better)", lambda r: f"{r['test']['ece']:.3f}"),
    ("PCA components for 95% of feature variance", lambda r: f"{r['pca']['components_for_variance']['95%']} of {r['pca']['feature_dim']}"),
]


def main():
    runs = [Path(p) for p in sys.argv[1:]]
    if not runs:
        raise SystemExit(__doc__)
    reports = [json.loads((p / "report.json").read_text()) for p in runs]
    head = ["", *[f"{p.name} ({r['method']})" for p, r in zip(runs, reports)]]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for name, fn in ROWS:
        lines.append("| " + " | ".join([name, *[fn(r) for r in reports]]) + " |")
    text = "\n".join(lines)
    print(text)
    Path("comparison.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
