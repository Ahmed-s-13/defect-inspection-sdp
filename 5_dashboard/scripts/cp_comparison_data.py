"""Export the saved equal-crop CP experiment for the Uncertainty page.

Run from 5_dashboard/: python scripts/cp_comparison_data.py
Run archB_equal_crop_cp.py first if the model or calibration has changed.
"""
import json
from pathlib import Path

DASH = Path(__file__).resolve().parents[1]
SOURCE = DASH.parent / "1_main_model_resnet18/enhanced_res/arch_a/results/archB/equal_crop_cp/results.json"


def main():
    report = json.loads(SOURCE.read_text())
    variants = report["variants"]
    data = {
        "source": str(SOURCE.relative_to(DASH.parent)),
        "alpha": variants["family_weighted"]["alpha"],
        "calibration_crops": report["accepted_calibration_crops"],
        "test_crops": variants["family_weighted"]["accepted"]["n"],
        "test_families": report["test_families"],
        "variants": {name: {"probability_cutoff": row["probability_cutoff"],
                            **{key: row["accepted"][key] for key in (
                                "coverage", "macro_coverage", "family_coverage", "mean_set_size", "singleton_rate")}}
                     for name, row in variants.items()},
    }
    (DASH / "web/src/data/cp_comparison.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
