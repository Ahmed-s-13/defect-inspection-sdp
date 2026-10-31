"""Export the saved Architecture B DINOv2 OOD comparison for Uncertainty.
Run: python 5_dashboard/scripts/ood_comparison_data.py
"""
import json
from pathlib import Path

DASH = Path(__file__).resolve().parents[1]
SOURCE = DASH.parent / '1_main_model_resnet18/enhanced_res/arch_a/results/archB/dinov2_ood/results.json'


def main():
    report = json.loads(SOURCE.read_text())
    names = [
        ('ResNet AoN (fit-only)', 'ResNet + AoN'),
        ('ResNet kNN', 'ResNet + kNN'),
        ('ResNet AoN + kNN (fit-only)', 'ResNet + AoN + kNN'),
        ('DINOv2 AoN', 'DINOv2 + AoN'),
        ('DINOv2 kNN', 'DINOv2 + kNN'),
        ('DINOv2 AoN + kNN', 'DINOv2 + AoN + kNN'),
    ]
    data = {'source': str(SOURCE.relative_to(DASH.parent)), 'alpha_ood': 0.08,
            'test_crops': report['counts']['test'], 'd5_crops': report['counts']['d5'],
            'offtarget_crops': report['variants'][names[0][0]]['held_out_far']['n'],
            'variants': [{'name': label, 'selected': name == 'ResNet AoN + kNN (fit-only)',
                          'far': report['variants'][name]['held_out_far']['rate'],
                          'd1_frr': report['variants'][name]['d1_frr']['rate'],
                          'd5_frr': report['variants'][name]['d5_frr']['rate']}
                         for name, label in names]}
    (DASH / 'web/src/data/ood_comparison.json').write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    main()
