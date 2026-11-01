"""Export saved calibration-stage experiments for the Uncertainty page.
Run: python 5_dashboard/scripts/uncertainty_experiments_data.py
"""
import json
from pathlib import Path

DASH = Path(__file__).resolve().parents[1]
ROOT = DASH.parent
RESULTS = ROOT / '1_main_model_resnet18/enhanced_res/arch_a/results'


def table(path, first_header):
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('| ' + first_header + ' |'))
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith('|'):
            break
        rows.append([cell.strip() for cell in line.strip('|').split('|')])
    return rows


def main():
    calibrators = RESULTS / 'archB/historical_local_x/calibrators.md'
    guard = RESULTS / 'archB/choices.md'
    cal20 = RESULTS / 'archB/historical_local_x/cal20.md'
    gates = RESULTS / 'archB/historical_local_x/gates.md'
    metrics = table(calibrators, 'calibrator')
    cp = {row[0]: row for row in table(calibrators, 'calibrator | family coverage')}
    data = {
        'sources': [str(p.relative_to(ROOT)) for p in (calibrators, guard, cal20, gates)],
        'gate_combinations': [{'name': r[0], 'defect_frr': r[1], 'clean_frr': r[2], 'offtarget_shown': r[3]}
                              for r in table(gates, 'variant')[:3]],
        'ood_alpha': [{'alpha': r[0], 'defect_frr': r[1], 'clean_frr': r[2], 'seen_far': r[3],
                       'wallpaper_far': r[4], 'static_far': r[5]} for r in table(guard, 'OOD α')],
        'calibrators': [{'name': row[0], 'nll': row[1].split()[0], 'ece': row[3].split()[0],
                         'ece_delta_ci': row[3][row[3].find('['):] if '[' in row[3] else 'Reference',
                         'family_coverage': cp[row[0]][1], 'set_size': cp[row[0]][2]}
                        for row in metrics],
        'guard': [{'name': r[0], 'silent_miss': r[1], 'clean_silence': r[2]}
                  for r in table(guard, 'miss guard')],
        'calibration_split': [{'name': label, 'families': r[2], 'coverage': r[5], 'set_size': r[6],
                               'silent_miss': r[7], 'false_alarm': r[8], 'false_reject': r[9]}
                              for label, r in zip(['All train; 44 cal-half families', '80% train; held-out 20% for calibration',
                                                   '80% train; held-out 20% + 44 cal-half families'], table(cal20, 'setup'), strict=True)],
    }
    (DASH / 'web/src/data/uncertainty_experiments.json').write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    main()
