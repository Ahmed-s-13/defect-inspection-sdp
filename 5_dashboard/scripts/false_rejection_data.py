"""Export the audited family-533 example. Run from any directory with Python 3.
Requires the tracked rejection audit and its local crop file; no model inference.
"""
import csv
import json
import shutil
from pathlib import Path

DASH = Path(__file__).resolve().parents[1]
ROOT = DASH.parent
AUDIT = ROOT / '1_main_model_resnet18/enhanced_res/arch_a/results/archB/knn_sweep/rejected_crops.csv'


def main():
    rows = [r for r in csv.DictReader(AUDIT.open()) if r['src_group'] == '533']
    assert len(rows) == 9 and all(r['cls']=='corrosion' and r['check']=='AoN' for r in rows)
    row = rows[0]
    dest = DASH / 'web/public/uncertainty/family-533-false-reject.jpg'
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT / row['crop_path'],dest)
    data = {'family':533,'class':'corrosion','rejected_crops':len(rows),'check':'AoN',
            'image':'/uncertainty/'+dest.name,'source_crop':row['crop_path'],
            'original_image':row['image'],'p_aon':float(row['p_aon']),'p_knn':float(row['p_knn']),
            'ood_alpha':0.05,'per_detector_cutoff':0.025,'audit':str(AUDIT.relative_to(ROOT))}
    (DASH/'web/src/data/false_rejection.json').write_text(json.dumps(data,indent=2)+'\n')
    print('Exported family 533 example and detector p-values.')

if __name__ == '__main__':
    main()
