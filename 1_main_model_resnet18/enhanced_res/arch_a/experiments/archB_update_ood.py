"""Change B's OOD cutoff, refit LAC on accepted calibration crops, and refresh expected results.
Keeps the fitted model, AoN, temperature and CP alpha unchanged. Uses aligned cached outputs.
"""
import argparse
import copy
import json
import sys
from datetime import date
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import archB_common as BC
R = BC.R
OUT = HERE.parent / 'results/archB/ood_alpha_change'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--alpha', type=float, required=True)
    args = ap.parse_args()
    assert 0 < args.alpha < 1
    path = R.CALIBRATIONS['gc7']
    cfg = json.loads(path.read_text())
    if cfg['ood_gate']['alpha'] == args.alpha:
        print(f'OOD alpha is already {args.alpha}; keeping the current calibration and original change report.')
        return
    before = copy.deepcopy(cfg)
    assert cfg['cp']['alpha'] == 0.10 and not cfg['quality_gate']['enabled']
    assert 'miss_guard' not in cfg, 'Handle guard recalibration explicitly before changing this configuration.'
    va, cal, test = BC.halves(cfg['data']['split'])
    z = np.load(BC.C.CACHE['gc7'], allow_pickle=True)
    assert list(z['crop_path']) == list(va.crop_path) + list(cal.crop_path) + list(test.crop_path)
    a = R.ArchA(path)
    nv, nc = len(va), len(cal)
    sc, st = slice(nv, nv+nc), slice(nv+nc, None)
    cid = {c:i for i,c in enumerate(a.classes)}
    ycal, ytest = cal.cls.map(cid).to_numpy(), test.cls.map(cid).to_numpy()
    pc = R.temper(z['probs'][sc], a.T)
    passed = a.ood(z['knn_dist'][sc], z['feats'][sc])[0] > args.alpha
    cfg['ood_gate']['alpha'] = args.alpha
    cfg['cp']['q'] = {key: R.family_threshold(pc[passed], ycal[passed], cal.src_group.to_numpy()[passed], float(key)) for key in cfg['cp']['q']}
    cfg['created'] = str(date.today())
    cfg['cp']['accepted_calibration_crops'] = int(passed.sum())
    a.alpha_ood = args.alpha
    a.q = cfg['cp']['q'][f'{a.alpha:.2f}']
    dec = a.decide(z['probs'][st], knn_dist=z['knn_dist'][st], feats=z['feats'][st])
    cfg['expected_test_half'] = R.summarise(dec, a, z['probs'][st], ytest, test.src_group.to_numpy())
    assert cfg['ood_gate']['aon'] == before['ood_gate']['aon']
    assert cfg['temperature'] == before['temperature']
    assert cfg['model'] == before['model']
    OUT.mkdir(parents=True, exist_ok=True)
    # Keep a full reproducible pre-change snapshot; historical studies keep their original settings.
    old = OUT / f"calibration_ood_{before['ood_gate']['alpha']:.2f}.json"
    if not old.exists(): old.write_text(json.dumps(before)+'\n')
    path.write_text(json.dumps(cfg)+'\n')
    check = R.ArchA(path)
    assert check.alpha_ood == args.alpha and check.alpha == 0.1
    report = {'before_alpha': before['ood_gate']['alpha'], 'after_alpha': args.alpha,
              'cp_alpha': check.alpha, 'accepted_calibration_crops': int(passed.sum()),
              'before': before['expected_test_half'], 'after': cfg['expected_test_half'],
              'before_q': before['cp']['q'], 'after_q': cfg['cp']['q']}
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
