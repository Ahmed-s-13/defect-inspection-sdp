"""Architecture B OOD ablation; no deployed assets changed.
Run from repository root with the d1_sscd_grouped Python environment.
Fixed deployed alpha=0.08, quality off. Evaluation never tunes thresholds.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import archB_common as BC
import archA_ood_bench as OLD
R = BC.R
OUT = HERE.parent / 'results/archB/dinov2_ood'
CACHE = BC.CACHE / 'dinov2_ood'


def rate(mask):
    return {'n': int(len(mask)), 'count': int(np.sum(mask)), 'rate': float(np.mean(mask))}


def main():
    torch.set_num_threads(4)
    B = R.ArchA(R.CALIBRATIONS['gc7'])
    assert not B.use_quality and B.alpha_ood == 0.08
    va, cal, test = BC.halves(B.cal['data']['split'])
    split = BC.load_split(B.cal['data']['split'])
    train = split[split.split == 'train']
    assert not set(train.src_group) & set(cal.src_group)
    assert not set(cal.src_group) & set(test.src_group)
    man = pd.read_csv(BC.OFF / 'manifest.csv')
    man = man[~man.source.isin(['places365-plain', 'dtd'])].reset_index(drop=True)
    man['path'] = [str(BC.MAIN / p) for p in man.crop_path]
    d5, _ = BC.A5.d5_crops()
    items = {'train': list(train.crop_path), 'cal': list(cal.crop_path), 'test': list(test.crop_path),
             'bench': list(man.path), 'd5': list(d5.crop_path)}
    # Reuse only exact path matches, including corrected crop paths; never match by basename.
    cached = {}
    for directory in (OLD.CACHE, CACHE):
        for file in sorted(directory.glob('dinov2_*.npz')):
            z = np.load(file, allow_pickle=True)
            cached.update(zip(map(str, z['items']), z['x']))
    missing = list(dict.fromkeys(p for ps in items.values() for p in ps if p not in cached))
    # work_fix also relocates unchanged crops. Reuse these only after byte equality.
    reused = 0
    for p in missing:
        old = p.replace('/data/work_fix/', '/data/work/')
        if old != p and old in cached and Path(old).read_bytes() == Path(p).read_bytes():
            cached[p] = cached[old]
            reused += 1
    missing = [p for p in missing if p not in cached]
    print(f'Verified byte-identical relocated crops: {reused}', flush=True)
    print(f'DINO features: {len(missing)} missing, {len(cached)} cached', flush=True)
    if missing:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print('Device:', device, flush=True)
        ext = OLD.DINO(device)
        for i in range(0, len(missing), 64):
            paths = missing[i:i+64]
            ims = []
            for p in paths:
                with Image.open(p) as im:
                    ims.append(im.convert('RGB'))
            cached.update(zip(paths, ext(ims)))
            if i % 640 == 0: print(f'Extracted {min(i+64,len(missing))}/{len(missing)}', flush=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE / 'dinov2_added.npz', items=np.array(missing), x=np.stack([cached[p] for p in missing]))
    DF = {s: np.stack([cached[p] for p in ps]) for s, ps in items.items()}
    z = np.load(BC.C.CACHE['gc7'], allow_pickle=True)
    assert list(z['crop_path']) == list(va.crop_path) + items['cal'] + items['test']
    nv, nc = len(va), len(cal)
    EF = {'cal': z['feats'][nv:nv+nc], 'test': z['feats'][nv+nc:]}
    ED = {'cal': z['knn_dist'][nv:nv+nc], 'test': z['knn_dist'][nv+nc:]}
    for s in ('bench', 'd5'):
        zz = np.load(BC.CACHE / f'gc7_{s}.npz', allow_pickle=True)
        assert list(zz['crop_path']) == items[s], s
        EF[s], ED[s] = zz['feats'], zz['knn_dist']
    bank = torch.load(BC.MODELS['gc7'] / 'enhanced_res_head.pt', map_location='cpu')['knn_bank'].float().numpy()
    fit = (man.role == 'fit').to_numpy()
    held = (man.role == 'test').to_numpy()
    gc, gt = cal.src_group.to_numpy(), test.src_group.to_numpy()
    def aon(pos, features):
        X = np.concatenate([pos, features['bench'][fit]])
        mu, sd = X.mean(0), X.std(0) + 1e-6
        clf = LogisticRegression(C=0.5, class_weight='balanced', max_iter=3000)
        clf.fit((X-mu)/sd, np.r_[np.zeros(len(pos)), np.ones(fit.sum())])
        assert clf.n_iter_.max() < 3000
        return {s: clf.predict_proba((f-mu)/sd)[:,1] for s,f in features.items() if s != 'train'}
    scores = {'ResNet AoN (fit-only)': aon(bank, EF), 'DINOv2 AoN': aon(DF['train'], DF)}
    db = DF['train'] / np.linalg.norm(DF['train'], axis=1, keepdims=True)
    # Chunk queries to bound pairwise distance memory.
    scores['DINOv2 kNN'] = {s: np.concatenate([OLD.knn_dist(db, f[i:i+256], 5) for i in range(0,len(f),256)])
                              for s,f in DF.items() if s != 'train'}
    scores['ResNet kNN'] = ED
    pv = {n: {s: R.pvalue(sc['cal'], gc, sc[s]) for s in ('test','bench','d5')} for n,sc in scores.items()}
    for name, a, b in [('ResNet AoN + kNN (fit-only)', 'ResNet AoN (fit-only)', 'ResNet kNN'),
                        ('DINOv2 AoN + kNN', 'DINOv2 AoN', 'DINOv2 kNN')]:
        pv[name] = {s: np.minimum(1, 2*np.minimum(pv[a][s], pv[b][s])) for s in pv[a]}
    pv['Deployed B (reference only)'] = {s: B.ood(ED[s], EF[s])[0] for s in ('test','bench','d5')}
    results = {}
    for name, values in pv.items():
        stop = {s: p <= B.alpha_ood for s,p in values.items()}
        m = {'d1_frr': rate(stop['test']), 'd1_family_frr': BC.fam_mean(stop['test'], gt),
             'd1_frr_family_bootstrap_ci95': list(BC.family_ci(stop['test'],gt,n=2000)),
             'd5_frr': rate(stop['d5']), 'held_out_far': rate(~stop['bench'][held]),
             'by_category': {}}
        for mode in ('centre', 'box'):
            mask = held & (man['mode'] == mode).to_numpy()
            m[f'held_out_far_{mode}'] = rate(~stop['bench'][mask])
        for category in sorted(man.category[held].unique()):
            mask = held & (man.category == category).to_numpy()
            m['by_category'][category] = rate(~stop['bench'][mask])
        m['held_out_far_category_macro'] = float(np.mean([v['rate'] for v in m['by_category'].values()]))
        for label, category in [('d6_skin', 'aircraft skin patch (no damage)'),('d6_whole','whole aircraft (no damage)')]:
            mask = (man.source == 'fgvc-aircraft').to_numpy() & (man.category == category).to_numpy()
            m[label + '_rejection'] = rate(stop['bench'][mask])
        results[name] = m
    assert results['Deployed B (reference only)']['d1_frr']['count'] == 79
    report = {'protocol': 'Architecture B, quality off, fixed OOD alpha 0.08; family-weighted p-values; Bonferroni for two-score gates. No test tuning. FAR means gate passage, not visor defect display. AoN trained on fit categories only; role=test excluded. Deployed reference saw more off-target categories and is not a held-out comparison. D6 rejection need not be a false rejection of usable skin.',
              'features': 'Frozen pretrained vit_small_patch14_dinov2.lvd142m, 224 pixels, 384 features, no classification head. k=5. AoN logistic regression C=0.5, balanced; all D1 training features used.',
              'counts': {s: len(ps) for s,ps in items.items()}, 'cal_families': int(cal.src_group.nunique()),
              'test_families': int(test.src_group.nunique()), 'resnet_aon_positive_features': len(bank),
              'fit_categories': sorted(man.category[fit].unique()), 'held_out_categories': sorted(man.category[held].unique()),
              'variants': results}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    lines = ['# DINOv2 OOD gates in Architecture B', '', report['protocol'], '', report['features'], '',
             f'Calibration: {nc} crops / {cal.src_group.nunique()} families. Test: {len(test)} crops / {test.src_group.nunique()} families. D5: {len(d5)} crops. Held-out off-target: {held.sum()} crops.', '',
             '| Gate | D1 FRR | D1 family FRR | D5 FRR | Held-out FAR | FAR centre / box | D6 skin rejection | D6 whole rejection |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    pct = lambda v: f'{100*v:.2f}%'
    for name,m in results.items():
        lines.append('| '+name+' | '+' | '.join([pct(m['d1_frr']['rate']),pct(m['d1_family_frr']),pct(m['d5_frr']['rate']),pct(m['held_out_far']['rate']),pct(m['held_out_far_centre']['rate'])+' / '+pct(m['held_out_far_box']['rate']),pct(m['d6_skin_rejection']['rate']),pct(m['d6_whole_rejection']['rate'])])+' |')
    lines += ['', 'FAR here is pooled over crops; JSON also records per-category rates and their equal-category average. Older Architecture A tables averaged categories, so their FARs are not directly comparable. D1 intervals in JSON resample evaluation families, conditional on the fitted calibration. This experiment changes only the OOD gate: it does not recalibrate LAC or measure downstream CP/visor outcomes. No deployed model or calibration was changed.', '', 'Run: `1_main_model_resnet18/experiments/d1_sscd_grouped/.venv/bin/python 1_main_model_resnet18/enhanced_res/arch_a/experiments/archB_dinov2_ood.py`']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
