"""Audit deployed rejections and sweep OOD k without changing deployment.
Run from repo root with the model Python environment activated.
Requires existing gc7 calibration, model and aligned feature caches (same as archB_dinov2_ood).
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import archB_common as B
R = B.R
OUT = Path(__file__).resolve().parent.parent / 'results/archB/knn_sweep'
KS = (1, 3, 5, 7, 10, 15, 20, 30, 50)


def main():
    torch.set_num_threads(4)
    a = R.ArchA(R.CALIBRATIONS['gc7'])
    v, c, t = B.halves(a.cal['data']['split'])
    z = np.load(B.C.CACHE['gc7'], allow_pickle=True)
    assert list(z['crop_path']) == list(v.crop_path) + list(c.crop_path) + list(t.crop_path)
    nc, nv = len(c), len(v)
    f = {'cal': z['feats'][nv:nv+nc], 'test': z['feats'][nv+nc:]}
    man = pd.read_csv(B.OFF / 'manifest.csv')
    man = man[~man.source.isin(['places365-plain', 'dtd'])].reset_index(drop=True)
    d5, _ = B.A5.d5_crops()
    for key, paths in [('bench', [str(B.MAIN / p) for p in man.crop_path]), ('d5', list(d5.crop_path))]:
        zz = np.load(B.CACHE / f'gc7_{key}.npz', allow_pickle=True)
        assert list(zz['crop_path']) == paths
        f[key] = zz['feats']
    bank = torch.load(B.MODELS['gc7'] / 'enhanced_res_head.pt', map_location='cpu')['knn_bank'].float()
    normbank = torch.nn.functional.normalize(bank, dim=1)
    dist = {}
    with torch.no_grad():
        for key, feats in f.items():
            chunks = []
            for start in range(0, len(feats), 128):
                q = torch.nn.functional.normalize(torch.as_tensor(feats[start:start+128]).float(), dim=1)
                top = (q @ normbank.T).topk(max(KS), dim=1).values
                chunks.append((1 - top.cumsum(1) / torch.arange(1, max(KS)+1)).numpy())
            dist[key] = np.concatenate(chunks)
            print(f'Computed neighbours: {key}, {len(feats)} crops', flush=True)
    assert np.allclose(dist['cal'][:,4], z['knn_dist'][nv:nv+nc], atol=2e-6)
    assert np.allclose(dist['test'][:,4], z['knn_dist'][nv+nc:], atol=2e-6)
    # Controlled AoN: same protocol as the DINO comparison, excludes all role=test crops.
    fit = (man.role == 'fit').to_numpy()
    held = (man.role == 'test').to_numpy()
    X = np.concatenate([bank.numpy(), f['bench'][fit]])
    mu, sd = X.mean(0), X.std(0)+1e-6
    clf = LogisticRegression(C=.5, class_weight='balanced', max_iter=3000)
    clf.fit((X-mu)/sd, np.r_[np.zeros(len(bank)), np.ones(fit.sum())])
    assert clf.n_iter_.max() < 3000
    scores = {key: clf.predict_proba((val-mu)/sd)[:,1] for key,val in f.items()}
    g = c.src_group.to_numpy()
    families = np.unique(g)
    rng = np.random.default_rng(0)
    masks = [np.isin(g, rng.choice(families, len(families)//2, replace=False)) for _ in range(100)]
    rows = []
    for k in KS:
        fr = []
        for fitcal in masks:
            pk = R.pvalue(dist['cal'][fitcal,k-1], g[fitcal], dist['cal'][~fitcal,k-1])
            pa = R.pvalue(scores['cal'][fitcal], g[fitcal], scores['cal'][~fitcal])
            stop = 2*np.minimum(pk,pa) <= a.alpha_ood
            fr.append(B.fam_mean(stop, g[~fitcal]))
        rows.append({'k': k, 'cal_family_frr_mean': float(np.mean(fr)),
                     'cal_family_frr_split_p05_p95': np.quantile(fr,[.05,.95]).tolist()})
    # Lock choice before consulting evaluation metrics. Tie-break prefers current k, then smaller k.
    chosen = min(rows, key=lambda r: (r['cal_family_frr_mean'], r['k'] != 5, r['k']))['k']
    for row in rows:
        k = row['k']
        p = {key: np.minimum(1,2*np.minimum(R.pvalue(dist['cal'][:,k-1],g,val[:,k-1]),
             R.pvalue(scores['cal'],g,scores[key]))) for key,val in dist.items() if key != 'cal'}
        row.update(d1_frr=float(np.mean(p['test'] <= a.alpha_ood)),
                   d5_frr=float(np.mean(p['d5'] <= a.alpha_ood)),
                   held_out_far=float(np.mean(p['bench'][held] > a.alpha_ood)),
                   d1_rejected=int(np.sum(p['test'] <= a.alpha_ood)),
                   d5_rejected=int(np.sum(p['d5'] <= a.alpha_ood)),
                   nonaircraft_accepted=int(np.sum(p['bench'][held] > a.alpha_ood)),
                   gate_auprc=float(average_precision_score(np.r_[np.zeros(len(t)),np.ones(held.sum())],
                       np.r_[1-p['test'],1-p['bench'][held]])))
    # Also evaluate the deployed AoN, keeping these training-exposed FARs separate.
    deployed_pa = {key: R.pvalue(a.lr['cal_scores'], a.o_groups, a.aon_score(val))
                   for key, val in f.items() if key != 'cal'}
    for row in rows:
        k = row['k']
        p = {key: np.minimum(1, 2*np.minimum(pa, R.pvalue(dist['cal'][:,k-1], g, dist[key][:,k-1])))
             for key, pa in deployed_pa.items()}
        row['deployed_aon_reference'] = {
            'd1_frr': float(np.mean(p['test'] <= a.alpha_ood)),
            'd5_frr': float(np.mean(p['d5'] <= a.alpha_ood)),
            'training_exposed_far': float(np.mean(p['bench'][held] > a.alpha_ood))}
    # Exact deployed k=5 audit uses original saved scores, avoiding numeric rank/tie changes.
    pk = R.pvalue(a.o_cal,a.o_groups,z['knn_dist'][nv+nc:])
    pa = R.pvalue(a.lr['cal_scores'],a.o_groups,a.aon_score(f['test']))
    stop = np.minimum(pk,pa)*2 <= a.alpha_ood
    audit = t.loc[stop,['crop_path','image','cls','src_group']].copy()
    audit['crop_path'] = [str(Path(p).relative_to(B.MAIN.parent)) for p in audit.crop_path]
    audit['p_aon'], audit['p_knn'] = pa[stop], pk[stop]
    audit['check'] = np.where((pa[stop]<=a.alpha_ood/2)&(pk[stop]<=a.alpha_ood/2),'both',np.where(pa[stop]<=a.alpha_ood/2,'AoN','kNN'))
    assert len(audit) == round(a.cal['expected_test_half']['gates_false_reject']*len(t))
    OUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT/'rejected_crops.csv', index=False)
    report = {'scope':'OOD k only; classifier k=5 unchanged; deployment unchanged', 'ood_alpha':a.alpha_ood,
              'selection':'Lowest mean family FRR on 100 seeded 44/44 family halvings of calibration; no test selection. This diagnostic tunes on calibration, so deployment would require independent recalibration.',
              'protocol':'Controlled fit-only AoN, fixed ResNet features and bank. Each k has separately calibrated p-values. Test comparisons are exploratory; existing test data were examined in previous work. No downstream CP claims.',
              'selected_k':chosen,'cal_crops':len(c),'cal_families':len(families),'test_crops':len(t),
              'd5_crops':len(d5),'nonaircraft_test_crops':int(held.sum()),'rows':rows}
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    lines = ['# Architecture B OOD neighbour-count experiment','',report['scope'],'',report['selection'],'',report['protocol'],'',
             f'Calibration-selected candidate: k={chosen}. This is not proof of an ideal k. Gate alpha={a.alpha_ood}.', '',
             '| k | Calibration family FRR | D1 test FRR | D5 FRR | Held-out non-aircraft FAR | Gate AUPRC |',
             '|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+str(r['k'])+' | '+' | '.join(f'{100*r[key]:.3f}%' for key in ['cal_family_frr_mean','d1_frr','d5_frr','held_out_far','gate_auprc'])+' |')
    lines += ['', '## Deployed AoN reference (non-aircraft categories were training-exposed)', '',
              '| k | D1 FRR | D5 FRR | Training-exposed FAR |', '|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+str(r['k'])+' | '+' | '.join(f'{100*r["deployed_aon_reference"][key]:.3f}%'
                     for key in ['d1_frr','d5_frr','training_exposed_far'])+' |')
    lines += ['', 'The first table uses the controlled, fit-only AoN; the second uses the deployed AoN. The 5%–95% calibration split ranges in JSON describe split variation, not confidence intervals. Minimizing FRR alone does not optimize detection; compare FAR as well.', '',
              '## Exact deployed rejected crops','', 'The CSV records each rejected D1 test crop, original image identifier, class, family and both detector p-values. Paths are relative to the repository root. A detector fires at p <= 0.025. Reproduce with `python 1_main_model_resnet18/enhanced_res/arch_a/experiments/archB_knn_sweep.py`.','',
              '[Rejected crop manifest](rejected_crops.csv)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__ == '__main__':
    main()
