"""Compare crop- and family-weighted validation NLL temperature fits; deployment unchanged.
Run from repo root with the model Python environment activated.
Uses the existing aligned gc7 calibration-output cache, model split and calibration JSON.
"""
import copy
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, str(Path(__file__).resolve().parent))
import archB_common as B
R = B.R
OUT = Path(__file__).resolve().parent.parent / 'results/archB/family_temperature'


def weights(groups):
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv]


def metrics(p, y, g):
    correct = p.argmax(1) == y
    loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))
    confidence = p.max(1)
    result = {'accuracy': float(correct.mean()), 'mean_confidence': float(confidence.mean())}
    for name, w in [('crop', np.ones(len(y))), ('family', weights(g))]:
        w = w / w.sum()
        ece = 0.
        edges = np.linspace(0, 1, 16)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (confidence > lo) & (confidence <= hi)
            if m.any():
                ece += abs(np.sum(w[m]*(confidence[m]-correct[m])))
        result[name+'_nll'] = float(np.sum(w*loss))
        result[name+'_ece_15_bins'] = float(ece)
    return result


def main():
    a = R.ArchA(R.CALIBRATIONS['gc7'])
    v, c, t = B.halves(a.cal['data']['split'])
    z = np.load(B.C.CACHE['gc7'], allow_pickle=True)
    assert list(z['crop_path']) == list(v.crop_path)+list(c.crop_path)+list(t.crop_path)
    nv, nc = len(v), len(c)
    pv, pc, pt = z['probs'][:nv], z['probs'][nv:nv+nc], z['probs'][nv+nc:]
    fc, ft = z['feats'][nv:nv+nc], z['feats'][nv+nc:]
    dc, dt = z['knn_dist'][nv:nv+nc], z['knn_dist'][nv+nc:]
    cid = {cls:i for i,cls in enumerate(a.classes)}
    yv, yc, yt = (d.cls.map(cid).to_numpy() for d in [v,c,t])
    gv, gc, gt = (d.src_group.to_numpy() for d in [v,c,t])
    assert not set(gv)&set(gc) and not set(gv)&set(gt) and not set(gc)&set(gt)
    passedc = a.ood(dc,fc)[0]>a.alpha_ood
    passedt = a.ood(dt,ft)[0]>a.alpha_ood
    results, arrays = {}, {}
    for name,w in [('crop_weighted',np.ones(nv)),('family_weighted',weights(gv))]:
        w = w/w.sum()
        objective = lambda logt: float(np.sum(w * -np.log(np.clip(R.temper(pv,np.exp(logt))[np.arange(nv),yv],1e-12,None))))
        fit = minimize_scalar(objective, bounds=(np.log(.05),np.log(20)),method='bounded')
        assert fit.success
        arch = copy.deepcopy(a)
        arch.T = float(np.exp(fit.x))
        arch.q = R.family_threshold(R.temper(pc[passedc],arch.T),yc[passedc],gc[passedc],arch.alpha)
        deployed_name = 'family_weighted' if a.cal['temperature'].get('weighting') else 'crop_weighted'
        if name==deployed_name:
            assert np.isclose(arch.T,a.T,atol=1e-7,rtol=0)
            assert np.isclose(arch.q,a.q,atol=1e-8,rtol=0)
        p,sets = arch.sets(pt)
        hit,size = sets[np.arange(len(yt)),yt],sets.sum(1)
        dec = arch.decide(pt,knn_dist=dt,feats=ft)
        summary = R.summarise(dec,arch,pt,yt,gt)
        if name==deployed_name:
            assert abs(summary['cp_coverage']-a.cal['expected_test_half']['cp_coverage'])<1e-8
        result = {'T':arch.T,'q_hat':arch.q,'probability_cutoff':1-arch.q,'optimizer_evaluations':fit.nfev,
                  'validation':metrics(R.temper(pv,arch.T),yv,gv),'test_all':metrics(p,yt,gt),
                  'test_accepted':metrics(p[passedt],yt[passedt],gt[passedt]),'pipeline':summary}
        result['test_accepted'].update(coverage=float(hit[passedt].mean()),
            family_coverage=float(np.average(hit[passedt],weights=weights(gt[passedt]))),
            macro_coverage=float(np.mean([hit[passedt & (yt==i)].mean() for i in range(len(cid))])),
            mean_set_size=float(size[passedt].mean()),singleton_rate=float(np.mean(size[passedt]==1)),
            singleton_accuracy=float(hit[passedt & (size==1)].mean()))
        result['unmodified_empty_sets'] = int(np.sum(np.all(p < 1-arch.q,axis=1)))
        results[name] = result
        arrays[name] = {'hit':hit,'sets':sets,'actions':np.array([d.action for d in dec]),'p':p}
    base, alt = arrays['crop_weighted'],arrays['family_weighted']
    _, inv = np.unique(gt[passedt],return_inverse=True)
    counts = np.bincount(inv)
    diff = np.bincount(inv,weights=alt['hit'][passedt].astype(float)-base['hit'][passedt])
    rng=np.random.default_rng(0)
    idx=rng.integers(0,len(counts),(5000,len(counts)))
    boot=diff[idx].sum(1)/counts[idx].sum(1)
    paired={'accepted_sets_changed':int(np.any(base['sets']!=alt['sets'],axis=1)[passedt].sum()),
            'accepted_newly_covered':int(np.sum(passedt & ~base['hit'] & alt['hit'])),
            'accepted_newly_missed':int(np.sum(passedt & base['hit'] & ~alt['hit'])),
            'visor_actions_changed':int(np.sum(base['actions']!=alt['actions'])),
            'family_minus_crop_coverage_difference_ci95':np.quantile(boot,[.025,.975]).tolist()}
    report={'protocol':'Same hybrid probabilities, OOD alpha 0.05, CP alpha 0.10 and family-weighted LAC. Only the temperature fitting objective changes. Both temperatures fitted on all validation crops, using identical optimizer and bounds. Refit CP cutoff separately on the same accepted calibration crops. Test data used only for evaluation. No deployment files changed.',
            'validation_crops':nv,'validation_families':len(np.unique(gv)),
            'validation_family_size_min_median_max':np.quantile(np.unique(gv,return_counts=True)[1],[0,.5,1]).tolist(),
            'accepted_calibration_crops':int(passedc.sum()),'test_crops':len(t),'accepted_test_crops':int(passedt.sum()),
            'variants':results,'paired':paired}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Family-balanced validation for temperature scaling','',report['protocol'],'',
           f'Validation: {nv} crops / {report["validation_families"]} families. Accepted calibration: {passedc.sum()} crops. Test: {len(t)} crops ({passedt.sum()} accepted).','',
           '| Metric | Crop-weighted temperature | Family-weighted temperature |','|---|---:|---:|']
    def add(label,section,key,percent=False):
        vals=[r[section][key] if section else r[key] for r in results.values()]
        lines.append('| '+label+' | '+' | '.join(f'{x*100:.4f}%' if percent else f'{x:.6f}' for x in vals)+' |')
    add('Temperature',None,'T')
    for key in ['crop_nll','family_nll','crop_ece_15_bins','family_ece_15_bins']:
        add('All test: '+key,'test_all',key)
    for key in ['coverage','family_coverage','macro_coverage','singleton_rate','singleton_accuracy']:
        add('Accepted: '+key,'test_accepted',key,True)
    add('Accepted: mean set size','test_accepted','mean_set_size')
    for key in ['silent_miss','no_defect_false_alarm','human_attention']:
        add(key,'pipeline',key,True)
    lines += ['', 'Paired comparison:', '```json',json.dumps(paired,indent=2),'```','',
              'ECE uses 15 equal-width confidence bins. Family-weighted metrics assign each observed family equal total weight. The paired coverage interval resamples test families conditional on this fitted pipeline; it does not include training/calibration uncertainty. Test comparisons are exploratory, not proof of superiority. OOD FRR/FAR cannot change because neither OOD detector uses temperature.','',
              'Run from repository root: `python 1_main_model_resnet18/enhanced_res/arch_a/experiments/archB_family_temperature.py`.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    main()
