"""Adopt family-weighted temperature fitting without refitting the model or OOD gate.
Run from repo root using the model Python environment; needs aligned gc7 output cache.
"""
import copy
import json
import sys
from datetime import date
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import archB_common as B
R=B.R
OUT=Path(__file__).resolve().parent.parent/'results/archB/temperature_change'


def main():
    path=R.CALIBRATIONS['gc7']
    cfg=json.loads(path.read_text())
    a=R.ArchA(path)
    assert a.alpha_ood==.05 and a.alpha==.1 and 'miss_guard' not in cfg
    v,c,t=B.halves(cfg['data']['split'])
    z=np.load(B.C.CACHE['gc7'],allow_pickle=True)
    assert list(z['crop_path'])==list(v.crop_path)+list(c.crop_path)+list(t.crop_path)
    nv,nc=len(v),len(c)
    cid={cls:i for i,cls in enumerate(a.classes)}
    fit=B.C.fit_temperature(z['probs'][:nv],v.cls.map(cid).to_numpy(),v.src_group.to_numpy())
    assert fit.success
    temperature=float(np.exp(fit.x))
    if cfg['temperature'].get('weighting') and np.isclose(a.T,temperature,rtol=0,atol=1e-10):
        print('Family-weighted temperature already deployed; no changes.')
        return
    before=copy.deepcopy(cfg)
    a.T=temperature
    sc,st=slice(nv,nv+nc),slice(nv+nc,None)
    passed=a.ood(z['knn_dist'][sc],z['feats'][sc])[0]>a.alpha_ood
    cfg['temperature']={'kind':'probability temperature: p ** (1/T), renormalised; fitted on val by family-weighted NLL',
        'weighting':'each validation family has total weight 1; each crop has weight 1/family_size',
        'validation_families':int(v.src_group.nunique()),'T':temperature}
    cfg['cp']['q']={k:R.family_threshold(R.temper(z['probs'][sc][passed],temperature),c.cls.map(cid).to_numpy()[passed],c.src_group.to_numpy()[passed],float(k)) for k in cfg['cp']['q']}
    a.q=cfg['cp']['q'][f'{a.alpha:.2f}']
    dec=a.decide(z['probs'][st],knn_dist=z['knn_dist'][st],feats=z['feats'][st])
    cfg['expected_test_half']=R.summarise(dec,a,z['probs'][st],t.cls.map(cid).to_numpy(),t.src_group.to_numpy())
    cfg['created']=str(date.today())
    assert cfg['ood_gate']==before['ood_gate'] and cfg['model']==before['model']
    OUT.mkdir(parents=True,exist_ok=True)
    snapshot=OUT/'calibration_crop_weighted.json'
    if not snapshot.exists():snapshot.write_text(json.dumps(before)+'\n')
    path.write_text(json.dumps(cfg)+'\n')
    report={'before_temperature':before['temperature'],'after_temperature':cfg['temperature'],
        'before':before['expected_test_half'],'after':cfg['expected_test_half'],
        'before_q':before['cp']['q'],'after_q':cfg['cp']['q']}
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'README.md').write_text('# Family-weighted temperature deployment\n\nCurrent B fits temperature with equal total weight per validation family. The model and OOD gate are unchanged; CP cutoffs were refitted on the same 2,065 accepted calibration crops.\n\nThe JSON records the before/after values; the old calibration is a historical reproducibility snapshot. See [the paired comparison](../family_temperature/README.md). Run `archB_update_temperature.py` from the model environment to reproduce the update from cached probabilities.\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
