"""Report block-weighted test metrics and paired block-bootstrap differences."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/d1_sscd_grouped/results';WORK=ROOT/'data/d1_sscd_grouped/work'
def metrics(cm):
 tp=np.diag(cm);true=cm.sum(1);pred=cm.sum(0)
 return {'accuracy':float(tp.sum()/cm.sum()),'macro_f1':float(np.mean(2*tp/np.maximum(true+pred,1e-12))),'balanced_accuracy':float(np.mean(tp/np.maximum(true,1e-12)))}
def main():
 z=np.load(OUT/'embeddings.npz');keep=z['split']=='test';ck=torch.load(WORK/'model_best.pt',map_location='cpu');classes=ck['classes'];ys=np.array([classes.index(c) for c in z['cls'][keep]])
 assert ck['split_sha256']==hashlib.sha256((WORK/'d1_split.csv').read_bytes()).hexdigest()
 fc=(z['emb'][keep]@ck['state']['fc.weight'].numpy().T+ck['state']['fc.bias'].numpy()).argmax(1)
 csv=pd.read_csv(OUT/'mlp/test_predictions.csv').set_index('crop_path');paths=z['crop_path'][keep];pred=np.array([classes.index(c) for c in csv.loc[paths,'pred']]);assert list(csv.loc[paths,'true'])==list(z['cls'][keep])
 groups=z['group'][keep];unique,gi=np.unique(groups,return_inverse=True);n=len(unique);cm={}
 for name,ps in [('resnet18',fc),('mlp_knn',pred)]:
  raw=np.zeros((n,6,6));np.add.at(raw,(gi,ys,ps),1);cm[name]=raw/raw.sum((1,2))[:,None,None]
 report={'test_crops':len(ys),'test_blocks':n,'definition':'Each conservative split block has equal total weight; its crops share that weight. Metrics are calculated from the resulting weighted confusion matrix.','models':{name:metrics(v.sum(0)) for name,v in cm.items()},'bootstrap_seed':42,'bootstrap_samples':2000,'limitations':'Intervals assume split blocks are independent; grouping errors can violate this. Broad blocks need not be individual scenes. One trained backbone seed; intervals do not represent training-seed uncertainty.'}
 rng=np.random.default_rng(42);samples={name:[] for name in cm};deltas=[]
 for _ in range(2000):
  ix=rng.integers(0,n,n);m={name:metrics(v[ix].sum(0)) for name,v in cm.items()}
  for name in cm:samples[name].append(list(m[name].values()))
  deltas.append(np.array(list(m['mlp_knn'].values()))-np.array(list(m['resnet18'].values())))
 keys=['accuracy','macro_f1','balanced_accuracy']
 for name in cm:report['models'][name]['bootstrap_95_percentile_ci']={key:np.quantile(np.array(samples[name])[:,k],[.025,.975]).tolist() for k,key in enumerate(keys)}
 report['paired_difference_mlp_minus_resnet']={key:{'estimate':report['models']['mlp_knn'][key]-report['models']['resnet18'][key],'bootstrap_95_percentile_ci':np.quantile(np.array(deltas)[:,k],[.025,.975]).tolist()} for k,key in enumerate(keys)}
 (OUT/'block_metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
