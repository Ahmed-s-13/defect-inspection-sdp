"""Exact best cross-split scores, independent of the global top-k candidate cap."""
import json,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 m=json.loads((ROOT/'reports/images.json').read_text());x=np.load(ROOT/'reports/sscd_embeddings.npy');x=x/np.linalg.norm(x,axis=1,keepdims=True)
 splits=np.array([r['split'] for r in m]);records=[];summary={}
 for split in ['train','val','test']:
  q=np.flatnonzero(splits==split);db=np.flatnonzero(splits!=split);best=[]
  for start in range(0,len(q),128):
   sims=x[q[start:start+128]]@x[db].T;idx=sims.argmax(axis=1)
   for i,j,s in zip(q[start:start+128],db[idx],sims[np.arange(len(idx)),idx]):
    records.append({'image_id':int(i),'split':split,'other_id':int(j),'other_split':splits[j],'cosine':float(s)});best.append(float(s))
  summary[split]={'n':len(q),'threshold_counts':{str(t):int(np.sum(np.array(best)>=t)) for t in [.7,.75,.8,.85,.9,.95,.99]}}
 with (ROOT/'reports/best_cross_split.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
 (ROOT/'reports/cross_split_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
