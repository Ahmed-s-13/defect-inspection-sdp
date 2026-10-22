"""Freeze a class-balanced group split and materialize raw links + crop manifest."""
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from build_groups import UF
OUT=ROOT/'data/d1_sscd_grouped'
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--seed',type=int,default=20261021);ap.add_argument('--colocate-threshold',type=float,default=.45);a=ap.parse_args()
 if (OUT/'split_manifest.csv').exists():raise SystemExit('Frozen split already exists. Preserve it; use a new version directory for a new split.')
 groups=pd.read_csv(OUT/'groups.csv');meta=json.loads((ROOT/'reports/images.json').read_text());n=len(meta);uf=UF(n)
 for _,r in groups.groupby('group_id'):
  ids=r.image_id.tolist()
  for j in ids[1:]:uf.union(ids[0],j)
 # Conservative co-location blocks: high-similarity but unconfirmed pairs stay together.
 # This does not declare them copies. False positive links reduce data efficiency, not separation.
 x=np.load(ROOT/'reports/sscd_embeddings.npy');extra=0
 for start in range(0,n,128):
  sim=x[start:start+128]@x.T
  for local in range(len(sim)):
   i=start+local
   for j in np.flatnonzero(sim[local,i+1:]>=a.colocate_threshold)+i+1:
    if uf.find(i)!=uf.find(int(j)):uf.union(i,int(j));extra+=1
 manual = OUT / 'review_colocate.csv'
 for links in [manual, OUT / 'refined_links.csv']:
  if links.exists():
   for row in csv.DictReader(links.open()):uf.union(int(row['a']),int(row['b']))
 roots={};block=np.array([roots.setdefault(uf.find(i),len(roots)) for i in range(n)])
 groups['split_block']=block
 # Absolute crop paths make scripts safe to run from any working directory.
 crops=pd.read_csv(OUT/'work/crops/D1.csv');crops['crop_path']=[str((ROOT/p).resolve()) if not Path(p).is_absolute() else p for p in crops.crop_path]
 paths=[ROOT/r['path'] for r in meta];rawroot=ROOT/'data/D1';rel_to_id={str(p.relative_to(rawroot)):i for i,p in enumerate(paths)}
 image_ids=np.array([rel_to_id[s] for s in crops.image]);crops['image_id']=image_ids;crops['source_group']=groups.group_id.to_numpy()[image_ids];crops['group_id']=block[image_ids]
 classes=sorted(crops.cls.unique());class_id={c:i for i,c in enumerate(classes)};g=len(roots)
 counts=np.zeros((g,len(classes)+2),float)
 for gid,cls in zip(crops.group_id,crops.cls):counts[int(gid),class_id[cls]]+=1
 np.add.at(counts[:,-2],block,1);counts[:,-1]=1
 fractions=np.array([.7,.15,.15]);target=fractions[:,None]*counts.sum(0);scale=np.maximum(target,1)
 # Repeated randomized greedy allocation + moves; objective uses counts only, never model scores.
 weight=np.array([1.]*len(classes)+[.5,.1]);best=None
 def objective(cur):return float((((cur-target)/scale)**2*weight).sum())
 for seed in range(24):
  rng=np.random.default_rng(a.seed+seed);importance=(counts[:,:len(classes)]/np.maximum(counts[:,:len(classes)].sum(0),1)).max(1)
  order=np.argsort(-(importance+rng.random(g)*importance.max()*.04));assign=np.zeros(g,int);cur=np.zeros_like(target)
  for gi in order:
   opts=[]
   for sp in range(3):
    tmp=cur.copy();tmp[sp]+=counts[gi];opts.append(objective(tmp))
   sp=int(np.argmin(opts));assign[gi]=sp;cur[sp]+=counts[gi]
  for _ in range(8):
   changed=False
   for gi in rng.permutation(g):
    old=assign[gi];base=objective(cur);chosen=old
    for sp in range(3):
     if sp==old:continue
     tmp=cur.copy();tmp[old]-=counts[gi];tmp[sp]+=counts[gi];score=objective(tmp)
     if score<base-1e-10:base=score;chosen=sp
    if chosen!=old:cur[old]-=counts[gi];cur[chosen]+=counts[gi];assign[gi]=chosen;changed=True
   if not changed:break
  score=objective(cur)
  if best is None or score<best[0]:best=(score,assign.copy(),cur.copy())
 _,assign,cur=best
 # Pairwise swaps can improve class balance where one-group moves get stuck.
 rng=np.random.default_rng(a.seed+1000);current=objective(cur)
 for _ in range(100000):
  gi,gj=rng.integers(0,g,size=2);si,sj=assign[gi],assign[gj]
  if si==sj:continue
  tmp=cur.copy();delta=counts[gj]-counts[gi];tmp[si]+=delta;tmp[sj]-=delta;score=objective(tmp)
  if score<current-1e-12:assign[gi],assign[gj]=sj,si;cur=tmp;current=score
 # Mixed-integer allocation minimizes class/count imbalance globally, without model scores.
 from scipy.optimize import milp, Bounds, LinearConstraint
 from scipy.sparse import lil_matrix
 d=counts.shape[1];nv=g*3+3*d*2;A=lil_matrix((g+3*d,nv));rhs=np.zeros(g+3*d);cost=np.zeros(nv);upper=np.full(nv,np.inf);upper[:g*3]=1.;lower=np.zeros(nv)
 for gi in range(g):
  A[gi,gi*3:gi*3+3]=1;rhs[gi]=1
  # A single block should not dominate a small held-out split.
  if counts[gi,-2]>.5*target[1,-2]:lower[gi*3]=1;upper[gi*3+1:gi*3+3]=0
 for sp in range(3):
  for k in range(d):
   row=g+sp*d+k;A[row,np.arange(g)*3+sp]=counts[:,k];rhs[row]=target[sp,k]
   plus=g*3+(sp*d+k)*2;minus=plus+1;A[row,plus]=1;A[row,minus]=-1
   cost[plus]=cost[minus]=weight[k]/scale[sp,k]
 result=milp(cost,integrality=np.r_[np.ones(g*3),np.zeros(3*d*2)],bounds=Bounds(lower,upper),constraints=LinearConstraint(A.tocsr(),rhs,rhs),options={'time_limit':30.,'mip_rel_gap':.01})
 if result.x is not None:
  candidate=result.x[:g*3].reshape(g,3)
  if np.allclose(candidate,np.round(candidate),atol=1e-5):assign=candidate.argmax(axis=1)
 splits=np.array(['train','val','test']);groups['split']=splits[assign[block]];crops['split']=groups.split.to_numpy()[image_ids]
 # The classifier's group_id refers to conservative split blocks, source_group to verified components.
 assert crops.groupby('group_id').split.nunique().max()==1
 assert groups.groupby('group_id').split.nunique().max()==1
 assert len(crops)==len(set(crops.crop_path))
 assert (pd.crosstab(crops.cls,crops.split)>0).all().all()
 for s in splits:
  (OUT/'images'/s).mkdir(parents=True,exist_ok=True);(OUT/'labels'/s).mkdir(parents=True,exist_ok=True)
 groups['sha256']=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
 for r in groups.itertuples():
  src=ROOT/r.path;label=Path(str(src).replace('/images/','/labels/')).with_suffix('.txt')
  for source,folder in [(src,'images'),(label,'labels')]:
   dest=OUT/folder/r.split/source.name
   if not dest.exists():dest.symlink_to(os.path.relpath(source,dest.parent))
 groups.to_csv(OUT/'split_manifest.csv',index=False)
 crops.to_csv(OUT/'work/crops/D1.csv',index=False)
 crops[['crop_path','cls','image','group_id','split','image_id','source_group']].to_csv(OUT/'work/d1_split.csv',index=False)
 for s in splits:crops[crops.split==s].to_csv(OUT/f'{s}_crops.csv',index=False)
 names=['crack','dent','corrosion','scratch','missing-head','paint-peel-off']
 (OUT/'data.yaml').write_text('path: '+str(OUT)+'\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 6\nnames: '+json.dumps(names)+'\n')
 # Full score audit, not limited to retrieved top20.
 crossmax=-1.;crosscounts={str(t):0 for t in [a.colocate_threshold,.75,.8,.9,.95]};ss=groups.split.to_numpy()
 for start in range(0,n,128):
  sim=x[start:start+128]@x.T;mask=ss[start:start+128,None]!=ss[None,:];v=sim[mask]
  if len(v):crossmax=max(crossmax,float(v.max()))
  for t in crosscounts:crosscounts[t]+=int((v>=float(t)).sum())
 assert crosscounts[str(a.colocate_threshold)]==0
 evidence=pd.read_csv(OUT/'pair_evidence.csv');accepted=evidence[evidence.accepted]
 assert all(ss[int(a)]==ss[int(b)] for a,b in zip(accepted.a,accepted.b))
 for links in [manual, OUT / 'refined_links.csv']:
  if links.exists():
   for row in csv.DictReader(links.open()):assert ss[int(row['a'])]==ss[int(row['b'])]
 summary={'seed':a.seed,'colocate_threshold':a.colocate_threshold,'allocation_solver_message':str(result.message),'images':n,'crops':len(crops),'verified_components':int(groups.group_id.nunique()),'conservative_split_blocks':g,'extra_colocation_merges':extra,'largest_split_blocks':groups.split_block.value_counts().head(20).tolist(),'image_counts':groups.split.value_counts().to_dict(),'crop_counts':crops.split.value_counts().to_dict(),'block_counts':groups.groupby('split').split_block.nunique().to_dict(),'class_crop_counts':pd.crosstab(crops.cls,crops.split).to_dict(),'max_cross_split_single_view_sscd':crossmax,'cross_split_directed_pairs_above_threshold':crosscounts,'accepted_cross_split_edges':0,'split_sha256':hashlib.sha256((OUT/'work/d1_split.csv').read_bytes()).hexdigest(),'limitation':f'Zero detected accepted-edge leaks and no single-view cosine >= {a.colocate_threshold} cross-split pairs. Not proof that all transformed copies are found. Broad similarity blocks deliberately co-locate uncertain pairs and may contain different scenes. All D4 pHash <=10 candidates were considered; geometric SSCD candidates were top20 >=0.5.'}
 (OUT/'split_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
