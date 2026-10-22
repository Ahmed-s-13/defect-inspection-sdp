"""Check low-score candidates with geometric alignment + pixel agreement; save co-location links."""
import csv,json,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from build_groups import UF,features,gray
O=ROOT/'data/d1_sscd_grouped';cv2.setNumThreads(1)
def check(job):
 a,b,pa,pb=job;ka,da,sa,_=features(pa,False);ia=gray(pa)
 for flip in [False,True]:
  kb,db,sb,_=features(pb,flip)
  if min(len(da),len(db))<4:continue
  matcher=cv2.BFMatcher(cv2.NORM_L2);ab=matcher.knnMatch(da,db,k=2);ba=matcher.knnMatch(db,da,k=2)
  rev={(m.queryIdx,m.trainIdx) for m,n in ba if m.distance<.75*n.distance}
  good=[m for m,n in ab if m.distance<.75*n.distance and (m.trainIdx,m.queryIdx) in rev]
  if len(good)<6:continue
  p=np.float32([ka[m.queryIdx] for m in good]);q=np.float32([kb[m.trainIdx] for m in good]);H,mask=cv2.findHomography(p,q,cv2.RANSAC,3.)
  if H is None or mask is None:continue
  keep=mask.ravel().astype(bool);ni=int(keep.sum());ratio=ni/len(good)
  if ni<6 or ratio<.6:continue
  cover=min(cv2.contourArea(cv2.convexHull(p[keep]))/np.prod(sa),cv2.contourArea(cv2.convexHull(q[keep]))/np.prod(sb))
  if cover<.02:continue
  ib=gray(pb)
  if flip:ib=cv2.flip(ib,1)
  warped=cv2.warpPerspective(ia,H,(ib.shape[1],ib.shape[0]));valid=cv2.warpPerspective(np.ones_like(ia)*255,H,(ib.shape[1],ib.shape[0]))>250
  valid=cv2.erode(valid.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
  if valid.mean()<.2:continue
  v=warped[valid].astype(float);w=ib[valid].astype(float);v-=v.mean();w-=w.mean();corr=float(v@w/(np.linalg.norm(v)*np.linalg.norm(w)+1e-9))
  if corr>=.9:return {'a':a,'b':b,'inliers':ni,'ratio':ratio,'coverage':float(cover),'pixel_correlation':corr,'reason':'SIFT homography plus aligned pixel correlation >=0.9; conservative co-location'}
 return None
def main():
 if (O/'FROZEN').exists():raise SystemExit('Dataset is frozen. Use a separately versioned output for new refinement.')
 m=json.loads((ROOT/'reports/images.json').read_text());x=np.load(ROOT/'reports/sscd_embeddings.npy');g=pd.read_csv(O/'split_manifest.csv');uf=UF(len(m));paths=[str(ROOT/r['path']) for r in m]
 for _,r in g.groupby('split_block'):
  ids=r.image_id.tolist()
  for i in ids[1:]:uf.union(ids[0],i)
 pairs={}
 # Broader top-50 retrieval without the pilot's 0.5 cutoff.
 for start in range(0,len(x),128):
  sim=x[start:start+128]@x.T
  for k in range(len(sim)):
   i=start+k;js=np.argpartition(sim[k],-51)[-51:];js=js[np.argsort(-sim[k,js])];kept=0
   for j in js:
    j=int(j)
    if i==j or sim[k,j]<.35 or uf.find(i)==uf.find(j):continue
    pair=tuple(sorted((i,j)));pairs[pair]=float(sim[k,j]);kept+=1
    if kept>=5:break
 ordered=sorted(pairs,key=lambda k:-pairs[k]);print('Candidates',len(ordered),flush=True);dest=O/'refined_links.csv';t=time.time()
 with dest.open('w') as f,ThreadPoolExecutor(6) as ex:
  fields=['a','b','inliers','ratio','coverage','pixel_correlation','reason'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();accepted=0
  for start in range(0,len(ordered),128):
   jobs=[(a,b,paths[a],paths[b]) for a,b in ordered[start:start+128] if uf.find(a)!=uf.find(b)]
   for r in ex.map(check,jobs):
    if r:writer.writerow(r);uf.union(r['a'],r['b']);accepted+=1
   f.flush()
   if start%2560==0:print(start,'links',accepted,'seconds',round(time.time()-t),flush=True)
 print('DONE',accepted,'additional links',flush=True)
if __name__=='__main__':main()
