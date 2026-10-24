"""SSCD candidate + D4 pHash retrieval; conservative geometric grouping with resumable evidence."""
import csv,json,time,os
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/d1_sscd_grouped';CACHE=ROOT/'reports/grouping_cache';cv2.setNumThreads(1)
class UF:
 def __init__(self,n):self.p=list(range(n));self.size=[1]*n
 def find(self,a):
  while self.p[a]!=a:self.p[a]=self.p[self.p[a]];a=self.p[a]
  return a
 def union(self,a,b):
  a,b=self.find(a),self.find(b)
  if a==b:return
  if self.size[a]<self.size[b]:a,b=b,a
  self.p[b]=a;self.size[a]+=self.size[b]
def gray(path):
 im=cv2.imread(str(path),0)
 if im is None:raise ValueError(path)
 h,w=im.shape;s=min(1,640/max(h,w));return cv2.resize(im,(round(w*s),round(h*s)))
def fingerprint(path):
 im=cv2.resize(gray(path),(32,32));out=[]
 for flip in [False,True]:
  base=np.fliplr(im) if flip else im
  for k in range(4):
   d=cv2.dct(np.float32(np.rot90(base,k)))[:8,:8].ravel();bits=d>np.median(d[1:]);bits[0]=False
   out.append(int.from_bytes(np.packbits(bits).tobytes(),'big'))
 return out
@lru_cache(maxsize=256)
def features(path,flip):
 p=Path(path);dest=CACHE/(p.stem+('_f' if flip else '')+'.npz')
 if dest.exists():
  with np.load(dest) as z:return z['pts'],z['desc'],z['shape'],z['gray']
 im=gray(p)
 if flip:im=cv2.flip(im,1)
 sift=cv2.SIFT_create(nfeatures=1200);kp,d=sift.detectAndCompute(im,None)
 pts=np.float32([k.pt for k in kp]).reshape(-1,2);d=np.zeros((0,128),np.uint8) if d is None else d.astype(np.uint8)
 shape=np.array(im.shape);small=cv2.resize(im,(128,128))
 tmp=dest.with_name(dest.name+f'.{os.getpid()}.{__import__("threading").get_ident()}.npz');np.savez(tmp,pts=pts,desc=d,shape=shape,gray=small);os.replace(tmp,dest)
 return pts,d,shape,small
def verify(job):
 i,j,score,hd,pa,pb=job;best={'a':i,'b':j,'cosine':score,'phash':hd,'accepted':False,'method':'none','inliers':0,'ratio':0.,'coverage':0.,'flip':False,'aligned_ncc':0.}
 # Pixel correlation under all quarter turns/reflections supports low-texture exact copies.
 ia=cv2.resize(gray(pa),(96,96)).astype(float);ib=cv2.resize(gray(pb),(96,96)).astype(float)
 def corr(a,b):
  a=a.ravel()-a.mean();b=b.ravel()-b.mean();return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
 ncc=max(corr(ia,np.rot90(np.fliplr(ib) if f else ib,k)) for f in [0,1] for k in range(4));best['aligned_ncc']=ncc
 if hd<=6 and ncc>=.97:
  best.update(accepted=True,method='D4_pixel_phash');return best
 ka,da,sa,_=features(pa,False)
 for flip in [False,True]:
  kb,db,sb,_=features(pb,flip)
  if min(len(da),len(db))<4:continue
  match=cv2.BFMatcher(cv2.NORM_L2);ab=match.knnMatch(da,db,k=2);ba=match.knnMatch(db,da,k=2)
  rev={(m.queryIdx,m.trainIdx) for m,n in ba if m.distance<.72*n.distance}
  good=[m for m,n in ab if m.distance<.72*n.distance and (m.trainIdx,m.queryIdx) in rev]
  if len(good)<8:continue
  a=np.float32([ka[m.queryIdx] for m in good]);b=np.float32([kb[m.trainIdx] for m in good]);H,mask=cv2.findHomography(a,b,cv2.RANSAC,3.,maxIters=1500)
  if H is None or mask is None:continue
  keep=mask.ravel().astype(bool);n=int(keep.sum());ratio=n/len(good)
  cov=min(cv2.contourArea(cv2.convexHull(a[keep]))/np.prod(sa),cv2.contourArea(cv2.convexHull(b[keep]))/np.prod(sb)) if n>=3 else 0.
  accepted=n>=20 and ratio>=.55 and cov>=.08
  if n>best['inliers']:best.update(inliers=n,ratio=ratio,coverage=float(cov),flip=flip)
  if accepted:best.update(accepted=True,method='mutual_SIFT_RANSAC',flip=flip);return best
 return best
def main():
 if (OUT/'FROZEN').exists():raise SystemExit('Dataset is frozen. Use a separately versioned output for regrouping.')
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 meta=json.loads((ROOT/'reports/images.json').read_text());paths=[str(ROOT/r['path']) for r in meta];n=len(paths);t=time.time()
 hp=CACHE/'phashes.npy'
 with ThreadPoolExecutor(6) as ex:
  if hp.exists():h=np.load(hp)
  else:h=np.array(list(ex.map(fingerprint,paths)),dtype=np.uint64);np.save(hp,h)
 print('D4 hashes ready',round(time.time()-t),flush=True)
 # All-pairs D4 pHash supplementation. All eight transforms of B versus identity A.
 candidates={};z=np.load(ROOT/'reports/neighbors.npz');idx=z['indices'];sc=z['scores']
 for i in range(n):
  for j,s in zip(idx[i],sc[i]):
   if s>=.50:
    a,b=sorted((i,int(j)));candidates[a,b]=max(candidates.get((a,b),-1),float(s))
 x=np.load(ROOT/'reports/sscd_embeddings.npy')
 for i in range(n):
  ds=np.bitwise_count(h[i,0]^h[i+1:]).min(axis=1)
  for off in np.flatnonzero(ds<=10):
   j=i+1+int(off)
   if (i,j) not in candidates:candidates[i,j]=float(x[i]@x[j])
 print('candidates',len(candidates),'seconds',round(time.time()-t),flush=True)
 pairs=sorted(candidates,key=lambda k:-candidates[k]);np.savez(OUT/'candidate_pairs.npz',pairs=np.array(pairs,dtype=np.int32),cosine=np.array([candidates[k] for k in pairs],dtype=np.float32))
 uf=UF(n);done=set();evidence=OUT/'pair_evidence.csv'
 if evidence.exists():
  for r in csv.DictReader(evidence.open()):
   a,b=int(r['a']),int(r['b']);done.add((a,b))
   if r['accepted']=='True':uf.union(a,b)
 fields=list(verify((0,0,1.,0,paths[0],paths[0])))
 with evidence.open('a') as file,ThreadPoolExecutor(6) as ex:
  writer=csv.DictWriter(file,fieldnames=fields)
  if file.tell()==0:writer.writeheader()
  tested=len(done);accepted=0
  for start in range(0,len(pairs),128):
   batch=[]
   for a,b in pairs[start:start+128]:
    if (a,b) in done or uf.find(a)==uf.find(b):continue
    hd=int(np.bitwise_count(h[a,0]^h[b]).min());batch.append((a,b,candidates[a,b],hd,paths[a],paths[b]))
   for r in ex.map(verify,batch):
    writer.writerow(r);tested+=1
    if r['accepted']:uf.union(r['a'],r['b']);accepted+=1
   file.flush()
   if start%2560==0:print('candidate',start,'tested',tested,'groups',len({uf.find(i) for i in range(n)}),'seconds',round(time.time()-t),flush=True)
 roots={};rows=[]
 for i,r in enumerate(meta):
  group=roots.setdefault(uf.find(i),len(roots));rows.append(dict(image_id=i,path=r['path'],original_split=r['split'],group_id=group))
 with (OUT/'groups.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 sizes=sorted([uf.size[r] for r in set(uf.find(i) for i in range(n))],reverse=True)
 report={'images':n,'groups':len(roots),'candidate_pairs':len(candidates),'largest_groups':sizes[:30],'seconds':round(time.time()-t),'status':'algorithmic groups; review and cross-split audit required'}
 (OUT/'grouping_summary.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
if __name__=='__main__':main()
