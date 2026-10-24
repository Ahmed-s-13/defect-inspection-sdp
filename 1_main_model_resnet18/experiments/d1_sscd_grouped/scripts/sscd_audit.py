"""Reproducible SSCD pilot: embeddings, nearest neighbours, geometric checks, review sheets."""
import argparse,csv,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps,ImageDraw
from torchvision import transforms as T
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parents[1]
class Images(Dataset):
 def __init__(self,paths):
  self.paths=paths
  self.tf=T.Compose([T.Resize((320,320)),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 def __len__(self):return len(self.paths)
 def __getitem__(self,i):
  with Image.open(self.paths[i]) as im: return self.tf(ImageOps.exif_transpose(im).convert('RGB'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--batch',type=int,default=32);a=ap.parse_args()
 torch.set_num_threads(4)
 paths=sorted((ROOT/'data/D1').rglob('*.jpg'));out=ROOT/'reports';out.mkdir(exist_ok=True)
 manifest=[{'id':i,'path':str(p.relative_to(ROOT)),'split':p.parent.name} for i,p in enumerate(paths)]
 (out/'images.json').write_text(json.dumps(manifest,indent=2))
 device='mps' if torch.backends.mps.is_available() else 'cpu'
 print('device',device,'images',len(paths),flush=True)
 cache=out/'sscd_embeddings.npy'
 if not cache.exists():
  model=torch.jit.load(str(ROOT/'models/sscd_disc_mixup.torchscript.pt'),map_location='cpu').eval().to(device)
  loader=DataLoader(Images(paths),batch_size=a.batch,num_workers=0)
  emb=[];start=time.time()
  with torch.inference_mode():
   for step,x in enumerate(loader):
    y=model(x.to(device)).cpu().numpy();emb.append(y)
    if step%20==0:print('embedded',min((step+1)*a.batch,len(paths)),'seconds',round(time.time()-start),flush=True)
  np.save(cache,np.concatenate(emb))
 x=np.load(cache);x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
 bank=torch.from_numpy(x).to(device);ids=[];scores=[]
 with torch.inference_mode():
  for start in range(0,len(x),256):
   s=bank[start:start+256]@bank.T
   s[torch.arange(len(s),device=device),torch.arange(start,start+len(s),device=device)]=-2
   v,i=s.topk(20,dim=1);ids.append(i.cpu().numpy());scores.append(v.cpu().numpy())
 ids=np.concatenate(ids);scores=np.concatenate(scores)
 np.savez(out/'neighbors.npz',indices=ids,scores=scores)
 rows=[]
 for i in range(len(x)):
  for rank,(j,s) in enumerate(zip(ids[i],scores[i])):
   rows.append(dict(a=i,b=int(j),rank=rank+1,cosine=float(s),cross_split=manifest[i]['split']!=manifest[j]['split']))
 with (out/'neighbors.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 summary={'images':len(x),'device':device,'views':1,'preprocessing':'RGB, direct resize 320x320, ImageNet normalization','thresholds_are_exploratory':True}
 for threshold in [.7,.75,.8,.85,.9,.95]:
  hits=[r for r in rows if r['cross_split'] and r['cosine']>=threshold]
  summary[str(threshold)]={'images_with_cross_split_candidate':len(set(r['a'] for r in hits)),'unique_cross_split_pairs':len(set(tuple(sorted([r['a'],r['b']])) for r in hits))}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
 # Deterministically sample across score bands; never infer confirmed copies from cosine alone.
 rng=np.random.default_rng(42);chosen=[]
 unique=[r for r in rows if r['a']<r['b'] and r['cross_split']]
 for lo,hi in [(.95,1.01),(.9,.95),(.8,.9),(.7,.8),(.5,.7)]:
  pool=[r for r in unique if lo<=r['cosine']<hi]
  if pool:chosen.extend(pool[int(k)] for k in rng.choice(len(pool),min(12,len(pool)),replace=False))
 import cv2
 cv2.setNumThreads(4);sift=cv2.SIFT_create(nfeatures=2500)
 def features(p):
  im=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE);h,w=im.shape
  im=cv2.resize(im,(round(w*min(1,800/max(h,w))),round(h*min(1,800/max(h,w)))))
  return sift.detectAndCompute(im,None)
 for r in chosen:
  ka,da=features(paths[r['a']]);kb,db=features(paths[r['b']]);good=[]
  if da is not None and db is not None and len(db)>=2:
   good=[m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2) if m.distance<.75*n.distance]
  r['matches']=len(good);r['inliers']=0;r['inlier_ratio']=0.
  if len(good)>=4:
   pa=np.float32([ka[m.queryIdx].pt for m in good]);pb=np.float32([kb[m.trainIdx].pt for m in good])
   H,mask=cv2.findHomography(pa,pb,cv2.RANSAC,4.)
   if mask is not None:r['inliers']=int(mask.sum());r['inlier_ratio']=float(mask.mean())
 with (out/'review_pairs.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(chosen[0])+['human_label']);w.writeheader();w.writerows(chosen)
 for page in range((len(chosen)+9)//10):
  sheet=Image.new('RGB',(1000,10*190),'white');draw=ImageDraw.Draw(sheet)
  for row,r in enumerate(chosen[page*10:(page+1)*10]):
   for col,key in enumerate(['a','b']):
    with Image.open(paths[r[key]]) as im:
     im=ImageOps.contain(im.convert('RGB'),(460,155));sheet.paste(im,(col*500,row*190+30))
   draw.text((5,row*190+4),f"{r['a']} / {r['b']} cos={r['cosine']:.3f} SIFT inliers={r['inliers']}/{r['matches']}",fill='black')
  sheet.save(out/f'review_{page+1:02d}.jpg')
 print('DONE: embeddings, top20 candidates and stratified review pairs saved',flush=True)
if __name__=='__main__':main()
