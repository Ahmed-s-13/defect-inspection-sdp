"""Reflection-aware geometric diagnostics for review pairs; does not assign groups."""
import csv,json
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
cv2.setNumThreads(4)
def main():
 meta=json.loads((ROOT/'reports/images.json').read_text());rows=list(csv.DictReader((ROOT/'reports/review_pairs.csv').open()));sift=cv2.SIFT_create(nfeatures=3000);cache={}
 def features(i,flip=False):
  key=(i,flip)
  if key not in cache:
   im=cv2.imread(str(ROOT/meta[i]['path']),0);h,w=im.shape;scale=min(1,800/max(h,w));im=cv2.resize(im,(round(w*scale),round(h*scale)))
   if flip:im=cv2.flip(im,1)
   k,d=sift.detectAndCompute(im,None);cache[key]=(k,d,im.shape)
  return cache[key]
 for r in rows:
  ka,da,sa=features(int(r['a']));best={'verified_inliers':0,'verified_matches':0,'verified_ratio':0.,'coverage_a':0.,'coverage_b':0.,'reflection':False,'median_error':None}
  for flip in [False,True]:
   kb,db,sb=features(int(r['b']),flip)
   if da is None or db is None or len(db)<2 or len(da)<2:continue
   matcher=cv2.BFMatcher();ab=matcher.knnMatch(da,db,k=2);ba=matcher.knnMatch(db,da,k=2)
   reverse={(m.queryIdx,m.trainIdx) for m,n in ba if m.distance<.75*n.distance}
   good=[m for m,n in ab if m.distance<.75*n.distance and (m.trainIdx,m.queryIdx) in reverse]
   if len(good)<4:continue
   pa=np.float32([ka[m.queryIdx].pt for m in good]);pb=np.float32([kb[m.trainIdx].pt for m in good]);H,mask=cv2.findHomography(pa,pb,cv2.RANSAC,4.)
   if H is None or mask is None:continue
   keep=mask.ravel().astype(bool);count=int(keep.sum())
   if count<=best['verified_inliers']:continue
   def coverage(p,shape):return float(cv2.contourArea(cv2.convexHull(p)))/(shape[0]*shape[1]) if len(p)>=3 else 0.
   projected=cv2.perspectiveTransform(pa[keep,None,:],H)[:,0,:]
   best=dict(verified_inliers=count,verified_matches=len(good),verified_ratio=count/len(good),coverage_a=coverage(pa[keep],sa),coverage_b=coverage(pb[keep],sb),reflection=flip,median_error=float(np.median(np.linalg.norm(projected-pb[keep],axis=1))))
  r.update(best)
 with (ROOT/'reports/review_geometry.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print('Reflection-aware geometric diagnostics saved for',len(rows),'pairs')
if __name__=='__main__':main()
