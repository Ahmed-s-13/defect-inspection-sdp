"""Create contact sheets for largest groups and highest residual cross-split scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'data/d1_sscd_grouped';R=ROOT/'reports/final_split_review';R.mkdir(exist_ok=True)
m=pd.read_csv(O/'split_manifest.csv');x=np.load(ROOT/'reports/sscd_embeddings.npy');sp=m.split.to_numpy()
# Largest source groups: evenly spaced members by image index, 12 each.
for page in range(2):
 sheet=Image.new('RGB',(1200,6*170),'white');d=ImageDraw.Draw(sheet)
 for row,g in enumerate(m.group_id.value_counts().index[page*6:(page+1)*6]):
  members=m[m.group_id==g];idx=np.linspace(0,len(members)-1,6,dtype=int)
  d.text((4,row*170+2),f'Source group {g}: {len(members)} images',fill='black')
  for col,j in enumerate(idx):
   r=members.iloc[j];im=ImageOps.contain(Image.open(ROOT/r.path).convert('RGB'),(194,135));sheet.paste(im,(col*200,row*170+28))
 sheet.save(R/f'largest_groups_{page+1}.jpg')
pairs=[]
for start in range(0,len(x),128):
 s=x[start:start+128]@x.T;s[sp[start:start+128,None]==sp[None,:]]=-2
 js=s.argmax(1)
 for k,j in enumerate(js):
  i=start+k
  if i<j:pairs.append((float(s[k,j]),i,int(j)))
pairs.sort(reverse=True);chosen=[];seen=set()
for score,a,b in pairs:
 ga,gb=int(m.iloc[a].group_id),int(m.iloc[b].group_id);key=tuple(sorted((ga,gb)))
 if key in seen:continue
 seen.add(key);chosen.append((score,a,b))
 if len(chosen)==30:break
pd.DataFrame(chosen,columns=['cosine','a','b']).to_csv(R/'residual_pairs.csv',index=False)
for page in range(3):
 sheet=Image.new('RGB',(700,10*150),'white');d=ImageDraw.Draw(sheet)
 for row,(score,a,b) in enumerate(chosen[page*10:(page+1)*10]):
  d.text((4,row*150+2),f'{a} / {b} cos={score:.3f}',fill='black')
  for col,i in enumerate([a,b]):
   im=ImageOps.contain(Image.open(ROOT/m.iloc[i].path).convert('RGB'),(340,125));sheet.paste(im,(col*350,row*150+22))
 sheet.save(R/f'residual_{page+1}.jpg')
print('Created final split review sheets')
