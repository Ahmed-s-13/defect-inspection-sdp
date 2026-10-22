"""Extract fresh ResNet-18 embeddings for the frozen grouped D1 split."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from klshift.common import CropDataset,embed_batch,pick_device
from evaluate import load_model
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--work',type=Path,default=ROOT/'data/d1_sscd_grouped/work');ap.add_argument('--results',type=Path,default=ROOT/'data/d1_sscd_grouped/results');ap.add_argument('--batch',type=int,default=64);ap.add_argument('--workers',type=int,default=0);a=ap.parse_args()
 digest=hashlib.sha256((a.work/'d1_split.csv').read_bytes()).hexdigest();ck=torch.load(a.work/'model_best.pt',map_location='cpu')
 if ck.get('split_sha256')!=digest:raise SystemExit('Checkpoint is not trained on this exact grouped split. Run main_model/train.py first.')
 device=pick_device();model,classes=load_model(a.work,device);df=pd.read_csv(a.work/'d1_split.csv');y=df.cls.map({c:i for i,c in enumerate(classes)})
 dl=DataLoader(CropDataset(df.crop_path,y),batch_size=a.batch,num_workers=a.workers);features=[]
 with torch.inference_mode():
  for i,(x,_) in enumerate(dl):
   features.append(embed_batch(model,x.to(device)).cpu().numpy())
   if i%50==0:print('embedded',min((i+1)*a.batch,len(df)),'/',len(df),flush=True)
 out={k:df[k].astype(str).to_numpy() for k in ['split','cls','image','crop_path']};out['dataset']=np.full(len(df),'D1');out['group']=df.group_id.astype(str).to_numpy();out['emb']=np.concatenate(features)
 a.results.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.results/'embeddings.npz',**out)
 (a.results/'embedding_provenance.json').write_text(json.dumps({'split_sha256':digest,'checkpoint_sha256':hashlib.sha256((a.work/'model_best.pt').read_bytes()).hexdigest(),'rows':len(df)},indent=2));print('Saved',a.results/'embeddings.npz')
if __name__=='__main__':main()
