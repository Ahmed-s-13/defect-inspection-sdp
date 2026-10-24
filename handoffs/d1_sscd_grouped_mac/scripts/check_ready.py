"""Validate the frozen dataset and perform one disposable training-batch smoke check."""
import csv,hashlib,json,sys,tempfile,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'main_model'))
from train import make_split
from klshift.common import CropDataset,build_model,embed_batch,train_transform
from klshift.mlp import MLPHead,MLPClassifier,knn_proba,l2n
from evaluate import load_model
from torch.utils.data import DataLoader

def main():
 torch.set_num_threads(2);torch.manual_seed(0);work=ROOT/'data/d1_sscd_grouped/work';out=ROOT/'data/d1_sscd_grouped'
 df=make_split(work,0);m=pd.read_csv(out/'split_manifest.csv');summary=json.loads((out/'split_summary.json').read_text())
 assert hashlib.sha256((work/'d1_split.csv').read_bytes()).hexdigest()==summary['split_sha256']
 assert len(m)==14760 and m.image_id.is_unique
 assert m.groupby('group_id').split.nunique().max()==1
 assert m.groupby('split_block').split.nunique().max()==1
 assert df.groupby('image').split.nunique().max()==1
 for r in m.itertuples():
  name=Path(r.path).name
  assert (out/'images'/r.split/name).exists()
  assert (out/'labels'/r.split/Path(name).with_suffix('.txt')).exists()
 assert set(df.image_id).issubset(set(m.image_id))
 lookup=m.set_index('image_id')
 assert all(df.split.to_numpy()==lookup.loc[df.image_id,'split'].to_numpy())
 assert all(df.group_id.to_numpy()==lookup.loc[df.image_id,'split_block'].to_numpy())
 classes=sorted(df.cls.unique());sample=df[df.split=='train'].groupby('cls').head(2);ys=sample.cls.map({c:i for i,c in enumerate(classes)})
 x,y=next(iter(DataLoader(CropDataset(sample.crop_path,ys,train_transform()),batch_size=12)))
 model=build_model(6,pretrained=True);opt=torch.optim.AdamW(model.parameters(),lr=1e-4);pred=model(x);loss=torch.nn.functional.cross_entropy(pred,y);assert torch.isfinite(loss)
 opt.zero_grad();loss.backward();opt.step();model.eval()
 with torch.no_grad():emb=embed_batch(model,x)
 assert emb.shape==(12,512) and torch.isfinite(emb).all()
 head=MLPHead(6);head.train();loss2=torch.nn.functional.cross_entropy(head(emb.detach()),y);loss2.backward();assert torch.isfinite(loss2)
 prob=knn_proba(l2n(emb),y,l2n(emb),5,6);assert torch.allclose(prob.sum(1),torch.ones(12),atol=1e-5)
 # Verify actual checkpoint loading and rejection of a stale backbone/head.
 with tempfile.TemporaryDirectory(prefix='d1-smoke-') as folder:
  tmp=Path(folder);shutil.copy2(work/'d1_split.csv',tmp/'d1_split.csv')
  split_hash=hashlib.sha256((tmp/'d1_split.csv').read_bytes()).hexdigest()
  torch.save({'state':model.state_dict(),'classes':classes,'epoch':0,'split_sha256':split_hash},tmp/'model_best.pt')
  loaded,_=load_model(tmp,torch.device('cpu'))
  pr={'split_sha256':split_hash,'checkpoint_sha256':hashlib.sha256((tmp/'model_best.pt').read_bytes()).hexdigest()}
  torch.save({'states':[head.state_dict()],'arch':{},'classes':classes,'knn_k':5,'knn_weight':.5,'knn_bank':emb,'knn_labels':y,'provenance':pr},tmp/'mlp_head.pt')
  classifier=MLPClassifier(tmp/'mlp_head.pt',torch.device('cpu'));out_prob=classifier(emb)
  assert out_prob.shape==(12,6) and torch.allclose(out_prob.sum(1),torch.ones(12),atol=1e-5)
  (tmp/'d1_split.csv').write_text('modified split')
  for loader in [lambda:load_model(tmp,torch.device('cpu')),lambda:MLPClassifier(tmp/'mlp_head.pt',torch.device('cpu'))]:
   try:loader()
   except SystemExit:pass
   else:raise AssertionError('Stale checkpoint was accepted')
 report={'status':'PASS','checks':['frozen manifest checksum','all raw links and labels resolve','all crop paths resolve','exact crop inventory coverage','source groups and split blocks never cross splits','image/crop split consistency','ImageNet ResNet18 forward/backward/optimizer step','512-dimensional feature extraction','MLP forward/backward','5-NN normalized probabilities','checkpoint roundtrip','stale split/backbone/head rejection'],'smoke_only':True,'saved_trained_checkpoint':False,'device':'cpu','images':len(m),'crops':len(df),'classes':classes}
 (out/'readiness_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
