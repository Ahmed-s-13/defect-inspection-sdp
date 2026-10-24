"""Save held-out predictions and crop/family-weighted overall and per-class metrics."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'main_model'))
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report,confusion_matrix
from evaluate import load_model
from train import predict
from klshift.common import CropDataset,pick_device
OUT=ROOT/'reports/resnet18_run';WORK=ROOT/'data/d1_sscd_grouped/work'
def main():
 torch.set_num_threads(4);device=pick_device();model,classes=load_model(WORK,device)
 df=pd.read_csv(WORK/'d1_split.csv');families=pd.read_csv(ROOT/'reports/unique_sources/source_estimate_manifest.csv').set_index('image_id').estimated_family
 df['family']=df.image_id.map(families);assert df.family.notna().all()
 report={'model':'ImageNet-initialized ResNet-18 with linear classifier; no MLP','split_sha256':hashlib.sha256((WORK/'d1_split.csv').read_bytes()).hexdigest(),'definitions':{'all_crops':'Each crop has equal weight.','equal_source_family':'Each estimated source family has total weight one; its crops divide that weight equally.'},'limitations':'Families are estimated, not ground truth. No calibrated grouping confidence or training-seed uncertainty. Test was not used for model selection.','splits':{}}
 md=['# ResNet-18 on d1_sscd_grouped','',report['model'],'','Best checkpoint selected by validation macro-F1. See d1_history.csv for every epoch.','']
 for split in ['val','test']:
  d=df[df.split==split].copy();y=d.cls.map({c:i for i,c in enumerate(classes)})
  yt,yp=predict(model,DataLoader(CropDataset(d.crop_path,y),batch_size=64,num_workers=4),device)
  d['true']=d.cls;d['predicted']=[classes[i] for i in yp];d.to_csv(OUT/f'{split}_predictions.csv',index=False)
  weights=1/d.groupby('family').family.transform('size').to_numpy();entry={'crops':len(d),'source_families':int(d.family.nunique())}
  for name,w in [('all_crops',None),('equal_source_family',weights)]:
   rep=classification_report(yt,yp,labels=list(range(len(classes))),target_names=classes,sample_weight=w,output_dict=True,zero_division=0)
   cm=confusion_matrix(yt,yp,labels=list(range(len(classes))),sample_weight=w)
   entry[name]={'accuracy':float(np.average(yt==yp,weights=w)),'macro_f1':rep['macro avg']['f1-score'],'balanced_accuracy':rep['macro avg']['recall'],'classification_report':rep,'confusion_matrix':cm.tolist()}
   m=entry[name];md += [f'## {split}: {name}','',f"Accuracy **{m['accuracy']:.4f}**; macro-F1 **{m['macro_f1']:.4f}**; balanced accuracy **{m['balanced_accuracy']:.4f}**.",'','| Class | Precision | Recall | F1 | Support (weighted for family metrics) |','|---|---:|---:|---:|---:|']
   for c in classes:
    r=rep[c];md.append(f"| {c} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {r['support']:.2f} |")
   md+=['']
  report['splits'][split]=entry
 md += ['Family-weighted metrics give each estimated source family equal influence; they do not establish that families are truly independent.','',report['limitations']]
 (OUT/'metrics.json').write_text(json.dumps(report,indent=2));(OUT/'README.md').write_text('\n'.join(md)+'\n')
if __name__=='__main__':main()
