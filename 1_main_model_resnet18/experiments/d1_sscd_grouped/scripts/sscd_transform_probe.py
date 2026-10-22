"""Known-positive transformations on a fixed 32-image sample; not real-pair recall."""
import json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps,ImageEnhance
from torchvision import transforms as T
ROOT=Path(__file__).resolve().parents[1]
def main():
 torch.set_num_threads(4);device='mps' if torch.backends.mps.is_available() else 'cpu'
 meta=json.loads((ROOT/'reports/images.json').read_text());emb=np.load(ROOT/'reports/sscd_embeddings.npy')
 rng=np.random.default_rng(123);chosen=rng.choice(len(meta),32,replace=False)
 model=torch.jit.load(str(ROOT/'models/sscd_disc_mixup.torchscript.pt'),map_location='cpu').eval().to(device)
 tf=T.Compose([T.Resize((320,320)),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 bank=torch.from_numpy(emb).to(device);bank=torch.nn.functional.normalize(bank,dim=1)
 ops={'flip':Image.Transpose.FLIP_LEFT_RIGHT,'rotate90':Image.Transpose.ROTATE_90,'rotate180':Image.Transpose.ROTATE_180,'rotate270':Image.Transpose.ROTATE_270,'transpose':Image.Transpose.TRANSPOSE,'transverse':Image.Transpose.TRANSVERSE,'flip_vertical':Image.Transpose.FLIP_TOP_BOTTOM}
 output={}
 with torch.inference_mode():
  for name in list(ops)+['center_crop_80','color']:
   batch=[]
   for i in chosen:
    im=Image.open(ROOT/meta[int(i)]['path']).convert('RGB')
    if name in ops:im=im.transpose(ops[name])
    elif name=='color':im=ImageEnhance.Color(ImageEnhance.Brightness(im).enhance(.7)).enhance(.5)
    else:
     w,h=im.size;im=im.crop((round(.1*w),round(.1*h),round(.9*w),round(.9*h)))
    batch.append(tf(im))
   y=torch.nn.functional.normalize(model(torch.stack(batch).to(device)),dim=1);sim=y@bank.T
   positive=sim[torch.arange(len(chosen),device=device),torch.tensor(chosen,device=device)]
   rank=(sim>positive[:,None]+1e-6).sum(dim=1)+1
   output[name]={'n':len(chosen),'median_same_image_cosine':float(positive.median().cpu()),'min_same_image_cosine':float(positive.min().cpu()),'source_in_top20':int((rank<=20).sum().cpu()),'source_top1':int((rank==1).sum().cpu())}
   print(name,output[name],flush=True)
 (ROOT/'reports/transform_probe.json').write_text(json.dumps({'seed':123,'image_ids':chosen.tolist(),'results':output},indent=2))
if __name__=='__main__':main()
