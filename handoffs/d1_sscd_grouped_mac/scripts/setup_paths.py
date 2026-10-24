"""Localize a frozen portable split without changing assignments."""
from pathlib import Path
import csv,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/d1_sscd_grouped'
def main():
 portable='--portable' in sys.argv
 for manifest,out in [('d1_split.portable.csv','work/d1_split.csv'),('crop_inventory.portable.csv','work/crops/D1.csv')]:
  with (ROOT/'manifests'/manifest).open() as f:rows=list(csv.DictReader(f))
  for r in rows:
   p=ROOT/r['crop_path'];assert p.is_file(),p
   if not portable:r['crop_path']=str(p)
  with (D/out).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 sha=hashlib.sha256((D/'work/d1_split.csv').read_bytes()).hexdigest()
 s=json.loads((D/'split_summary.json').read_text());s['split_sha256']=sha;s['migration_note']='Portable Mac handoff; local paths updated, assignments unchanged.';(D/'split_summary.json').write_text(json.dumps(s,indent=2))
 (D/'FROZEN').write_text('Frozen d1_sscd_grouped split. Active manifest SHA256: '+sha+'\n')
 (D/'data.yaml').write_text('path: '+json.dumps('data/d1_sscd_grouped' if portable else str(D))+'\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 6\nnames: [crack, dent, corrosion, scratch, missing-head, paint-peel-off]\n')
 print('Portable manifests restored.' if portable else 'Paths ready for this Mac. Split assignments unchanged.')
if __name__=='__main__':main()
