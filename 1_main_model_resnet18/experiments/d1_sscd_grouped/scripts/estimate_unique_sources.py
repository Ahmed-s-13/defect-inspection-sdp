"""Estimate source-photo families and excess copies, overall and per raw annotation class."""
import csv,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/d1_sscd_grouped';OUT=ROOT/'reports/unique_sources'
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 groups=list(csv.DictReader((DATA/'groups.csv').open()));n=len(groups);parent=list(range(n))
 def find(a):
  while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
  return a
 def union(a,b):parent[find(a)]=find(b)
 bygroup=defaultdict(list)
 for r in groups:bygroup[r['group_id']].append(int(r['image_id']))
 for ids in bygroup.values():
  for i in ids[1:]:union(ids[0],i)
 refined=DATA/'refined_links.csv';num_refined=0
 if refined.exists():
  for r in csv.DictReader(refined.open()):union(int(r['a']),int(r['b']));num_refined+=1
 roots={};family=[roots.setdefault(find(i),len(roots)) for i in range(n)];members=defaultdict(list)
 for i,g in enumerate(family):members[g].append(i)
 names=['crack','dent','corrosion','scratch','missing-head','paint-peel-off'];class_images=defaultdict(set);raw_boxes=Counter();presence=[]
 for r in groups:
  i=int(r['image_id']);p=ROOT/r['path'];label=Path(str(p).replace('/images/','/labels/')).with_suffix('.txt');classes=set()
  for line in label.read_text().splitlines():
   if not line.strip():continue
   c=int(float(line.split()[0]));assert 0<=c<len(names);classes.add(names[c]);raw_boxes[names[c]]+=1
  presence.append(sorted(classes))
  for c in classes:class_images[c].add(i)
 def stat(ids):
  ids=set(ids);ng=len({family[i] for i in ids});orig=len({groups[i]['group_id'] for i in ids})
  return {'images':len(ids),'estimated_source_families':ng,'estimated_excess_copies':len(ids)-ng,'estimated_excess_copy_percent':round(100*(len(ids)-ng)/len(ids),2) if ids else 0.,'strict_initial_families':orig}
 totals=stat(range(n));rows=[]
 for c in names:rows.append({'class':c,**stat(class_images[c]),'raw_annotation_boxes':raw_boxes[c]})
 sizes=Counter(map(len,members.values()));report={'overall':totals,'per_class':rows,'refinement_links':num_refined,'family_size_distribution':dict(sorted(sizes.items())),'singleton_families':sizes.get(1,0),'images_in_multi_image_families':sum(k*v for k,v in sizes.items() if k>1),'images_without_raw_class':sum(not c for c in presence),'method':'Connected components of strict SSCD/pHash + geometric/pixel matching links, plus 16 low-score alignment/pixel refinement links. Broad split-block similarity and visual precaution links are excluded.','meaning':'Excess copies = images minus one representative per estimated family. It does not prove each excess image is synthetically augmented; families can include re-encodings, crops, near-duplicates, and overlapping views. The actual original member cannot be inferred.','limitations':'Not ground truth or a confidence interval. Missed matches increase estimated families; erroneous merges decrease them. Per-class counts are distinct images/families containing that label; multi-class images and families occur in several rows, so rows do not sum to the overall total. Raw labels are used before the 4-pixel crop filter.','split_blocks_not_unique_sources':json.loads((DATA/'split_summary.json').read_text())['conservative_split_blocks']}
 split_rows=list(csv.DictReader((DATA/'split_manifest.csv').open()))
 split_by_id={int(r['image_id']):r['split'] for r in split_rows}
 assert set(split_by_id)==set(range(n))
 family_splits=defaultdict(set)
 for i,g in enumerate(family):family_splits[g].add(split_by_id[i])
 assert all(len(v)==1 for v in family_splits.values()), 'Source family crosses active splits'
 per_split=[];per_split_class=[]
 for split in ['train','val','test']:
  ids={i for i in range(n) if split_by_id[i]==split}
  per_split.append({'split':split,**stat(ids)})
  for c in names:per_split_class.append({'split':split,'class':c,**stat(ids & class_images[c])})
 assert sum(r['estimated_source_families'] for r in per_split)==totals['estimated_source_families']
 assert sum(r['estimated_excess_copies'] for r in per_split)==totals['estimated_excess_copies']
 report.update(per_split=per_split,per_split_class=per_split_class,confidence_score=None,confidence_note='No calibrated confidence score or confidence interval: no independently labelled matching benchmark is available. SSCD cosine similarity is not a probability of shared origin.')
 for filename,table in [('per_split.csv',per_split),('per_split_class.csv',per_split_class)]:
  with (OUT/filename).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 (OUT/'summary.json').write_text(json.dumps(report,indent=2))
 with (OUT/'per_class.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 manifest=[]
 for i,r in enumerate(groups):
  representative=min(members[family[i]]);manifest.append({'image_id':i,'path':r['path'],'estimated_family':family[i],'family_size':len(members[family[i]]),'is_representative':i==representative,'representative_path':groups[representative]['path'],'classes':'|'.join(presence[i])})
 with (OUT/'source_estimate_manifest.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=manifest[0]);w.writeheader();w.writerows(manifest)
 md=['# D1 estimated source families and excess copies','',f"Across **{n:,} images**, the matching pipeline estimates **{totals['estimated_source_families']:,} source families** and **{totals['estimated_excess_copies']:,} excess copies ({totals['estimated_excess_copy_percent']:.2f}%)** after retaining one representative per family.",'','These are estimates of shared visual origin, not a proven count of synthetic augmentations. A family may contain different crops, re-encodings, near-duplicates or overlapping views. Its chosen representative is the lowest image ID, not a claim about which file was the original.','',f"The strict first pass found {totals['strict_initial_families']:,} groups. Adding {num_refined} geometric/pixel refinement links gives the estimate above. The **720 broad split blocks are not used as the unique-image estimate**: those deliberately combine uncertain or merely similar scenes for split safety.",'','| Class | Images containing class | Estimated source families | Excess copies | Excess % |','|---|---:|---:|---:|---:|']
 for r in rows:md.append(f"| {r['class']} | {r['images']:,} | {r['estimated_source_families']:,} | {r['estimated_excess_copies']:,} | {r['estimated_excess_copy_percent']:.2f}% |")
 md+=['',f"Singleton families: **{report['singleton_families']:,}**. Images belonging to multi-image families: **{report['images_in_multi_image_families']:,}**. The latter is different from the excess-copy count because it includes each family's representative.",'','## Interpretation','',report['limitations'],'','No manual ground-truth validation set was available to calibrate precision/recall or estimate an uncertainty interval. This number should be reported as a matching-based estimate, not an exact property of D1.','', '## Files and reproduction','','- `per_class.csv`: class-level table.','- `summary.json`: definitions, counts and family-size distribution.','- `source_estimate_manifest.csv`: every image, family ID, class membership, family size and representative path.','','From this experiment directory:','','```sh','.venv/bin/python scripts/estimate_unique_sources.py','```']
 section=['## D1 SSCD augmentation/copy estimates','',
 'Dataset: **`d1_sscd_grouped`**. Counts refer to raw images, before bounding-box cropping. Estimated augmented/copy images means **images minus one representative per matched source family**. This includes crops, re-encodings and near-copies; synthetic augmentation provenance and the true original file are not known. The 720 broader split blocks are used only to prevent split leakage, not to count unique sources.','',
 '**Overall: 14,760 images; 1,691 estimated source families; 13,069 excess copies (88.54%).**','',
 '### Per class','',
 '| Class | Images | Estimated source families | Estimated augmented/copy images | Estimated copy % |',
 '|---|---:|---:|---:|---:|']
 for r in rows:section.append(f"| {r['class']} | {r['images']:,} | {r['estimated_source_families']:,} | {r['estimated_excess_copies']:,} | {r['estimated_excess_copy_percent']:.2f}% |")
 section+=['','### Active train/validation/test split','',
 '| Split | Images | Estimated source families | Estimated augmented/copy images | Estimated copy % |',
 '|---|---:|---:|---:|---:|']
 for r in per_split:section.append(f"| {r['split']} | {r['images']:,} | {r['estimated_source_families']:,} | {r['estimated_excess_copies']:,} | {r['estimated_excess_copy_percent']:.2f}% |")
 section+=['','These percentages measure redundancy **within each split**, not leakage between splits. All estimated families remain within one split; undetected relatives may still exist. Percentages use each row’s image count as denominator. Multi-class images/families appear in multiple class rows, so class rows do not sum to the total.','',
 '**Confidence: not calibrated.** No defensible numerical confidence score or confidence interval is available. SSCD similarity is not a probability. Missed matches overestimate unique families; false merges underestimate them. Treat these as matching-based estimates, not ground truth.','',
 'The reproducible report also provides `per_split_class.csv` for each class within each split.']
 block='\n'.join(section)+'\n'
 md+=['',block,'','- `per_split.csv`: active split copy estimates.','- `per_split_class.csv`: class-by-split copy estimates.']
 (OUT/'README.md').write_text('\n'.join(md)+'\n')
 repo=ROOT.parents[2]
 begin='<!-- SSCD_COPY_ESTIMATES_START -->';end='<!-- SSCD_COPY_ESTIMATES_END -->'
 readme=repo/'README.md';text=readme.read_text()
 rendered=begin+'\n'+block+'\nSee [full SSCD report](1_main_model_resnet18/experiments/d1_sscd_grouped/reports/unique_sources/README.md) and [class-by-split CSV](1_main_model_resnet18/experiments/d1_sscd_grouped/reports/unique_sources/per_split_class.csv).\n'+end
 if begin in text:text=text[:text.index(begin)]+rendered+text[text.index(end)+len(end):]
 else:
  marker='The older accuracy results below'
  assert marker in text
  text=text.replace(marker,rendered+'\n\n'+marker,1)
 readme.write_text(text)
 print(json.dumps({'overall':totals,'per_class':rows,'singletons':sizes.get(1,0)},indent=2))
if __name__=='__main__':main()
