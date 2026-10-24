"""Local, offline-capable ResNet-only training supervisor. Launched under caffeinate."""
from pathlib import Path
import os,sys,json,subprocess,datetime,time,hashlib,traceback
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/resnet18_run';REPO=ROOT.parents[2]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 OUT.mkdir(exist_ok=True)
 status={'status':'running','started_utc':now(),'supervisor_pid':os.getpid(),'model':'ResNet-18 linear classifier (no MLP)','split_sha256':hashlib.sha256((ROOT/'data/d1_sscd_grouped/work/d1_split.csv').read_bytes()).hexdigest(),'stages':[],'power':'caffeinate -is: prevents idle sleep; system-sleep protection on AC; cannot survive shutdown or forced/lid sleep','offline':True}
 def save():
  temp=OUT/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(OUT/'status.json')
 def doc(message):
  p=REPO/'README.md';s=p.read_text();begin='<!-- SSCD_RESNET_RUN_START -->';end='<!-- SSCD_RESNET_RUN_END -->';block=begin+'\n'+message+'\n'+end
  if begin in s:s=s[:s.index(begin)]+block+s[s.index(end)+len(end):]
  else:s=s.replace('## Current D1 grouping work — 22 October 2026','## Current D1 grouping work — 22 October 2026\n\n'+block,1)
  s=s.replace('- Training status: first attempt interrupted before a completed epoch/checkpoint. **No new grouped-split accuracy is available yet.**','- Current training status and results are recorded in the run section above.')
  p.write_text(s)
 try:
  assert not (ROOT/'data/d1_sscd_grouped/work/model_best.pt').exists(), 'Refusing to overwrite checkpoint'
  save();doc('**ResNet-18 training is running locally in a detached process.** [Live status](1_main_model_resnet18/experiments/d1_sscd_grouped/reports/resnet18_run/status.json) · [training log](1_main_model_resnet18/experiments/d1_sscd_grouped/reports/resnet18_run/train.log). Overall/per-class and family-weighted results will be written automatically after completion. Screen-off and offline execution are supported while the Mac remains awake and powered.')
  env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'))
  for name,args in [('train',['main_model/train.py','--workers','4','--results',str(OUT)]),('report',['scripts/report_resnet.py'])]:
   rec={'stage':name,'status':'running','started_utc':now()};status['stages'].append(rec);save()
   with (OUT/f'{name}.log').open('w') as log:
    p=subprocess.Popen([sys.executable,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    rec['pid']=p.pid;save()
    while p.poll() is None:status['heartbeat_utc']=now();save();time.sleep(15)
   rec.update(status='complete' if p.returncode==0 else 'failed',returncode=p.returncode,finished_utc=now());save()
   if p.returncode:raise RuntimeError(f'{name} exited {p.returncode}; see {name}.log')
  status.update(status='complete',finished_utc=now());save()
  doc('**ResNet-18 training and evaluation completed.** [Overall and per-class results](1_main_model_resnet18/experiments/d1_sscd_grouped/reports/resnet18_run/README.md), including both ordinary crop metrics and equal-source-family metrics. [Run status](1_main_model_resnet18/experiments/d1_sscd_grouped/reports/resnet18_run/status.json).')
 except BaseException as e:
  status.update(status='failed',error=str(e),finished_utc=now());save();doc('**ResNet-18 run failed or was interrupted.** Inspect [status and error](1_main_model_resnet18/experiments/d1_sscd_grouped/reports/resnet18_run/status.json) and the accompanying logs. No successful completion is claimed.');traceback.print_exc();raise
if __name__=='__main__':main()
