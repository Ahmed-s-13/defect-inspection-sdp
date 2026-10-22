"""Run the frozen-split baseline pipeline, recording durable status and separate logs."""
import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/d1_sscd_grouped/results';OUT.mkdir(parents=True,exist_ok=True)
STATUS=OUT/'pipeline_status.json'
def main():
 if (ROOT/'data/d1_sscd_grouped/work/model_best.pt').exists():raise SystemExit('Existing backbone checkpoint: refusing to overwrite. Inspect the previous run first.')
 env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
 stages=[('resnet18',['main_model/train.py','--workers','4']),('embeddings',['main_model/embed.py','--workers','4']),('mlp_knn',['main_model/train_mlp.py']),('evaluate_resnet18',['main_model/evaluate.py','--workers','0']),('evaluate_mlp',['main_model/evaluate_mlp.py'])]
 status={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'split_sha256':hashlib.sha256((ROOT/'data/d1_sscd_grouped/work/d1_split.csv').read_bytes()).hexdigest(),'stages':[],'status':'running'}
 try:
  for name,args in stages:
   rec={'name':name,'command':[sys.executable,*args],'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};status['stages'].append(rec);STATUS.write_text(json.dumps(status,indent=2));print('START',name,flush=True);t=time.time()
   with (OUT/f'{name}.log').open('w') as log:ret=subprocess.run([sys.executable,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
   rec.update(status='complete' if ret==0 else 'failed',returncode=ret,seconds=round(time.time()-t));STATUS.write_text(json.dumps(status,indent=2))
   if ret:raise RuntimeError(f'{name} exited {ret}; see {OUT/name}.log')
   print('DONE',name,rec['seconds'],'seconds',flush=True)
  status['status']='complete'
 except BaseException as exc:
  status['status']='failed';status['error']=str(exc);raise
 finally:STATUS.write_text(json.dumps(status,indent=2))
if __name__=='__main__':main()
