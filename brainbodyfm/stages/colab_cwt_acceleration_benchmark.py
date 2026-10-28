# @title Test numerical parity and speed of CWT execution backends
from pathlib import Path
import os,sys,json,subprocess,time,hashlib
base=Path('/content/nsrr');work=base/'work';code=base/'repo/brainbodyfm/stages';bench=base/'cwt_acceleration';bench.mkdir(exist_ok=True)
os.environ.update(SSQ_GPU='0',SSQ_PARALLEL='0',NUMBA_NUM_THREADS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
sys.path[:0]=[str(code),str(work/'entry_source'),str(work/'upstream_source/src')]
import numpy as np
from phi_challenge.embed import load_eeg_c4m1
from philosophers_stone.philosopher_utils import Config,preprocess_filter
from reconstruct_challenge_normalization import download_file
queue=json.loads((work/'queue_private.json').read_text());row=next(r for r in queue if str(r['record']['SiteID'])=='I0006');path=Path(row['phys_file']);relative='physiological_data/I0006/'+path.name
download_file(relative,path);cfg=Config(model_file=str(base/'SleepPhilosophersStone.ckpt'));signals,fs,_=load_eeg_c4m1(str(path),cfg);signals=preprocess_filter(signals,bandpass_high=cfg.f_high);np.save(bench/'input.npy',signals[cfg.channel].to_numpy());del signals
child=bench/'worker.py';child.write_text('''from pathlib import Path
import sys,os,time,json
sys.path[:0]=['/content/nsrr/repo/brainbodyfm/stages','/content/nsrr/work/upstream_source/src']
import numpy as np, torch
from philosophers_stone.philosopher_utils import Config
from reconstruction_wavelet import compute_specs
torch.set_num_threads(2)
cfg=Config(model_file='/content/nsrr/SleepPhilosophersStone.ckpt')
x=np.load('/content/nsrr/cwt_acceleration/input.npy');batch=int(sys.argv[1]);out=Path(sys.argv[2]);times=[]
for i in range(1):
 if torch.cuda.is_available():torch.cuda.synchronize()
 t=time.monotonic();spec=compute_specs(x,cfg,scale_batch=batch)
 if torch.cuda.is_available():torch.cuda.synchronize()
 times.append(time.monotonic()-t)
np.save(out,spec);out.with_suffix('.json').write_text(json.dumps(dict(seconds=times,batch=batch,SSQ_GPU=os.environ.get('SSQ_GPU'),shape=spec.shape)))
print('CWT_VARIANT_COMPLETE',out.stem,times,flush=True)
''')
result=dict(variants={},parity={},record_sha256=hashlib.sha256(str(row['record']['BidsFolder']).encode()).hexdigest())
for name,gpu,batch in [('cpu16','0',16),('cpu128','0',128),('gpu64','1',64)]:
 env=os.environ.copy();env['SSQ_GPU']=gpu
 print('CWT_VARIANT_START',name,flush=True)
 p=subprocess.run([sys.executable,'-u',str(child),str(batch),str(bench/(name+'.npy'))],env=env)
 if p.returncode:result['variants'][name]=dict(failed=True);continue
 result['variants'][name]=json.loads((bench/(name+'.json')).read_text())
 if name!='cpu16':
  a=np.load(bench/'cpu16.npy');b=np.load(bench/(name+'.npy'));difference=np.abs(a-b);result['parity'][name]=dict(allclose=bool(np.allclose(a,b,rtol=2e-5,atol=2e-6)),max_absolute=float(difference.max()),relative_L2=float(np.linalg.norm(a-b)/np.linalg.norm(a)));del a,b,difference
 (base/'cwt_acceleration_benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
result['complete']=True;(base/'cwt_acceleration_benchmark.json').write_text(json.dumps(result,indent=2)+'\n');print('CWT_ACCELERATION_COMPLETE',json.dumps(result),flush=True);path.unlink()
