# @title Benchmark identical HSP records and verify full-precision GPU extraction
from pathlib import Path
import os,sys,time,json,hashlib,gc,statistics,importlib.metadata
os.environ.update(SSQ_PARALLEL='0',NUMBA_NUM_THREADS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
base=Path('/content/nsrr');work=base/'work';code=base/'repo/brainbodyfm/stages'
sys.path[:0]=[str(code),str(work/'entry_source'),str(work/'upstream_source/src')]
import numpy as np
import torch
torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
from phi_challenge.embed import load_phi_runtime,load_eeg_c4m1,infer_feature_taps
from philosophers_stone.philosopher_utils import preprocess_filter
from reconstruction_wavelet import compute_specs
from reconstruct_challenge_normalization import download_file
from phi_challenge.heads import load_head_bank,project_batch
from analyze import STABLE4
os.environ['PHILOSOPHER_MODEL_FILE']=str(base/'SleepPhilosophersStone.ckpt')
state=json.loads((work/'state_private.json').read_text());queue=json.loads((work/'queue_private.json').read_text());bykey={str(x['record']['BidsFolder'])+'_ses-'+str(x['record']['SessionID']):x for x in queue}
chosen=[];sites=set()
for row in state['completed']:
 if row['site']not in sites:chosen.append(row);sites.add(row['site'])
assert len(chosen)==3
bank=load_head_bank(str(base/'SleepPhilosophersStone.ckpt'),device='cpu')
source=np.vstack([np.load(work/'latents'/r['latent_file'])for r in state['completed']]);projected_source=project_batch(bank,source);sd=np.column_stack([projected_source[h]for h in STABLE4]).std(0);sd=np.where(sd<1e-8,1.,sd)
runtime=load_phi_runtime(verbose=False);model=runtime['model'];cfg=runtime['cfg'];assert torch.cuda.is_available()and str(cfg.device).startswith('cuda')
receipt=dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,python=sys.version.split()[0],vcpus=os.cpu_count(),benchmark_cpu_threads=2,precision='float32_TF32_disabled_autocast_disabled',chosen_records=3,records=[],versions={n:importlib.metadata.version(n)for n in ['numpy','pandas','scipy','ssqueezepy','mne','numba']})
for ri,row in enumerate(chosen):
 item=bykey[row['key']];path=Path(item['phys_file']);relative='physiological_data/'+str(item['record']['SiteID'])+'/'+path.name
 tick=time.monotonic();download_file(relative,path);download_s=time.monotonic()-tick
 tick=time.monotonic();signals,fs,meta=load_eeg_c4m1(str(path),cfg);signals=preprocess_filter(signals,bandpass_high=cfg.f_high);edf_s=time.monotonic()-tick
 tick=time.monotonic();specs=compute_specs(signals[cfg.channel].to_numpy(),cfg);cwt_s=time.monotonic()-tick;del signals;gc.collect()
 torch.cuda.synchronize();tick=time.monotonic();warm=infer_feature_taps(model,specs,float(item['age']),int(item['sex']),cfg);torch.cuda.synchronize();warm_s=time.monotonic()-tick
 gpu_s=[]
 for _ in range(2):
  torch.cuda.synchronize();tick=time.monotonic();taps=infer_feature_taps(model,specs,float(item['age']),int(item['sex']),cfg);torch.cuda.synchronize();gpu_s.append(time.monotonic()-tick)
 cached=np.load(work/'latents'/row['latent_file']);relative_l2=float(np.linalg.norm(taps['latent']-cached)/np.linalg.norm(cached));projected=project_batch(bank,taps['latent'][None]);oldprojected=project_batch(bank,cached[None]);head_error=float(np.max(np.abs(np.array([projected[h][0]-oldprojected[h][0]for h in STABLE4]))/sd))
 model.to('cpu');cfg.device='cpu';tick=time.monotonic();cpu=infer_feature_taps(model,specs,float(item['age']),int(item['sex']),cfg);cpu_s=time.monotonic()-tick;cpu_relative=float(np.linalg.norm(cpu['latent']-cached)/np.linalg.norm(cached));model.to('cuda:0');cfg.device='cuda:0';torch.cuda.synchronize()
 entry=dict(source_center=row['site'],record_sha256=hashlib.sha256(row['key'].encode()).hexdigest(),download_seconds=download_s,edf_and_filter_seconds=edf_s,cwt_seconds=cwt_s,gpu_warmup_seconds=warm_s,gpu_forward_seconds=statistics.median(gpu_s),cpu_forward_seconds=cpu_s,end_to_end_gpu_seconds=download_s+edf_s+cwt_s+statistics.median(gpu_s),end_to_end_cpu_seconds=download_s+edf_s+cwt_s+cpu_s,gpu_latent_relative_L2_to_cache=relative_l2,cpu_latent_relative_L2_to_cache=cpu_relative,max_head_difference_in_source_SD=head_error,parity_pass=relative_l2<=1e-4 and head_error<=1e-3,first_record_includes_CWT_warmup=(ri==0))
 receipt['records'].append(entry);(base/'hardware_benchmark.json').write_text(json.dumps(receipt,indent=2)+'\n');print('BENCHMARK_RECORD_COMPLETE',json.dumps(entry),flush=True);path.unlink();del specs,taps,cpu,warm;gc.collect();torch.cuda.empty_cache()
receipt['all_parity_pass']=all(r['parity_pass']for r in receipt['records']);receipt['median_gpu_end_to_end_seconds']=statistics.median(r['end_to_end_gpu_seconds']for r in receipt['records']);receipt['median_cpu_end_to_end_seconds']=statistics.median(r['end_to_end_cpu_seconds']for r in receipt['records']);receipt['complete']=True
(base/'hardware_benchmark.json').write_text(json.dumps(receipt,indent=2)+'\n');print('HARDWARE_BENCHMARK_COMPLETE',json.dumps({k:v for k,v in receipt.items()if k!='records'}),flush=True)
