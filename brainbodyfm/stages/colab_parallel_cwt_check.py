"""Verify remote parallel chunks before continuing the HSP source extraction."""
def main():
 from pathlib import Path
 import os,sys,time,json,subprocess
 os.environ.update(SSQ_GPU='0',SSQ_PARALLEL='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='2')
 base=Path('/content/nsrr');work=base/'work';repo=base/'repo';code=repo/'brainbodyfm/stages'
 subprocess.run(['git','-C',str(repo),'fetch','--quiet','origin'],check=True);subprocess.run(['git','-C',str(repo),'checkout','--quiet','d54cc4e'],check=True)
 sys.path[:0]=[str(code),str(work/'entry_source'),str(work/'upstream_source/src')]
 import numpy as np,torch
 from phi_challenge.embed import load_eeg_c4m1
 from philosophers_stone.philosopher_utils import Config,preprocess_filter
 from reconstruct_challenge_normalization import download_file
 from parallel_reconstruction_wavelet import compute_specs,close_pool
 torch.set_num_threads(2)
 row=next(r for r in json.loads((work/'queue_private.json').read_text())if str(r['record']['SiteID'])=='I0006');path=Path(row['phys_file']);download_file('physiological_data/I0006/'+path.name,path)
 cfg=Config(model_file=str(base/'SleepPhilosophersStone.ckpt'));signals,fs,_=load_eeg_c4m1(str(path),cfg);signals=preprocess_filter(signals,bandpass_high=cfg.f_high);x=signals[cfg.channel].to_numpy();reference=np.load('/content/colab_cwt_reference.npy');timings=[]
 for i in range(2):
  tick=time.monotonic();spec=compute_specs(x,cfg);seconds=time.monotonic()-tick;exact=bool(np.array_equal(spec,reference));timings.append(seconds)
  print('PARALLEL_CWT_CHECK',json.dumps(dict(iteration=i,seconds=seconds,exact_equal=exact)),flush=True)
  if not exact:raise RuntimeError('Parallel CWT did not reproduce the original reference exactly')
 receipt=dict(pass_exact_equality=True,cold_seconds=timings[0],warm_seconds=timings[1],workers=3,scale_batch=128,original_CPU_seconds=115.51327527499984,original_CPU128_seconds=101.07801600200014)
 (base/'parallel_cwt_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');close_pool();path.unlink();print('PARALLEL_CWT_VALIDATED',json.dumps(receipt),flush=True)

if __name__=='__main__':main()
