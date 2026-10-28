"""Resume unchanged HSP task with independently verified remote execution."""
def main():
 from pathlib import Path
 import os,sys,json,tarfile,hashlib,time
 base=Path('/content/nsrr');work=base/'work';code=base/'repo/brainbodyfm/stages'
 benchmark=json.loads((base/'hardware_benchmark.json').read_text());parallel=json.loads((base/'parallel_cwt_receipt.json').read_text())
 assert benchmark.get('complete')and benchmark.get('all_parity_pass')and 'T4'in benchmark['gpu']
 assert parallel.get('pass_exact_equality')
 bundle=Path('/content/colab_stages_rank_inputs.tar.gz');assert hashlib.sha256(bundle.read_bytes()).hexdigest()=='25a21d6e486d91ecd2efb4886f6370b22ff6298d3b4041606ebfeef52bc3ea29'
 target=base/'stages';target.mkdir(exist_ok=True)
 with tarfile.open(bundle)as archive:archive.extractall(target,filter='data')
 os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',NUMBA_NUM_THREADS='2',SSQ_PARALLEL='0',SSQ_GPU='0',PHILOSOPHER_MODEL_FILE=str(base/'SleepPhilosophersStone.ckpt'))
 sys.path[:0]=[str(code),str(work/'entry_source'),str(work/'upstream_source/src')]
 import torch
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 import phi_challenge.embed as embed
 original=embed.embed_edf_feature_taps
 def full_precision(runtime,path,age,sex,**kwargs):kwargs['amp']=False;return original(runtime,path,age,sex,**kwargs)
 embed.embed_edf_feature_taps=full_precision;ordinary=embed.embed_edf_path
 def ordinary_full(runtime,path,age,sex,**kwargs):kwargs['amp']=False;return ordinary(runtime,path,age,sex,**kwargs)
 embed.embed_edf_path=ordinary_full
 import reconstruction_wavelet as wavelet
 from parallel_reconstruction_wavelet import compute_specs,close_pool
 wavelet.compute_specs=compute_specs
 import reconstruct_challenge_normalization as task
 original_atomic=task.atomic;exported=set()
 def export_snapshot():
  state=json.loads((work/'state_private.json').read_text());n=len(state['completed']);temporary=base/'hsp_checkpoint_export.tmp';destination=base/'hsp_checkpoint_export.tar.gz'
  with tarfile.open(temporary,'w:gz')as archive:
   for name in ['state_private.json','queue_private.json','wavelet_parity.json','environment.txt','data/demographics.csv']:
    archive.add(work/name,arcname=name)
   for row in state['completed']:
    for folder,key in [('latents','latent_file'),('feature_taps','feature_tap_file')]:
     if key in row:archive.add(work/folder/row[key],arcname=str(Path(folder)/row[key]))
   for name in ['reconstructed_normalization.json','completed_source_roster_private.json']:
    if (work/name).exists():archive.add(work/name,arcname=name)
  temporary.replace(destination);h=task.digest(destination)
  print('COLAB_CHECKPOINT_READY',json.dumps(dict(records=n,sha256=h,bytes=destination.stat().st_size)),flush=True)
 def audited_atomic(path,value):
  original_atomic(path,value)
  if Path(path).name=='state_private.json':
   n=len(value.get('completed',[]))
   if n and n%25==0 and n not in exported:exported.add(n);export_snapshot()
 task.atomic=audited_atomic
 state=json.loads((work/'state_private.json').read_text());receipt=dict(device=torch.cuda.get_device_name(0),precision='float32_autocast_and_TF32_disabled',CWT_backend='original_CPU_algorithm_three_chunks_parallel_batch128',start_n=len(state['completed']),benchmark=benchmark,parallel_validation=parallel,started_unix=time.time(),original_budget_seconds=state['extraction_budget_seconds'],prior_active_seconds=state['active_seconds'])
 (base/'gpu_resume_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 sys.argv=[str(code/'reconstruct_challenge_normalization.py'),'--entry-source',str(work/'entry_source'),'--upstream-source',str(work/'upstream_source'),'--checkpoint',str(base/'SleepPhilosophersStone.ckpt'),'--work',str(work),'--stages-root',str(target)]
 print('COLAB_HSP_RESUMING',receipt['start_n'],receipt['device'],flush=True)
 try:task.main()
 finally:export_snapshot();close_pool()

if __name__=='__main__':main()
