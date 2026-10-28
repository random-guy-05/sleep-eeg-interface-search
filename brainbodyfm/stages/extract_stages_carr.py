"""Outcome-blind, resumable exact submitted taps; never overwrites old caches."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import gc,hashlib,json,os
from pathlib import Path
import shutil,subprocess,sys,time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

BASE=Path(__file__).resolve().parent
FILES=['STAGES_CARR_REPRESENTATION_PLAN.md','extract_stages_carr.py','run_stages_carr_transfer.py']


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def atomic(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def receipt():
    repo=Path(subprocess.check_output(['git','-C',str(BASE),'rev-parse','--show-toplevel'],text=True).strip());commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();hashes={}
    for name in FILES:
        path=BASE/name;b=subprocess.check_output(['git','-C',str(repo),'show',commit+':'+str(path.relative_to(repo))])
        if b!=path.read_bytes():raise ValueError('Commit extraction plan and code before execution')
        hashes[name]=hashlib.sha256(b).hexdigest()
    return dict(commit=commit,files=hashes,started_utc=datetime.now(timezone.utc).isoformat())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--work',type=Path,required=True);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--run-transfer',action='store_true');a=ap.parse_args()
    freeze=receipt();root=a.data_root.resolve();work=a.work.resolve();out=root/'cache/carr_taps';out.mkdir(exist_ok=True,parents=True);status_path=out/'state_private.json'
    if status_path.exists():
        state=json.loads(status_path.read_text())
        if state['freeze']['files']!=freeze['files']:raise ValueError('Do not resume under changed extraction code')
    else:state=dict(freeze=freeze,status='preparing',completed=[],failures=[])
    manifest=json.loads((BASE/'data/reconstruction_source_manifest.json').read_text())
    for category,folder in [('entry',work/'entry_source'),('upstream',work/'upstream_source')]:
        for name,expected in manifest[category]['files'].items():
            if digest(folder/name)!=expected:raise ValueError('Submitted or pinned source differs: '+name)
    checkpoint=work.parent/'SleepPhilosophersStone.ckpt'
    old_meta=json.loads((root/'results/checkpoint_provenance.json').read_text())
    if digest(checkpoint)!=old_meta['sha256']:raise ValueError('Checkpoint differs from frozen STAGES checkpoint')
    sys.path.insert(0,str(work/'upstream_source/src'));sys.path.insert(0,str(work/'entry_source'));sys.path.insert(0,str(root))
    from signal_io import read_signal
    from wavelet import compute_specs
    from phi_challenge.embed import load_phi_runtime,infer_feature_taps
    from philosophers_stone.preprocessing_and_spectrograms import preprocess_filter
    ids=pd.read_csv(root/'results/provisional_assumed_uv/nested_transport/analysis_subjects.csv').subject_id
    cohort=pd.read_csv(root/'data/cohort_before_signal_qc.csv')
    columns=['subject_id','center','age','sex','edf_path']
    queue=cohort.loc[cohort.subject_id.isin(ids),columns].sort_values(['center','subject_id']).reset_index(drop=True)
    assert len(queue)==590 and queue.subject_id.is_unique
    done={r['subject_id'] for r in state['completed']};failed={r['subject_id'] for r in state['failures']}
    os.environ['PHILOSOPHER_MODEL_FILE']=str(checkpoint)
    original=torch.load
    def mapped_load(source,*args,**kwargs):
        if not isinstance(source,(str,Path)):
            name=getattr(source,'name',None)
            if name and Path(name).is_file():source=name
        if isinstance(source,(str,Path)):kwargs['mmap']=True
        return original(source,*args,**kwargs)
    torch.load=mapped_load
    try:runtime=load_phi_runtime(verbose=False)
    finally:torch.load=original
    cfg=runtime['cfg'];model=runtime['model'];started=time.monotonic();initial=state.get('active_seconds',0.);count=0
    for row in queue.itertuples(index=False):
        if row.subject_id in done or row.subject_id in failed:continue
        if a.limit and count>=a.limit:break
        if shutil.disk_usage(root).free<4*1024**3:raise RuntimeError('Less than 4 GiB free; stop before extracting more records')
        tick=time.monotonic();state['status']='extracting';state['current_subject_id']=row.subject_id;atomic(status_path,state)
        try:
            old_path=root/'cache/assumed_uv/subjects'/f'{row.subject_id}.npz';previous=json.loads(old_path.with_suffix('.json').read_text())
            eeg,qc=read_signal(row.edf_path,assume_missing_uv=True);filtered=preprocess_filter(pd.DataFrame({cfg.channel:eeg}),bandpass_high=cfg.f_high);specs=compute_specs(filtered[cfg.channel].to_numpy(),cfg)
            del eeg,filtered;gc.collect();spectrogram_sha=hashlib.sha256(specs.tobytes()).hexdigest()
            if spectrogram_sha!=previous['specs_sha256']:raise ValueError('Spectrogram hash differs from existing extraction')
            with torch.no_grad():taps=infer_feature_taps(model,specs,float(row.age),int(row.sex),cfg)
            with np.load(old_path,allow_pickle=False) as old:
                relative=float(np.linalg.norm(taps['latent']-old['latent'])/np.linalg.norm(old['latent']))
            if relative>1e-5:raise ValueError('New ordinary latent differs from existing cache')
            arrays={k:np.asarray(taps[k],np.float32) for k in ['latent','global','temporal','counterfactual']}
            shapes={'latent':(1024,),'global':(2048,),'temporal':(32,32),'counterfactual':(8,)}
            if any(arrays[k].shape!=shapes[k] or not np.isfinite(arrays[k]).all() for k in arrays):raise ValueError('Invalid submitted tap shapes or values')
            path=out/(hashlib.sha256(row.subject_id.encode()).hexdigest()+'.npz');temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as f:np.savez_compressed(f,**arrays)
            temporary.replace(path)
            state['completed'].append(dict(subject_id=row.subject_id,center=row.center,file=path.name,sha256=digest(path),spectrogram_sha256=spectrogram_sha,latent_relative_L2=relative,seconds=time.monotonic()-tick))
            print(json.dumps(dict(stage='STAGES_CARR_record_complete',n=len(state['completed']),total=590,center=row.center,latent_relative_L2=relative,seconds=round(time.monotonic()-tick,1))),flush=True)
            del specs,taps,arrays;gc.collect()
        except Exception as error:
            state['failures'].append(dict(subject_id=row.subject_id,center=row.center,error=type(error).__name__+':'+str(error)))
            print(json.dumps(dict(stage='STAGES_CARR_record_failed',failures=len(state['failures']),reason=type(error).__name__)),flush=True)
        count+=1;state['active_seconds']=initial+time.monotonic()-started;atomic(status_path,state)
    state['status']='complete' if len(state['completed'])==590 else 'partial_or_failures';state['active_seconds']=initial+time.monotonic()-started;atomic(status_path,state)
    print(json.dumps(dict(stage='STAGES_CARR_pass_complete',n=len(state['completed']),failures=len(state['failures']),status=state['status'])),flush=True)
    if a.run_transfer and state['status']=='complete':
        subprocess.run([sys.executable,str(BASE/'run_stages_carr_transfer.py'),'--data-root',str(root),'--entry-source',str(work/'entry_source')],check=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    with threadpool_limits(limits=1):main()
