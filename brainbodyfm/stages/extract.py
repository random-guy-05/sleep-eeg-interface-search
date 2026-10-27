"""Resumable, outcome-blind frozen EEG extraction from the mounted Drive copy."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

from signal_io import edf_header, plan_channel, read_signal
from wavelet import compute_specs

BASE=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)


@torch.no_grad()
def infer_latent(model,specs,age,sex,cfg):
    x=torch.as_tensor(specs,dtype=torch.float32,device=cfg.device)[None,None]
    cov=torch.tensor([[(float(age)-cfg.age_mean_tr_data)/cfg.age_std_tr_data,int(sex)]],dtype=torch.float32,device=cfg.device)
    _,_,_,encoded=model.stem(x)
    encoded=model.forward_maxvit(encoded)
    latent_tensor=model.forward_final_latent_space(encoded,cov)
    outputs={name.replace('final_head_',''):head(latent_tensor) for name,head in model.heads.items()}
    semantic={}
    for name,dimension,task in zip(model.target_names,model.target_output_dims,model.task_types):
        values=outputs[name]
        if task=='classification':values=torch.sigmoid(values)
        values=values.float().cpu().numpy().reshape(-1)
        for i,v in enumerate(values):
            suffix=f'_{i}' if dimension>1 else ''
            semantic[f'pred_{task}_{name}{suffix}']=float(v)
    return latent_tensor.float().cpu().numpy().reshape(-1), semantic


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--queue',type=Path,default=BASE/'data/pilot_queue.csv')
    parser.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--keep-specs',action='store_true')
    parser.add_argument('--unit-policy',choices=['strict','assumed_uv'],default='strict')
    parser.add_argument('--pilot-unit-sensitivity',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    cfg_data=json.loads((BASE/'protocol_v1.json').read_text())
    cfg_data.update(json.loads((BASE/'amendment_01.json').read_text())['overrides'])
    thresholds=cfg_data['signal_qc_thresholds']
    preflight=json.loads((BASE/'results/preprocessing_validation.json').read_text())
    if preflight['status']!='passed' or preflight['wavelet_source_sha256']!=digest(BASE/'wavelet.py'):
        raise RuntimeError('Preprocessing has not passed validation for these source bytes')
    from philosophers_stone.philosopher_utils import Config,load_model,_apply_head_weights
    meta=json.loads((BASE/'results/checkpoint_provenance.json').read_text())
    cfg=Config(model_file=meta['path']);cfg.device=args.device
    queue=pd.read_csv(args.queue).fillna('')
    if queue.subject_id.duplicated().any():raise ValueError('Duplicate extraction subjects')
    if any(c in queue for c in ['pcet_accuracy2','face_memory_correct','pvt_mean_reciprocal_rt']):
        raise ValueError('Extraction queue must not contain cognitive outcomes')
    if args.limit:queue=queue.head(args.limit)
    if args.unit_policy=='assumed_uv' and not args.pilot_unit_sensitivity:
        gate=json.loads((BASE/'results/unit_sensitivity_gate.json').read_text())
        if not gate['passed']:raise ValueError('Provisional unit sensitivity gate did not pass')
    cache=BASE/'cache' if args.unit_policy=='strict' else BASE/'cache/assumed_uv'
    cache.mkdir(parents=True,exist_ok=True)
    out=cache/'subjects';out.mkdir(parents=True,exist_ok=True)
    log_path=BASE/'results'/('extraction_progress.jsonl' if args.unit_policy=='strict' else 'assumed_uv_extraction_progress.jsonl')
    source_hashes={p.name:digest(p) for p in [BASE/'extract.py',BASE/'wavelet.py',BASE/'signal_io.py']}
    run_key=hashlib.sha256(json.dumps({'source':source_hashes,'checkpoint':meta['sha256'],'device':args.device,'unit_policy':args.unit_policy},sort_keys=True).encode()).hexdigest()
    def log(stage,**kw):
        entry={'utc':datetime.now(timezone.utc).isoformat(),'stage':stage,**kw}
        with log_path.open('a') as f:f.write(json.dumps(entry)+'\n');f.flush()
        print(json.dumps(entry),flush=True)
    model=None
    completed=quarantined=0
    taxonomy=pd.read_csv(BASE.parents[0]/'frozen/head_taxonomy.csv').sort_values('head_index').head_name.tolist()
    for row in queue.itertuples(index=False):
        sid=row.subject_id;start=time.time();marker=out/f'{sid}.json';payload=out/f'{sid}.npz'
        if marker.exists() and payload.exists():
            saved=json.loads(marker.read_text())
            if saved['run_key']!=run_key or saved['protocol_sha256']!=row.protocol_sha256:
                raise RuntimeError(f'Existing extraction provenance differs for {sid}; do not overwrite')
            if digest(payload)!=saved['payload_sha256']:raise RuntimeError(f'Extraction hash mismatch: {sid}')
            completed+=1;log('already_complete',subject_id=sid,center=row.center);continue
        try:
            if shutil.disk_usage(BASE).free < 2*1024**3:raise RuntimeError('Less than 2 GiB free disk; waiting for storage resolution')
            header=edf_header(row.edf_path)
            plan=plan_channel([c['label'] for c in header['channels']])
            if plan['kind']=='unavailable':raise ValueError('No genuine C4-M1 or explicit C4 minus M1 route')
            selected=[c for c in header['channels'] if c['label'] in plan['channels']]
            if any(c['unit'].lower() not in ['uv','µv','μv','mv','v'] and not (args.unit_policy=='assumed_uv' and c['unit']=='') for c in selected):
                raise ValueError('Voltage unit missing or unknown; calibration must be documented')
            log('read_signal',subject_id=sid,center=row.center,channel_kind=plan['kind'])
            eeg,qc=read_signal(row.edf_path,assume_missing_uv=args.unit_policy=='assumed_uv')
            if qc['duration_hours']<2 or qc['finite_fraction']!=1:
                raise ValueError('Recording duration or finite-signal QC failed')
            if not thresholds['minimum_std_uv']<=qc['std_uv']<=thresholds['maximum_std_uv']:
                raise ValueError('Signal amplitude QC failed')
            if qc['flat_difference_fraction']>thresholds['maximum_flat_difference_fraction'] or qc['absolute_gt500_uv_fraction']>thresholds['maximum_absolute_gt500_uv_fraction']:
                raise ValueError('Signal flatness or gross artifact QC failed')
            from philosophers_stone.preprocessing_and_spectrograms import preprocess_filter
            frame=preprocess_filter(pd.DataFrame({'c4-m1':eeg}),bandpass_high=cfg.f_high)
            log('wavelet',subject_id=sid,center=row.center,signal_qc=qc)
            specs=compute_specs(frame['c4-m1'].to_numpy(),cfg)
            del frame,eeg;gc.collect()
            spec_sha=hashlib.sha256(specs.tobytes()).hexdigest()
            if args.keep_specs:
                dest=cache/'specs';dest.mkdir(parents=True,exist_ok=True);np.save(dest/f'{sid}.npy',specs)
            if model is None:
                log('load_model',device=cfg.device);model=load_model(cfg)
            log('infer',subject_id=sid,center=row.center,device=cfg.device)
            latent,semantic=infer_latent(model,specs,row.age,row.sex,cfg)
            if args.pilot_unit_sensitivity:
                sensitivity=[]
                for factor in [1,1000,1000000]:
                    alt_latent,alt_semantic=infer_latent(model,specs*factor,row.age,row.sex,cfg)
                    sensitivity.append({'factor':factor,'latent_relative_L2':float(np.linalg.norm(alt_latent-latent)/np.linalg.norm(latent)),
                                        'max_semantic_absolute_change':float(max(abs(alt_semantic[k]-semantic[k]) for k in semantic))})
                atomic_json(cache/'unit_sensitivity'/f'{sid}.json',{'subject_id':sid,'center':row.center,'checks':sensitivity,'cognitive_outcomes_used':False,'run_key':run_key})
            if set(semantic)!=set(taxonomy):raise ValueError('Projected semantic head names differ from frozen taxonomy')
            values=np.array([semantic[k] for k in taxonomy],dtype=np.float32)
            if latent.shape!=(1024,) or not np.isfinite(latent).all() or not np.isfinite(values).all():
                raise ValueError('Invalid latent or semantic output')
            _,bhs=_apply_head_weights(latent,cfg,(float(row.age)-cfg.age_mean_tr_data)/cfg.age_std_tr_data,int(row.sex))
            tmp=payload.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,latent=latent,semantic=values,semantic_names=np.array(taxonomy),brain_health_score=bhs)
            tmp.replace(payload)
            record={'subject_id':sid,'center':row.center,'protocol_sha256':row.protocol_sha256,'run_key':run_key,'unit_policy':args.unit_policy,
                    'checkpoint_sha256':meta['sha256'],'source_hashes':source_hashes,'device':cfg.device,
                    'eeg_source_path':row.edf_path,'eeg_source_bytes':int(row.edf_bytes),
                    'specs_sha256':spec_sha,'signal_qc':qc,'elapsed_seconds':round(time.time()-start,2),
                    'payload_sha256':digest(payload),'semantic_features':len(values),'latent_features':len(latent)}
            atomic_json(marker,record);completed+=1
            (cache/'quarantine'/f'{sid}.json').unlink(missing_ok=True)
            log('complete',subject_id=sid,center=row.center,elapsed_seconds=record['elapsed_seconds'],completed=completed)
            del specs,latent,semantic,values;gc.collect()
        except Exception as e:
            quarantined+=1
            record={'subject_id':sid,'center':row.center,'reason':f'{type(e).__name__}: {e}','run_key':run_key,
                    'protocol_sha256':row.protocol_sha256,'utc':datetime.now(timezone.utc).isoformat()}
            atomic_json(cache/'quarantine'/f'{sid}.json',record)
            log('quarantine',**record)
            if isinstance(e,RuntimeError) and 'free disk' in str(e):raise
            traceback.print_exc()
    log('queue_finished',queue=str(args.queue),completed=completed,quarantined=quarantined,total=len(queue))


if __name__=='__main__':main()
