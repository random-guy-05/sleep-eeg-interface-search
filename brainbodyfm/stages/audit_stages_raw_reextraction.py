"""Re-extract four outcome-blind audit records without changing saved analyses."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--work',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    root=a.data_root.resolve();work=a.work.resolve()
    sys.path.insert(0,str(work/'upstream_source/src'));sys.path.insert(0,str(work/'entry_source'));sys.path.insert(0,str(root))
    from signal_io import read_signal,edf_header,plan_channel
    from wavelet import compute_specs
    from phi_challenge.embed import load_phi_runtime,infer_feature_taps,infer_latent_only
    from philosophers_stone.preprocessing_and_spectrograms import preprocess_filter
    from philosophers_stone.philosopher_utils import _apply_head_weights
    from analyze import STABLE4
    os.environ['PHILOSOPHER_MODEL_FILE']=str(work.parent/'SleepPhilosophersStone.ckpt')
    original=torch.load
    def memory_mapped_load(source,*args,**kwargs):
        # Only checkpoint deserialization changes; weights and model setup do not.
        if not isinstance(source,(str,Path)):
            candidate=getattr(source,'name',None)
            if candidate and Path(candidate).is_file():source=candidate
        if isinstance(source,(str,Path)):kwargs['mmap']=True
        return original(source,*args,**kwargs)
    torch.load=memory_mapped_load
    try:runtime=load_phi_runtime(verbose=False)
    finally:torch.load=original
    cfg=runtime['cfg'];model=runtime['model']
    cohort=pd.read_csv(root/'data/cohort_before_signal_qc.csv')
    ids=pd.read_csv(root/'results/provisional_assumed_uv/nested_transport/analysis_subjects.csv').subject_id
    cohort=cohort[cohort.subject_id.isin(ids)]
    # Shortest available record in each center, with ID tie-break: no scores or
    # outcomes participate in choosing this small engineering audit sample.
    chosen=cohort.sort_values(['center','edf_bytes','subject_id']).groupby('center',sort=True).head(1)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),selection='smallest EDF byte count per center, ID tie-break, independent of outcomes and predictions',records=[],runtime_weight_loading='memory-mapped deserialization, exact submitted loader and model setup',voltage_units='user-confirmed microvolts; missing metadata receives the existing documented uV override')
    import mne
    for row in chosen.itertuples():
        print(json.dumps({'stage':'read_raw','center':row.center}),flush=True)
        header=edf_header(row.edf_path);plan=plan_channel([x['label'] for x in header['channels']]);wanted=[x for x in header['channels'] if x['label'] in plan['channels']]
        overrides={x['label']:'uV' for x in wanted if not x['unit']}
        raw=mne.io.read_raw_edf(row.edf_path,include=plan['channels'],preload=False,units=overrides or None,verbose='ERROR')
        # Independently decode EDF digital samples and the physical calibration
        # for one data record; compare with MNE before filtering/resampling.
        block=np.memmap(row.edf_path,dtype='<i2',mode='r',offset=header['header_bytes'],shape=(sum(int(x['samples_per_record']) for x in header['channels']),))
        offsets=np.cumsum([0]+[int(x['samples_per_record']) for x in header['channels']]);calibration=[]
        for c in wanted:
            j=header['channels'].index(c);dig=np.array(block[offsets[j]:offsets[j+1]],float)
            unit=c['unit'].lower();factor=1. if unit in ['', 'uv','µv','μv'] else 1000. if unit=='mv' else 1000000.
            physical=(dig-float(c['digital_min']))*(float(c['physical_max'])-float(c['physical_min']))/(float(c['digital_max'])-float(c['digital_min']))+float(c['physical_min'])
            if abs(len(physical)/header['record_seconds']-float(raw.info['sfreq']))<1e-8:
                mne_uv=raw.get_data(picks=[c['label']],start=0,stop=len(physical))[0]*1e6
                calibration.append(float(np.max(np.abs(mne_uv-physical*factor))))
        del raw,block
        eeg,qc=read_signal(row.edf_path,assume_missing_uv=True)
        filtered=preprocess_filter(pd.DataFrame({cfg.channel:eeg}),bandpass_high=cfg.f_high)
        specs=compute_specs(filtered[cfg.channel].to_numpy(),cfg)
        del eeg,filtered;gc.collect()
        print(json.dumps({'stage':'infer','center':row.center}),flush=True)
        with torch.no_grad():
            fresh=infer_feature_taps(model,specs,float(row.age),int(row.sex),cfg)['latent']
            # Ordinary submitted forward is compared on one record; same weights,
            # same tensor, two code paths. This catches a missing regression layer.
            ordinary=infer_latent_only(model,specs,float(row.age),int(row.sex),cfg) if not report['records'] else None
            x=torch.as_tensor(fresh,dtype=torch.float32)[None,:]
            four=[]
            for name in STABLE4:
                layer=model.heads['final_head_'+name.removeprefix('pred_regression_')]
                four.append(float(layer(x).reshape(-1)[0]))
        cache=root/'cache/assumed_uv/subjects'/f'{row.subject_id}.npz'
        meta=json.loads(cache.with_suffix('.json').read_text())
        with np.load(cache,allow_pickle=False) as z:
            ix=[z['semantic_names'].tolist().index(k) for k in STABLE4]
            head_error=np.asarray(four)-z['semantic'][ix]
            relative=float(np.linalg.norm(fresh-z['latent'])/np.linalg.norm(z['latent']))
            _,fresh_bhs=_apply_head_weights(fresh,cfg,(row.age-cfg.age_mean_tr_data)/cfg.age_std_tr_data,row.sex)
            bhs_error=float(fresh_bhs-z['brain_health_score'])
        record=dict(center=row.center,duration_hours=qc['duration_hours'],channel_kind=qc['kind'],
            direct_EDF_calibration_max_error_uv=max(calibration) if calibration else None,
            fresh_latent_relative_L2=relative,four_output_absolute_errors=np.abs(head_error).tolist(),
            brain_health_absolute_error=abs(bhs_error),
            spectrogram_sha_matches=hashlib.sha256(specs.tobytes()).hexdigest()==meta['specs_sha256'],
            source_input_bytes_match=Path(row.edf_path).stat().st_size==meta['eeg_source_bytes'],
            cached_checkpoint_sha256=meta['checkpoint_sha256'])
        if ordinary is not None:record['submitted_tap_vs_ordinary_relative_L2']=float(np.linalg.norm(fresh-ordinary)/np.linalg.norm(ordinary))
        report['records'].append(record)
        a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'stage':'checked',**record}),flush=True)
        del specs,fresh,x;gc.collect()
    report['status']='complete';a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    with threadpool_limits(limits=1):main()
