"""Strict central EEG resolution and bounded reads of the supplied STAGES EDFs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent


def edf_header(path):
    with open(path, 'rb') as f:
        fixed = f.read(256)
        if len(fixed) != 256 or fixed[:8].strip() != b'0':
            raise ValueError('Not a supported 16-bit EDF')
        n = int(fixed[252:256])
        count = int(fixed[236:244])
        duration = float(fixed[244:252])
        header_bytes = int(fixed[184:192])
        variable = f.read(header_bytes - 256)
    fields = [('label',16),('transducer',80),('unit',8),('physical_min',8),('physical_max',8),
              ('digital_min',8),('digital_max',8),('prefilter',80),('samples_per_record',8),('reserved',32)]
    channels = [{} for _ in range(n)]
    pos = 0
    for key,width in fields:
        for i in range(n):
            value = variable[pos+i*width:pos+(i+1)*width].decode('latin1').strip()
            channels[i][key] = value
        pos += n*width
    if count < 0:
        count = (Path(path).stat().st_size-header_bytes)//(2*sum(int(c['samples_per_record']) for c in channels))
    return {'n_channels':n,'n_records':count,'record_seconds':duration,'duration_hours':count*duration/3600,
            'header_bytes':header_bytes,'channels':channels}


def normalize(label):
    s = label.lower().strip().replace('‑','-').replace('–','-').replace('−','-')
    s = re.sub(r'^eeg[\s_]*', '', s)
    return re.sub(r'\s+', '', s).replace('/','-').replace(':','-')


def plan_channel(labels, allow_contralateral=False):
    norm = {normalize(s):s for s in labels}
    def native(aliases):
        return next((norm[a] for a in aliases if a in norm),None)
    c = native(['c4-m1','c4-a1','c4m1','c4a1','c4(m1)','c4(a1)'])
    if c:
        return {'kind':'native_c4m1','channels':[c]}
    cathode = native(['m1','a1'])
    if 'c4' in norm and cathode:
        return {'kind':'derived_c4m1','channels':[norm['c4'],cathode]}
    if allow_contralateral:
        c = native(['c3-m2','c3-a2','c3m2','c3a2','c3(m2)','c3(a2)'])
        if c:
            return {'kind':'native_c3m2','channels':[c]}
        cathode = native(['m2','a2'])
        if 'c3' in norm and cathode:
            return {'kind':'derived_c3m2','channels':[norm['c3'],cathode]}
    return {'kind':'unavailable','channels':[]}


def read_signal(path, *, allow_contralateral=False, maximum_hours=11, assume_missing_uv=False):
    import mne
    from philosophers_stone.philosopher_utils import _resample_raw
    header = edf_header(path)
    plan = plan_channel([r['label'] for r in header['channels']],allow_contralateral)
    if plan['kind']=='unavailable':
        raise ValueError('No strictly referenced C4-M1 route')
    wanted = [c for c in header['channels'] if c['label'] in plan['channels']]
    unit_overrides={c['label']:'uV' for c in wanted if c['unit']=='' and assume_missing_uv}
    if any(c['unit'].lower() not in ['uv','µv','μv','mv','v'] and c['label'] not in unit_overrides for c in wanted):
        raise ValueError('Unknown voltage units in selected channels')
    # Read only the selected electrodes, avoiding preload of every PSG channel.
    raw = mne.io.read_raw_edf(path,include=plan['channels'],preload=False,units=unit_overrides or None,verbose='ERROR')
    if raw.n_times/float(raw.info['sfreq']) > maximum_hours*3600:
        raw.crop(tmin=0,tmax=maximum_hours*3600-1/float(raw.info['sfreq']))
    raw.load_data(verbose='ERROR')
    raw = _resample_raw(raw,200)
    data = raw.get_data(picks=plan['channels']) * 1e6
    x = data[0] - data[1] if plan['kind'].startswith('derived') else data[0]
    qc = {'duration_hours':len(x)/200/3600,'finite_fraction':float(np.isfinite(x).mean()),
          'std_uv':float(np.std(x)),'flat_difference_fraction':float(np.mean(np.diff(x)==0)),
          'absolute_gt500_uv_fraction':float(np.mean(np.abs(x)>500)),
          'voltage_unit_status':'assumed_uV' if unit_overrides else 'explicit_EDF_dimension',
          'voltage_unit_overrides':unit_overrides,**plan}
    return x, qc


def pilot_headers(base=BASE):
    queue=pd.read_csv(base/'data/pilot_queue.csv')
    results=[]
    def worker(r):
        start=time.time()
        try:
            header=edf_header(r.edf_path)
            primary=plan_channel([c['label'] for c in header['channels']])
            alternate=plan_channel([c['label'] for c in header['channels']],True)
            return {'subject_id':r.subject_id,'center':r.center,'primary':primary,'sensitivity':alternate,
                    'read_seconds':round(time.time()-start,3),**header}
        except Exception as e:
            return {'subject_id':r.subject_id,'center':r.center,'error':f'{type(e).__name__}: {e}'}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for future in as_completed([executor.submit(worker,r) for r in queue.itertuples(index=False)]):
            r=future.result();results.append(r)
            print(json.dumps({k:v for k,v in r.items() if k in ['subject_id','center','primary','sensitivity','read_seconds','duration_hours','error']}),flush=True)
    (base/'results/pilot_headers.json').write_text(json.dumps(sorted(results,key=lambda x:x['center']),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot-headers',action='store_true');args=parser.parse_args()
    if args.pilot_headers:pilot_headers()
