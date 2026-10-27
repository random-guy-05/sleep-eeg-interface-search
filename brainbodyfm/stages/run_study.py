"""Coordinate the gated provisional study; persist each stage and retry I/O failures."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
import pandas as pd
from analyze import readiness

BASE=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)


def verify_unit_gate():
    policy_path=BASE/'amendment_02_provisional_units.json';policy=json.loads(policy_path.read_text())
    expected=pd.read_csv(BASE/'data/pilot_queue.csv')
    records=[]
    for row in expected.itertuples(index=False):
        p=BASE/'cache/assumed_uv/unit_sensitivity'/f'{row.subject_id}.json'
        if not p.exists():return {'passed':False,'status':'awaiting_pilot','missing_center':row.center}
        r=json.loads(p.read_text());assert r['center']==row.center and not r['cognitive_outcomes_used']
        assert [x['factor'] for x in r['checks']]==policy['tested_alternative_unit_factors']
        marker=BASE/'cache/assumed_uv/subjects'/f'{row.subject_id}.json'
        if not marker.exists():return {'passed':False,'status':'awaiting_pilot_commit','missing_center':row.center}
        meta=json.loads(marker.read_text());assert r['run_key']==meta['run_key']
        for name,sha in meta['source_hashes'].items():
            if digest(BASE/name)!=sha:raise ValueError('Pilot inference source changed')
        records.append(r)
    assert len(records)==4 and len({x['run_key'] for x in records})==1
    bounds=policy['unit_sensitivity_gate']
    maximum_latent=max(c['latent_relative_L2'] for r in records for c in r['checks'])
    maximum_head=max(c['max_semantic_absolute_change'] for r in records for c in r['checks'])
    result={'passed':bool(maximum_latent<=bounds['maximum_latent_relative_L2_difference'] and maximum_head<=bounds['maximum_semantic_head_absolute_difference']),
            'status':'evaluated','pilots':records,'maximum_latent_relative_L2':maximum_latent,
            'maximum_semantic_absolute_difference':maximum_head,'amendment02_sha256':digest(policy_path),
            'interpretation':'Common unit conversion sensitivity only; physical units remain unconfirmed'}
    save(BASE/'results/unit_sensitivity_gate.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-gate',action='store_true');args=parser.parse_args()
    gate=verify_unit_gate()
    print(json.dumps({k:v for k,v in gate.items() if k!='pilots'},indent=2),flush=True)
    if args.check_gate:return
    if not gate['passed']:raise RuntimeError('Unit sensitivity pilot is incomplete or failed')
    lock=(BASE/'results/study.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def state(stage,**extra):
        r={'stage':stage,'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat(),
           'unit_policy':'assumed_uv','results_provisional':True,**extra}
        save(BASE/'results/study_status.json',r);print(json.dumps(r),flush=True)
    def run(script,*arguments):
        cmd=[sys.executable,str(BASE/script),*map(str,arguments)]
        result=subprocess.run(cmd,check=False)
        if result.returncode:raise RuntimeError(f'{script} exited {result.returncode}')
    try:
        queue=pd.read_csv(BASE/'data/extraction_queue.csv').fillna('')
        headers=pd.read_csv(BASE/'data/signal_header_audit.csv')
        valid=set(headers.loc[(headers.channel_kind!='unavailable')&(headers.duration_hours>=2),'subject_id'])
        queue=queue[queue.subject_id.isin(valid)].copy()
        # Stable round-robin ordering prevents an interrupted prefix from containing one center only.
        queue['order_hash']=queue.subject_id.map(lambda s:hashlib.sha256(f'STAGES-transport-v1:{s}'.encode()).hexdigest())
        groups=[part.sort_values('order_hash').drop(columns='order_hash') for _,part in queue.groupby('center',sort=True)]
        rows=[]
        for i in range(max(len(g) for g in groups)):
            rows.extend(g.iloc[[i]] for g in groups if i<len(g))
        queue=pd.concat(rows,ignore_index=True);queue_path=BASE/'data/provisional_extraction_queue.csv';queue.to_csv(queue_path,index=False)
        save(BASE/'results/provisional_run_inputs.json',{
            'queue_sha256':digest(queue_path),'cohort_sha256':digest(BASE/'data/cohort_before_signal_qc.csv'),
            'unit_policy_sha256':digest(BASE/'amendment_02_provisional_units.json'),
            'unit_gate_sha256':digest(BASE/'results/unit_sensitivity_gate.json'),
            'coordinator_sha256':digest(__file__),'inference_run_key':gate['pilots'][0]['run_key'],
            'potential_recordings':len(queue),'by_center':queue.groupby('center').size().to_dict()})
        state('extraction_running',potential_recordings=len(queue))
        run('extract.py','--unit-policy','assumed_uv','--queue',queue_path)
        # Retry remote read failures only. Technical QC exclusions remain exclusions.
        for attempt in [2,3]:
            retry=[]
            for row in queue.itertuples(index=False):
                p=BASE/'cache/assumed_uv/quarantine'/f'{row.subject_id}.json'
                if p.exists():
                    reason=json.loads(p.read_text())['reason'].lower()
                    if any(x in reason for x in ['timeouterror','operation timed out','input/output error','connectionerror']):retry.append(row.subject_id)
            if not retry:break
            path=BASE/'data'/f'io_retry_pass{attempt}.csv';queue[queue.subject_id.isin(retry)].to_csv(path,index=False)
            state('retrying_remote_IO',attempt=attempt,n=len(retry));run('extract.py','--unit-policy','assumed_uv','--queue',path)
        status,_=readiness(unit_policy='assumed_uv')
        if status['status']!='ready':
            state('analysis_not_estimable',readiness=status);return
        for outcome in ['pcet_accuracy2','face_memory_correct','pvt_mean_reciprocal_rt']:
            state('nested_analysis_running',outcome=outcome)
            run('analyze.py','--unit-policy','assumed_uv','--outcome',outcome,'--run')
        state('reporting');run('report.py')
        state('provisional_analysis_complete',report=str(BASE/'results/provisional_assumed_uv/RESULTS.md'))
    except BaseException as exc:
        state('needs_attention',error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc();raise


if __name__=='__main__':main()
