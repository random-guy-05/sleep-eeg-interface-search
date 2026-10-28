"""Separate, frozen original-protocol timing sensitivity; private new-patient cache."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, shutil, subprocess, sys, time, zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from analyze import STABLE4
from stable4_extension import CENTERS
from age_conditioned_transfer import eligible_pairs, rank_auc, continuous_concordance, clean_json
from general_cognitive_prediction import source_reference
from unexpected_cognition_discovery import inputs, endpoint, ROSTER, SIGNED

BASE=Path(__file__).resolve().parent
FILES=['TIMING14_FIXED_SCORE_PLAN.md','timing14_fixed_scores.py']
TASKS={'executive':('pcet_accuracy2',),'memory':('face_memory_correct',),'composite':('pcet_accuracy2','face_memory_correct'),'attention':('attention_better',)}

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def atomic(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(clean_json(d),indent=2,allow_nan=False)+'\n');tmp.replace(p)

def receipt():
 repo=Path(subprocess.check_output(['git','-C',str(BASE),'rev-parse','--show-toplevel'],text=True).strip());sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();files={}
 for name in FILES:
  p=BASE/name;b=subprocess.check_output(['git','-C',str(repo),'show',sha+':'+str(p.relative_to(repo))])
  if b!=p.read_bytes():raise ValueError('Commit timing plan/code before computing new associations')
  files[name]=hashlib.sha256(b).hexdigest()
 return dict(commit=sha,files=files,utc=datetime.now(timezone.utc).isoformat())

def prepare(root,freeze):
 import prepare as original
 out=root/'cache/timing14';scratch=out/'preparation';(scratch/'data').mkdir(parents=True,exist_ok=True);(scratch/'results').mkdir(exist_ok=True)
 source=root/'data/source';target=scratch/'data/source'
 if not target.exists():target.symlink_to(source,target_is_directory=True)
 for name in ['psg_drive_inventory.json','psg_local_inventory.json']:
  target=scratch/'data'/name
  if not target.exists():target.symlink_to(root/'data'/name)
 cfg=json.loads((root/'protocol_v1.json').read_text());assert 'cnb_window_14_days' in cfg['sensitivity_analyses'];cfg['primary_population']['max_absolute_psg_cnb_calendar_days']=14
 atomic(scratch/'protocol_v1.json',cfg);shutil.copyfile(root/'amendment_01.json',scratch/'amendment_01.json')
 subprocess.run([sys.executable,str(root/'prepare.py'),'--base',str(scratch)],check=True,stdout=subprocess.DEVNULL)
 cohort=pd.read_csv(scratch/'data/cohort_before_signal_qc.csv');old=pd.read_csv(root/'data/cohort_before_signal_qc.csv');new=cohort.loc[~cohort.subject_id.isin(old.subject_id)].copy()
 assert len(new)==87 and new.subject_id.is_unique
 attr=pd.read_csv(root/'data/attrition_by_subject.csv').set_index('subject_id')
 assert attr.loc[new.subject_id,'exclusion_reasons'].eq('PCET_PSG_date_window_not_met').all()
 assert new.cnb_psg_days.abs().between(2,14).all()
 # Recover only the exact previously selected session, preserving source row ties.
 selected={}
 with zipfile.ZipFile(source/'stages_cnb.zip') as z:
  for member in sorted(z.namelist()):
   if not member.endswith('.xlsx') or 'inlab' not in member:continue
   for rownum,r in original.excel_rows(BytesIO(z.read(member)),max_col=310):
    sid=original.text(r.get('test_sessions.subid'))
    if sid not in set(new.subject_id):continue
    target=new.loc[new.subject_id.eq(sid)].iloc[0]
    dt=original.day(r.get('modified.test_sessions_v.dotest'))
    if dt is None or dt.isoformat()!=target.cnb_date or original.text(r.get('test_sessions.datasetid'))!=str(target.cnb_dataset_id):continue
    session=original.num(r.get('test_sessions_v.session'))
    if session!=target.cnb_session:continue
    if original.text(r.get('PCET_A.status'))!='C1' or original.text(r.get('PCET_A.valid_code'))!='V' or original.truth(r.get('PCET_A.skipped')):continue
    if sid not in selected:selected[sid]=r
 assert len(selected)==len(new)
 extra=[]
 for sid in new.subject_id:
  r=selected[sid];extra.append(dict(subject_id=sid,**{k:r.get(k) for k in ['SPCPTNL.status','SPCPTNL.valid_code','SPCPTNL.skipped','SPCPTNL.SCPT_TPRT']}))
 new=new.merge(pd.DataFrame(extra),on='subject_id',validate='one_to_one',sort=False).sort_values(['center','subject_id']).reset_index(drop=True)
 new.to_csv(out/'additional_cohort_private.csv',index=False)
 cols=['subject_id','center','age','sex','edf_path','edf_name','edf_bytes'];new[cols].to_csv(out/'extraction_queue_private.csv',index=False)
 atomic(out/'preparation_private.json',dict(freeze=freeze,n=len(new),by_center=new.center.value_counts().to_dict(),source_builder_sha256=digest(root/'prepare.py'),queue_sha256=digest(out/'extraction_queue_private.csv'),cohort_sha256=digest(out/'additional_cohort_private.csv'),original_roster_sha256=digest(root/'data/cohort_before_signal_qc.csv'),source_manifest_sha256=digest(source/'SHA256SUMS.json'),associations_computed=False))
 print(json.dumps(dict(stage='timing14_prepared',n=len(new),by_center=new.center.value_counts().to_dict(),associations_computed=False)),flush=True)

def extract(root,work,freeze):
 import torch
 out=root/'cache/timing14';meta=json.loads((out/'preparation_private.json').read_text());assert meta['freeze']['files']==freeze['files']
 queue=pd.read_csv(out/'extraction_queue_private.csv');assert digest(out/'extraction_queue_private.csv')==meta['queue_sha256']
 allowed={'subject_id','center','age','sex','edf_path','edf_name','edf_bytes'};assert set(queue)==allowed and len(queue)==87
 checkpoint=work.parent/'SleepPhilosophersStone.ckpt';assert digest(checkpoint)==json.loads((root/'results/checkpoint_provenance.json').read_text())['sha256']
 manifest=json.loads((BASE/'data/reconstruction_source_manifest.json').read_text())
 for kind,folder in [('entry',work/'entry_source'),('upstream',work/'upstream_source')]:
  for name,sha in manifest[kind]['files'].items():
   if digest(folder/name)!=sha:raise ValueError('Pinned source changed')
 sys.path.insert(0,str(work/'upstream_source/src'));sys.path.insert(0,str(work/'entry_source'));sys.path.insert(0,str(root))
 from signal_io import read_signal
 from wavelet import compute_specs
 from phi_challenge.embed import load_phi_runtime,infer_feature_taps
 from philosophers_stone.philosopher_utils import _apply_head_weights
 from philosophers_stone.preprocessing_and_spectrograms import preprocess_filter
 cfgdata=json.loads((root/'amendment_01.json').read_text())['overrides']['signal_qc_thresholds'];os.environ['PHILOSOPHER_MODEL_FILE']=str(checkpoint);torch.set_num_threads(1);torch.set_grad_enabled(False)
 original=torch.load
 def mapped(source,*args,**kw):
  if not isinstance(source,(str,Path)) and getattr(source,'name',None):source=source.name
  if isinstance(source,(str,Path)):kw['mmap']=True
  return original(source,*args,**kw)
 torch.load=mapped
 try:runtime=load_phi_runtime(verbose=False)
 finally:torch.load=original
 cfg=runtime['cfg'];model=runtime['model'];statepath=out/'extraction_state_private.json'
 state=json.loads(statepath.read_text()) if statepath.exists() else dict(freeze=freeze,completed=[],failures=[],status='extracting')
 assert state['freeze']['files']==freeze['files'];done={r['subject_id'] for r in state['completed']+state['failures']}
 for row in queue.itertuples(index=False):
  if row.subject_id in done:continue
  if shutil.disk_usage(root).free<4*1024**3:raise RuntimeError('Less than 4 GiB free disk')
  tick=time.monotonic()
  try:
   eeg,qc=read_signal(row.edf_path,assume_missing_uv=True)
   if qc['duration_hours']<2 or qc['finite_fraction']!=1 or not cfgdata['minimum_std_uv']<=qc['std_uv']<=cfgdata['maximum_std_uv'] or qc['flat_difference_fraction']>cfgdata['maximum_flat_difference_fraction'] or qc['absolute_gt500_uv_fraction']>cfgdata['maximum_absolute_gt500_uv_fraction']:raise ValueError('Original signal QC failed')
   filtered=preprocess_filter(pd.DataFrame({cfg.channel:eeg}),bandpass_high=cfg.f_high);specs=compute_specs(filtered[cfg.channel].to_numpy(),cfg);del eeg,filtered;gc.collect()
   with torch.no_grad():
    taps=infer_feature_taps(model,specs,float(row.age),int(row.sex),cfg);latent=torch.as_tensor(taps['latent'],dtype=torch.float32)[None];projected={name.replace('final_head_',''):head(latent) for name,head in model.heads.items()};semantic={}
    for name,dimension,task in zip(model.target_names,model.target_output_dims,model.task_types):
     values=projected[name];values=torch.sigmoid(values) if task=='classification' else values
     for i,v in enumerate(values.cpu().numpy().reshape(-1)):semantic[f'pred_{task}_{name}'+(f'_{i}' if dimension>1 else '')]=float(v)
   _,bhs=_apply_head_weights(taps['latent'],cfg,(float(row.age)-cfg.age_mean_tr_data)/cfg.age_std_tr_data,int(row.sex));names=sorted(semantic);values=np.array([semantic[n] for n in names],np.float32)
   assert len(names)==841 and taps['latent'].shape==(1024,) and np.isfinite(values).all() and np.isfinite(bhs) and all(np.isfinite(v).all() for v in taps.values())
   path=out/(hashlib.sha256(row.subject_id.encode()).hexdigest()+'.npz');temporary=path.with_suffix('.tmp')
   with temporary.open('wb') as f:np.savez_compressed(f,**taps,semantic=values,semantic_names=np.array(names),brain_health_score=bhs)
   temporary.replace(path);state['completed'].append(dict(subject_id=row.subject_id,center=row.center,file=path.name,sha256=digest(path),signal_qc=qc,spectrogram_sha256=hashlib.sha256(specs.tobytes()).hexdigest(),seconds=time.monotonic()-tick));del specs,taps,projected,latent;gc.collect()
   print(json.dumps(dict(stage='timing14_record_complete',n=len(state['completed']),total=87,center=row.center)),flush=True)
  except Exception as error:
   state['failures'].append(dict(subject_id=row.subject_id,center=row.center,error=type(error).__name__+':'+str(error)));print(json.dumps(dict(stage='timing14_record_failed',reason=type(error).__name__)),flush=True)
  atomic(statepath,state)
 state['status']='all_records_attempted';atomic(statepath,state)

def evaluate(original,new,oh,nh,ob,nb,columns):
 combined=pd.concat([original,new],ignore_index=True);centers=combined.center.to_numpy();n=len(original);scores=np.empty((len(combined),2));scores[:,0]=np.r_[ob,nb];heads=np.vstack([oh,nh]);labels=np.empty(len(combined),bool);zvalues=np.empty(len(combined))
 for c in CENTERS:
  source=original.center.ne(c).to_numpy();target=centers==c;_,z,_=source_reference(original.loc[source],combined.loc[target],columns);labels[target]=z<=-1.;zvalues[target]=z;mean=oh[source].mean(0);sd=oh[source].std(0);sd=np.where(sd<1e-8,1.,sd);scores[target,1]=((heads[target]-mean)/sd)@SIGNED
 output={}
 for sample,take in [('additional',np.arange(n,len(combined))),('timing14',np.arange(len(combined)))]:
  f=combined.iloc[take];y=labels[take];s=scores[take];pairs=eligible_pairs(f.age.to_numpy(float),f.center.to_numpy());age,npairs=rank_auc(y,s,pairs);i,j=np.triu_indices(len(f),1);keep=f.center.to_numpy()[i]==f.center.to_numpy()[j];within,_=rank_auc(y,s,(i[keep],j[keep]))
  for k,score in enumerate(['brain_health_risk','stable4']):
   for metric,value in [('ordinary_auc',roc_auc_score(y,s[:,k]) if len(np.unique(y))==2 else np.nan),('within_center_ordinary',within[k]),('age_conditioned',age[k]),('continuous_concordance',continuous_concordance(zvalues[take],s[:,k],pairs))]:output[sample+'__'+score+'__'+metric]=float(value)
  output[sample+'__counts']=dict(n=len(f),cases=int(y.sum()),eligible_pairs=npairs,by_center={c:dict(n=int(f.center.eq(c).sum()),cases=int(y[f.center.eq(c)].sum())) for c in CENTERS})
 return output

def analyze(root,freeze):
 out=root/'cache/timing14';state=json.loads((out/'extraction_state_private.json').read_text());assert state['status']=='all_records_attempted' and state['freeze']['files']==freeze['files']
 original,semantic,names,scores,score_names,_=inputs(root);slow,ok=endpoint(original,next(x for x in ROSTER if x[0]=='attention_slowness'));original['attention_better']=-slow
 cohort=pd.read_csv(out/'additional_cohort_private.csv');cohort=cohort.loc[cohort.subject_id.isin([r['subject_id'] for r in state['completed']])].reset_index(drop=True);nh=[];nb=[];ahi=[];roster={r['subject_id']:r for r in state['completed']}
 for sid in cohort.subject_id:
  path=out/roster[sid]['file'];assert digest(path)==roster[sid]['sha256']
  with np.load(path) as z:
   ns=z['semantic_names'].tolist();nh.append(z['semantic'][[ns.index(h) for h in STABLE4]]);nb.append(-float(z['brain_health_score']));ahi.append(float(z['semantic'][ns.index('pred_regression_f_ahi')]))
 nh=np.array(nh);nb=np.array(nb);oh=semantic[:,[names.index(h) for h in STABLE4]];ob=scores[:,score_names.index('negated_brain_health')];slow,nok=endpoint(cohort,next(x for x in ROSTER if x[0]=='attention_slowness'));cohort['attention_better']=-slow
 destination=root/'results/timing14_fixed_scores';destination.mkdir(exist_ok=True);summary=dict(status='running',freeze=freeze,original_n=590,additional_attempted=87,additional_extracted=len(cohort),extraction_failures=len(state['failures']),analyses={},posthoc=True)
 for ti,(task,columns) in enumerate(TASKS.items()):
  if task=='attention':a,b=ok,nok
  elif task in ['memory','composite']:a=original['CPF_A.status'].eq('C1').to_numpy()&original['CPF_A.valid_code'].eq('V').to_numpy()&original.face_memory_correct.notna().to_numpy();b=cohort.face_memory_status.eq('C1').to_numpy()&cohort.face_memory_validation.eq('V').to_numpy()&cohort.face_memory_correct.notna().to_numpy()
  else:a=np.ones(len(original),bool);b=np.ones(len(cohort),bool)
  f=original.loc[a].reset_index(drop=True);g=cohort.loc[b].reset_index(drop=True);point=evaluate(f,g,oh[a],nh[b],ob[a],nb[b],columns);boot=[]
  groups=lambda h:[np.flatnonzero(h.center.eq(c)) for c in CENTERS]
  for seed in np.random.SeedSequence(2026102811+ti).generate_state(2000):
   rng=np.random.default_rng(seed);resample=lambda h:np.concatenate([rng.choice(i,size=len(i),replace=True) for i in groups(h)]);i=resample(f);j=resample(g);boot.append(evaluate(f.iloc[i].reset_index(drop=True),g.iloc[j].reset_index(drop=True),oh[a][i],nh[b][j],ob[a][i],nb[b][j],columns))
  intervals={}
  for key,value in point.items():
   if isinstance(value,dict):continue
   v=np.array([r[key] for r in boot]);v=v[np.isfinite(v)];ci=dict(valid_replicates=len(v))
   if len(v):
    lo,hi=np.quantile(v,[.025,.975]);ci.update(ci_low=float(lo),ci_high=float(hi))
    if key.endswith('__age_conditioned'):lo,hi=np.quantile(v,[.025/8,1-.025/8]);ci.update(familywise8_low=float(lo),familywise8_high=float(hi))
   intervals[key]=ci
  summary['analyses'][task]=dict(point=point,intervals=intervals);atomic(destination/'summary.json',summary);print(json.dumps(dict(stage='timing14_outcome_complete',task=task)),flush=True)
 # Descriptive AHI check uses the same new participants without outcome search.
 y=pd.to_numeric(cohort.ahi,errors='coerce').to_numpy();s=np.array(ahi);take=np.isfinite(y)&np.isfinite(s);auc,npairs=rank_auc(y[take]>=15,s[take,None],eligible_pairs(cohort.age.to_numpy(float)[take],cohort.center.to_numpy()[take]));summary['additional_ahi_check']=dict(n=int(take.sum()),eligible_pairs=npairs,age_conditioned_auc=float(auc[0]));summary['status']='complete';atomic(destination/'summary.json',summary)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--work',type=Path,required=True);ap.add_argument('--stage',choices=['prepare','extract','analyze','after-carr'],required=True);a=ap.parse_args();freeze=receipt();root=a.data_root.resolve()
 if a.stage=='prepare':prepare(root,freeze)
 elif a.stage=='extract':extract(root,a.work.resolve(),freeze)
 elif a.stage=='analyze':analyze(root,freeze)
 else:
  print('TIMING14_QUEUED_AFTER_CARR',flush=True)
  while True:
   state=root/'cache/carr_taps/state_private.json'
   if state.exists():
    d=json.loads(state.read_text())
    if d['status']=='complete':break
    if d['status']=='partial_or_failures':raise RuntimeError('CARR extraction did not finish cleanly; inspect before running additional extraction')
   time.sleep(30)
  extract(root,a.work.resolve(),freeze);analyze(root,freeze)

if __name__=='__main__':
 with threadpool_limits(limits=1):main()
