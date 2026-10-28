"""Bounded discovery; commit selections before opening new MedSleep outcomes."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm,t
from threadpoolctl import threadpool_limits
import stable4_extension
from stable4_extension import load_inputs
from analyze import covariate_design,STABLE4

BASE=Path(__file__).resolve().parent
SIGNED=np.array([.360,.270,-.235,-.135])
FILES=['UNEXPECTED_COGNITION_DISCOVERY_PLAN.md','unexpected_cognition_discovery.py','test_unexpected_cognition_discovery.py']
# Endpoint construction is outcome-only and frozen before association tests.
ROSTER=[
 ('executive_concept','PCET_A','negative','PCET_A.PCET_WIS_CLEVEL_RES',True),
 ('executive_perseveration','PCET_A','log1p','PCET_A.PCET_WIS_PER_ER',True),
 ('executive_set_failure','PCET_A','log1p','PCET_A.PCET_WIS_FMS',True),
 ('executive_slowness','PCET_A','log','PCET_A.PCET_RTCR',True),
 ('memory_slowness','CPF_A','log','CPF_A.CPF_RTCR',True),
 ('memory_confident_errors','CPF_A','confidence','',True),
 ('memory_discrimination','CPF_A','dprime','',True),
 ('attention_discrimination','SPCPTNL','dprime','',True),
 ('attention_slowness','SPCPTNL','log','SPCPTNL.SCPT_TPRT',True),
 ('attention_variability','SPCPTNL','cv','',True),
 ('working_accuracy','SLNB12_60','negative','SLNB12_60.SLNB2_MCR',True),
 ('working_slowness','SLNB12_60','log','SLNB12_60.SLNB2_MRTC',True),
 ('working_load_cost','SLNB12_60','load','',True),
 ('vigilance_lapses','PVTB','identity','PVTB.PVTB_500MS_LAPPROB',True),
 ('vigilance_variability','PVTB','cv','',True),
 ('vigilance_slow_tail','PVTB','negative','PVTB.PVTB_MEAN_S10PC_RRT',True),
 ('executive_accuracy_context','PCET_A','negative','PCET_A.PCET_ACC2',False),
 ('memory_accuracy_context','CPF_A','negative','CPF_A.CPF_CR',False),
 ('vigilance_speed_context','PVTB','negative','PVTB.PVTB_MEAN_RRT',False)]


def rank_normal(x):
    return norm.ppf((pd.Series(x).rank(method='average').to_numpy()-.5)/len(x))


def endpoint(frame,entry):
    name,task,transform,column,new=entry
    numeric=lambda k:pd.to_numeric(frame[k],errors='coerce').to_numpy(float)
    if transform in ['identity','negative','log','log1p']:
        y=numeric(column)
        if transform=='negative':y=-y
        elif transform=='log':y=np.log(np.where(y>0,y,np.nan))
        elif transform=='log1p':y=np.log1p(np.where(y>=0,y,np.nan))
    elif transform=='confidence':
        error=numeric('CPF_A.CPF_DFP')+numeric('CPF_A.CPF_DFN')
        denominator=error+numeric('CPF_A.CPF_DTP')+numeric('CPF_A.CPF_DTN')
        y=error/np.where(denominator>0,denominator,np.nan)
    elif transform=='dprime':
        prefix='CPF_A.CPF_' if task=='CPF_A' else 'SPCPTNL.SCPT_'
        hit,miss,fa,cr=[numeric(prefix+k) for k in ['TP','FN','FP','TN']]
        y=-(norm.ppf((hit+.5)/(hit+miss+1))-norm.ppf((fa+.5)/(fa+cr+1)))
        y[(hit+miss<=0)|(fa+cr<=0)]=np.nan
    elif transform=='cv':
        prefix='PVTB.PVTB_' if task=='PVTB' else 'SPCPTNL.SCPT_'
        den=numeric(prefix+'MEAN_RT') if task=='PVTB' else numeric(prefix+'RT_MEAN')
        numerator=numeric(prefix+'STDEV_RT') if task=='PVTB' else numeric(prefix+'RT_STDEV')
        y=numerator/np.where(den>0,den,np.nan)
    elif transform=='load':
        two=numeric('SLNB12_60.SLNB2_RTC2');one=numeric('SLNB12_60.SLNB2_RTC1')
        y=np.log(np.where((two>0)&(one>0),two/one,np.nan))
    status=frame[task+'.status'].fillna('').astype(str).eq('C1').to_numpy()
    code=frame[task+'.valid_code'].fillna('').astype(str)
    valid=code.isin(['0','0.0']).to_numpy() if task in ['PVTB','SLNB12_60'] else code.eq('V').to_numpy()
    skipped=frame[task+'.skipped'].fillna(0).astype(str).isin(['1','1.0','True','true']).to_numpy()
    keep=status&valid&~skipped&np.isfinite(y)
    return y,keep


def nuisance(frame,clinical=True,center=True):
    a,_=covariate_design(frame,frame,clinical=clinical)
    duration=frame.duration_hours.to_numpy(float);duration=np.where(np.isfinite(duration),duration,np.nanmedian(duration))
    columns=[np.ones((len(frame),1)),a]
    if clinical:columns.append(((duration-duration.mean())/max(duration.std(),1e-8))[:,None])
    if center:
        values=sorted(frame.center.unique())
        if len(values)>1:columns.append(np.column_stack([frame.center.eq(c).to_numpy(float) for c in values[1:]]))
    return np.column_stack(columns)


def partial_scan(x,y,design):
    if x.ndim==1:x=x[:,None]
    q,s,_=np.linalg.svd(design,full_matrices=False)
    q=q[:,s>max(design.shape)*np.finfo(float).eps*s.max()]
    xr=x-q@(q.T@x);yr=y-q@(q.T@y)
    xx=np.sum(xr*xr,axis=0);yy=float(yr@yr);xy=xr.T@yr
    r=xy/np.sqrt(np.maximum(xx*yy,1e-30));r=np.clip(r,-1+1e-15,1-1e-15)
    df=len(y)-q.shape[1]-1;stat=r*np.sqrt(df/np.maximum(1-r*r,1e-30));p=2*t.sf(np.abs(stat),df)
    beta=xy/np.maximum(xx,1e-30)*x.std(axis=0)/max(y.std(),1e-12)
    z=np.arctanh(r);se=1/np.sqrt(max(len(y)-q.shape[1]-2,1));lo=np.tanh(z-1.96*se);hi=np.tanh(z+1.96*se)
    return dict(r=r,p=p,beta=beta,ci_low=lo,ci_high=hi,df=df,nuisance_rank=q.shape[1])


def adjust_p(values,method='bh'):
    a=np.asarray(values,float);order=np.argsort(a,kind='stable');out=np.empty(len(a));n=len(a)
    if method=='bh':out[order]=np.minimum(1.,np.minimum.accumulate((a[order]*n/np.arange(1,n+1))[::-1])[::-1])
    else:out[order]=np.minimum(1.,np.maximum.accumulate(a[order]*(n-np.arange(n))))
    return out


def freeze():
    repo=Path(subprocess.check_output(['git','-C',str(BASE),'rev-parse','--show-toplevel'],text=True).strip());sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();files={}
    for name in FILES:
        p=BASE/name;b=subprocess.check_output(['git','-C',str(repo),'show',sha+':'+str(p.relative_to(repo))])
        if b!=p.read_bytes():raise ValueError('Commit discovery plan/code before any association calculation')
        files[name]=hashlib.sha256(b).hexdigest()
    return dict(commit=sha,files=files,utc=datetime.now(timezone.utc).isoformat())


def inputs(root):
    stable4_extension.BASE=root;stable4_extension.OUT=root/'results/stable4_extension'
    f,semantic,latent,bhs,names=load_inputs()
    extra=pd.read_csv(root/'results/unexpected_cognition_discovery/cnb_extra_summary_private.csv')
    f=f.merge(extra,on='subject_id',how='left',validate='one_to_one',sort=False)
    durations=pd.read_csv(root/'data/signal_header_audit.csv')[['subject_id','duration_hours']]
    f=f.merge(durations,on='subject_id',how='left',validate='one_to_one',sort=False)
    assert len(f)==590
    # load_inputs order is preserved by the left merges; verify independently.
    z=np.load(root/'results/stable4_extension/private_input_cache.npz')
    assert f.subject_id.tolist()==z['ids'].tolist()
    discovery=f.center.ne('MedSleep').to_numpy()
    heads=semantic[:,[names.index(h) for h in STABLE4]]
    mu=heads[discovery].mean(0);sd=heads[discovery].std(0);sd=np.where(sd<1e-8,1.,sd)
    fixed=(heads-mu)/sd
    scores=np.column_stack([fixed@SIGNED,-bhs,fixed*np.sign(SIGNED)])
    score_names=['stable4','negated_brain_health']+['signed_component_'+str(i+1) for i in range(4)]
    return f,semantic,names,scores,score_names,dict(mean=mu.tolist(),sd=sd.tolist(),source_n=int(discovery.sum()))


def select_candidates(frame,track):
    frame=frame.sort_values(['p','endpoint_order','marker'],kind='stable')
    selected=[];outcomes=set();families=set()
    for r in frame.itertuples(index=False):
        if r.endpoint in outcomes or (track=='physiology' and r.family in families):continue
        selected.append(dict(endpoint=r.endpoint,marker=r.marker,track=track,family=r.family,
            discovery_r=r.r,discovery_p=r.p,discovery_q=r.q,
            discovery_direction=1 if r.r>0 else -1))
        outcomes.add(r.endpoint);families.add(r.family)
        if len(selected)==3:break
    return selected


def discover(args):
    receipt=freeze();root=args.data_root.resolve();out=root/'results/unexpected_cognition_discovery';out.mkdir(exist_ok=True)
    if (out/'discovery_selection.json').exists():raise ValueError('Discovery selection already exists; do not select again after confirmation')
    f,semantic,names,scores,score_names,normalizer=inputs(root)
    take=f.center.ne('MedSleep').to_numpy();source=f.loc[take].reset_index(drop=True)
    scores=scores[take];semantic=semantic[take]
    taxonomy=pd.read_csv(root.parent/'frozen/head_taxonomy.csv')
    taxonomy=taxonomy[taxonomy.include_physiology_only.astype(str).str.lower().isin(['true','1'])]
    allowed=taxonomy.head_name.tolist();assert len(allowed)==539
    phys_index=[names.index(h) for h in allowed];families=taxonomy.set_index('head_name').physiological_family.to_dict()
    fixedrows=[];physrows=[];availability=[]
    for ei,entry in enumerate(ROSTER):
        y,keep=endpoint(source,entry);g=source.loc[keep].reset_index(drop=True);yy=rank_normal(y[keep])
        availability.append(dict(endpoint=entry[0],new_endpoint=entry[4],n=int(keep.sum()),discovery_by_center=g.center.value_counts().to_dict()))
        if len(g)<50 or np.std(yy)<1e-8:continue
        for adjustment,design in [('unadjusted',np.ones((len(g),1))),('demographic',nuisance(g,False)),('clinical',nuisance(g,True))]:
            result=partial_scan(scores[keep],yy,design)
            for j,name in enumerate(score_names):
                fixedrows.append(dict(endpoint=entry[0],endpoint_order=ei,new_endpoint=entry[4],marker=name,family='fixed_score',adjustment=adjustment,n=len(g),r=float(result['r'][j]),p=float(result['p'][j]),standardized_beta=float(result['beta'][j]),ci_low=float(result['ci_low'][j]),ci_high=float(result['ci_high'][j])))
        if entry[4]:
            result=partial_scan(semantic[keep][:,phys_index],yy,nuisance(g,True))
            for j,name in enumerate(allowed):physrows.append(dict(endpoint=entry[0],endpoint_order=ei,marker=name,family=families[name],n=len(g),r=float(result['r'][j]),p=float(result['p'][j]),standardized_beta=float(result['beta'][j]),ci_low=float(result['ci_low'][j]),ci_high=float(result['ci_high'][j])))
    fixed=pd.DataFrame(fixedrows);phys=pd.DataFrame(physrows)
    test=fixed.new_endpoint & fixed.marker.isin(score_names[:2]) & fixed.adjustment.eq('clinical')
    fixed['q']=np.nan;fixed.loc[test,'q']=adjust_p(fixed.loc[test,'p'])
    phys['q']=adjust_p(phys.p)
    selected=select_candidates(fixed.loc[test],'fixed')+select_candidates(phys,'physiology')
    fixed.to_csv(out/'discovery_fixed_scores.csv',index=False);phys.to_csv(out/'discovery_all_physiology.csv',index=False)
    result=dict(stage='discovery_complete_confirmation_not_read',freeze=receipt,source_normalizer=normalizer,
        discovery_n=len(source),confirmation_n=int((~take).sum()),new_endpoint_count=16,
        fixed_test_family=int(test.sum()),physiology_test_family=len(phys),selected=selected,
        availability=availability,discovery_output_sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['discovery_fixed_scores.csv','discovery_all_physiology.csv']})
    (out/'discovery_selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def confirm(args):
    # This command only runs after the discovery list itself is Git committed.
    freeze();root=args.data_root.resolve();out=root/'results/unexpected_cognition_discovery';selection=json.loads((out/'discovery_selection.json').read_text())
    repo=Path(subprocess.check_output(['git','-C',str(BASE),'rev-parse','--show-toplevel'],text=True).strip())
    selected_path='brainbodyfm/stages/results/aggregate/unexpected_cognition_discovery/discovery_selection.json'
    committed=subprocess.check_output(['git','-C',str(repo),'show','HEAD:'+selected_path])
    if committed!=(out/'discovery_selection.json').read_bytes():raise ValueError('Commit exact discovery selections before MedSleep confirmation')
    sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    f,sem,names,scores,score_names,normalizer=inputs(root);take=f.center.eq('MedSleep').to_numpy()
    target=f.loc[take].reset_index(drop=True);sem=sem[take];scores=scores[take]
    rows=[]
    for item in selection['selected']:
        entry=next(x for x in ROSTER if x[0]==item['endpoint']);y,keep=endpoint(target,entry);g=target.loc[keep].reset_index(drop=True);yy=rank_normal(y[keep])
        marker=scores[keep,score_names.index(item['marker'])] if item['track']=='fixed' else sem[keep,names.index(item['marker'])]
        for adjustment,design in [('unadjusted',np.ones((len(g),1))),('demographic',nuisance(g,False)),('clinical',nuisance(g,True))]:
            r=partial_scan(marker,yy,design)
            rows.append(dict(**item,adjustment=adjustment,n=len(g),r=float(r['r'][0]),p=float(r['p'][0]),standardized_beta=float(r['beta'][0]),ci_low=float(r['ci_low'][0]),ci_high=float(r['ci_high'][0]),same_direction=bool(r['r'][0]*item['discovery_direction']>0)))
    rows=pd.DataFrame(rows);rows['holm_p']=np.nan
    for track in ['fixed','physiology']:
        ix=rows.track.eq(track)&rows.adjustment.eq('clinical');rows.loc[ix,'holm_p']=adjust_p(rows.loc[ix,'p'],'holm')
    rows['replicated']=rows.same_direction & rows.holm_p.lt(.05)
    rows.to_csv(out/'confirmation_selected.csv',index=False)
    summary=dict(stage='confirmation_complete',selection_commit=sha,source_normalizer=normalizer,selected_count=len(selection['selected']),
        fixed_replicated=rows.loc[rows.track.eq('fixed')&rows.replicated].to_dict('records'),
        physiology_replicated=rows.loc[rows.track.eq('physiology')&rows.replicated].to_dict('records'),
        confirmation_rows=rows.replace({np.nan:None}).to_dict('records'),earlier_all_cohort_outcomes_previously_examined=True)
    (out/'confirmation_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(rows[rows.adjustment.eq('clinical')].to_string(index=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['discover','confirm']);ap.add_argument('--data-root',type=Path,required=True);args=ap.parse_args()
    with threadpool_limits(limits=1):discover(args) if args.stage=='discover' else confirm(args)
