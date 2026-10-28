"""Paired, same-participant comparison of frozen-score cognitive domains."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from stable4_extension import CENTERS
from analyze import STABLE4
from unexpected_cognition_discovery import inputs,endpoint,ROSTER,SIGNED
from general_cognitive_prediction import source_reference
from age_conditioned_transfer import eligible_pairs,rank_auc,continuous_concordance,clean_json

BASE=Path(__file__).resolve().parent
FILES=['COGNITIVE_DOMAIN_SPECIFICITY_PLAN.md','cognitive_domain_specificity.py']
TASKS={'executive':('pcet_accuracy2',),'memory':('face_memory_correct',),'composite':('pcet_accuracy2','face_memory_correct'),'attention':('attention_better',)}
SCORES=['brain_health_risk','stable4']


def receipt():
    repo=Path(subprocess.check_output(['git','-C',str(BASE),'rev-parse','--show-toplevel'],text=True).strip());sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();files={}
    for name in FILES:
        p=BASE/name;b=subprocess.check_output(['git','-C',str(repo),'show',sha+':'+str(p.relative_to(repo))])
        if b!=p.read_bytes():raise ValueError('Commit new domain comparison before running it')
        files[name]=hashlib.sha256(b).hexdigest()
    return dict(commit=sha,files=files,utc=datetime.now(timezone.utc).isoformat())


def evaluate(f,heads,bhs,details=False):
    centers=f.center.to_numpy();scores=np.empty((len(f),2));scores[:,0]=bhs
    labels={k:np.empty(len(f),bool) for k in TASKS};values={k:np.empty(len(f)) for k in TASKS}
    for c in CENTERS:
        tr=centers!=c;te=~tr;sd=heads[tr].std(0);sd=np.where(sd<1e-8,1.,sd)
        scores[te,1]=((heads[te]-heads[tr].mean(0))/sd)@SIGNED
        for task,cols in TASKS.items():
            _,z,_=source_reference(f.loc[tr],f.loc[te],cols);labels[task][te]=z<=-1.;values[task][te]=z
    pairs=eligible_pairs(f.age.to_numpy(float),centers);a,b=np.triu_indices(len(f),1);keep=centers[a]==centers[b];allpairs=(a[keep],b[keep])
    result={};counts={}
    for task in TASKS:
        y=labels[task];age,n=rank_auc(y,scores,pairs);ordinary,_=rank_auc(y,scores,allpairs)
        for j,score in enumerate(SCORES):
            for metric,value in [('ordinary_auc',roc_auc_score(y,scores[:,j])),('within_center_ordinary',ordinary[j]),('age_conditioned',age[j]),('continuous_concordance',continuous_concordance(values[task],scores[:,j],pairs))]:
                result[score+'__'+task+'__'+metric]=float(value)
        counts[task]=dict(cases=int(y.sum()),noncases=int((~y).sum()),eligible_pairs=n,by_center={c:dict(n=int((centers==c).sum()),cases=int(y[centers==c].sum())) for c in CENTERS})
    for score in SCORES:
        for domain in ['executive','memory','composite']:
            for metric in ['ordinary_auc','within_center_ordinary','age_conditioned','continuous_concordance']:
                result[score+'__attention_minus_'+domain+'__'+metric]=result[score+'__attention__'+metric]-result[score+'__'+domain+'__'+metric]
    return (result,counts,scores,labels) if details else result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);a=ap.parse_args();freeze=receipt();root=a.data_root.resolve();out=root/'results/cognitive_domain_specificity';out.mkdir(exist_ok=True)
    f,sem,names,scores,score_names,normalizer=inputs(root)
    slow,valid=endpoint(f,next(r for r in ROSTER if r[0]=='attention_slowness'))
    memory=f['CPF_A.status'].eq('C1').to_numpy() & f['CPF_A.valid_code'].eq('V').to_numpy() & f.face_memory_correct.notna().to_numpy()
    take=valid&memory;heads=sem[take][:,[names.index(h) for h in STABLE4]];bh=scores[take,score_names.index('negated_brain_health')]
    g=f.loc[take].reset_index(drop=True);g['attention_better']=-slow[take]
    point,counts,predictions,labels=evaluate(g,heads,bh,True)
    np.savez_compressed(out/'paired_private_inputs.npz',ids=g.subject_id.to_numpy(str),scores=predictions,**{k:np.asarray(v) for k,v in labels.items()})
    seeds=np.random.SeedSequence(2026102798).generate_state(2000);groups=[np.flatnonzero(g.center.eq(c)) for c in CENTERS];boot=[];started=time.time()
    for i,seed in enumerate(seeds):
        rng=np.random.default_rng(seed);ix=np.concatenate([rng.choice(group,size=len(group),replace=True) for group in groups]);boot.append(evaluate(g.iloc[ix].reset_index(drop=True),heads[ix],bh[ix]))
        if (i+1)%200==0:print('paired-domain bootstrap',i+1,'seconds',round(time.time()-started,1),flush=True)
    intervals={};private={}
    for metric in point:
        draws=np.array([r[metric] for r in boot]);draws=draws[np.isfinite(draws)];private[metric]=draws
        c=dict(ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975)),valid_replicates=len(draws))
        if 'attention_minus_' in metric:
            if metric.endswith('__age_conditioned'):
                q=np.quantile(draws,[.025/6,1-.025/6]);c['familywise_six_low']=float(q[0]);c['familywise_six_high']=float(q[1])
            q=np.quantile(draws,[.025/24,1-.025/24]);c['familywise_24_low']=float(q[0]);c['familywise_24_high']=float(q[1])
        intervals[metric]=c
    summary=dict(status='complete',freeze=freeze,n=len(g),common_by_center=g.center.value_counts().to_dict(),counts=counts,point=point,intervals=intervals,bootstrap=2000,all_data_fitted_steps_repeated=True,score_weights_frozen=True,posthoc=True,completed_utc=datetime.now(timezone.utc).isoformat())
    np.savez_compressed(out/'paired_private_bootstrap.npz',**private)
    (out/'summary.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:[v,intervals[k]] for k,v in point.items() if 'brain_health' in k and ('age_conditioned' in k or 'continuous_concordance' in k)},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
