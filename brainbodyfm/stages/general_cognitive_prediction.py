"""General low-performance prediction with source-only outcome references."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from analyze import covariate_design,STABLE4
from stable4_extension import CENTERS,load_inputs
import stable4_extension
from age_conditioned_transfer import eligible_pairs,rank_auc,clean_json
import cinc_method_replication as prior

BASE=Path(__file__).resolve().parent
FILES=['GENERAL_COGNITIVE_PREDICTION_CORRECTION.md','general_cognitive_prediction.py','test_general_cognitive_prediction.py']


def source_reference(source,target,columns):
    a=source[list(columns)].to_numpy(float);b=target[list(columns)].to_numpy(float)
    mean=a.mean(axis=0);sd=a.std(axis=0);sd=np.where(sd<1e-8,1.,sd)
    ca=((a-mean)/sd).mean(axis=1);cb=((b-mean)/sd).mean(axis=1)
    cm=float(ca.mean());cs=float(ca.std());cs=cs if cs>=1e-8 else 1.
    return (ca-cm)/cs,(cb-cm)/cs,dict(task_mean=mean.tolist(),task_sd=sd.tolist(),composite_mean=cm,composite_sd=cs)


def predict(frame,heads,bhs,columns):
    centers=frame.center.to_numpy();ages=frame.age.to_numpy(float);sex=frame.sex.to_numpy(float)
    scores=np.empty((len(frame),len(prior.MODELS)));labels=np.empty(len(frame),bool);references={}
    for center in CENTERS:
        te=centers==center;tr=~te
        za,zb,reference=source_reference(frame.loc[tr],frame.loc[te],columns)
        ya=za<=-1.;labels[te]=zb<=-1.
        fixed,artifact=prior.fixed_fold_scores(heads[tr],heads[te],sex[tr],sex[te])
        scores[te,:10]=fixed;scores[te,10]=-bhs[te];scores[te,11]=ages[te]
        mean=heads[tr].mean(0);sd=heads[tr].std(0);sd=np.where(sd<1e-8,1.,sd)
        ha,hb=(heads[tr]-mean)/sd,(heads[te]-mean)/sd
        da,db=covariate_design(frame.loc[tr],frame.loc[te],clinical=False)
        ca,cb=covariate_design(frame.loc[tr],frame.loc[te],clinical=True)
        for name,a,b in [('demographics',da,db),('clinical',ca,cb),('demographics_plus_four',np.c_[da,ha],np.c_[db,hb]),('clinical_plus_four',np.c_[ca,ha],np.c_[cb,hb])]:
            scores[te,prior.MODELS.index(name)]=prior.classifier_predict(a,ya,ages[tr],b)
        references[center]=dict(source_n=int(tr.sum()),source_cases=int(ya.sum()),held_n=int(te.sum()),held_cases=int(labels[te].sum()),**reference)
    return labels,scores,references


def evaluate(frame,heads,bhs,columns,details=False):
    labels,scores,references=predict(frame,heads,bhs,columns)
    centers=frame.center.to_numpy();pairs=eligible_pairs(frame.age.to_numpy(float),centers)
    values,npairs=rank_auc(labels,scores,pairs);bycenter={};counts={}
    for center in CENTERS:
        ix=centers==center;a,b=pairs;keep=ix[a]
        v,n=rank_auc(labels,scores,(a[keep],b[keep]));bycenter[center]=dict(zip(prior.MODELS,map(float,v)))
        counts[center]=dict(n=int(ix.sum()),cases=int(labels[ix].sum()),noncases=int((~labels[ix]).sum()),eligible_pairs=n)
    macro=np.array([[bycenter[c][m] for m in prior.MODELS] for c in CENTERS]).mean(0)
    a,b=np.triu_indices(len(frame),1);keep=centers[a]==centers[b]
    ordinary,_=rank_auc(labels,scores,(a[keep],b[keep]))
    result=dict(age_conditioned=dict(zip(prior.MODELS,map(float,values))),equal_center=dict(zip(prior.MODELS,map(float,macro))),ordinary_within_center=dict(zip(prior.MODELS,map(float,ordinary))))
    result['pooled_ordinary_auc']={m:float(roc_auc_score(labels,scores[:,j])) for j,m in enumerate(prior.MODELS)}
    result['paired_differences']={m+'__minus__'+b:result['age_conditioned'][m]-result['age_conditioned'][b] for m,b in prior.CONTRASTS}
    if details:
        result.update(n=len(frame),cases=int(labels.sum()),eligible_pairs=npairs,counts=counts,by_center=bycenter,source_outcome_references=references)
        return result,labels,pairs,scores
    return result


def bootstrap_one(job):
    frame,heads,bhs,columns,seed=job;rng=np.random.default_rng(seed)
    c=frame.center.to_numpy();ix=np.concatenate([rng.choice(np.flatnonzero(c==v),size=int((c==v).sum()),replace=True) for v in CENTERS])
    with threadpool_limits(limits=1):return evaluate(frame.iloc[ix].reset_index(drop=True),heads[ix],bhs[ix],columns)


def receipt():
    repo=Path(subprocess.check_output(['git','-C',str(BASE),'rev-parse','--show-toplevel'],text=True).strip());sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();files={}
    for name in FILES:
        p=BASE/name;b=subprocess.check_output(['git','-C',str(repo),'show',sha+':'+str(p.relative_to(repo))])
        if b!=p.read_bytes():raise ValueError('Commit corrected plan/code before execution')
        files[name]=hashlib.sha256(b).hexdigest()
    return dict(commit=sha,files=files,execution_started_utc=datetime.now(timezone.utc).isoformat())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--entry-source',type=Path,required=True);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
    freeze=receipt();prior.load_entry(a.entry_source)
    root=a.data_root.resolve();out=root/'results/general_cognitive_prediction';out.mkdir(exist_ok=True)
    stable4_extension.BASE=root;stable4_extension.OUT=root/'results/stable4_extension'
    frame,semantic,_,bhs,names=load_inputs();heads=semantic[:,[names.index(h) for h in STABLE4]]
    summary=dict(status='running',freeze=freeze,models=prior.MODELS,method='source_only_raw_cognitive_reference_no_outcome_covariate_regression',prior_results_known=True,analyses={})
    tasks=[('composite',('pcet_accuracy2','face_memory_correct'),'secondary_face_memory_correct'),('executive_function',('pcet_accuracy2',),'nested_transport'),('face_memory',('face_memory_correct',),'secondary_face_memory_correct')]
    pkeys=[];pvalues=[]
    for ti,(task,columns,folder) in enumerate(tasks):
        ids=pd.read_csv(root/'results/provisional_assumed_uv'/folder/'analysis_subjects.csv').subject_id;use=frame.subject_id.isin(ids).to_numpy()
        f,h,b=frame.loc[use].reset_index(drop=True),heads[use],bhs[use]
        with threadpool_limits(limits=1):result,labels,pairs,scores=evaluate(f,h,b,columns,True)
        print(task,'point',json.dumps({m:result['age_conditioned'][m] for m in ['stable4','demographics','demographics_plus_four','clinical','clinical_plus_four']}),'cases',int(labels.sum()),flush=True)
        np.savez_compressed(out/(task+'_private_predictions.npz'),ids=f.subject_id.to_numpy(str),labels=labels,scores=scores,names=np.array(prior.MODELS))
        seeds=np.random.SeedSequence(2026102761+ti).generate_state(2000);jobs=[(f,h,b,columns,int(seed)) for seed in seeds];boot=[];started=time.time()
        with ProcessPoolExecutor(max_workers=a.workers,initializer=prior.worker_init,initargs=(str(a.entry_source),)) as pool:
            for v in pool.map(bootstrap_one,jobs,chunksize=8):
                boot.append(v)
                if len(boot)%200==0:print(task,'bootstrap',len(boot),'seconds',round(time.time()-started,1),flush=True)
        result['intervals']={};private={}
        for metric in ['age_conditioned','equal_center','ordinary_within_center','pooled_ordinary_auc','paired_differences']:
            result['intervals'][metric]={}
            for model in result[metric]:
                draws=[v[metric][model] for v in boot];ci=prior.intervals(draws)
                if metric=='paired_differences':ci['familywise_95_bonferroni']=prior.intervals(draws,9)
                result['intervals'][metric][model]=ci;private[metric+'__'+model]=np.asarray(draws)
        np.savez_compressed(out/(task+'_private_bootstrap.npz'),**private)
        with threadpool_limits(limits=1):ps=prior.permutation_fixed(labels,scores,pairs,f.center.to_numpy(),10000,2026102771+ti)
        result['permutation_p_one_sided']=dict(zip(prior.FIXED_NAMES,map(float,ps)))
        for m,p in zip(prior.FIXED_NAMES,ps):pkeys.append((task,m));pvalues.append(p)
        summary['analyses'][task]=result;(out/'summary.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False)+'\n')
        print(task,'complete',flush=True)
    for (task,m),p in zip(pkeys,prior.holm(pvalues)):summary['analyses'][task].setdefault('permutation_p_holm_33',{})[m]=float(p)
    summary['status']='complete';summary['completed_utc']=datetime.now(timezone.utc).isoformat();(out/'summary.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False)+'\n')
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
