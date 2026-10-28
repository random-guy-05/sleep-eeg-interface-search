"""Evaluate exact available CinC branches and nested full-representation readouts."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from analyze import covariate_design,STABLE4
import stable4_extension
from stable4_extension import CENTERS,load_inputs
import cinc_method_replication as original
from general_cognitive_prediction import source_reference
from age_conditioned_transfer import eligible_pairs,rank_auc,clean_json
from unexpected_cognition_discovery import inputs as expanded_inputs,endpoint,ROSTER

BASE=Path(__file__).resolve().parent
L2S=[.01,.1,1.,10.]
READOUTS=['cognition23','physiology539','latent1024','all841']
MODELS=['age','demographics','clinical','negated_brain_health','stable4','submitted_available','submitted_latent']+['clinical_plus_'+n for n in READOUTS]+['source_selected_linear']
FILES=['CINC_TRAINABLE_TRANSFER_PLAN.md','cinc_trainable_transfer.py','test_cinc_trainable_transfer.py']


def receipt():
    repo=Path(subprocess.check_output(['git','-C',str(BASE),'rev-parse','--show-toplevel'],text=True).strip());sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();files={}
    for name in FILES:
        p=BASE/name;b=subprocess.check_output(['git','-C',str(repo),'show',sha+':'+str(p.relative_to(repo))])
        if b!=p.read_bytes():raise ValueError('Commit new method plan/code before execution: '+name)
        files[name]=hashlib.sha256(b).hexdigest()
    return dict(commit=sha,files=files,utc=datetime.now(timezone.utc).isoformat())


def label_split(frame,train,test,columns):
    ya,yb,reference=source_reference(frame.iloc[train],frame.iloc[test],columns)
    return ya<=-1.,yb<=-1.,reference


def design_split(frame,representation,train,test,clinical):
    a,b=covariate_design(frame.iloc[train],frame.iloc[test],clinical=clinical)
    if representation is not None:
        x=representation[train];mean=x.mean(0);sd=x.std(0);sd=np.where(sd<1e-8,1.,sd)
        a=np.c_[a,(x-mean)/sd];b=np.c_[b,(representation[test]-mean)/sd]
    return a,b


def fit_predict(a,y,age,b,l2):
    w,intercept,_,_=original.ENTRY.fit_linear(a,y.astype(float),age,age_window=2.,rank_weight=0.,bce_weight=1.,l2=l2)
    return b@w+intercept


def choose_penalty(values):
    values=np.asarray(values,float);best=np.nanmax(values)
    return max(i for i,v in enumerate(values) if np.isfinite(v) and v>=best-1e-6)


def fit_linear_models(frame,reps,train,test,columns):
    center=frame.center.to_numpy();age=frame.age.to_numpy(float)
    configs=[('demographics',None,False),('clinical',None,True)]+[('clinical_plus_'+n,reps[n],True) for n in READOUTS]
    tuning={name:[] for name,_,_ in configs}
    for held in sorted(set(center[train])):
        inner_train=train[center[train]!=held];inner_test=train[center[train]==held]
        ya,yb,_=label_split(frame,inner_train,inner_test,columns)
        for name,rep,clinical in configs:
            a,b=design_split(frame,rep,inner_train,inner_test,clinical)
            aucs=[]
            for l2 in L2S:
                score=fit_predict(a,ya,age[inner_train],b,l2)
                aucs.append(float(roc_auc_score(yb,score)) if len(np.unique(yb))==2 else np.nan)
            tuning[name].append(aucs)
    ya,yb,reference=label_split(frame,train,test,columns);predictions={};details={}
    for name,rep,clinical in configs:
        mean=np.nanmean(tuning[name],axis=0);index=choose_penalty(mean);penalty=L2S[index]
        a,b=design_split(frame,rep,train,test,clinical)
        predictions[name]=fit_predict(a,ya,age[train],b,penalty)
        details[name]=dict(l2=penalty,inner_equal_center_ordinary_auc=float(mean[index]),inner_auc_grid=mean.tolist())
    # Selection is exclusively on source-center inner results.
    selected=READOUTS[0];best=-np.inf
    for name in READOUTS:
        value=details['clinical_plus_'+name]['inner_equal_center_ordinary_auc']
        if value>best+1e-6:best=value;selected=name
    predictions['source_selected_linear']=predictions['clinical_plus_'+selected]
    details['source_selected_linear']=dict(selected_representation=selected,inner_ordinary_auc=best)
    return predictions,details,ya,yb,reference


def submitted_predict(frame,semantic,latent,names,train,test,labels):
    model=original.ENTRY.tune_and_fit(names,semantic[train],labels.astype(int),frame.age.to_numpy(float)[train],frame.sex.to_numpy(float)[train],
        strategy='b_primary',tune=True,latent_matrix=latent[train],sites=frame.center.to_numpy()[train],verbose=False)
    family={}
    for key,heads in model.family_columns().items():
        if key=='derived':
            gap=original.ENTRY.brain_age_gap_batch(names,semantic[test],frame.age.to_numpy(float)[test],age_mean=model.age_mean_tr_data,age_std=model.age_std_tr_data)
            if gap is not None:family[key]=gap[:,None]
        else:family[key]=semantic[test][:,[names.index(h) for h in heads]]
    score=model.predict_scores(family,frame.sex.to_numpy(float)[test],latent=latent[test],age=frame.age.to_numpy(float)[test])
    detail={k:getattr(model,k) for k in ['recipe','deployment_streams','fusion_weights','site_loso_auc','min_site_auc','stable4_accepted','latent_accepted','b_rem_gate_pass','final_gate_pass']}
    # Standalone latent uses exact submitted defaults and final seed, not a
    # selected initialization or a new architecture.
    native=original.ENTRY.fit_latent_ranker(latent[train],labels.astype(int),frame.age.to_numpy(float)[train],sites=frame.center.to_numpy()[train],
        epochs=60,seed=original.ENTRY.SEED+991)
    standalone=native.score_matrix(latent[test])
    return score,standalone,detail


def metrics(labels,scores,ages,centers):
    ordinary={m:float(roc_auc_score(labels,scores[:,i])) for i,m in enumerate(MODELS)}
    ap={m:float(average_precision_score(labels,scores[:,i])) for i,m in enumerate(MODELS)}
    pairs=eligible_pairs(ages,centers);age_auc,npairs=rank_auc(labels,scores,pairs)
    a,b=np.triu_indices(len(labels),1);keep=centers[a]==centers[b];within,_=rank_auc(labels,scores,(a[keep],b[keep]))
    return dict(ordinary_auc=ordinary,within_center_ordinary=dict(zip(MODELS,map(float,within))),age_conditioned=dict(zip(MODELS,map(float,age_auc))),average_precision=ap,eligible_pairs=npairs)


def conditional_bootstrap(frame,scores,columns,seeds):
    """Repeat reference labels; condition on fitted predictions and selection."""
    centers=frame.center.to_numpy();groups=[np.flatnonzero(centers==c) for c in CENTERS];output=[]
    for seed in seeds:
        rng=np.random.default_rng(seed);ix=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups]);f=frame.iloc[ix].reset_index(drop=True);s=scores[ix];c=f.center.to_numpy();y=np.empty(len(f),bool)
        for center in CENTERS:
            te=np.flatnonzero(c==center);tr=np.flatnonzero(c!=center);_,yb,_=label_split(f,tr,te,columns);y[te]=yb
        output.append(metrics(y,s,f.age.to_numpy(float),c))
    return output


def diagnostics(root,out,frame,semantic,latent,bhs,names,reps,freeze):
    ids=pd.read_csv(root/'results/provisional_assumed_uv/secondary_face_memory_correct/analysis_subjects.csv').subject_id
    keep=frame.subject_id.isin(ids).to_numpy();f=frame.loc[keep].reset_index(drop=True);sem=semantic[keep];ll=latent[keep];bh=bhs[keep];rr={k:v[keep] for k,v in reps.items()}
    cv=StratifiedKFold(n_splits=4,shuffle=True,random_state=2026102795);center=f.center.to_numpy();columns=('pcet_accuracy2','face_memory_correct');scores=np.empty((len(f),len(MODELS)));labels=np.empty(len(f),bool);folds={}
    center_names=['demographics','clinical','cognition23','physiology539','latent1024'];center_scores={k:np.empty(len(f),object) for k in center_names}
    for fold,(tr,te) in enumerate(cv.split(np.zeros(len(f)),center)):
        print('patient_fold',fold,'fitting',flush=True)
        with threadpool_limits(limits=1):
            fitted,detail,ya,yb,reference=fit_linear_models(f,rr,tr,te,columns)
            stable,_=original.fixed_fold_scores(sem[tr][:,[names.index(h) for h in STABLE4]],sem[te][:,[names.index(h) for h in STABLE4]],f.sex.to_numpy(float)[tr],f.sex.to_numpy(float)[te])
            fitted.update(age=f.age.to_numpy(float)[te],negated_brain_health=-bh[te],stable4=stable[:,0])
            ss,nn,source_detail=submitted_predict(f,sem,ll,names,tr,te,ya);fitted.update(submitted_available=ss,submitted_latent=nn)
            for name,value in fitted.items():scores[te,MODELS.index(name)]=value
            labels[te]=yb;folds[str(fold)]=dict(train_n=len(tr),test_n=len(te),cases=int(yb.sum()),reference=reference,linear_tuning=detail,submitted_source_selection=source_detail)
            for name in center_names:
                if name in ['demographics','clinical']:a,b=covariate_design(f.iloc[tr],f.iloc[te],clinical=name=='clinical')
                else:
                    scaler=StandardScaler();a=scaler.fit_transform(rr[name][tr]);b=scaler.transform(rr[name][te])
                classifier=LogisticRegression(C=1.,max_iter=2000);classifier.fit(a,center[tr]);center_scores[name][te]=classifier.predict(b)
        print('patient_fold',fold,'complete',flush=True)
    result=metrics(labels,scores,f.age.to_numpy(float),center);result.update(n=len(f),cases=int(labels.sum()),folds=folds)
    site={k:dict(accuracy=float(accuracy_score(center,pred)),balanced_accuracy=float(balanced_accuracy_score(center,pred)),confusion_matrix=confusion_matrix(center,pred,labels=CENTERS).tolist(),class_order=CENTERS) for k,pred in center_scores.items()}
    np.savez_compressed(out/'patient_cv_private_predictions.npz',ids=f.subject_id.to_numpy(str),labels=labels,scores=scores)
    (out/'diagnostics_summary.json').write_text(json.dumps(clean_json(dict(patient_cv=result,center_prediction=site,freeze=freeze)),indent=2,allow_nan=False)+'\n')
    print('DIAGNOSTICS_COMPLETE',json.dumps(result['ordinary_auc']),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--entry-source',type=Path,required=True);ap.add_argument('--bootstrap',type=int,default=2000);ap.add_argument('--stage',choices=['main','diagnostics'],default='main');args=ap.parse_args()
    if args.bootstrap!=2000:raise ValueError('The frozen conditional-bootstrap count is 2000')
    freeze=receipt();original.load_entry(args.entry_source);root=args.data_root.resolve();out=root/'results/cinc_trainable_transfer';out.mkdir(exist_ok=True)
    stable4_extension.BASE=root;stable4_extension.OUT=root/'results/stable4_extension';frame,sem,latent,bhs,names=load_inputs()
    taxonomy=pd.read_csv(root.parent/'frozen/head_taxonomy.csv');phys=taxonomy.loc[taxonomy.include_physiology_only.astype(str).str.lower().isin(['true','1']),'head_name'].tolist();cog=taxonomy.loc[taxonomy.category.eq('target_proximal_clinical_cognitive'),'head_name'].tolist()
    assert len(phys)==539 and len(cog)==23
    reps={'cognition23':sem[:,[names.index(n) for n in cog]],'physiology539':sem[:,[names.index(n) for n in phys]],'latent1024':latent,'all841':sem}
    if args.stage=='diagnostics':
        diagnostics(root,out,frame,sem,latent,bhs,names,reps,freeze);return
    expanded,*_=expanded_inputs(root);assert expanded.subject_id.tolist()==frame.subject_id.tolist()
    y,attention_ok=endpoint(expanded,next(x for x in ROSTER if x[0]=='attention_slowness'));frame['attention_better']=-y
    tasks=[('composite',('pcet_accuracy2','face_memory_correct'),'secondary_face_memory_correct'),('executive_function',('pcet_accuracy2',),'nested_transport'),('face_memory',('face_memory_correct',),'secondary_face_memory_correct'),('attention_slowness',('attention_better',),None)]
    summary=dict(status='running',freeze=freeze,entry_commit='76a36e54251cfce85889a3b2390018ffc23329ee',models=MODELS,analyses={},reused_STAGES_cohort=True,original_hidden_site_scores_not_reassigned_to_new_models=True)
    (out/'execution_receipt.json').write_text(json.dumps(freeze,indent=2)+'\n')
    for ti,(task,columns,folder) in enumerate(tasks):
        cache=out/(task+'_private_predictions.npz');details_file=out/(task+'_point.json')
        if cache.exists() and details_file.exists():
            d=json.loads(details_file.read_text());assert d['freeze']==freeze['files'];z=np.load(cache);ix=pd.Index(frame.subject_id).get_indexer(z['ids']);f=frame.iloc[ix].reset_index(drop=True);scores=z['scores'];labels=z['labels'];points=d['point'];folds=d['folds']
        else:
            if folder:
                ids=pd.read_csv(root/'results/provisional_assumed_uv'/folder/'analysis_subjects.csv').subject_id;keep=frame.subject_id.isin(ids).to_numpy()
            else:keep=attention_ok
            ix=np.flatnonzero(keep);f=frame.iloc[ix].reset_index(drop=True);semantic=sem[ix];ll=latent[ix];bh=bhs[ix];rr={k:v[ix] for k,v in reps.items()}
            scores=np.empty((len(f),len(MODELS)));labels=np.empty(len(f),bool);folds={};center=f.center.to_numpy()
            for outer in CENTERS:
                start=time.time();tr=np.flatnonzero(center!=outer);te=np.flatnonzero(center==outer)
                print(task,outer,'linear fits',flush=True)
                with threadpool_limits(limits=1):
                    fitted,detail,ya,yb,reference=fit_linear_models(f,rr,tr,te,columns)
                    stable,_=original.fixed_fold_scores(semantic[tr][:,[names.index(h) for h in STABLE4]],semantic[te][:,[names.index(h) for h in STABLE4]],f.sex.to_numpy(float)[tr],f.sex.to_numpy(float)[te])
                    fitted.update(age=f.age.to_numpy(float)[te],negated_brain_health=-bh[te],stable4=stable[:,0])
                    print(task,outer,'exact submitted fitting',flush=True)
                    submitted,native,source_detail=submitted_predict(f,semantic,ll,names,tr,te,ya)
                    fitted.update(submitted_available=submitted,submitted_latent=native)
                for name,value in fitted.items():scores[te,MODELS.index(name)]=value
                labels[te]=yb;folds[outer]=dict(source_n=len(tr),held_n=len(te),source_cases=int(ya.sum()),held_cases=int(yb.sum()),outcome_reference=reference,linear_tuning=detail,submitted_source_selection=source_detail,seconds=time.time()-start)
                print(task,outer,'held metrics',json.dumps({m:float(roc_auc_score(yb,scores[te,MODELS.index(m)])) for m in ['clinical','submitted_available','submitted_latent','source_selected_linear']}),'seconds',round(time.time()-start,1),flush=True)
            points=metrics(labels,scores,f.age.to_numpy(float),center);points.update(n=len(f),cases=int(labels.sum()))
            points['by_center']={}
            for c in CENTERS:
                use=center==c;points['by_center'][c]=metrics(labels[use],scores[use],f.age.to_numpy(float)[use],center[use])
            np.savez_compressed(cache,ids=f.subject_id.to_numpy(str),scores=scores,labels=labels,names=np.array(MODELS))
            details_file.write_text(json.dumps(clean_json(dict(freeze=freeze['files'],point=points,folds=folds)),indent=2,allow_nan=False)+'\n')
        print(task,'all held-center ordinary',json.dumps(points['ordinary_auc']),flush=True)
        with threadpool_limits(limits=1):draws=conditional_bootstrap(f,scores,columns,np.random.SeedSequence(2026102791+ti).generate_state(2000))
        intervals={};arrays={}
        for metric in ['ordinary_auc','within_center_ordinary','age_conditioned','average_precision']:
            intervals[metric]={}
            for m in MODELS:
                values=np.array([v[metric][m] for v in draws]);values=values[np.isfinite(values)];intervals[metric][m]=dict(ci_low=float(np.quantile(values,.025)),ci_high=float(np.quantile(values,.975)),valid_replicates=len(values),conditional_on_fitted_models=True);arrays[metric+'__'+m]=values
        contrasts={}
        for m in ['submitted_available','source_selected_linear']:
            d=np.array([v['ordinary_auc'][m]-v['ordinary_auc']['clinical'] for v in draws]);lo,hi=np.quantile(d,[.025,.975]);fl,fh=np.quantile(d,[.025/8,1-.025/8]);contrasts[m+'__minus__clinical']=dict(delta=points['ordinary_auc'][m]-points['ordinary_auc']['clinical'],ci_low=float(lo),ci_high=float(hi),familywise_low=float(fl),familywise_high=float(fh),conditional_on_fitted_models=True)
        signals=[m for m in ['submitted_available','source_selected_linear'] if task=='composite' and (points['ordinary_auc'][m]>=.65 or contrasts[m+'__minus__clinical']['ci_low']>0)]
        summary['analyses'][task]=dict(point=points,intervals=intervals,contrasts=contrasts,folds=folds,requires_full_refit_followup=signals)
        np.savez_compressed(out/(task+'_private_conditional_bootstrap.npz'),**arrays)
        (out/'summary.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False)+'\n');print(task,'complete',flush=True)
    summary['status']='complete';summary['completed_utc']=datetime.now(timezone.utc).isoformat();(out/'summary.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False)+'\n');print('COMPLETE',flush=True)


if __name__=='__main__':main()
