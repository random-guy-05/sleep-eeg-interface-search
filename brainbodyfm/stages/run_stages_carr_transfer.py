"""Run the predeclared CARR additions only after all 590 audited taps exist."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from analyze import STABLE4
from stable4_extension import CENTERS,load_inputs
import stable4_extension
import cinc_method_replication as submitted
from general_cognitive_prediction import source_reference
from unexpected_cognition_discovery import inputs as expanded_inputs,endpoint,ROSTER
from age_conditioned_transfer import eligible_pairs,rank_auc,clean_json
from sklearn.metrics import roc_auc_score

MODELS=['carr_enabled_system','standalone_carr','clinical','stable4']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_metrics(labels,scores,f):
    age,pairs=rank_auc(labels,scores,eligible_pairs(f.age.to_numpy(float),f.center.to_numpy()))
    return dict(ordinary_auc={m:float(roc_auc_score(labels,scores[:,i])) for i,m in enumerate(MODELS)},age_conditioned=dict(zip(MODELS,map(float,age))),eligible_pairs=pairs)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--entry-source',type=Path,required=True);a=ap.parse_args();root=a.data_root.resolve();cache=root/'cache/carr_taps';state=json.loads((cache/'state_private.json').read_text())
    if state['status']!='complete' or len(state['completed'])!=590 or state['failures']:raise ValueError('All 590 audited CARR tap records must be complete before analysis')
    submitted.load_entry(a.entry_source);stable4_extension.BASE=root;stable4_extension.OUT=root/'results/stable4_extension';frame,semantic,latent,bhs,names=load_inputs()
    roster={r['subject_id']:r for r in state['completed']};global_x=[];temporal=[];cf=[]
    for i,sid in enumerate(frame.subject_id):
        row=roster[sid];path=cache/row['file']
        if digest(path)!=row['sha256']:raise ValueError('CARR feature cache hash mismatch')
        with np.load(path) as z:
            if np.linalg.norm(z['latent']-latent[i])/np.linalg.norm(latent[i])>1e-5:raise ValueError('CARR input latent differs from frozen ordinary input')
            global_x.append(z['global']);temporal.append(z['temporal']);cf.append(z['counterfactual'])
    global_x=np.array(global_x);temporal=np.array(temporal);cf=np.array(cf)
    expanded,*_=expanded_inputs(root);assert expanded.subject_id.tolist()==frame.subject_id.tolist();slow,attention=endpoint(expanded,next(x for x in ROSTER if x[0]=='attention_slowness'));frame['attention_better']=-slow
    out=root/'results/stages_carr_transfer';out.mkdir(exist_ok=True)
    prior_root=root/'results/cinc_trainable_transfer';prior=json.loads((prior_root/'summary.json').read_text())
    tasks=[('composite',('pcet_accuracy2','face_memory_correct'),'secondary_face_memory_correct'),('executive_function',('pcet_accuracy2',),'nested_transport'),('face_memory',('face_memory_correct',),'secondary_face_memory_correct'),('attention_slowness',('attention_better',),None)]
    summary=dict(status='running',tap_freeze=state['freeze'],tap_state_sha256=digest(cache/'state_private.json'),tap_records=590,models=MODELS,analyses={})
    for ti,(task,columns,folder) in enumerate(tasks):
        if folder:
            ids=pd.read_csv(root/'results/provisional_assumed_uv'/folder/'analysis_subjects.csv').subject_id;keep=frame.subject_id.isin(ids).to_numpy()
        else:keep=attention
        ix=np.flatnonzero(keep);f=frame.iloc[ix].reset_index(drop=True);sem=semantic[ix];ll=latent[ix];gg=global_x[ix];tt=temporal[ix];cc=cf[ix];centers=f.center.to_numpy();age=f.age.to_numpy(float);sex=f.sex.to_numpy(float)
        old=np.load(prior_root/(task+'_private_predictions.npz'));assert old['ids'].tolist()==f.subject_id.tolist()
        scores=np.empty((len(f),4));scores[:,2]=old['scores'][:,old['names'].tolist().index('clinical')];scores[:,3]=old['scores'][:,old['names'].tolist().index('stable4')];labels=np.empty(len(f),bool);folds={}
        for center in CENTERS:
            start=time.time();tr=centers!=center;te=~tr;za,zb,reference=source_reference(f.loc[tr],f.loc[te],columns);ya=za<=-1.;labels[te]=zb<=-1.
            print(task,center,'CARR fitting',flush=True)
            with threadpool_limits(limits=1):
                model=submitted.ENTRY.tune_and_fit(names,sem[tr],ya.astype(int),age[tr],sex[tr],strategy='b_primary',tune=True,latent_matrix=ll[tr],carr_global_matrix=gg[tr],carr_temporal_matrix=tt[tr],carr_counterfactual_matrix=cc[tr],sites=centers[tr],verbose=False)
                families={}
                for family,columns_in_family in model.family_columns().items():
                    if family=='derived':
                        gap=submitted.ENTRY.brain_age_gap_batch(names,sem[te],age[te],age_mean=model.age_mean_tr_data,age_std=model.age_std_tr_data)
                        if gap is not None:families[family]=gap[:,None]
                    else:families[family]=sem[te][:,[names.index(h) for h in columns_in_family]]
                features=submitted.ENTRY.carr_feature_matrix(gg[te],tt[te],cc[te]);scores[te,0]=model.predict_scores(families,sex[te],latent=ll[te],age=age[te],carr_features=features)
                stable_indices=[names.index(h) for h in STABLE4];base=submitted.ENTRY._fit_stable4_artifact(sem[tr][:,stable_indices],sex[tr]);source_baseline=base.score_matrix(sem[tr][:,stable_indices],sex[tr]);test_baseline=base.score_matrix(sem[te][:,stable_indices],sex[te])
                carr=submitted.ENTRY.fit_carr_mil(gg[tr],tt[tr],cc[tr],ya.astype(int),age[tr],centers[tr],source_baseline,epochs=80,seed=submitted.ENTRY.SEED+8801);scores[te,1]=carr.score_matrix(features,test_baseline)
            folds[center]=dict(n=int(te.sum()),cases=int(labels[te].sum()),recipe=model.recipe,deployment_streams=model.deployment_streams,carr_accepted=model.carr_accepted,final_gate_pass=model.final_gate_pass,carr_candidate_auc=model.carr_candidate_auc,carr_candidate_min_site_auc=model.carr_candidate_min_site_auc,seconds=time.time()-start)
        np.testing.assert_array_equal(labels,old['labels']);point=score_metrics(labels,scores,f)
        np.savez_compressed(out/(task+'_private_predictions.npz'),ids=f.subject_id.to_numpy(str),labels=labels,scores=scores,names=np.array(MODELS))
        groups=[np.flatnonzero(centers==c) for c in CENTERS];boot=[]
        for seed in np.random.SeedSequence(2026102801+ti).generate_state(2000):
            rng=np.random.default_rng(seed);take=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups]);ff=f.iloc[take].reset_index(drop=True);y=np.empty(len(ff),bool);c=ff.center.to_numpy()
            for center in CENTERS:
                tr=c!=center;te=~tr;_,z,_=source_reference(ff.loc[tr],ff.loc[te],columns);y[te]=z<=-1.
            boot.append(score_metrics(y,scores[take],ff))
        intervals={}
        for metric in ['ordinary_auc','age_conditioned']:
            intervals[metric]={}
            for model in MODELS:
                v=np.array([r[metric][model] for r in boot]);v=v[np.isfinite(v)];lo,hi=np.quantile(v,[.025,.975]);intervals[metric][model]=dict(ci_low=float(lo),ci_high=float(hi),valid_replicates=len(v),conditional_on_fitted_models=True)
        contrasts={}
        for model,base in [('carr_enabled_system','clinical'),('standalone_carr','clinical'),('standalone_carr','stable4')]:
            v=np.array([r['ordinary_auc'][model]-r['ordinary_auc'][base] for r in boot]);q=np.quantile(v,[.025,.975,.025/12,1-.025/12]);contrasts[model+'__minus__'+base]=dict(delta=point['ordinary_auc'][model]-point['ordinary_auc'][base],ci_low=float(q[0]),ci_high=float(q[1]),familywise12_low=float(q[2]),familywise12_high=float(q[3]),conditional_on_fitted_models=True)
        summary['analyses'][task]=dict(n=len(f),cases=int(labels.sum()),point=point,intervals=intervals,contrasts=contrasts,folds=folds)
        (out/'summary.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False)+'\n');print(task,'complete',json.dumps(point['ordinary_auc']),flush=True)
    summary['status']='complete';summary['completed_utc']=datetime.now(timezone.utc).isoformat();(out/'summary.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False)+'\n')
    print('STAGES_CARR_TRANSFER_COMPLETE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
