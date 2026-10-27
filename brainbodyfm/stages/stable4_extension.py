"""Post hoc STAGES fixed-score comparisons with patient bootstrap refitting.

The prespecified search-gap analysis is left intact. This extension implements
CinC's fixed signed Stable4 combination, source-only normalization and a fitted
cognition link; it does not treat four refitted feature coefficients as Stable4.
Participant-level caches remain local. Only aggregate summaries may be released.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from analyze import covariate_design, ridge_path, choose_alpha, STABLE4, config

BASE=Path(__file__).resolve().parent
OUT=BASE/'results/stable4_extension'
TASKS=[('executive_function','nested_transport','pcet_accuracy2'),('face_memory','secondary_face_memory_correct','face_memory_correct'),('vigilance','secondary_pvt_mean_reciprocal_rt','pvt_mean_reciprocal_rt')]
MODELS=['intercept','demographics','clinical','brain_health','stable4_score','stable4_four_refitted','selected_four']
SIGNED=np.array([.36,.27,-.235,-.135])
CENTERS=['Geisinger','MedSleep','St_Lukes','Stanford']


def load_inputs():
    cohort=pd.read_csv(BASE/'data/cohort_before_signal_qc.csv')
    ids=pd.read_csv(BASE/'results/provisional_assumed_uv/nested_transport/analysis_subjects.csv')
    f=ids.merge(cohort,on=['subject_id','center'],suffixes=('','_source'),validate='one_to_one')
    assert len(f)==590
    dest=OUT/'private_input_cache.npz'
    if dest.exists():
        z=np.load(dest,allow_pickle=False)
        assert z['ids'].tolist()==f.subject_id.tolist()
        return f,z['semantic'],z['latent'],z['bhs'],z['names'].tolist()
    def read(sid):
        with np.load(BASE/'cache/assumed_uv/subjects'/f'{sid}.npz',allow_pickle=False) as z:
            return z['semantic'],z['latent'],float(z['brain_health_score']),z['semantic_names']
    with ThreadPoolExecutor(max_workers=8) as pool:r=list(pool.map(read,f.subject_id))
    names=r[0][3].tolist();assert all(x[3].tolist()==names for x in r)
    semantic=np.array([x[0] for x in r],float);latent=np.array([x[1] for x in r],float);bhs=np.array([x[2] for x in r],float)
    assert np.isfinite(semantic).all() and np.isfinite(latent).all() and np.isfinite(bhs).all()
    np.savez_compressed(dest,ids=f.subject_id.to_numpy(str),semantic=semantic,latent=latent,bhs=bhs,names=np.array(names))
    return f,semantic,latent,bhs,names


def reps_for_split(stable,bhs,selected,train,test):
    # Stable4's feature names/signs/weights are frozen; its source normalization
    # follows the CinC procedure. A scalar ridge link to each task is learned.
    scaler=StandardScaler();sa=scaler.fit_transform(stable[train]);sb=scaler.transform(stable[test])
    fixed_a=sa@SIGNED;fixed_b=sb@SIGNED
    scale=StandardScaler();ba=scale.fit_transform(bhs[train,None]);bb=scale.transform(bhs[test,None])
    scale=StandardScaler();qa=scale.fit_transform(selected[train]);qb=scale.transform(selected[test])
    # Rescale the one-dimensional Stable4 score, preserving its composition.
    scale=StandardScaler();fa=scale.fit_transform(fixed_a[:,None]);fb=scale.transform(fixed_b[:,None])
    return {'brain_health':(ba,bb),'stable4_score':(fa,fb),'stable4_four_refitted':(sa,sb),'selected_four':(qa,qb)}


def evaluate(frame,semantic,bhs,names,selection,weights=None):
    """Resampling repeats preprocessing, penalty tuning, all model fits and scoring.

    The already-source-selected quartet is held fixed for the selected_four arm.
    Its feature-selection uncertainty is explicitly conditional; the fixed-score
    comparison has no within-STAGES feature selection to repeat.
    """
    cfg=config();alphas=np.asarray(cfg['ridge_alpha_grid']);y=frame.target.to_numpy(float);c=frame.center.to_numpy();n=len(frame)
    if weights is None:sample=np.arange(n)
    else:sample=np.repeat(np.arange(n),weights)
    idx={center:sample[c[sample]==center] for center in CENTERS}
    named={v:i for i,v in enumerate(names)};stable=semantic[:,[named[x] for x in STABLE4]]
    errors={m:{} for m in MODELS};predictions={m:np.empty(n) for m in MODELS};penalties={}
    for outer in CENTERS:
        source=[x for x in CENTERS if x!=outer];train=np.concatenate([idx[x] for x in source]);test=idx[outer]
        selected=semantic[:,[named[x] for x in selection[outer]]]
        grid={m:[] for m in MODELS if m!='intercept'}
        for held in source:
            tr=np.concatenate([idx[x] for x in source if x!=held]);te=idx[held]
            da,db=covariate_design(frame.iloc[tr],frame.iloc[te],clinical=False)
            ca,cb=covariate_design(frame.iloc[tr],frame.iloc[te],clinical=True)
            designs={'demographics':(da,db),'clinical':(ca,cb)}
            for m,(a,b) in reps_for_split(stable,bhs,selected,tr,te).items():designs[m]=(np.c_[ca,a],np.c_[cb,b])
            for m,(a,b) in designs.items():grid[m].append(np.mean(np.abs(ridge_path(a,y[tr],b,alphas)-y[te,None]),axis=0))
        chosen={m:float(alphas[choose_alpha(np.mean(v,axis=0),alphas,cfg['tie_tolerance'])]) for m,v in grid.items()}
        da,db=covariate_design(frame.iloc[train],frame.iloc[test],clinical=False);ca,cb=covariate_design(frame.iloc[train],frame.iloc[test],clinical=True)
        designs={'demographics':(da,db),'clinical':(ca,cb)}
        for m,(a,b) in reps_for_split(stable,bhs,selected,train,test).items():designs[m]=(np.c_[ca,a],np.c_[cb,b])
        for m in MODELS:
            pred=np.full(len(test),np.mean(y[train])) if m=='intercept' else ridge_path(*[designs[m][0],y[train],designs[m][1],[chosen[m]]])[:,0]
            errors[m][outer]=float(np.mean(np.abs(pred-y[test])))
            if weights is None:predictions[m][test]=pred
        penalties[outer]=chosen
    equal=np.array([np.mean([errors[m][x] for x in CENTERS]) for m in MODELS]);counts=np.array([len(idx[x]) for x in CENTERS]);patient=np.array([np.average([errors[m][x] for x in CENTERS],weights=counts) for m in MODELS])
    return equal,patient,errors,predictions,penalties


def run_bootstrap_task(args):
    frame,semantic,bhs,names,selection,seed=args
    with threadpool_limits(limits=1):
        rng=np.random.default_rng(seed);c=frame.center.to_numpy();w=np.zeros(len(frame),int)
        for center in CENTERS:
            ix=np.flatnonzero(c==center);w[ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)))
        a,b,*_=evaluate(frame,semantic,bhs,names,selection,w)
    return a,b


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bootstrap',type=int,default=2000);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--task',choices=[x[0] for x in TASKS]+['all'],default='all');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);frame,semantic,latent,bhs,names=load_inputs();bank=json.loads((BASE/'data/candidate_bank.json').read_text())['candidates'];all_results={}
    for ti,(task,folder,column) in enumerate(TASKS):
        if args.task not in ['all',task]:continue
        d=BASE/'results/provisional_assumed_uv'/folder;f=pd.read_csv(d/'analysis_subjects.csv')
        indices=frame.set_index('subject_id').index.get_indexer(f.subject_id);assert np.all(indices>=0)
        g=frame.iloc[indices].reset_index(drop=True).copy();g['target']=f[column].to_numpy();sem=semantic[indices];bh=bhs[indices]
        sel=pd.read_csv(d/'search_budget_selections.csv');selection={c:bank[int(sel.loc[(sel.outer==c)&(sel.budget==500),'draw'].iloc[0])]['heads'] for c in CENTERS}
        with threadpool_limits(limits=1):equal,patient,errors,predictions,penalties=evaluate(g,sem,bh,names,selection)
        original=pd.read_csv(d/'baseline_endpoints.csv')
        for m,old in [('demographics','demographics'),('clinical','clinical'),('brain_health','clinical_plus_frozen_BHS'),('stable4_four_refitted','clinical_plus_historical_Stable4_descriptive')]:
            for c in CENTERS:assert abs(errors[m][c]-original.loc[(original.outer==c)&(original.interface==old),'target_mae'].iloc[0])<1e-6,(task,m,c)
        (OUT/f'{task}_point.json').write_text(json.dumps({'n':len(g),'models':MODELS,'equal_mae':equal.tolist(),'patient_mae':patient.tolist(),'center_mae':errors,'penalties':penalties,'selection':selection},indent=2)+'\n')
        np.savez_compressed(OUT/f'{task}_private_predictions.npz',**predictions)
        print(task,'point',dict(zip(MODELS,np.round(equal,6))),flush=True)
        seeds=np.random.SeedSequence(2026102701+ti).generate_state(args.bootstrap).tolist();bs=[];start=time.time()
        existing=OUT/f'{task}_refit_bootstrap.npz'
        if existing.exists():
            z=np.load(existing);bs=list(zip(z['equal'],z['patient']));assert len(bs)<=args.bootstrap
            np.testing.assert_array_equal(z['seed'],np.array(seeds[:len(bs)]))
        tasks=[(g,sem,bh,names,selection,int(seed)) for seed in seeds[len(bs):]]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(run_bootstrap_task,tasks,chunksize=8):
                bs.append(result)
                if len(bs)%100==0 or len(bs)==args.bootstrap:
                    np.savez_compressed(existing,equal=np.array([v[0] for v in bs]),patient=np.array([v[1] for v in bs]),seed=np.array(seeds[:len(bs)]))
                    print(task,'bootstrap',len(bs),'elapsed',round(time.time()-start,1),flush=True)
        boot_equal=np.array([v[0] for v in bs]);boot_patient=np.array([v[1] for v in bs]);rows=[]
        for weighting,point,boot in [('equal_center',equal,boot_equal),('patient',patient,boot_patient)]:
            for model in ['brain_health','stable4_score','stable4_four_refitted','selected_four']:
                comparators=['demographics','clinical']+(['selected_four'] if model=='stable4_score' else [])
                for comparator in comparators:
                    delta=boot[:,MODELS.index(model)]-boot[:,MODELS.index(comparator)];lo,hi=np.quantile(delta,[.025,.975]);rows.append(dict(task=task,weighting=weighting,model=model,comparator=comparator,n=len(g),delta_mae=float(point[MODELS.index(model)]-point[MODELS.index(comparator)]),ci_low=float(lo),ci_high=float(hi),replicates=len(bs),refit=True,selection_repeated=('selected_four' not in (model,comparator)),selection_scope=('conditional_original_quartet' if 'selected_four' in (model,comparator) else 'no_outcome_feature_selection')))
            # The positive control is observed demographic prediction versus a
            # training-mean intercept; it is empirical and may itself be null.
            delta=boot[:,MODELS.index('demographics')]-boot[:,MODELS.index('intercept')];lo,hi=np.quantile(delta,[.025,.975]);rows.append(dict(task=task,weighting=weighting,model='demographics',comparator='intercept',n=len(g),delta_mae=float(point[1]-point[0]),ci_low=float(lo),ci_high=float(hi),replicates=len(bs),refit=True,selection_repeated=True,selection_scope='no_outcome_feature_selection'))
        pd.DataFrame(rows).to_csv(OUT/f'{task}_paired_refit_summary.csv',index=False)
        print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':main()
