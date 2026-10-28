"""Finish every selected discovery with conditional bootstrap and prediction QA."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm,rankdata,t
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from analyze import covariate_design,STABLE4
from age_conditioned_transfer import eligible_pairs,rank_auc,clean_json
from unexpected_cognition_discovery import inputs,endpoint,ROSTER,nuisance,partial_scan,rank_normal,SIGNED


def outcome_reference(source,target):
    """Empirical source normal scores, applied without target-fitted mapping."""
    order=np.argsort(source,kind='stable');x=source[order]
    q=(rankdata(source,method='average')-.5)/len(source)
    unique,index=np.unique(x,return_index=True)
    uq=q[order][index]
    return norm.ppf(q),norm.ppf(np.interp(target,unique,uq,left=.5/len(source),right=1-.5/len(source)))


def covariates(source,target):
    a,b=covariate_design(source,target,clinical=True)
    da=source.duration_hours.to_numpy(float);db=target.duration_hours.to_numpy(float)
    median=float(np.nanmedian(da));da=np.where(np.isfinite(da),da,median);db=np.where(np.isfinite(db),db,median)
    mean=float(da.mean());sd=max(float(da.std()),1e-8)
    return np.c_[np.ones(len(a)),a,(da-mean)/sd],np.c_[np.ones(len(b)),b,(db-mean)/sd]


def score_vector(frame,semantic,names,bhs,item):
    if item['marker']=='stable4':
        head=semantic[:,[names.index(h) for h in STABLE4]];tr=frame.center.ne('MedSleep').to_numpy()
        sd=head[tr].std(0);sd=np.where(sd<1e-8,1.,sd)
        return ((head-head[tr].mean(0))/sd)@SIGNED
    if item['marker']=='negated_brain_health':return bhs
    return semantic[:,names.index(item['marker'])]


def hc3(marker,y,design):
    x=(marker-marker.mean())/max(marker.std(),1e-12);yy=(y-y.mean())/max(y.std(),1e-12)
    full=np.c_[design,x];inverse=np.linalg.pinv(full);coefficients=inverse@yy
    residual=yy-full@coefficients;leverage=np.sum(full*inverse.T,axis=1)
    influence=inverse*(residual/np.maximum(1-leverage,1e-8))[None,:]
    se=float(np.sqrt(np.sum(influence[-1]**2)));df=len(y)-np.linalg.matrix_rank(full)
    beta=float(coefficients[-1]);critical=t.ppf(.975,df)
    return dict(standardized_beta=beta,se_HC3=se,p_HC3=float(2*t.sf(abs(beta/se),df)),
        ci_low=float(beta-critical*se),ci_high=float(beta+critical*se),df=int(df))


def evaluate(f,semantic,names,bhs,selection,extra=False):
    output={}
    for item in selection:
        label=item['track']+'__'+item['marker']+'__'+item['endpoint']
        risk=score_vector(f,semantic,names,bhs,item)
        entry=next(e for e in ROSTER if e[0]==item['endpoint']);y,valid=endpoint(f,entry)
        source=valid&f.center.ne('MedSleep').to_numpy();target=valid&f.center.eq('MedSleep').to_numpy()
        a=f.loc[source].reset_index(drop=True);b=f.loc[target].reset_index(drop=True)
        xa,xb=risk[source],risk[target];ya,yb=y[source],y[target]
        ra=partial_scan(xa,rank_normal(ya),nuisance(a,True));rb=partial_scan(xb,rank_normal(yb),nuisance(b,True))
        result={'discovery_partial_r':float(ra['r'][0]),'confirmation_partial_r':float(rb['r'][0]),'confirmation_beta':float(rb['beta'][0]),'partial_r_drop':float(ra['r'][0]-rb['r'][0])}
        aa,bb=covariates(a,b);za,zb=outcome_reference(ya,yb)
        # Small fixed ridge protects singular nuisance columns and source-only
        # spline extrapolation. The penalty is not tuned to target performance.
        def fit(train,target):
            mean=train.mean(0);scale=train.std(0);scale=np.where(scale<1e-8,1.,scale)
            tr=(train-mean)/scale;te=(target-mean)/scale
            design=np.c_[np.ones(len(tr)),tr];other=np.c_[np.ones(len(te)),te]
            penalty=np.eye(design.shape[1]);penalty[0,0]=0.
            return other@np.linalg.solve(design.T@design+penalty,design.T@za)
        baseline=fit(aa[:,1:],bb[:,1:]);augmented=fit(np.c_[aa[:,1:],xa],np.c_[bb[:,1:],xb])
        cutoff=float(np.quantile(ya,.841344746));labels=yb>=cutoff
        oriented=xb if item['track']=='fixed' else xb*item['discovery_direction']
        scores=np.c_[oriented,baseline,augmented]
        if labels.any() and (~labels).any():
            auc=np.array([roc_auc_score(labels,scores[:,i]) for i in range(3)])
        else:auc=np.full(3,np.nan)
        pairs=eligible_pairs(b.age.to_numpy(float),b.center.to_numpy());ageauc,npairs=rank_auc(labels,scores,pairs)
        result.update(marker_ordinary_auc=float(auc[0]),clinical_ordinary_auc=float(auc[1]),augmented_ordinary_auc=float(auc[2]),delta_ordinary_auc=float(auc[2]-auc[1]),
            marker_age_conditioned_auc=float(ageauc[0]),clinical_age_conditioned_auc=float(ageauc[1]),augmented_age_conditioned_auc=float(ageauc[2]),delta_age_conditioned_auc=float(ageauc[2]-ageauc[1]),
            clinical_mae=float(np.mean(np.abs(zb-baseline))),augmented_mae=float(np.mean(np.abs(zb-augmented))),delta_mae=float(np.mean(np.abs(zb-augmented))-np.mean(np.abs(zb-baseline))))
        if extra:
            result.update(discovery_n=int(source.sum()),confirmation_n=int(target.sum()),source_threshold=cutoff,confirmation_cases=int(labels.sum()),eligible_pairs=npairs,
                discovery_HC3=hc3(xa,rank_normal(ya),nuisance(a,True)),confirmation_HC3=hc3(xb,rank_normal(yb),nuisance(b,True)),centers={})
            for c in sorted(f.center.unique()):
                ix=valid&f.center.eq(c).to_numpy();g=f.loc[ix].reset_index(drop=True);design=nuisance(g,True)
                if len(g)-np.linalg.matrix_rank(design)<12:
                    result['centers'][c]=dict(n=len(g),status='too_few_residual_degrees_of_freedom');continue
                estimate=partial_scan(risk[ix],rank_normal(y[ix]),design)
                result['centers'][c]=dict(n=len(g),r=float(estimate['r'][0]),p=float(estimate['p'][0]),ci_low=float(estimate['ci_low'][0]),ci_high=float(estimate['ci_high'][0]))
        output[label]=result
    return output


def bootstrap(job):
    frame,semantic,names,bhs,selection,seed=job;rng=np.random.default_rng(seed)
    ix=np.concatenate([rng.choice(np.flatnonzero(frame.center.eq(c)),size=int(frame.center.eq(c).sum()),replace=True) for c in sorted(frame.center.unique())])
    with threadpool_limits(limits=1):return evaluate(frame.iloc[ix].reset_index(drop=True),semantic[ix],names,bhs[ix],selection)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
    root=a.data_root.resolve();out=root/'results/unexpected_cognition_discovery';plan=json.loads((out/'discovery_selection.json').read_text())
    f,sem,names,scores,score_names,normalizer=inputs(root);bhs=scores[:,score_names.index('negated_brain_health')]
    with threadpool_limits(limits=1):points=evaluate(f,sem,names,bhs,plan['selected'],True)
    seeds=np.random.SeedSequence(2026102781).generate_state(2000);jobs=[(f,sem,names,bhs,plan['selected'],int(seed)) for seed in seeds]
    draws=[];started=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for value in pool.map(bootstrap,jobs,chunksize=8):
            draws.append(value)
            if len(draws)%100==0:print('selected bootstrap',len(draws),'seconds',round(time.time()-started,1),flush=True)
    summary=dict(stage='inference_complete',bootstrap=2000,seed=2026102781,
        selection_conditioned=True,selection_freeze=plan['freeze'],source_normalizer=normalizer,
        predictive_ridge_penalty=1.,prediction_context='source-fitted outcome-normal-score model; clinical with or without frozen selected marker',
        results=points,completed_utc=datetime.now(timezone.utc).isoformat())
    arrays={}
    for key,r in points.items():
        r['bootstrap_intervals']={}
        for metric in draws[0][key]:
            values=np.array([v[key][metric] for v in draws]);arrays[key+'__'+metric]=values;good=values[np.isfinite(values)]
            r['bootstrap_intervals'][metric]=dict(ci_low=float(np.quantile(good,.025)) if len(good) else None,ci_high=float(np.quantile(good,.975)) if len(good) else None,valid_replicates=len(good),requested_replicates=2000)
    np.savez_compressed(out/'selected_private_bootstrap.npz',**arrays)
    (out/'selected_inference_summary.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False)+'\n')
    print('inference complete',flush=True)


if __name__=='__main__':main()
