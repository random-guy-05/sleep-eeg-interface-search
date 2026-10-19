"""Paired participant/run uncertainty and fixed-selection metric sensitivity."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

PACKAGE=Path(__file__).resolve().parents[2]
REV=PACKAGE/'ieee_access_revision';OUT=REV/'results'
SITES=('I0002','I0006','S0001')
DRAWS=50000; SEED=2026101801

def pair_surface(y, ages, scores, tolerance):
    """One participant pair surface; ties receive half credit."""
    y=np.asarray(y);ages=np.asarray(ages);scores=np.asarray(scores)
    if not np.isfinite(scores).all():raise ValueError('Nonfinite score')
    p=np.flatnonzero(y==1);n=np.flatnonzero(y==0)
    eligible=(np.abs(ages[p,None]-ages[None,n])<=tolerance).astype(float)
    d=scores[p,None]-scores[None,n]
    return ((d>0)+.5*(d==0))*eligible,eligible

def participant_draws(surfaces, eligible, rng, draws=DRAWS):
    """Subject resampling via multiplicities; keep matched arms paired."""
    if eligible.sum()==0:raise ValueError('No eligible age-matched pairs')
    np_,nn=eligible.shape
    if not np_ or not nn:raise ValueError('Both outcome classes required')
    output={k:np.empty(draws) for k in surfaces};cursor=0;invalid=0
    while cursor<draws:
        size=min(1000,draws-cursor)
        wp=rng.multinomial(np_,np.full(np_,1/np_),size=size)
        wn=rng.multinomial(nn,np.full(nn,1/nn),size=size)
        denom=np.einsum('bi,ij,bj->b',wp,eligible,wn,optimize=True)
        good=denom>0;invalid+=int((~good).sum());take=int(good.sum())
        for name,surface in surfaces.items():
            numer=np.einsum('bi,ij,bj->b',wp,surface,wn,optimize=True)
            output[name][cursor:cursor+take]=numer[good]/denom[good]
        cursor+=take
    return output,invalid

def summary(draws):
    return dict(ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975)))

def targets():
    x=pd.read_parquet(PACKAGE/'sealed/target_labels.parquet')
    return {s:x[x.held_site.eq(s)].sort_values('subject_id').reset_index(drop=True) for s in SITES}

def surfaces_for(site, target, pred, selections, tolerance, group):
    surf={};eligible=None
    for name,g in selections.groupby(group):
        sums=[]
        for r in g.itertuples():
            if group=='method':
                sub=pred[(pred.held_site.eq(site))&(pred.quartet_key.eq(r.quartet_key))&np.isclose(pred.selected_C,r.selected_C)]
            else:
                sub=pred[(pred.site.eq(site))&(pred.arm.eq(name))&(pred.seed.eq(r.seed))]
            score=sub.assign(subject_id=sub.subject_id.astype(str)).set_index('subject_id').score.reindex(target.subject_id.astype(str)).to_numpy()
            if not np.isfinite(score).all():raise ValueError(f'Predictions missing for {site}/{name}')
            win,eligible=pair_surface(target.outcome,target.age,score,tolerance)
            sums.append(win)
        surf[name]=np.mean(sums,axis=0)
    return surf,eligible

def sensitivity():
    metrics=pd.read_csv(PACKAGE/'analysis/output/final_selection_metrics.csv')
    mask=(metrics.budget.eq(10)&(((metrics.method.eq('full_agent'))&metrics.selector.eq('robust'))|
          ((metrics.method.eq('random_source_robust'))&metrics.selector.eq('source_robust'))|
          ((metrics.method.isin(['taxonomy_diverse_robust','evolutionary_robust']))&metrics.selector.eq('robust'))))
    selected=metrics[mask]
    pred=pd.read_parquet(PACKAGE/'analysis/output/participant_predictions.parquet')
    ts=targets();rows=[];contrasts=[];invalids=[]
    for tol in [1.,2.,3.,float('inf')]:
        rng=np.random.default_rng(SEED+int(tol if np.isfinite(tol) else 99))
        macro={};site_diffs={}
        for site in SITES:
            surf,e=surfaces_for(site,ts[site],pred,selected[selected.held_site.eq(site)],tol,'method')
            boot,invalid=participant_draws(surf,e,rng)
            invalids.append(dict(tolerance=str(tol),site=site,rejected_zero_pair_draws=invalid,accepted_draws=DRAWS))
            for name,w in surf.items():
                value=float(w.sum()/e.sum())
                rows.append(dict(tolerance=str(tol),site=site,method=name,auc=value,eligible_pairs=int(e.sum()),**summary(boot[name])))
                macro.setdefault(name,[]).append(boot[name])
            for comp in surf:
                if comp=='full_agent':continue
                diff=boot['full_agent']-boot[comp]
                contrasts.append(dict(tolerance=str(tol),site=site,comparison='full_agent_minus_'+comp,difference=float((surf['full_agent']-surf[comp]).sum()/e.sum()),**summary(diff)))
        for name,arrays in macro.items():
            value=np.mean([r['auc'] for r in rows if r['tolerance']==str(tol) and r['method']==name and r['site'] in SITES])
            rows.append(dict(tolerance=str(tol),site='macro',method=name,auc=float(value),eligible_pairs=None,**summary(np.mean(arrays,axis=0))))
        for comp in macro:
            if comp=='full_agent':continue
            diff=np.mean(macro['full_agent'],axis=0)-np.mean(macro[comp],axis=0)
            points={r['method']:r['auc'] for r in rows if r['site']=='macro' and r['tolerance']==str(tol)}
            contrasts.append(dict(tolerance=str(tol),site='macro',comparison='full_agent_minus_'+comp,difference=points['full_agent']-points[comp],**summary(diff)))
        print('Sensitivity tolerance',tol,'complete',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'metric_sensitivity.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'metric_sensitivity_contrasts.csv',index=False)
    pd.DataFrame(invalids).to_csv(OUT/'metric_sensitivity_resampling.csv',index=False)

def feedback():
    results=pd.read_csv(OUT/'paired_target_results.csv')
    pred=pd.read_parquet(OUT/'paired_predictions_private.parquet')
    ts=targets();rng=np.random.default_rng(SEED);macro=[];rows=[];run_macro=[]
    for site in SITES:
        selected=results[results.site.eq(site)]
        surf,e=surfaces_for(site,ts[site],pred,selected,2.,'arm')
        boot,invalid=participant_draws(surf,e,rng)
        diff=boot['feedback']-boot['no_feedback'];macro.append(diff)
        pivot=selected.pivot(index='seed',columns='arm',values='target_auc').sort_index()
        if len(pivot)!=10 or pivot.isna().any().any():raise ValueError('Incomplete paired grid')
        d=(pivot.feedback-pivot.no_feedback).to_numpy()
        run=rng.choice(d,size=(DRAWS,len(d)),replace=True).mean(axis=1);run_macro.append(run)
        rows.append(dict(site=site,n_pairs=len(d),feedback_auc=float(pivot.feedback.mean()),
             no_feedback_auc=float(pivot.no_feedback.mean()),difference=float(d.mean()),
             participant_ci_low=summary(diff)['ci_low'],participant_ci_high=summary(diff)['ci_high'],
             run_ci_low=summary(run)['ci_low'],run_ci_high=summary(run)['ci_high'],
             rejected_zero_pair_draws=invalid,draws=DRAWS))
    p=np.mean(macro,axis=0);r=np.mean(run_macro,axis=0)
    rows.append(dict(site='macro',n_pairs=30,feedback_auc=float(np.mean([x['feedback_auc'] for x in rows])),
         no_feedback_auc=float(np.mean([x['no_feedback_auc'] for x in rows])),difference=float(np.mean([x['difference'] for x in rows])),
         participant_ci_low=summary(p)['ci_low'],participant_ci_high=summary(p)['ci_high'],
         run_ci_low=summary(r)['ci_low'],run_ci_high=summary(r)['ci_high'],
         rejected_zero_pair_draws=sum(x['rejected_zero_pair_draws'] for x in rows),draws=DRAWS))
    pd.DataFrame(rows).to_csv(OUT/'feedback_ablation_summary.csv',index=False)
    source=pd.read_csv(OUT/'source_choices.csv')
    for field in ['min_source_auc','mean_source_auc','continuation_best']:
        p=source.pivot(index=['site','seed'],columns='arm',values=field)
        d=p.feedback-p.no_feedback
        print(field,'feedback minus no_feedback',d.mean(),flush=True)
    source.groupby(['site','arm'])[['min_source_auc','mean_source_auc','continuation_best']].mean().to_csv(OUT/'feedback_source_summary.csv')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['sensitivity','feedback']);args=p.parse_args()
    with threadpool_limits(limits=1):globals()[args.phase]()
