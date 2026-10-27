"""Prespecified continuous-outcome, nested center-holdout evaluation.

The command refuses primary analysis when required centers or input dispositions
are incomplete. No target row enters a fitted transform or alpha selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,SplineTransformer,StandardScaler

BASE=Path(__file__).resolve().parent
STABLE4=['pred_regression_f_total_power_rem_rel_theta','pred_regression_f_fs_chirp_c4m1',
         'pred_regression_f_sigma-9-std','pred_regression_f_sigma_bandpower_kurtosis_c_n1']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config(base=BASE):
    c=json.loads((base/'protocol_v1.json').read_text())
    c.update(json.loads((base/'amendment_01.json').read_text())['overrides'])
    return c


def covariate_design(train,other,*,clinical=True):
    numeric=['sex']+(['bmi','ahi','ess','phq9'] if clinical else [])
    impute=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)
    a=impute.fit_transform(train[numeric]);b=impute.transform(other[numeric])
    age_imp=SimpleImputer(strategy='median')
    age_a=age_imp.fit_transform(train[['age']]);age_b=age_imp.transform(other[['age']])
    spline=SplineTransformer(n_knots=4,degree=3,knots='quantile',include_bias=False,extrapolation='linear')
    a=np.c_[a,spline.fit_transform(age_a)];b=np.c_[b,spline.transform(age_b)]
    encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    ea=train[['education']].fillna('missing').astype(str);eb=other[['education']].fillna('missing').astype(str)
    a=np.c_[a,encoder.fit_transform(ea)];b=np.c_[b,encoder.transform(eb)]
    scale=StandardScaler();return scale.fit_transform(a),scale.transform(b)


def ridge_path(train,y,other,alphas):
    """The exact ridge path from one SVD, with an unpenalized intercept."""
    mean=train.mean(axis=0);ym=float(np.mean(y));x=train-mean
    u,s,vt=np.linalg.svd(x,full_matrices=False)
    coefs=(vt.T[None,:,:]*(s[None,:]/(s[None,:]**2+np.asarray(alphas)[:,None]))[:,None,:]) @ (u.T@(y-ym))
    return (other-mean)@coefs.T+ym


def choose_alpha(errors,alphas,tolerance):
    eligible=np.flatnonzero(errors<=float(np.min(errors))+tolerance)
    return int(max(eligible,key=lambda i:float(alphas[i])))


def select_candidate(source_mae,ids,tolerance):
    ids=np.asarray(ids,dtype=int);best=float(np.min(source_mae[ids]))
    return int(np.min(ids[source_mae[ids]<=best+tolerance]))


def build_design(frame,representation,train_idx,eval_idx,*,clinical=True,pca=False):
    a,b=covariate_design(frame.iloc[train_idx],frame.iloc[eval_idx],clinical=clinical)
    if representation.shape[1]:
        scaler=StandardScaler();ra=scaler.fit_transform(representation[train_idx]);rb=scaler.transform(representation[eval_idx])
        if pca:
            reducer=PCA(n_components=8,svd_solver='full');ra=reducer.fit_transform(ra);rb=reducer.transform(rb)
        a=np.c_[a,ra];b=np.c_[b,rb]
    return a,b


def fit_outer(frame,representation,outer,cfg,*,clinical=True,pca=False):
    centers=frame.center.to_numpy();y=frame[cfg.get('target_column','pcet_accuracy2')].to_numpy(float)
    target=np.flatnonzero(centers==outer);sources=np.flatnonzero(centers!=outer)
    alphas=np.asarray(cfg['ridge_alpha_grid'],float);inner_scores=[];validation=[]
    for held in sorted(set(centers[sources])):
        train=np.flatnonzero((centers!=outer)&(centers!=held));val=np.flatnonzero(centers==held)
        if set(train)&set(target) or set(train)&set(val):raise AssertionError('Center leakage')
        a,b=build_design(frame,representation,train,val,clinical=clinical,pca=pca)
        predictions=ridge_path(a,y[train],b,alphas)
        inner_scores.append(np.mean(np.abs(predictions-y[val,None]),axis=0));validation.append((val,predictions))
    scores=np.mean(inner_scores,axis=0)
    chosen=choose_alpha(scores,alphas,cfg['tie_tolerance'])
    a,b=build_design(frame,representation,sources,target,clinical=clinical,pca=pca)
    predictions=ridge_path(a,y[sources],b,[alphas[chosen]])[:,0]
    all_predictions=np.empty(len(frame),float);all_predictions[target]=predictions
    for val,pred in validation:all_predictions[val]=pred[:,chosen]
    # Only after source selection and refitting do target labels enter scoring.
    target_mae=float(np.mean(np.abs(predictions-y[target])))
    return {'outer':outer,'source_mae':float(scores[chosen]),'target_mae':target_mae,
            'target_rmse':float(np.sqrt(np.mean((predictions-y[target])**2))),
            'target_Pearson_r':float(pearsonr(y[target],predictions).statistic),
            'target_Spearman_r':float(spearmanr(y[target],predictions).statistic),
            'target_n':len(target),'alpha':float(alphas[chosen]),
            'alpha_grid_source_mae':scores,'predictions':all_predictions}


def readiness(base=BASE,unit_policy='strict'):
    c=config(base);frame=pd.read_csv(base/'data/cohort_before_signal_qc.csv')
    header_path=base/'data/signal_header_audit.csv'
    headers=pd.read_csv(header_path).set_index('subject_id') if header_path.exists() else None
    cache=base/'cache' if unit_policy=='strict' else base/'cache/assumed_uv'
    complete=[];pending=[];excluded=[]
    for row in frame.itertuples(index=False):
        marker=cache/'subjects'/f'{row.subject_id}.json';payload=marker.with_suffix('.npz')
        quarantine=cache/'quarantine'/f'{row.subject_id}.json'
        if marker.exists() and payload.exists():
            m=json.loads(marker.read_text())
            if m['protocol_sha256']!=row.protocol_sha256 or digest(payload)!=m['payload_sha256']:
                raise ValueError(f'Cache provenance mismatch for {row.subject_id}')
            complete.append(row.subject_id)
        elif headers is not None and row.subject_id in headers.index and not (
            bool(headers.loc[row.subject_id,'header_eligible']) if unit_policy=='strict'
            else headers.loc[row.subject_id,'channel_kind']!='unavailable' and float(headers.loc[row.subject_id,'duration_hours'])>=2):
            excluded.append(row.subject_id)
        elif quarantine.exists():excluded.append(row.subject_id)
        else:pending.append(row.subject_id)
    counts=frame[frame.subject_id.isin(complete)].groupby('center').size().to_dict()
    missing={center:counts.get(center,0) for center in c['primary_outer_centers'] if counts.get(center,0)<c['minimum_target_n_for_summary']}
    status='ready' if not pending and not missing else ('awaiting_extraction' if pending else 'not_estimable_required_centers_below_minimum')
    result={'status':status,'unit_policy':unit_policy,'results_provisional':unit_policy=='assumed_uv',
            'cohort_before_signal_QC':len(frame),'complete_extractions':len(complete),
            'pending_extractions':len(pending),'excluded_or_quarantined':len(excluded),
            'completed_by_center':counts,'required_centers_below_minimum':missing,
            'required_primary_centers':c['primary_outer_centers'],'header_audit_complete':headers is not None}
    readiness_name='analysis_readiness.json' if unit_policy=='strict' else 'assumed_uv_analysis_readiness.json'
    (base/'results'/readiness_name).write_text(json.dumps(result,indent=2)+'\n')
    return result,frame[frame.subject_id.isin(complete)].reset_index(drop=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true')
    parser.add_argument('--outcome',choices=['pcet_accuracy2','face_memory_correct','pvt_mean_reciprocal_rt'],default='pcet_accuracy2')
    parser.add_argument('--unit-policy',choices=['strict','assumed_uv'],default='strict')
    args=parser.parse_args()
    status,frame=readiness(unit_policy=args.unit_policy);print(json.dumps(status,indent=2),flush=True)
    if not args.run or status['status']!='ready':return
    cfg=config();cfg['target_column']=args.outcome
    cache=BASE/'cache' if args.unit_policy=='strict' else BASE/'cache/assumed_uv'
    if args.outcome=='face_memory_correct':
        frame=frame[(frame.face_memory_status=='C1')&(frame.face_memory_validation=='V')&frame.face_memory_correct.notna()].reset_index(drop=True)
    elif args.outcome=='pvt_mean_reciprocal_rt':
        frame=frame[(frame.pvt_status=='C1')&frame.pvt_mean_reciprocal_rt.notna()].reset_index(drop=True)
    counts=frame.groupby('center').size().to_dict()
    if any(counts.get(c,0)<cfg['minimum_target_n_for_summary'] for c in cfg['primary_outer_centers']):
        missing={'status':'outcome_not_estimable_required_centers_below_minimum','outcome':args.outcome,'by_center':counts,'unit_policy':args.unit_policy}
        (BASE/'results'/f'{args.unit_policy}_{args.outcome}_readiness.json').write_text(json.dumps(missing,indent=2)+'\n')
        print(json.dumps(missing));return
    candidates=json.loads((BASE/'data/candidate_bank.json').read_text())['candidates']
    latent=[];semantic=[];bhs=[];names=None;hashes=[]
    for sid in frame.subject_id:
        path=cache/'subjects'/f'{sid}.npz';z=np.load(path,allow_pickle=False)
        if names is None:names=z['semantic_names'].tolist()
        if z['semantic_names'].tolist()!=names:raise ValueError('Semantic order changed')
        latent.append(z['latent']);semantic.append(z['semantic']);bhs.append(float(z['brain_health_score']));hashes.append(digest(path))
    latent=np.asarray(latent);semantic=np.asarray(semantic);bhs=np.asarray(bhs)[:,None]
    taxonomy=pd.read_csv(BASE.parents[0]/'frozen/head_taxonomy.csv')
    allowed=set(taxonomy.loc[taxonomy.include_physiology_only.astype(str).str.lower().isin(['true','1']),'head_name'])
    physiology=semantic[:,[i for i,n in enumerate(names) if n in allowed]]
    ids={name:i for i,name in enumerate(names)}
    for candidate in candidates:
        if not set(candidate['heads']).issubset(allowed):raise ValueError('Candidate bank includes a forbidden head')
    run_hash=hashlib.sha256(json.dumps({'cohort':digest(BASE/'data/cohort_before_signal_qc.csv'),
                                      'candidates':digest(BASE/'data/candidate_bank.json'),
                                      'analysis':digest(__file__),'outcome':args.outcome,'unit_policy':args.unit_policy,'cache_hashes':hashes},sort_keys=True).encode()).hexdigest()
    parent=BASE/'results' if args.unit_policy=='strict' else BASE/'results/provisional_assumed_uv'
    out=parent/('nested_transport' if args.outcome=='pcet_accuracy2' else f'secondary_{args.outcome}');out.mkdir(parents=True,exist_ok=True)
    key=out/'run_key.json'
    if key.exists() and json.loads(key.read_text())['run_key']!=run_hash:raise ValueError('Existing analysis has different inputs')
    key.write_text(json.dumps({'run_key':run_hash,'protocol_sha256':frame.protocol_sha256.iloc[0]},indent=2)+'\n')
    summaries=[];panels={};baseline_predictions={}
    baseline_reps={'demographics':(np.empty((len(frame),0)),False,False),
                   'clinical':(np.empty((len(frame),0)),True,False),
                   'clinical_plus_frozen_BHS':(bhs,True,False),
                   'clinical_plus_physiology_ridge':(physiology,True,False),
                   'clinical_plus_latent_ridge':(latent,True,False),
                   'clinical_plus_training_only_PCA8':(latent,True,True),
                   'clinical_plus_historical_Stable4_descriptive':(semantic[:,[ids[n] for n in STABLE4]],True,False)}
    for outer in cfg['primary_outer_centers']:
        destination=out/f'{outer}.npz'
        if destination.exists():
            z=np.load(destination,allow_pickle=False)
            panel={k:z[k] for k in ['predictions','source_mae','target_mae','alpha']}
            summaries+=json.loads((out/f'{outer}_baselines.json').read_text())
            baseline_predictions[outer]=json.loads((out/f'{outer}_baseline_predictions.json').read_text())
        else:
            panel={'predictions':np.empty((len(candidates),len(frame)),np.float32),
                   'source_mae':np.empty(len(candidates)), 'target_mae':np.empty(len(candidates)), 'alpha':np.empty(len(candidates))}
            tuning=[]
            for candidate in candidates:
                draw=candidate['draw'];rep=semantic[:,[ids[n] for n in candidate['heads']]]
                fit=fit_outer(frame,rep,outer,cfg)
                for k in panel:panel[k][draw]=fit[k]
                tuning.append({'draw':draw,'source_grid_MAE':fit['alpha_grid_source_mae'].tolist()})
                if (draw+1)%50==0:print(json.dumps({'center':outer,'candidates_fitted':draw+1}),flush=True)
            bs=[];bp={}
            for name,(rep,clinical,pca) in baseline_reps.items():
                fit=fit_outer(frame,rep,outer,cfg,clinical=clinical,pca=pca)
                bs.append({k:fit[k] for k in ['outer','source_mae','target_mae','target_rmse','target_Pearson_r','target_Spearman_r','target_n','alpha']}|{'interface':name})
                bp[name]=fit['predictions'].tolist()
            (out/f'{outer}_baselines.json').write_text(json.dumps(bs,indent=2)+'\n')
            (out/f'{outer}_baseline_predictions.json').write_text(json.dumps(bp)+'\n')
            (out/f'{outer}_tuning.json').write_text(json.dumps(tuning)+'\n')
            tmp=destination.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,**panel)
            tmp.replace(destination);summaries+=bs;baseline_predictions[outer]=bp
        panels[outer]=panel
    pd.DataFrame(summaries).to_csv(out/'baseline_endpoints.csv',index=False)
    rng=np.random.default_rng(cfg['permutation_seed'])
    permutations=np.array([rng.permutation(len(candidates)) for _ in range(cfg['candidate_permutations'])])
    np.savetxt(out/'candidate_permutations.csv',permutations,delimiter=',',fmt='%d')
    selections=[];chosen_by_budget={}
    for outer,panel in panels.items():
        chosen_by_budget[outer]={}
        for budget in cfg['search_budgets']:
            chosen=[select_candidate(panel['source_mae'],p[:budget],cfg['tie_tolerance']) for p in permutations]
            chosen_by_budget[outer][budget]=chosen
            for order,draw in enumerate(chosen):
                selections.append({'outer':outer,'budget':budget,'permutation':order,'draw':draw,
                                   'source_mae':float(panel['source_mae'][draw]),'target_mae':float(panel['target_mae'][draw])})
    selected=pd.DataFrame(selections);selected['gap']=selected.target_mae-selected.source_mae
    selected.to_csv(out/'search_budget_selections.csv',index=False)
    macro=selected.groupby(['budget','permutation'])[['source_mae','target_mae','gap']].mean().reset_index()
    macro.to_csv(out/'search_budget_macro.csv',index=False)
    centers=frame.center.to_numpy();y=frame[args.outcome].to_numpy(float)
    # Shared participant bootstrap draws preserve dependence across overlapping outer folds.
    changes={}
    for outer,panel in panels.items():
        e0=np.abs(panel['predictions'][chosen_by_budget[outer][1]]-y[None,:]).mean(axis=0)
        e1=np.abs(panel['predictions'][chosen_by_budget[outer][500]]-y[None,:]).mean(axis=0)
        changes[outer]=e1-e0
    def gap_change(weights):
        values=[]
        for outer,delta in changes.items():
            target=centers==outer
            t=np.average(delta[target],weights=weights[target])
            s=np.mean([np.average(delta[centers==other],weights=weights[centers==other])
                       for other in cfg['primary_outer_centers'] if other!=outer])
            values.append(t-s)
        return float(np.mean(values))
    rng=np.random.default_rng(cfg['bootstrap_seed']);boot=[]
    for _ in range(cfg['bootstrap_replicates']):
        weights=np.zeros(len(frame),int)
        for center in cfg['primary_outer_centers']:
            ix=np.flatnonzero(centers==center);weights[ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)))
        boot.append(gap_change(weights))
    result={'primary_delta_gap_MAE_500_minus1':gap_change(np.ones(len(frame))),
            'conditional_patient_bootstrap_95CI':np.quantile(boot,[.025,.975]).tolist(),
            'bootstrap_replicates':len(boot),'run_key':run_hash,'outcome':args.outcome,'n':len(frame),
            'unit_policy':args.unit_policy,'results_provisional':args.unit_policy=='assumed_uv',
            'by_center':frame.groupby('center').size().to_dict(),
            'interpretation':'MAE target-minus-source transport diagnostic, not a causal estimate of selection optimism'}
    (out/'primary_result.json').write_text(json.dumps(result,indent=2)+'\n')
    frame[['subject_id','center',args.outcome]].to_csv(out/'analysis_subjects.csv',index=False)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
