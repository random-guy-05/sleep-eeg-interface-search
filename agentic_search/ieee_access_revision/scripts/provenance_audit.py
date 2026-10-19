"""Match published model-development lists; report conditional exclusion sensitivity."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .statistical_analysis import pair_surface,participant_draws,summary,surfaces_for,targets,SEED,DRAWS

ROOT=Path(__file__).resolve().parents[3];PACKAGE=ROOT/'agentic_search'
REV=PACKAGE/'ieee_access_revision';OUT=REV/'results';PROV=REV/'provenance'
EXPECTED={'s0001_released_checkpoint_training_recordings.csv':'3522a510a2b88ff24b85b05b802dd0dd42ec366109e232db312179e42373d4b7',
          's0001_conservative_development_subject_exclusions.csv':'29cd446b5ccd58fd9199d889dcce32674c40679148a42ca0c5570ccd657fbbbb'}

def main():
    for name,h in EXPECTED.items():
        assert hashlib.sha256((PROV/name).read_bytes()).hexdigest()==h
    m=pd.read_csv(ROOT/'brainbodyfm/cache/common_interface_manifest.csv',dtype={'subject_id':str})
    tr=pd.read_csv(PROV/'s0001_released_checkpoint_training_recordings.csv',dtype=str)
    ex=pd.read_csv(PROV/'s0001_conservative_development_subject_exclusions.csv',dtype=str)
    m['participant_id']='sub-'+m.site+m.subject_id
    m['training_subject_match']=m.participant_id.isin(set(tr.participant_id))
    m['training_record_match']=m.record_id.isin(set(tr.recording_id))
    m['development_subject_match']=m.participant_id.isin(set(ex.participant_id))
    rows=[]
    for site,g in m.groupby('site'):
        kept=g[~g.development_subject_match];p=kept[kept.outcome.eq(1)];n=kept[kept.outcome.eq(0)]
        pairs=int((np.abs(p.age.to_numpy()[:,None]-n.age.to_numpy()[None,:])<=2).sum())
        rows.append(dict(site=site,n=len(g),training_subject_matches=int(g.training_subject_match.sum()),
           training_record_matches=int(g.training_record_match.sum()),development_subject_matches=int(g.development_subject_match.sum()),
           retained=len(kept),retained_positive=len(p),retained_negative=len(n),retained_age_pairs=pairs))
    table=pd.DataFrame(rows);table.to_csv(OUT/'pretraining_overlap_summary.csv',index=False)
    m[['subject_id','site','training_subject_match','training_record_match','development_subject_match']].to_parquet(OUT/'pretraining_membership_private.parquet',index=False)
    metrics=pd.read_csv(PACKAGE/'analysis/output/final_selection_metrics.csv')
    mask=metrics.budget.eq(10)&(((metrics.method.eq('full_agent'))&metrics.selector.eq('robust'))|
         ((metrics.method.eq('random_source_robust'))&metrics.selector.eq('source_robust'))|
         ((metrics.method.isin(['taxonomy_diverse_robust','evolutionary_robust']))&metrics.selector.eq('robust')))
    selected=metrics[mask];pred=pd.read_parquet(PACKAGE/'analysis/output/participant_predictions.parquet')
    ts=targets();results=[];contrasts=[];macro={};macro_points={};invalids=[]
    rng=np.random.default_rng(SEED+109)
    for row in rows:
        site=row['site'];t=ts[site]
        kept_ids=set(m[(m.site.eq(site))&~m.development_subject_match].subject_id)
        t=t[t.subject_id.astype(str).isin(kept_ids)].copy()
        if row['retained_age_pairs']==0:
            for method in selected.method.unique():
                results.append(dict(site=site,method=method,n_target=len(t),eligible_pairs=0,auc=None,status='not_estimable_no_eligible_age_pairs'))
            continue
        surf,e=surfaces_for(site,t,pred,selected[selected.held_site.eq(site)],2.,'method')
        boot,invalid=participant_draws(surf,e,rng);invalids.append(dict(site=site,rejected_zero_pair_draws=invalid,accepted_draws=DRAWS))
        for name,s in surf.items():
            point=float(s.sum()/e.sum())
            results.append(dict(site=site,method=name,n_target=len(t),eligible_pairs=int(e.sum()),auc=point,status='estimable',**summary(boot[name])))
            macro.setdefault(name,[]).append(boot[name]);macro_points.setdefault(name,[]).append(point)
        for name in surf:
            if name=='full_agent':continue
            contrasts.append(dict(site=site,comparison='full_agent_minus_'+name,difference=float((surf['full_agent']-surf[name]).sum()/e.sum()),**summary(boot['full_agent']-boot[name])))
    if len(invalids)==3:
        for name,arrays in macro.items():
            results.append(dict(site='macro_three_retained_sites',method=name,auc=float(np.mean(macro_points[name])),status='estimable_but_sparse_retained_S0001',**summary(np.mean(arrays,axis=0))))
    else:
        results.append(dict(site='macro_three_retained_sites',method='all',auc=None,status='not_estimable_one_or_more_sites_without_pairs'))
    # This separate descriptive summary is always restricted to I0002 and I0006.
    for name in selected.method.unique():
        observed=[r['auc'] for r in results if r['site'] in ['I0002','I0006'] and r['method']==name]
        results.append(dict(site='macro_I0002_I0006_only',method=name,auc=float(np.mean(observed)),status='secondary_two_site_summary'))
    pd.DataFrame(results).to_csv(OUT/'overlap_exclusion_sensitivity.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'overlap_exclusion_contrasts.csv',index=False)
    pd.DataFrame(invalids).to_csv(OUT/'overlap_exclusion_resampling.csv',index=False)
    (OUT/'provenance_audit.json').write_text(json.dumps(dict(status='overlap_detected',published_checksums_verified=True,
        model_manifest_checkpoint_sha256='b2a9b8dab3ae8543241d613a80cd6a85a6dfca15573f897a1096861b3915af87',
        historical_extraction_checkpoint_identity_verified=False,
        scope='The upstream lists reconstruct exposure for the released checkpoint. They establish identifier overlap with the model-development population; historical extraction checkpoint bytes are unavailable.',
        source='https://github.com/bdsp-core/philosophers-stone/blob/master/paper/provenance/README.md',
        interpretation='Search-controller target blindness remains a distinct verified property. Foundation-model development independence is not established for the original S0001 fold. Zero I0002/I0006 matches do not exclude cross-hospital identity or unrecorded development exposure.',
        site_summary=rows,checksums=EXPECTED),indent=2)+'\n')
    print(table.to_string(index=False));print(pd.DataFrame(results).to_string(index=False))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
