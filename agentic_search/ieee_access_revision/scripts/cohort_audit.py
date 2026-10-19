"""Reconstruct recoverable cohort assembly; never invent a population denominator."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
REV=ROOT/'agentic_search/ieee_access_revision'
PS=ROOT/'experiments/philosophers_stone'

def main():
    inputs={}
    def read(rel):
        p=PS/rel
        inputs[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        columns=pd.read_csv(p,nrows=0).columns
        return pd.read_csv(p,usecols=[c for c in ['file_id','subject_id','site_id','age','target'] if c in columns],dtype={'file_id':str,'subject_id':str})
    stages={
        'initial_recovered':read('shard74_api_embeddings_73_valid.csv'),
        'positive_batch1_requested':read('batch1/batch1_positives_manifest.csv'),
        'positive_batch1_completed':pd.concat([read(f'batch1/embedding{k}/embeddings.csv') for k in [1,2]]),
        'positive_batch2_requested':read('batch2/batch2_positives_manifest.csv'),
        'positive_batch2_completed':pd.concat([read(f'batch2/embedding{k}/embeddings.csv') for k in [1,2]]),
        'expanded_cache':read('embeddings_expanded_c4m1fix.csv'),
        'negative_extension_requested':read('age_matched_negatives_top200_portable.csv'),
        'negative_extension_completed':read('negatives_balanced_v1/embeddings.csv')}
    assembled=pd.concat([stages[s] for s in ['initial_recovered','positive_batch1_completed','positive_batch2_completed','negative_extension_completed']])
    fpath=ROOT/'brainbodyfm/cache/common_interface_manifest.csv'
    frozen=pd.read_csv(fpath,dtype={'subject_id':str,'record_id':str})
    inputs[str(fpath.relative_to(ROOT))]=hashlib.sha256(fpath.read_bytes()).hexdigest()
    assert len(assembled)==338 and assembled.file_id.nunique()==338 and assembled.subject_id.nunique()==338
    assert set(assembled.file_id)==set(frozen.record_id)
    compare=assembled.set_index('file_id').reindex(frozen.record_id)
    assert (compare.target.to_numpy()==frozen.outcome.to_numpy()).all()
    assert (compare.site_id.to_numpy()==frozen.site.to_numpy()).all()
    assert np.isfinite(frozen.age).all()
    assert frozen[['included_primary_semantic','has_semantic_841','has_latent_1024','has_stable4_heads']].all().all()
    rows=[]
    for name,f in stages.items():
        assert f.file_id.nunique()==len(f)
        for site in ['I0002','I0006','S0001']:
            g=f[f.site_id.eq(site)]
            rows.append(dict(stage=name,site=site,n=len(g),positive=int(g.target.sum()),negative=int(g.target.eq(0).sum())))
    for site,g in frozen.groupby('site'):
        y=g.outcome.to_numpy();a=g.age.to_numpy()
        pairs=int((np.abs(a[y==1,None]-a[None,y==0])<=2).sum())
        rows.append(dict(stage='final_frozen',site=site,n=len(g),positive=int(g.outcome.sum()),negative=int(g.outcome.eq(0).sum()),age_pairs=pairs))
    flow=pd.DataFrame(rows);flow.to_csv(REV/'results/cohort_flow.csv',index=False)
    missing=[]
    for base in ['positive_batch1','positive_batch2','negative_extension']:
        req=stages[base+'_requested'];done=stages[base+'_completed']
        assert set(done.file_id)<=set(req.file_id)
        for site in ['I0002','I0006','S0001']:
            r=req[req.site_id.eq(site)];d=done[done.site_id.eq(site)]
            missing.append(dict(stage=base,site=site,requested=len(r),completed=len(d),not_recovered=len(set(r.file_id)-set(d.file_id)),reason='Not established by recovered artifacts' if len(r)!=len(d) else 'None'))
    pd.DataFrame(missing).to_csv(REV/'results/cohort_extraction_accounting.csv',index=False)
    demo_path=ROOT/'demographics.csv'
    demo=pd.read_csv(demo_path,dtype={'BDSPPatientID':str})
    demo['record_id']=demo.BidsFolder+'_ses-'+demo.SessionID.astype(str)
    demo['included']=demo.record_id.isin(set(frozen.record_id))
    joined=frozen.merge(demo,on='record_id',validate='one_to_one')
    assert len(joined)==338
    assert (joined.outcome==joined.Cognitive_Impairment.astype(int)).all()
    positive_times=joined.loc[joined.outcome.eq(1),'Time_to_Event']
    negative_times=joined.loc[joined.outcome.eq(0),'Time_to_Last_Visit']
    assert positive_times.between(365.25,6*365.25).all()
    assert negative_times.ge(6*365.25).all()
    demo_summary=demo.groupby('SiteID').agg(available_records=('record_id','size'),available_positive=('Cognitive_Impairment','sum'),included_records=('included','sum')).reset_index()
    demo_summary['not_in_frozen_cache']=demo_summary.available_records-demo_summary.included_records
    demo_summary.to_csv(REV/'results/available_demographic_frame.csv',index=False)
    inputs[str(demo_path.relative_to(ROOT))]=hashlib.sha256(demo_path.read_bytes()).hexdigest()
    obj=dict(status='verified_recoverable_assembly',n=338,n_positive=83,n_negative=255,
        checks={'unique_subjects':True,'unique_records':True,'exact_final_manifest_union':True,'binary_label_identity':True,'site_identity':True,'finite_ages':True,'all_final_rows_have_semantic_features':True},
        mechanism='Outcome-aware historical extraction: initial 73-record cache plus positive-only expansions, followed by 199 of 200 requested same-site age-matched negative records. Negative prioritization used positive-partner counts and site pair balance.',
        demographics_frame={'n_records':len(demo),'n_unique_subjects':demo.BDSPPatientID.nunique(),'not_selected':int((~demo.included).sum()),'scope':'Locally available labeled demographic frame; not the complete HSP source population.'},
        outcome_timing_check={'all_cached_labels_match_supplied_Cognitive_Impairment':True,'positive_first_diagnosis_days_range':[float(positive_times.min()),float(positive_times.max())],'negative_last_encounter_min_days':float(negative_times.min()),'two_diagnoses_at_least_week_apart_independently_verified':False},
        unknowns=['Complete eligible HSP source-population denominator and initial extraction selection/failure log','Reasons for unrecovered batch1 and negative-extension records','Exact historical checkpoint bytes, extraction timestamp and inference commit','Two qualifying diagnoses separated by a week cannot be re-adjudicated from the available summary fields'],
        inference='Conditional comparison on this selected benchmark; not population prevalence, calibration or prospective clinical validation.',inputs=inputs)
    (REV/'results/cohort_audit.json').write_text(json.dumps(obj,indent=2)+'\n')
    print(flow.to_string(index=False))
    print(pd.DataFrame(missing).to_string(index=False))

if __name__=='__main__':main()
