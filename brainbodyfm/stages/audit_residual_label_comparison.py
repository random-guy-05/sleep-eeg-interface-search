"""Simulation audit of full-cohort residual labels in center-held-out prediction.

Uses STAGES covariate distributions/centers only, not cognition or EEG values.
No observed endpoint is changed or recomputed.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from analyze import covariate_design
from age_conditioned_transfer import eligible_pairs,rank_auc
from stable4_extension import CENTERS
from cinc_method_replication import load_entry,classifier_predict


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--entry-source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--replicates',type=int,default=1000);a=ap.parse_args()
    load_entry(a.entry_source)
    cohort=pd.read_csv(a.data_root/'data/cohort_before_signal_qc.csv')
    ids=pd.read_csv(a.data_root/'results/provisional_assumed_uv/secondary_face_memory_correct/analysis_subjects.csv').subject_id
    # Access only design variables after matching the existing analysis roster.
    frame=cohort.loc[cohort.subject_id.isin(ids),['center','age','sex','education','bmi','ahi','ess','phq9']].reset_index(drop=True)
    full,_=covariate_design(frame,frame,clinical=False)
    centers=frame.center.to_numpy();ages=frame.age.to_numpy(float)
    x=np.column_stack([np.ones(len(frame)),full]+[frame.center.eq(c).to_numpy(float) for c in CENTERS[1:]])
    u,s,_=np.linalg.svd(x,full_matrices=False);q=u[:,s>max(x.shape)*np.finfo(float).eps*s.max()]
    folds=[]
    for c in CENTERS:
        te=centers==c;tr=~te;sa,sb=covariate_design(frame.loc[tr],frame.loc[te],clinical=False)
        folds.append((tr,te,sa,sb))
    pairs=eligible_pairs(ages,centers);rng=np.random.default_rng(2026102752)
    results={'full_cohort_fitted_residual_cases':[],'independent_true_norm_residual_cases':[]}
    for rep in range(a.replicates):
        error=rng.normal(size=len(frame))
        fitted_residual=error-q@(q.T@error)
        fitted_residual/=np.std(fitted_residual)
        definitions={'full_cohort_fitted_residual_cases':fitted_residual<=-1.,
                     'independent_true_norm_residual_cases':error<=-1.}
        for name,labels in definitions.items():
            predictions=np.empty(len(frame))
            for tr,te,sa,sb in folds:
                predictions[te]=classifier_predict(sa,labels[tr],ages[tr],sb)
            metric,_=rank_auc(labels,predictions,pairs)
            results[name].append(float(metric[0]))
        if (rep+1)%100==0:print('simulation replicates',rep+1,flush=True)
    summary={'simulation_replicates':a.replicates,'seed':2026102752,'n':len(frame),
        'real_cognitive_or_EEG_values_used':False,
        'purpose':'Check whether full-cohort residual-case construction depresses out-of-center demographic AUROC under a known null.',
        'statistics':{}}
    for name,values in results.items():
        v=np.asarray(values)
        summary['statistics'][name]={'mean_age_conditioned_auc':float(v.mean()),'mc_standard_error_of_mean':float(v.std(ddof=1)/np.sqrt(len(v))),
            'median':float(np.median(v)),'simulation_95_range':np.quantile(v,[.025,.975]).tolist(),'share_below_chance':float(np.mean(v<.5))}
    delta=np.array(results['full_cohort_fitted_residual_cases'])-np.array(results['independent_true_norm_residual_cases'])
    summary['paired_mean_design_effect']=float(delta.mean())
    summary['paired_mc_se']=float(delta.std(ddof=1)/np.sqrt(len(delta)))
    a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
