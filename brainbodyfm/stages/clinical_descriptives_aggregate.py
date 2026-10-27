"""Aggregate baseline characteristics in the final held-center analysis sample."""
from pathlib import Path
import pandas as pd
B=Path(__file__).resolve().parent
cohort=pd.read_csv(B/'data/cohort_before_signal_qc.csv');ids=pd.read_csv(B/'results/provisional_assumed_uv/nested_transport/analysis_subjects.csv')
f=ids.merge(cohort,on=['subject_id','center'],validate='one_to_one');assert len(f)==590
rows=[]
for center,g in [('Overall',f)]+[(n,g) for n,g in f.groupby('center',sort=True)]:
 r={'center':center,'n':len(g),'age_mean':g.age.mean(),'age_sd':g.age.std(ddof=1),'male_n':int(g.sex.sum())}
 for col in ['bmi','ahi','ess','phq9']:
  x=pd.to_numeric(g[col],errors='coerce');r[col+'_median']=x.median();r[col+'_q1']=x.quantile(.25);r[col+'_q3']=x.quantile(.75);r[col+'_missing']=int(x.isna().sum())
 rows.append(r)
out=B/'results/paired_cognitive_comparisons';out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(out/'clinical_descriptives.csv',index=False)
print(pd.DataFrame(rows).round(2).to_string(index=False))
