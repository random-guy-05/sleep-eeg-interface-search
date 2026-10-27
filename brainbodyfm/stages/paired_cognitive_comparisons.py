"""Conditional paired held-center MAE contrasts from the saved frozen fits.

Post hoc, exploratory analysis. Participant resampling is stratified within each
center, with a common participant draw for each model pair. The fitted models,
source-center tuning, and selected four-feature candidates are not refitted.
Only aggregate outputs are written; no individual data are made public.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
ROOT=BASE/'results/provisional_assumed_uv'
OUTPUT=BASE/'results/paired_cognitive_comparisons'
TASKS=[('executive_function','nested_transport','pcet_accuracy2'),('face_memory','secondary_face_memory_correct','face_memory_correct'),('vigilance','secondary_pvt_mean_reciprocal_rt','pvt_mean_reciprocal_rt')]
CENTERS=['Geisinger','MedSleep','St_Lukes','Stanford']
CONTRASTS=[('brain_health_vs_clinical','clinical_plus_frozen_BHS','clinical'),('selected_eeg_vs_clinical','selected_four_eeg','clinical'),('selected_eeg_vs_demographics','selected_four_eeg','demographics')]
N_BOOT=20000; SEED=2026102607

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load_and_check(directory,outcome):
 f=pd.read_csv(directory/'analysis_subjects.csv'); y=f[outcome].to_numpy(float);centers=f.center.to_numpy()
 assert len(f)==len(set(f.subject_id)) and np.isfinite(y).all() and set(centers)==set(CENTERS)
 sel=pd.read_csv(directory/'search_budget_selections.csv');base=pd.read_csv(directory/'baseline_endpoints.csv')
 errors={model:{} for _,a,b in CONTRASTS for model in (a,b)}
 for c in CENTERS:
  mask=centers==c
  pred=json.loads((directory/f'{c}_baseline_predictions.json').read_text())
  rows=sel[(sel.outer==c)&(sel.budget==500)]
  assert len(rows)==100 and rows.draw.nunique()==1
  draw=int(rows.draw.iloc[0]);panel=np.load(directory/f'{c}.npz',allow_pickle=False)
  for model in errors:
   full=panel['predictions'][draw] if model=='selected_four_eeg' else np.asarray(pred[model],float)
   assert len(full)==len(f) and np.isfinite(full[mask]).all()
   err=np.abs(full[mask]-y[mask]);errors[model][c]=err
   expected=float(rows.target_mae.iloc[0]) if model=='selected_four_eeg' else float(base.loc[(base.outer==c)&(base.interface==model),'target_mae'].iloc[0])
   assert abs(err.mean()-expected)<1e-6,(c,model,err.mean(),expected)
 return f,errors

def analyze(name,directory,outcome,seed):
 f,errors=load_and_check(directory,outcome);rng=np.random.default_rng(seed)
 weights={c:rng.multinomial(len(errors['clinical'][c]),np.full(len(errors['clinical'][c]),1/len(errors['clinical'][c])),size=N_BOOT) for c in CENTERS}
 rows=[]
 for label,a,b in CONTRASTS:
  within={c:errors[a][c]-errors[b][c] for c in CENTERS}
  center_delta={c:float(np.mean(within[c])) for c in CENTERS}
  point=float(np.mean(list(center_delta.values())))
  boot=np.mean([weights[c]@within[c]/len(within[c]) for c in CENTERS],axis=0)
  low,high=np.quantile(boot,[.025,.975])
  rows.append(dict(outcome=name,contrast=label,n=len(f),delta_mae=point,ci_lower=float(low),ci_upper=float(high),
                   bootstrap_replicates=N_BOOT,seed=seed,by_center_delta_mae=center_delta))
 return rows

OUTPUT.mkdir(parents=True,exist_ok=True)
all_rows=[]
for i,(name,folder,outcome) in enumerate(TASKS):all_rows.extend(analyze(name,ROOT/folder,outcome,SEED+i))
jsondoc={'description':'Exploratory paired model-minus-comparator held-center MAE; negative favors the model. Equal weighting of four observed center families. Within-center patient bootstrap conditions on all fitted models, preprocessing, hyperparameters and feature selection; it does not resample centers or refit models. No multiplicity adjustment.','rows':all_rows,
         'input_aggregate_result_sha256':{folder:sha(ROOT/folder/'primary_result.json') for _,folder,_ in TASKS}}
(OUTPUT/'paired_comparison_summary.json').write_text(json.dumps(jsondoc,indent=2)+'\n')
pd.DataFrame([{k:v for k,v in r.items() if k!='by_center_delta_mae'} for r in all_rows]).to_csv(OUTPUT/'paired_comparison_summary.csv',index=False)
print(pd.DataFrame(all_rows).drop(columns=['by_center_delta_mae']).round(5).to_string(index=False))
