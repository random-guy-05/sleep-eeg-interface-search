"""Export all completed follow-up results; never fit or select a model."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={
 'age':'Age', 'demographics':'Demographics', 'clinical':'Clinical',
 'negated_brain_health':'Brain-health risk', 'stable4':'Stable4',
 'submitted_available':'Submitted available branches', 'submitted_latent':'Submitted latent network',
 'clinical_plus_cognition23':'Clinical + cognition outputs (23)',
 'clinical_plus_physiology539':'Clinical + physiology outputs (539)',
 'clinical_plus_latent1024':'Clinical + latent representation (1024)',
 'clinical_plus_all841':'Clinical + all outputs (841)',
 'source_selected_linear':'Source-selected linear representation'}
DOMAINS={'executive':'Executive accuracy','memory':'Face-memory accuracy',
 'composite':'Executive + memory','attention':'Attention slowing'}

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 root=a.data_root;out=a.out;out.mkdir(exist_ok=True,parents=True)
 source={n:root/'results'/n for n in ['cinc_trainable_transfer/summary.json','cinc_trainable_transfer/diagnostics_summary.json','cognitive_domain_specificity/summary.json','cinc_trainable_transfer/additional_sample_availability_audit.json']}
 data={n:json.loads(p.read_text()) for n,p in source.items()}
 train=data['cinc_trainable_transfer/summary.json'];diag=data['cinc_trainable_transfer/diagnostics_summary.json'];dom=data['cognitive_domain_specificity/summary.json']
 assert train['status']==dom['status']=='complete'
 rows=[];counts=[];gates=[];contrasts=[]
 for task,result in train['analyses'].items():
  point=result['point'];counts.append(dict(task=task,n=point['n'],cases=point['cases'],eligible_age_pairs=point['eligible_pairs']))
  for metric in ['ordinary_auc','within_center_ordinary','age_conditioned','average_precision']:
   for model,value in point[metric].items():
    rows.append(dict(task=task,model=model,label=LABELS[model],metric=metric,estimate=value,**result['intervals'][metric][model]))
  for center,f in result['folds'].items():
   gates.append(dict(task=task,center=center,source_n=f['source_n'],held_n=f['held_n'],held_cases=f['held_cases'],**f['submitted_source_selection']))
  for name,v in result['contrasts'].items():contrasts.append(dict(task=task,contrast=name,**v))
 pd.DataFrame(rows).to_csv(out/'all_model_metrics.csv',index=False)
 pd.DataFrame(counts).to_csv(out/'outcome_counts.csv',index=False)
 pd.DataFrame(gates).to_csv(out/'submitted_branch_gates.csv',index=False)
 # Additional contrasts are explicitly post hoc, using paired saved draws.
 for task in ['composite','attention_slowness']:
  with np.load(root/'results/cinc_trainable_transfer'/(task+'_private_conditional_bootstrap.npz')) as z:
   for model in ['negated_brain_health','clinical_plus_latent1024','clinical_plus_cognition23']:
    v=z['ordinary_auc__'+model]-z['ordinary_auc__clinical'];v=v[np.isfinite(v)];q=np.quantile(v,[.025,.975,.025/24,1-.025/24])
    contrasts.append(dict(task=task,contrast=model+'__minus__clinical',delta=train['analyses'][task]['point']['ordinary_auc'][model]-train['analyses'][task]['point']['ordinary_auc']['clinical'],ci_low=q[0],ci_high=q[1],familywise_low=q[2],familywise_high=q[3],conditional_on_fitted_models=True,additional_posthoc=True,adjustment_family=24))
 pd.DataFrame(contrasts).to_csv(out/'paired_model_contrasts.csv',index=False)
 drows=[]
 for key,value in dom['point'].items():
  score,domain,metric=key.split('__');drows.append(dict(score=score,domain_or_contrast=domain,metric=metric,estimate=value,**dom['intervals'][key]))
 pd.DataFrame(drows).to_csv(out/'same_participant_domain_metrics_and_contrasts.csv',index=False)
 pd.DataFrame([dict(task=k,**{x:v for x,v in c.items() if x!='by_center'}) for k,c in dom['counts'].items()]).to_csv(out/'same_participant_outcome_counts.csv',index=False)
 pd.DataFrame([dict(representation=k,**{x:v for x,v in c.items() if x not in ['confusion_matrix','class_order']}) for k,c in diag['center_prediction'].items()]).to_csv(out/'center_prediction.csv',index=False)
 pd.DataFrame([dict(model=m,label=LABELS[m],center_held_out_auc=train['analyses']['composite']['point']['ordinary_auc'][m],random_patient_cv_auc=diag['patient_cv']['ordinary_auc'][m]) for m in train['models']]).to_csv(out/'patient_vs_center_cv.csv',index=False)
 public=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_hashes={n:digest(p) for n,p in source.items()},freezes=dict(trainable=train['freeze'],domain_contrasts=dom['freeze']),cohort_n=590,common_domain_n=dom['n'],common_by_center=dom['common_by_center'],bootstrap_replicates=2000,trainable_intervals_conditional_on_fitted_models=True,domain_analysis_repeats_all_data_fitted_steps=True,timing_availability=data['cinc_trainable_transfer/additional_sample_availability_audit.json'],original_endpoints_unchanged=True,new_analyses_posthoc=True)
 (out/'aggregate_provenance.json').write_text(json.dumps(public,indent=2)+'\n')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
 def save(fig,name):
  fig.savefig(out/(name+'.png'),dpi=200,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12.8,6),sharey=True,layout='constrained')
 models=['clinical','negated_brain_health','stable4','clinical_plus_cognition23','clinical_plus_physiology539','clinical_plus_latent1024','submitted_available','submitted_latent']
 for ax,task,title in zip(axes,['composite','attention_slowness'],['Executive + face-memory low performance','Sustained-attention slowing']):
  result=train['analyses'][task];y=np.arange(len(models))[::-1]
  for yy,m in zip(y,models):
   p=result['point']['ordinary_auc'][m];ci=result['intervals']['ordinary_auc'][m];color='#1f577b' if m!='stable4' else '#963d36'
   ax.errorbar(p,yy,xerr=[[p-ci['ci_low']],[ci['ci_high']-p]],fmt='o',color=color,capsize=3,lw=1.4)
   ax.text(.797,yy,f"{p:.3f}",ha='right',va='center',fontsize=9)
  ax.axvline(.5,color='.55',ls='--',lw=1);ax.set_xlim(.37,.81);ax.set_title(title+'\n'+f"n = {result['point']['n']}; cases = {result['point']['cases']}",fontsize=11,weight='bold');ax.set_xlabel('Ordinary AUROC (95% CI)');ax.set_yticks(y,[LABELS[m] for m in models]);ax.grid(axis='x',alpha=.15)
 fig.suptitle('Held-center cognitive discrimination across fixed and fitted readouts',fontsize=14,weight='bold')
 fig.supxlabel('Exploratory follow-up. Intervals for fitted readouts condition on their trained models and selections.',fontsize=9)
 save(fig,'figure_1_full_representation')
 fig,axes=plt.subplots(1,2,figsize=(11.4,5.2),layout='constrained')
 for ax,metric,title in zip(axes,['age_conditioned','continuous_concordance'],['Low-performance cases','Threshold-free concordance']):
  for j,score in enumerate(['brain_health_risk','stable4']):
   for i,domain in enumerate(DOMAINS):
    key=score+'__'+domain+'__'+metric;p=dom['point'][key];ci=dom['intervals'][key];y=3-i+(.10 if j==0 else -.10)
    ax.errorbar(p,y,xerr=[[p-ci['ci_low']],[ci['ci_high']-p]],fmt='o' if j==0 else 's',color=['#1f577b','#963d36'][j],capsize=3,label=['Brain-health risk','Stable4'][j] if i==0 else None)
  ax.set_yticks(range(4),list(DOMAINS.values())[::-1]);ax.set_xlim(.38,.77);ax.axvline(.5,ls='--',color='.55',lw=1);ax.grid(axis='x',alpha=.15);ax.set_title(title,weight='bold');ax.set_xlabel('Age-conditioned discrimination (95% CI)')
 axes[1].legend(loc='upper right',frameon=False,fontsize=9)
 fig.suptitle('Cognitive domains evaluated in the same 561 participants',weight='bold',fontsize=14)
 fig.supxlabel('All fitted outcome references, normalization and pairs repeated in 2,000 center-stratified bootstraps.',fontsize=9)
 save(fig,'figure_2_paired_domains')
 fig,axes=plt.subplots(1,2,figsize=(11.5,5.6),layout='constrained',gridspec_kw={'width_ratios':[1.3,1]})
 names=['clinical','negated_brain_health','stable4','clinical_plus_cognition23','clinical_plus_physiology539','clinical_plus_latent1024','submitted_available','submitted_latent'];y=np.arange(len(names))[::-1]
 for i,m in enumerate(names):
  p=train['analyses']['composite']['point']['ordinary_auc'][m];q=diag['patient_cv']['ordinary_auc'][m];axes[0].plot([p,q],[y[i],y[i]],color='.65');axes[0].scatter(p,y[i],color='#1f577b',label='Held-center' if i==0 else None);axes[0].scatter(q,y[i],color='#d68a32',marker='s',label='Random patient folds' if i==0 else None)
 axes[0].set_yticks(y,[LABELS[m] for m in names]);axes[0].set_xlabel('Executive + memory ordinary AUROC');axes[0].set_xlim(.45,.70);axes[0].axvline(.5,ls='--',color='.6');axes[0].legend(frameon=False,loc='lower right',fontsize=9);axes[0].set_title('Validation-strategy diagnostic',weight='bold')
 center=diag['center_prediction'];reps=list(center);yy=np.arange(len(reps))[::-1];axes[1].barh(yy,[center[k]['balanced_accuracy'] for k in reps],color='#1f577b',height=.6);axes[1].set_yticks(yy,['Demographics','Clinical','Cognition outputs','Physiology outputs','Latent representation']);axes[1].axvline(.25,ls='--',color='.45');axes[1].set_xlim(0,1);axes[1].set_xlabel('Center classification balanced accuracy');axes[1].set_title('Center information is retained',weight='bold')
 for i,k in enumerate(reps):axes[1].text(center[k]['balanced_accuracy']+.02,yy[i],f"{center[k]['balanced_accuracy']:.3f}",va='center',fontsize=9)
 fig.suptitle('Center information and cognitive prediction are distinct',weight='bold',fontsize=14)
 fig.supxlabel('Point diagnostics; label-reference case counts differ across validation strategies. Center information can reflect case mix.',fontsize=9)
 save(fig,'figure_3_validation_and_center_information')
 print(json.dumps(dict(out=str(out),model_metric_rows=len(rows),domain_rows=len(drows),figures=3)))

if __name__=='__main__':main()
