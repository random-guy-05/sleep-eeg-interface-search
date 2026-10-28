"""Complete report of corrected prediction, metric decomposition, and why audit."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_cinc_method_replication import LABELS,TASKS


def cell(r,metric,model):
    c=r['intervals'][metric][model]
    return f"{r[metric][model]:.3f} ({c['ci_low']:.3f}–{c['ci_high']:.3f})"


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--project-root',type=Path,required=True);a=ap.parse_args()
    s=json.loads((a.results/'summary.json').read_text());assert s['status']=='complete'
    why=json.loads((a.results/'explanatory_audit.json').read_text());paired=json.loads((a.results/'posthoc_ordinary_paired_diagnostics.json').read_text())
    out=a.output;out.mkdir(exist_ok=True,parents=True)
    for name in ['summary.json','explanatory_audit.json','posthoc_ordinary_paired_diagnostics.json']:shutil.copy2(a.results/name,out/name)
    for name in ['GENERAL_COGNITIVE_PREDICTION_CORRECTION.md','general_cognitive_prediction.py','test_general_cognitive_prediction.py','audit_residual_label_comparison.py','report_general_cognitive_prediction.py']:
        shutil.copy2(Path(__file__).with_name(name),out/name)
    design=a.project_root/'brainbodyfm/stages/results/cinc_method_replication/residual_label_design_audit.json'
    shutil.copy2(design,out/design.name)
    r=s['analyses']['composite']
    lines=['# Ordinary and age-conditioned cognitive prediction in STAGES','',
      f"**Without age conditioning**, the fixed Stable4 score has ordinary AUROC **{cell(r,'pooled_ordinary_auc','stable4')}**. The negated brain-health score has ordinary AUROC **{cell(r,'pooled_ordinary_auc','negated_brain_health')}**. This uses the corrected, source-reference-defined low-performance composite; it is a different target from the earlier full-cohort residual phenotype.",'',
      f"There are {r['cases']} low-performance cases among {r['n']} eligible participants, with {r['eligible_pairs']:,} eligible age-matched within-center pairs. The new plan/code were frozen at `{s['freeze']['commit']}` before this extension was computed. Earlier results were already known. All new results are exploratory; the original endpoints are retained.",'',
      '## The effect of center pairing and age conditioning','',
      '| Model | Ordinary, all pairs | Ordinary, within center | Age-conditioned, within center |',
      '|---|---:|---:|---:|']
    selected=['stable4','negated_brain_health','age','demographics','demographics_plus_four','clinical','clinical_plus_four']
    for m in selected:lines.append('| '+LABELS[m]+' | '+' | '.join(cell(r,metric,m) for metric in ['pooled_ordinary_auc','ordinary_within_center','age_conditioned'])+' |')
    d=paired['pooled_ordinary_auc__negated_brain_health__minus__demographics']
    lines += ['',f"The brain-health score minus demographics is {d['estimate']:+.3f} AUROC (paired 95% CI {d['ci_low']:+.3f} to {d['ci_high']:+.3f}). This additional post hoc diagnostic interval is not multiplicity-adjusted. The positive ordinary-AUROC association therefore does not by itself establish incremental prediction beyond demographics.",'',
      'Changing from all-pair ordinary AUROC to within-center age-conditioned AUROC changes both center pairing and age pairing. The middle column makes that distinction visible. These differences are descriptive effects of pair eligibility and weighting, not a causal decomposition of the score.','',
      '## Why the expectation of at least 0.667 is not supported','',
      '1. The organizer-scored test entry had ordinary AUROC 0.749 and age-conditioned AUROC 0.622. A minimum age-conditioned AUROC of 0.667 would already exceed that test result. The validation figures were 0.738 and 0.723. [Official Challenge results](https://moody-challenge.physionet.org/2026/results/).','',
      f"2. The HSP development sample is substantially older: mean {why['age_profiles']['HSP_development']['mean']:.1f} years versus {why['age_profiles']['STAGES']['mean']:.1f} in STAGES; {100*why['age_profiles']['HSP_development']['share_age60plus']:.1f}% versus {100*why['age_profiles']['STAGES']['share_age60plus']:.1f}% are at least 60. These are the selected development cache and STAGES sample, not the unknown hidden-site age distributions.",'',
      '3. The Challenge outcome is a later clinical diagnosis; STAGES supplies concurrent task performance. The corrected target is low raw executive-plus-memory performance relative to source-center reference values. It does not create a clinical MCI or dementia diagnosis. A score can discriminate later clinical diagnoses yet show weak concurrent task associations in a younger clinical sample. This is an explanation to investigate, not a proven mechanism.','',
      f"4. The actual Stable4 associations with the available raw tests are small: Pearson r = {why['correlations']['stable4']['raw_PCET']['Pearson']:.3f} for PCET and {why['correlations']['stable4']['raw_face_memory']['Pearson']:.3f} for face memory (higher risk should correlate negatively with better performance). They remain weak before outcome residualization, so residualization alone cannot explain the weak fixed-score discrimination.",'',
      f"5. Brain-health risk correlates with age at r = {why['correlations']['negated_brain_health']['age']['Pearson']:.3f}. Its ordinary discrimination decreases with within-center age matching, while the fixed Stable4 result changes little. Ordinary and age-independent discrimination answer different questions.",'',
      '6. Four outputs and their weights were selected on the same 338-person development benchmark used for its reported source cross-validation. That development estimate is conditional on selection. The organizer-scored hidden test is the more independent reference for expectation; its selected deployment branch still needs original logs or a clearly labeled reconstruction.','',
      '## Checks against an inference or data-processing mistake','',
      'The separate input audit verified all 590 participant/feature joins and cognitive source rows; independent checkpoint projection agrees to under 0.000001 output SD. Fresh EDF extraction at one record per center reproduces the cached outputs exactly. The same scorer reproduces HSP development macro AUROC 0.754144.','']
    oldnew=why['current_vs_historical_HSP_inference']
    if oldnew.get('matched_original_HSP_development_records',0):
        lines += [f"Current source inference was also compared with {oldnew['matched_original_HSP_development_records']} matched historical HSP participants. Ages and sex codes agree; four-head new/old correlations exceed 0.999997. Median latent relative difference is {oldnew['latent_relative_L2']['median']:.6f}. This matched subset gives no evidence of a large inference-path shift; it is not a raw-signal audit of every historical record.",'']
    lines += ['A real design issue affected the newly fitted classifiers against full-cohort residual labels. A 1,000-replicate null simulation gave demographic AUROC 0.478 with full-cohort fitted residual cases versus 0.499 with independently defined residual cases. Those older fitted binary contrasts are retained as audit diagnostics. In this corrected extension, every label reference, feature normalization, and coefficient fit uses only the source centers.','',
      'The reconstructed HSP-normalization endpoint is still pending. The source reconstruction now retains CARR feature taps; the first 102 records need separate backfill for a full branch replay. This does not change any earlier fixed-score result. The exact original organizer artifact and selected branch remain unavailable.','',
      '## All outcomes and all models','',
      'Intervals are nominal 95% patient bootstrap intervals (2,000 resamples stratified by center). Outcome reference values, labels, source preprocessing and model fits are repeated in each resample. No models, features, thresholds or outcomes were selected from this table.','']
    rows=[]
    for task,t in s['analyses'].items():
        lines += ['### '+TASKS[task],'',f"n = {t['n']}; low-performance cases = {t['cases']}; age-matched pairs = {t['eligible_pairs']:,}.",'','| Model | Ordinary, all pairs | Ordinary, within center | Age-conditioned | Holm p for age-conditioned fixed score |','|---|---:|---:|---:|---:|']
        for m in s['models']:
            p=t.get('permutation_p_holm_33',{}).get(m);pv=f'{p:.4f}' if p is not None else '—'
            lines.append('| '+LABELS[m]+' | '+' | '.join(cell(t,k,m) for k in ['pooled_ordinary_auc','ordinary_within_center','age_conditioned'])+' | '+pv+' |')
            for metric in ['pooled_ordinary_auc','ordinary_within_center','age_conditioned','equal_center']:
                rows.append(dict(outcome=task,model=m,metric=metric,estimate=t[metric][m],**t['intervals'][metric][m]))
        lines += ['','| Center | n | Cases | Age-matched pairs | Stable4 | Brain-health risk |','|---|---:|---:|---:|---:|---:|']
        for c,counts in t['counts'].items():lines.append(f"| {c.replace('_',' ')} | {counts['n']} | {counts['cases']} | {counts['eligible_pairs']} | {t['by_center'][c]['stable4']:.3f} | {t['by_center'][c]['negated_brain_health']:.3f} |")
        lines += ['']
    lines += ['## Every prespecified contrast in this extension','','Positive differences favor the first model. Familywise intervals use Bonferroni correction for the nine comparisons in this extension; they do not provide study-wide error control over the full post hoc analysis history.','','| Outcome | Comparison | Age-conditioned difference | Nominal 95% CI | Familywise 95% CI |','|---|---|---:|---:|---:|']
    for task,t in s['analyses'].items():
        for key,v in t['paired_differences'].items():
            m,b=key.split('__minus__');c=t['intervals']['paired_differences'][key];f=c['familywise_95_bonferroni']
            lines.append(f"| {TASKS[task]} | {LABELS[m]} vs {LABELS[b]} | {v:+.3f} | {c['ci_low']:+.3f} to {c['ci_high']:+.3f} | {f['ci_low']:+.3f} to {f['ci_high']:+.3f} |")
    lines += ['','![Ordinary and age-conditioned results](ordinary_vs_age_conditioned.png)','','The dashed horizontal line marks chance (0.5); the dotted line marks the 0.667 comparison reference. Error bars are nominal 95% bootstrap intervals. The plotted outcomes and held-center predictions are identical across the three pair-selection metrics.','']
    (out/'ORDINARY_AUROC_AND_WHY_REPORT.md').write_text('\n'.join(lines));pd.DataFrame(rows).to_csv(out/'all_model_metrics.csv',index=False)
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,6.3));metrics=['pooled_ordinary_auc','ordinary_within_center','age_conditioned']
    colors={'stable4':'#164e75','negated_brain_health':'#963e32','age':'#37735c','demographics':'#555555'}
    for j,m in enumerate(colors):
        x=np.arange(3)+(j-1.5)*.045;v=np.array([r[k][m] for k in metrics]);lo=np.array([r['intervals'][k][m]['ci_low'] for k in metrics]);hi=np.array([r['intervals'][k][m]['ci_high'] for k in metrics])
        ax.errorbar(x,v,yerr=np.array([v-lo,hi-v]),fmt='o-',color=colors[m],capsize=3,markersize=5,lw=1.5,label=LABELS[m])
    ax.axhline(.5,ls='--',color='black',lw=1);ax.axhline(.667,ls=':',color='#888888',lw=1)
    ax.set_xticks(range(3),['Ordinary AUROC\nAll case–noncase pairs','Ordinary AUROC\nWithin-center pairs','Age-conditioned AUROC\nWithin-center pairs, ±2 years'])
    ax.set_ylabel('AUROC (nominal 95% CI)');ax.set_ylim(.35,.78);ax.set_xlim(-.25,2.25);ax.grid(axis='y',alpha=.15)
    ax.set_title('Concurrent low executive–memory performance in STAGES',fontweight='bold',pad=16)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,frameon=False)
    fig.subplots_adjust(left=.1,right=.97,top=.9,bottom=.26)
    for suffix in ['png','svg']:fig.savefig(out/f'ordinary_vs_age_conditioned.{suffix}',dpi=300,facecolor='white')
    plt.close(fig)
    print(out/'ORDINARY_AUROC_AND_WHY_REPORT.md')


if __name__=='__main__':main()
