"""Report the complete discovery/confirmation ledger without favorable selection."""
from __future__ import annotations
import argparse,json,shutil,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'attention_slowness':'Sustained-attention response slowing','vigilance_variability':'Vigilance response-time variability','vigilance_slow_tail':'Vigilance slow-response tail','executive_slowness':'Executive-task response slowing'}
MARKERS={'stable4':'Stable4 (unchanged CinC recipe)','negated_brain_health':'Brain-health risk score','pred_regression_f_fs_chirp_c4m1':'Fast-spindle chirp output','pred_regression_f_sigma_bandpower_kurtosis_c_w':'Wake sigma-kurtosis output','pred_regression_f_total_power_rem_rel_delta':'REM relative-delta output'}


def interval(r,metric):
    c=r['bootstrap_intervals'][metric];return f"{r[metric]:.3f} ({c['ci_low']:.3f} to {c['ci_high']:.3f})"


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    source=a.data_root/'results/unexpected_cognition_discovery';out=a.output;out.mkdir(exist_ok=True,parents=True)
    selection=json.loads((source/'discovery_selection.json').read_text());confirmation=json.loads((source/'confirmation_summary.json').read_text());inference=json.loads((source/'selected_inference_summary.json').read_text())
    assert confirmation['stage']=='confirmation_complete' and inference['stage']=='inference_complete'
    fixed=pd.read_csv(source/'discovery_fixed_scores.csv');phys=pd.read_csv(source/'discovery_all_physiology.csv');confirm=pd.read_csv(source/'confirmation_selected.csv')
    clinical=confirm[confirm.adjustment.eq('clinical')];replicated=clinical[clinical.replicated]
    main=fixed[fixed.new_endpoint & fixed.marker.isin(['stable4','negated_brain_health']) & fixed.adjustment.eq('clinical')]
    lead=main.sort_values('p').iloc[0]
    lines=['# STAGES unexpected cognitive-signature discovery: completed results','',
       f"The strongest discovery was **higher unchanged Stable4 risk associated with slower correct sustained-attention responses in the US centers**: adjusted partial r = {lead.r:.3f}, p = {lead.p:.6f}, discovery FDR q = {lead.q:.4f}. **It did not meet the frozen confirmation criterion in MedSleep.**",'',
       f"The completed pass tested {selection['fixed_test_family']} fixed-score/new-endpoint associations and {selection['physiology_test_family']:,} physiology-output/new-endpoint associations. Six candidates were selected using US results alone; {len(replicated)} of six met same-direction, Holm-adjusted p < .05 at MedSleep. This pass did not produce a confirmed unexpected finding suitable for a positive validation headline.",'',
       '## What was searched and how confirmation was protected','',
       f"The discovery cohort contained {selection['discovery_n']} patients from Stanford, Geisinger and St Luke’s; the confirmation cohort contained {selection['confirmation_n']} MedSleep patients. Task validity and endpoint-specific missingness reduce these counts. This is a center-held replication of previously unanalyzed endpoints, not a wholly untouched study: the earlier accuracy, memory-correct and mean-vigilance results had already been inspected at all centers.",'',
       'The 16 additional measures covered cognitive flexibility, perseveration, failures to maintain set, response speed, recognition discrimination, confidence-related errors, sustained-attention variability, working memory, vigilance lapses, and the slow-response tail. No new score weights, thresholds or subgroups were selected after confirmation. Verbal memory and the longer working-memory version were excluded for inadequate availability.','',
       'Main associations adjusted for an age spline, sex, education, BMI, AHI, Epworth Sleepiness Scale, PHQ-9, recording duration, and discovery-center indicators. Outcomes were rank-normalized within the analysis sample. Results are cross-sectional associations, not effects of an intervention.','',
       f"Plan and code freeze: `{selection['freeze']['commit']}`. The exact six discovery selections and all discovery outputs were committed at `{confirmation['selection_commit']}` before the MedSleep associations were computed. This is a dated post hoc analysis, not preregistration of the original study.",'',
       '## Every selected finding','',
       'Intervals below are nominal 95% partial-correlation intervals. Discovery q-values adjust the full fixed-score or physiology discovery family. Confirmation p-values are Holm-adjusted separately over the three frozen selections in each track.','',
       '| Track / marker | Cognitive measure | Discovery n | Discovery r (95% CI) | Discovery q | Confirmation n | Confirmation r (95% CI) | Confirmation Holm p | Confirmed |',
       '|---|---|---:|---:|---:|---:|---:|---:|---|']
    rows=[]
    for item in selection['selected']:
        d=fixed[(fixed.marker==item['marker'])&(fixed.endpoint==item['endpoint'])&fixed.adjustment.eq('clinical')] if item['track']=='fixed' else phys[(phys.marker==item['marker'])&(phys.endpoint==item['endpoint'])]
        d=d.iloc[0];c=clinical[(clinical.marker==item['marker'])&(clinical.endpoint==item['endpoint'])].iloc[0]
        lines.append(f"| {item['track']}: {MARKERS[item['marker']]} | {LABELS.get(item['endpoint'],item['endpoint'])} | {d.n} | {d.r:.3f} ({d.ci_low:.3f} to {d.ci_high:.3f}) | {d.q:.4f} | {c.n} | {c.r:.3f} ({c.ci_low:.3f} to {c.ci_high:.3f}) | {c.holm_p:.4f} | {'Yes' if c.replicated else 'No'} |")
        rows.append(dict(**item,discovery_n=int(d.n),discovery_ci_low=float(d.ci_low),discovery_ci_high=float(d.ci_high),confirmation_n=int(c.n),confirmation_r=float(c.r),confirmation_ci_low=float(c.ci_low),confirmation_ci_high=float(c.ci_high),confirmation_holm_p=float(c.holm_p),replicated=bool(c.replicated)))
    pd.DataFrame(rows).to_csv(out/'selected_discovery_confirmation.csv',index=False)
    lines += ['',f"One of {len(main)} fixed-score discovery tests survived FDR q < .05. None of the {len(phys):,} physiological-output tests did. The physiology track's best nominal associations should therefore be treated as hypothesis-generating even before their confirmation results.",'',
       '## Held-center prediction diagnostics','',
       'Each diagnostic defines low performance using the US 84.1344746th percentile of endpoint risk. The threshold, feature normalization and clinical predictor coefficients are fit in the US cohort and then applied to MedSleep. There is no held-center threshold tuning. A fixed ridge penalty of 1 is used for the source-fitted clinical predictor and clinical-plus-marker predictor; this routine implementation choice was not tuned to MedSleep performance. These comparisons are conditional on the frozen discovery selections, are exploratory, and are not new confirmatory endpoints.','',
       '| Marker / outcome | MedSleep n / cases | Marker ordinary AUROC | Clinical ordinary AUROC | Clinical + marker ordinary AUROC | Paired AUROC difference | Paired MAE difference |',
       '|---|---:|---:|---:|---:|---:|---:|']
    for item in selection['selected']:
        key=item['track']+'__'+item['marker']+'__'+item['endpoint'];r=inference['results'][key]
        lines.append(f"| {MARKERS[item['marker']]} / {LABELS.get(item['endpoint'],item['endpoint'])} | {r['confirmation_n']} / {r['confirmation_cases']} | {interval(r,'marker_ordinary_auc')} | {interval(r,'clinical_ordinary_auc')} | {interval(r,'augmented_ordinary_auc')} | {interval(r,'delta_ordinary_auc')} | {interval(r,'delta_mae')} |")
    lines += ['','Positive AUROC differences favor adding the marker; negative MAE differences favor adding the marker. A marker may be associated with a continuous outcome without discriminating the source-defined low-performance group. The marker-only score retains the CinC direction for Stable4 and the published orientation for brain-health risk. A new physiology output uses its discovery direction, which was frozen before confirmation.','',
       '## Robustness and uncertainty','',
       'The 2,000 participant bootstrap resamples were stratified by center and repeated source score normalization, endpoint ranking, nuisance adjustment, source outcome references, and prediction fitting. They conditioned on the frozen selected findings. The selection itself was not repeated. Each candidate also received heteroskedasticity-robust HC3 inference.','',
       '| Marker / outcome | Discovery partial r bootstrap CI | Confirmation partial r bootstrap CI | Confirmation HC3 p |',
       '|---|---:|---:|---:|']
    for item in selection['selected']:
        key=item['track']+'__'+item['marker']+'__'+item['endpoint'];r=inference['results'][key]
        lines.append(f"| {MARKERS[item['marker']]} / {LABELS.get(item['endpoint'],item['endpoint'])} | {interval(r,'discovery_partial_r')} | {interval(r,'confirmation_partial_r')} | {r['confirmation_HC3']['p_HC3']:.4f} |")
    lines += ['','## How the CinC results can be used','',
       'The prior organizer-scored validation/test estimates, age-conditioned AUROC 0.723 and 0.622, concern later impairment diagnoses from submission 2593. They remain evidence about that submitted system. Whether standalone Stable4 produced those scores requires the deployment artifact/logs or a clearly labeled reconstruction. The submitted date-free system also contained guarded alternative branches. [Official Challenge results](https://moody-challenge.physionet.org/2026/results/).','',
       'If the exact fixed score is verified as the deployed branch, the CinC results can corroborate the same score’s cognitive-risk discrimination in a different population. They do not independently validate a newly discovered slowing mechanism, domain contrast, subgroup, or new EEG output. Testing any such claim in HSP requires the corresponding endpoint and frozen analysis.','',
       'For this completed pass, the additional obstacle is empirical: the new STAGES slowing association did not confirm at MedSleep. Calling it a CinC-validated unexpected finding would overstate both the STAGES confirmation and the meaning of the old Challenge estimates.','',
       '## Novelty assessment','',
       'Sleep EEG associations with processing speed are established in prior work; a new paper would need more than discovering that speed correlates with EEG. The potentially distinctive finding here was a previously selected compact risk score associating with sustained-attention speed despite weak accuracy prediction. Because that pattern did not confirm at the held center, it remains a candidate rather than an established new cognitive signature. See Djonlagic et al., [Macro and micro sleep architecture and cognitive performance in older adults](https://doi.org/10.1038/s41562-020-00964-y), Nature Human Behaviour. 2021;5:123–145, and Ujma et al., [Multivariate prediction of cognitive performance from the sleep electroencephalogram](https://doi.org/10.1016/j.neuroimage.2023.120319), NeuroImage. 2023;279:120319. Titles, authors, journals and identifiers were checked against primary publisher/PubMed records on October 27, 2026.','',
       'The original continuous endpoints, fixed-score rank analyses, corrected general-performance analyses, provenance uncertainties, and all new negative confirmations are retained. No new positive headline was inserted into the SLEEP manuscript.','',
       '![Frozen candidates in discovery and confirmation](discovery_confirmation_forest.png)','']
    (out/'DISCOVERY_AND_CINC_VALIDATION_REPORT.md').write_text('\n'.join(lines))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(12,6));colors=['#1b5576','#9c4434']
    for i,r in enumerate(rows):
        for j,(estimate,lo,hi) in enumerate([(r['discovery_r'],r['discovery_ci_low'],r['discovery_ci_high']),(r['confirmation_r'],r['confirmation_ci_low'],r['confirmation_ci_high'])]):
            yy=i+(-.12 if j==0 else .12)
            ax.plot([lo,hi],[yy,yy],color=colors[j],lw=1.5);ax.scatter(estimate,yy,color=colors[j],marker='o' if j==0 else 'D',s=30,label=['US discovery','MedSleep confirmation'][j] if i==0 else None)
    ax.axvline(0,color='black',ls='--',lw=1);ax.set_yticks(range(len(rows)),[MARKERS[r['marker']]+'\n'+LABELS.get(r['endpoint'],r['endpoint']) for r in rows]);ax.invert_yaxis();ax.set_xlim(-.4,.45);ax.grid(axis='x',alpha=.15);ax.set_xlabel('Clinical-adjusted partial correlation (nominal 95% CI)')
    ax.set_title('Cognitive-signature discovery and held-center confirmation',fontweight='bold',pad=16);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2,frameon=False)
    fig.subplots_adjust(left=.39,right=.97,top=.88,bottom=.23)
    for suffix in ['png','svg']:fig.savefig(out/f'discovery_confirmation_forest.{suffix}',dpi=300,facecolor='white')
    plt.close(fig)
    for name in ['discovery_selection.json','discovery_fixed_scores.csv','discovery_all_physiology.csv','confirmation_selected.csv','confirmation_summary.json','selected_inference_summary.json','availability_audit.json']:
        shutil.copy2(source/name,out/name)
    for name in ['UNEXPECTED_COGNITION_DISCOVERY_PLAN.md','unexpected_cognition_discovery.py','test_unexpected_cognition_discovery.py','unexpected_cognition_inference.py','report_unexpected_cognition.py']:
        shutil.copy2(Path(__file__).with_name(name),out/name)
    verification=dict(discovery_code_commit=selection['freeze']['commit'],selection_before_confirmation_commit=confirmation['selection_commit'],discovery_n=selection['discovery_n'],confirmation_n=selection['confirmation_n'],fixed_tests=32,physiology_tests=8624,confirmed_candidates=int(clinical.replicated.sum()),complete_bootstrap=2000,all_candidates_retained=True,participant_data_in_package=False,summary_sha256=hashlib.sha256((source/'selected_inference_summary.json').read_bytes()).hexdigest())
    (out/'VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(out/'DISCOVERY_AND_CINC_VALIDATION_REPORT.md')


if __name__=='__main__':main()
