"""Create an aggregate-only author report after the bounded analysis completes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {
 'stable4':'Stable4, fixed CinC weights', 'equal_weights':'Same four outputs, equal weights',
 'without_1':'Stable4 without REM theta', 'without_2':'Stable4 without spindle chirp',
 'without_3':'Stable4 without hour-9 sigma variability', 'without_4':'Stable4 without N1 kurtosis',
 'component_1':'REM theta alone', 'component_2':'Spindle chirp alone',
 'component_3':'Hour-9 sigma variability alone (risk direction)', 'component_4':'N1 kurtosis alone (risk direction)',
 'negated_brain_health':'Negated brain-health score', 'age':'Age alone',
 'demographics':'Demographics', 'clinical':'Clinical covariates',
 'demographics_plus_four':'Demographics + four refitted outputs',
 'clinical_plus_four':'Clinical + four refitted outputs'}
TASKS = {'composite':'Executive–memory composite', 'executive_function':'Executive function', 'face_memory':'Face memory'}


def fmt(v):return f'{v:.3f}'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    source=a.results/'summary.json';s=json.loads(source.read_text())
    if s['status']!='complete':raise SystemExit('Do not render an incomplete analysis as final')
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,out/'summary.json');shutil.copy2(a.results/'aggregate_model_results.csv',out/'aggregate_model_results.csv')
    base=Path(__file__).resolve().parent
    for f in ['CINC_METHOD_REPLICATION_PLAN.md','cinc_method_replication.py','test_cinc_method_replication.py','report_cinc_method_replication.py']:
        shutil.copy2(base/f,out/f)
    r=s['analyses']['composite'];value=r['age_conditioned']['stable4'];ci=r['intervals']['age_conditioned']['stable4']
    stable_positive=ci['ci_low']>.5
    adaptation=[]
    for task,t in s['analyses'].items():
        for contrast,point in t['paired_differences'].items():
            family=t['intervals']['paired_differences'][contrast]['familywise_95_bonferroni']
            if family['ci_low']>0:adaptation.append((task,contrast,point))
    decision = ('The fixed published method discriminated low composite performance in the expected direction at nominal 95% confidence.' if stable_positive else
                'The published Stable4 method did not establish discrimination of low composite performance above chance in STAGES.')
    directions=sum(v['stable4']>.5 for v in r['by_center'].values())
    center_pattern='; '.join(f"{name.replace('_',' ')} {values['stable4']:.3f}" for name,values in r['by_center'].items())
    lines=['# STAGES validation of the CinC cognitive-risk method','',decision,'',
        f"Among {r['n']} participants, {r['cases']} met the existing low-composite definition. Stable4 produced an age-conditioned AUROC of **{value:.3f} (95% CI {ci['ci_low']:.3f}–{ci['ci_high']:.3f})**, based on {r['eligible_pairs']:,} eligible case–noncase pairs.",'',
        f"The center-level point estimates were in the expected direction in {directions} of four centers: {center_pattern}. This is directional concordance, with an imprecise pooled estimate. Geisinger and St Luke’s contributed only 15 and 28 eligible pairs, respectively.",'',
        'The four outputs, signs, and weights were unchanged. Normalization was estimated exclusively from the other three centers in each fold, using the exact submitted implementation. This replicates the CinC development method; it does not use the still-pending reconstructed HSP normalizer. The endpoint is concurrent sample-relative low cognitive performance, rather than the Challenge’s later clinical diagnosis.','',
        f"The extension was committed before its first metric was computed: `{s['freeze']['commit']}`. Earlier continuous and within-center-normalization results were already known; this extension is explicitly post hoc. The analysis completed at {s['completed_utc']}.",'',
        '## Complete model results','',
        'All intervals below are nominal percentile 95% intervals from 2,000 participant bootstrap samples stratified by center. Every sample repeats outcome adjustment, labeling, source normalization, and all trained fits. AUROC uses within-center pairs with age difference at most two years. Holm p-values adjust the 33 outcome-blind score/outcome tests together. No permutation p-value is assigned to a trained classifier.','']
    design_audit=a.results/'residual_label_design_audit.json'
    if design_audit.exists():
        design=json.loads(design_audit.read_text());shutil.copy2(design_audit,out/design_audit.name)
        lines[2:2]=['**Subsequent design-audit note.** The fitted binary comparisons below are retained as methodological diagnostics. Full-cohort residual-case construction removes demographic signal and, in the specified null simulation, depressed held-center demographic AUROC to 0.478 versus 0.499 with independently defined residual labels. These contrasts are not used to establish general incremental cognitive prediction. A separately frozen, source-reference-only general-performance analysis addresses that question. The outcome-blind fixed-score calculation still estimates discrimination of its explicitly defined residual phenotype.','']
    for task,t in s['analyses'].items():
        lines += [f'### {TASKS[task]}','',f"n = {t['n']}; cases = {t['cases']}; eligible pairs = {t['eligible_pairs']:,}.",'',
            '| Model | Age-conditioned AUROC (95% CI) | Equal-center AUROC | Ordinary within-center AUROC | Holm p |',
            '|---|---:|---:|---:|---:|']
        for m in s['models']:
            c=t['intervals']['age_conditioned'][m];p=t.get('permutation_p_holm_33',{}).get(m)
            lines.append(f"| {LABELS[m]} | {t['age_conditioned'][m]:.3f} ({c['ci_low']:.3f}–{c['ci_high']:.3f}) | {t['equal_center'][m]:.3f} | {t['ordinary_within_center'][m]:.3f} | {p:.4f} |" if p is not None else
                f"| {LABELS[m]} | {t['age_conditioned'][m]:.3f} ({c['ci_low']:.3f}–{c['ci_high']:.3f}) | {t['equal_center'][m]:.3f} | {t['ordinary_within_center'][m]:.3f} | — |")
        lines += ['']
    lines += ['## Incremental prediction: every planned comparison','',
        'Positive differences favor the model in the first column. The familywise intervals use Bonferroni adjustment for all nine planned contrasts across the three outcomes. The four-output adaptation arms learn new coefficients on source centers and do not validate the unchanged Stable4 score.','',
        '| Outcome | Comparison | AUROC difference | Nominal 95% CI | Familywise 95% CI |',
        '|---|---|---:|---:|---:|']
    comparison_rows=[]
    for task,t in s['analyses'].items():
        for key,v in t['paired_differences'].items():
            m,b=key.split('__minus__');c=t['intervals']['paired_differences'][key];f=c['familywise_95_bonferroni']
            lines.append(f"| {TASKS[task]} | {LABELS[m]} vs {LABELS[b]} | {v:+.3f} | {c['ci_low']:+.3f} to {c['ci_high']:+.3f} | {f['ci_low']:+.3f} to {f['ci_high']:+.3f} |")
            comparison_rows.append(dict(outcome=task,model=m,comparator=b,delta_auc=v,ci_low=c['ci_low'],ci_high=c['ci_high'],familywise_low=f['ci_low'],familywise_high=f['ci_high']))
    pd.DataFrame(comparison_rows).to_csv(out/'aggregate_paired_comparisons.csv',index=False)
    lines += ['','## Center-level results for the composite','',
        '| Center | n | Cases | Eligible pairs | Stable4 | Demographics | Demographics + four |',
        '|---|---:|---:|---:|---:|---:|---:|']
    centers=[]
    for c,n in r['counts'].items():
        v=r['by_center'][c]
        lines.append(f"| {c.replace('_',' ')} | {n['n']} | {n['cases']} | {n['eligible_pairs']:,} | {v['stable4']:.3f} | {v['demographics']:.3f} | {v['demographics_plus_four']:.3f} |")
        for m in s['models']:centers.append(dict(center=c,model=m,age_conditioned_auc=v[m],**n))
    pd.DataFrame(centers).to_csv(out/'aggregate_composite_centers.csv',index=False)
    valid=r['intervals']['equal_center']['stable4']['valid_replicates']
    lines += ['',f'Equal-center Stable4 intervals had {valid:,}/2,000 valid replicates; all four centers must contribute pairs for a replicate to count. The pooled metric retains every eligible pair. Pair counts are not independent sample sizes.','',
        '## What this adds to the manuscript','',
        'This is a direct, outcome-aligned replication of the published four-output cognitive-risk method. It tests whether a compact score developed for later impairment ranks concurrent low performance at held-out sleep centers. It also tests, separately, whether source-center adaptation of the same output roster improves prediction.','',
        'Cognitive prediction from STAGES PSG is already represented in the literature. Khajehpiri and colleagues used EEG and ECG in 817 participants, dichotomized PCET concept-level responses at the sample median, and used subject-based ten-fold cross-validation. Their task and metric differ from the age-conditioned, adjusted low-performance endpoint here. The distinctive question for this study is transport of a previously selected four-output score across centers. [Khajehpiri et al., IEEE EMBC 2025; author manuscript](https://arxiv.org/html/2506.00279v1).','',
        ('At least one planned incremental contrast retained a positive familywise interval; assess the complete table, model discrimination, and centers before selecting its scientific interpretation.' if adaptation else
         'None of the nine planned incremental contrasts had a familywise interval wholly above zero. This extension therefore does not supply evidence for incremental cross-center prediction after accounting for the full comparison family.'),'',
        'The result does not determine what the original HSP-normalized score will show; that reconstruction remains a separate analysis. No additional endpoint, threshold, subgroup, sign flip, model family, or feature search was added in response to these results.','',
        'The existing AHI pipeline check remains 0.761 (95% CI 0.704–0.812). It demonstrates detection of that respiratory signal under the rank procedure; it does not establish power for a smaller cognitive association.','',
        '## Figures','',
        'The forest plot displays every model and outcome, without selecting favorable rows. Intervals are nominal 95% percentile intervals. The dashed line marks chance. No Challenge and STAGES estimates are pooled.','',
        '![All models and cognitive outcomes](cinc_method_all_models_forest.png)','']
    input_audit=a.results/'input_audit.json';raw_audit=a.results/'raw_reextraction_audit.json'
    audits_complete=False
    if input_audit.exists() and raw_audit.exists():
        audit=json.loads(input_audit.read_text());raw=json.loads(raw_audit.read_text())
        audits_complete=audit.get('status')=='complete' and raw.get('status')=='complete'
        if audits_complete:
            shutil.copy2(input_audit,out/input_audit.name);shutil.copy2(raw_audit,out/raw_audit.name)
            c=audit['checks'];cache=c['every_subject_cache_matches_joined_input'];cognitive=c['original_cognitive_rows'];projection=c['independent_checkpoint_projection'];historical=c['historical_HSP_development_reproduction']
            lines += ['## Input and implementation audit after the result was challenged','',
                '| Check | Evidence |','|---|---|',
                f"| Participant alignment and cached features | All {cache['checked']} subject caches match the analysis inputs exactly; all have the same checkpoint and extraction source hashes. |",
                f"| Original CNB records | All {cognitive['checked']} selected source rows recovered uniquely; PCET and face-memory scores match exactly; PCET formula differences are limited to source rounding (maximum {cognitive['pcet_formula_max_abs_rounding_error']:.5f}). |",
                f"| Independent projection from checkpoint tensors | Four outputs recomputed from all 590 latents; maximum absolute difference {max(projection['max_absolute_error_by_head']):.3g}, under {max(projection['max_error_in_head_sd_units']):.3g} output SD. |",
                '| Raw EDF re-extraction | One outcome-blind selected record per center; all four reproduce the cached spectrogram hash, latent, four outputs, and brain-health score exactly. Both native and derived C4–M1 routes are represented. |',
                f"| Original HSP development result | The submitted scorer reproduces macro age-conditioned AUROC {historical['macro_auc']:.6f} and each reported site estimate. |",
                '| Submitted metric | The Challenge entry’s independent pair-AUROC function exactly reproduces all three new STAGES estimates. |','',
                'These checks identified no scoring, indexing, head-mapping, voltage-scaling, or selected-record extraction discrepancy. Fresh raw-signal re-extraction covers four audit records, not all 590 EDFs.','',
                'A separate method-reproduction omission was identified: the source reconstruction had discarded feature taps required by the optional CARR branch. Amendment 07 retains future taps and preserves the first 102 latent files without changes. This affects our ability to reconstruct the full submitted system; it does not change the standalone Stable4 estimates above.','',
                'The current STAGES case definition is a covariate-adjusted concurrent-performance phenotype. It is not the later diagnosis label used in the Challenge. The original organizer-trained artifact and deployment logs remain unavailable, and the reconstructed HSP-normalization result is pending. The 0.533 estimate must not be presented as a completed evaluation of the exact organizer-trained system.','']
    (out/'RESULTS_AND_NOVELTY_ASSESSMENT.md').write_text('\n'.join(lines))
    order=['stable4','negated_brain_health','demographics','demographics_plus_four','clinical','clinical_plus_four','age','equal_weights']+[f'without_{i+1}' for i in range(4)]+[f'component_{i+1}' for i in range(4)]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(16,9),sharey=True)
    for ax,(task,t) in zip(axes,s['analyses'].items()):
        for i,m in enumerate(order):
            value=t['age_conditioned'][m];c=t['intervals']['age_conditioned'][m]
            color='#154c79' if m=='stable4' else '#823b34' if m.endswith('plus_four') else '#4a4a4a'
            ax.plot([c['ci_low'],c['ci_high']],[i,i],color=color,lw=1.5)
            ax.scatter(value,i,color=color,s=30,zorder=3,marker='D' if m=='stable4' else 'o')
        ax.axvline(.5,color='black',lw=1,ls='--');ax.set_xlim(.2,.85);ax.set_xticks([.2,.3,.4,.5,.6,.7,.8]);ax.grid(axis='x',alpha=.15)
        ax.set_title(TASKS[task]+f"\nn = {t['n']}; cases = {t['cases']}",fontweight='bold',pad=12)
        ax.set_xlabel('Age-conditioned AUROC (95% CI)')
    axes[0].set_yticks(range(len(order)),[LABELS[m] for m in order]);axes[0].invert_yaxis()
    fig.suptitle('CinC method evaluated at held-out STAGES centers',fontsize=17,fontweight='bold',x=.6,y=.98)
    fig.text(.30,.025,'Fixed Stable4: source-center normalization; unchanged outputs, directions and weights.\nAdaptation arms: coefficients learned only at the other centers. All rows are exploratory.',fontsize=10)
    fig.subplots_adjust(left=.30,right=.98,bottom=.12,top=.88,wspace=.14)
    for suffix in ['png','svg']:fig.savefig(out/f'cinc_method_all_models_forest.{suffix}',dpi=300,facecolor='white')
    plt.close(fig)
    verification=dict(source_summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        freeze=s['freeze'],completed_utc=s['completed_utc'],outcomes=len(s['analyses']),
        models_per_outcome=len(s['models']),fixed_score_permutation_family=33,incremental_contrast_family=9,
        fixed_composite_nominal_ci_above_chance=stable_positive,composite_centers_above_chance=directions,
        positive_familywise_contrasts=adaptation,participant_level_files_in_package=False,
        input_and_four_record_raw_audits_complete=audits_complete)
    (out/'VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(out/'RESULTS_AND_NOVELTY_ASSESSMENT.md')


if __name__=='__main__':main()
