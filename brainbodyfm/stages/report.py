"""Render the completed provisional results from saved numerical artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent


def main():
    root=BASE/'results/provisional_assumed_uv';primary=root/'nested_transport'
    confirmation_path=BASE/'results/voltage_units_confirmation.json'
    confirmed=confirmation_path.exists() and json.loads(confirmation_path.read_text()).get('status')=='confirmed_by_user'
    result=json.loads((primary/'primary_result.json').read_text())
    macro=pd.read_csv(primary/'search_budget_macro.csv')
    baselines=pd.read_csv(primary/'baseline_endpoints.csv')
    participants=pd.read_csv(primary/'analysis_subjects.csv')
    cohort=pd.read_csv(BASE/'data/cohort_before_signal_qc.csv')
    cohort=cohort[cohort.subject_id.isin(participants.subject_id)]
    table1=[]
    for center,part in cohort.groupby('center',sort=True):
        row={'center':center,'n':len(part),'age_mean':part.age.mean(),'age_SD':part.age.std(),
             'male_n':int((part.sex==1).sum()),'bmi_median':part.bmi.median(),
             'ahi_median':part.ahi.median(),'ahi_missing_n':int(part.ahi.isna().sum()),
             'ESS_median':part.ess.median(),'PHQ9_median':part.phq9.median()}
        table1.append(row)
    pd.DataFrame(table1).to_csv(root/'cohort_descriptives.csv',index=False)
    summary=macro.groupby('budget')[['source_mae','target_mae','gap']].mean()
    fig,ax=plt.subplots(figsize=(6.4,4.2),layout='constrained')
    for column,label,color in [('source_mae','Source validation','#255C8C'),('target_mae','Held center','#B05A23')]:
        grouped=macro.groupby('budget')[column];x=summary.index.to_numpy()
        ax.plot(x,grouped.mean(),marker='o',color=color,label=label)
        lo=grouped.quantile(.025).to_numpy();hi=grouped.quantile(.975).to_numpy()
        ax.fill_between(x,lo,hi,color=color,alpha=.13,linewidth=0)
    ax.set_xscale('log');ax.set_xticks(summary.index);ax.set_xticklabels(summary.index)
    ax.set_xlabel('Number of candidate probes searched');ax.set_ylabel('Mean absolute error (PCET accuracy2)')
    ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False)
    ax.set_title('STAGES: probe search and held-center transport' if confirmed else 'Provisional analysis: missing EDF units assigned µV',fontsize=10)
    for ext in ['png','pdf','svg']:fig.savefig(root/f'search_budget.{ext}',dpi=250)
    plt.close(fig)
    lo,hi=result['conditional_patient_bootstrap_95CI']
    unit_note=('**The user confirmed the microvolt interpretation on October 26, 2026.** The completed run used µV for blank EEG dimension fields. Its original provisional directory and run flags are retained as the audit trail.'
               if confirmed else '**Missing EDF voltage dimensions were assigned µV in memory. Physical units remain unconfirmed.**')
    lines=['# STAGES transport results' if confirmed else '# Provisional STAGES transport results','',
           unit_note,
           'These results use the recorded voltage interpretation; the original explicit-header-only analysis remains separately documented.','',
           f"Analysis participants: {result['n']}; centers: {json.dumps(result['by_center'],sort_keys=True)}.",'',
           f"The change in the macro target-minus-source MAE gap at budget 500 versus 1 was {result['primary_delta_gap_MAE_500_minus1']:.4f} (conditional patient bootstrap 95% interval {lo:.4f} to {hi:.4f}).",
           'This is a transport diagnostic and does not isolate selection optimism from differences in center composition or training sample size.','',
           '| Budget | Source MAE | Held-center MAE | Target minus source |','| --- | --- | --- | --- |']
    for budget,row in summary.iterrows():lines.append(f'| {budget} | {row.source_mae:.4f} | {row.target_mae:.4f} | {row.gap:.4f} |')
    lines+=['','The figure bands describe candidate-order sensitivity, not patient or new-center uncertainty.','',
            '| Fixed interface | Macro held-center MAE |','| --- | --- |']
    for name,g in baselines.groupby('interface',sort=True):lines.append(f'| {name} | {g.target_mae.mean():.4f} |')
    lines+=['','## Secondary outcomes','']
    for outcome in ['face_memory_correct','pvt_mean_reciprocal_rt']:
        path=root/f'secondary_{outcome}'/'primary_result.json'
        if path.exists():
            r=json.loads(path.read_text());interval=r['conditional_patient_bootstrap_95CI']
            lines.append(f"- {outcome}: n={r['n']}; change in MAE gap {r['primary_delta_gap_MAE_500_minus1']:.4f}; conditional 95% interval {interval[0]:.4f} to {interval[1]:.4f}.")
        else:lines.append(f'- {outcome}: no estimable result file; inspect its readiness report and study log.')
    review=('center mapping, backbone independence, signal-QC attrition, clinical relevance, and the completed numerical and graphical artifacts'
            if confirmed else 'the calibration assumption, center mapping, backbone independence, signal-QC attrition, clinical relevance, and the completed numerical and graphical artifacts')
    lines+=['',f'No journal-acceptance probability is estimated. A manuscript requires review of {review}.','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    print('wrote',root/'RESULTS.md',flush=True)


if __name__=='__main__':main()
