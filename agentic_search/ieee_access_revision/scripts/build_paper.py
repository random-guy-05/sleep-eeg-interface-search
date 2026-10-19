"""Generate journal tables, scientific figures and a consistent IEEE Access source."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PACKAGE=Path(__file__).resolve().parents[2];REV=PACKAGE/'ieee_access_revision'
OUT=REV/'results';PAPER=REV/'paper'
LABELS={'full_agent':'Agent + robust','random_source_robust':'Random + robust',
        'random_source_mean':'Random + mean','evolutionary_robust':'Evolutionary',
        'taxonomy_diverse_robust':'Taxonomy-diverse'}

def tex(s):
    return str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('&',r'\&').replace('%',r'\%').replace('#',r'\#')

def f(x):return f'{x:.3f}'
def ci(x,l,h):return f'{x:+.3f} [{l:+.3f}, {h:+.3f}]'
def table(name,headers,rows,spec,caption,label,wide=False):
    env='table*' if wide else 'table'
    text='\\begin{'+env+'}[t]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n'
    text+='\\caption{'+caption+'}\\label{'+label+'}\n'
    text+='\\begin{tabular}{'+spec+'}\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'
    text+='\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)
    text+='\n\\bottomrule\n\\end{tabular}\n\\end{'+env+'}\n'
    (PAPER/(name+'.tex')).write_text(text)

def main():
    PAPER.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    m=pd.read_csv(PACKAGE/'analysis/output/final_selection_metrics.csv')
    wanted=((m.method.eq('full_agent')&m.selector.eq('robust'))|
       (m.method.eq('random_source_robust')&m.selector.eq('source_robust'))|
       (m.method.eq('random_source_mean')&m.selector.eq('source_mean'))|
       (m.method.isin(['evolutionary_robust','taxonomy_diverse_robust'])&m.selector.eq('robust')))
    m=m[wanted];means=m.groupby(['method','budget','held_site']).target_auc.mean().groupby(['method','budget']).mean()
    methods=list(LABELS)
    rows=[]
    for method in methods:
        rows.append([LABELS[method]]+[f(means.loc[(method,b)]) if (method,b) in means.index else '---' for b in [10,25,50,500]])
    table('budget_table',['Method','$B=10$','$B=25$','$B=50$','$B=500$'],rows,'lrrrr',
          'Original full-cohort macro age-conditioned AUROC. These are historical benchmark results, including S0001 model-development overlap.','tab:budget')
    fig,ax=plt.subplots(figsize=(6.4,3.1))
    colors=['#2166ac','#b2182b','#ef8a62','#555555','#1b7837']
    for method,color in zip(methods,colors):
        s=means.loc[method];s=s[s.index.isin([10,25,50,100,500])]
        ax.plot(s.index,s.values,'o-',label=LABELS[method],color=color,lw=1.5,ms=4)
    ax.set_xscale('log');ax.set_xticks([10,25,50,100,500],[10,25,50,100,500]);ax.set_ylim(.53,.65)
    ax.set(xlabel='Candidate evaluation budget',ylabel='Macro age-conditioned AUROC')
    ax.legend(ncol=2,fontsize=8,loc='upper right');fig.tight_layout();fig.savefig(PAPER/'budget.pdf');plt.close(fig)
    contrasts=pd.read_csv(PACKAGE/'analysis/output/participant_bootstrap_contrasts.csv')
    rows=[]
    for r in contrasts.itertuples():
        comp='Random, B=500' if '500' in r.comparison else ('Random' if 'random' in r.comparison else ('Evolutionary' if 'evolution' in r.comparison else 'Taxonomy-diverse'))
        rows.append([comp,str(r.budget),ci(r.macro_difference,r.bootstrap_ci_2_5,r.bootstrap_ci_97_5)])
    table('contrast_table',['Comparator','$B$','Agent minus comparator [95\\% CI]'],rows,'llr',
        'Original participant-level intervals, conditional on fixed selections. B refers to the agent budget; the random-bank row compares B=10 with B=500. Contrasts are exploratory and intervals are pointwise.','tab:contrasts',True)
    flow=pd.read_csv(OUT/'cohort_flow.csv');demo=pd.read_csv(OUT/'available_demographic_frame.csv')
    rows=[['Available labeled frame']+[str(int(demo[demo.SiteID.eq(s)].available_records.iloc[0])) for s in ['I0002','I0006','S0001']]]
    for stage,label in [('initial_recovered','Initial recovered'),('positive_batch1_requested','Positive batch 1: requested'),('positive_batch1_completed','Positive batch 1: recovered'),('positive_batch2_requested','Positive batch 2: requested'),('positive_batch2_completed','Positive batch 2: recovered'),('negative_extension_requested','Negative extension: requested'),('negative_extension_completed','Negative extension: recovered'),('final_frozen','Final benchmark')]:
        rows.append([label]+[str(int(flow[(flow.stage.eq(stage))&flow.site.eq(s)].n.iloc[0])) for s in ['I0002','I0006','S0001']])
    rows.append(['Final positive / negative','7 / 36','20 / 147','56 / 72'])
    rows.append(['Eligible age-matched pairs','41','441','812'])
    table('cohort_table',['Stage','I0002','I0006','S0001'],rows,'lrrr',
        'Recoverable outcome-aware cohort assembly. Requested and recovered rows are alternatives, not additive stages. The initial cache plus the three recovered extensions sum to 338. The full HSP population denominator and initial extraction failures are unknown.','tab:cohort',True)
    overlap=pd.read_csv(OUT/'pretraining_overlap_summary.csv')
    rows=[[r.site,str(r.n),str(r.training_subject_matches),str(r.training_record_matches),str(r.development_subject_matches),f'{r.retained} ({r.retained_positive}/{r.retained_negative})',str(r.retained_age_pairs)] for r in overlap.itertuples()]
    table('overlap_table',['Site','$n$','Train subjects','Train records','Development subjects','Retained (pos/neg)','Pairs'],rows,'lrrrrlr',
        'Identifier matches to published reconstructed checkpoint training and conservative development manifests. Retained excludes every conservative-development subject. Exact historical extraction checkpoint bytes are unavailable.','tab:overlap',True)
    sensitivity=pd.read_csv(OUT/'metric_sensitivity.csv',dtype={'tolerance':str})
    sc=pd.read_csv(OUT/'metric_sensitivity_contrasts.csv',dtype={'tolerance':str});rows=[]
    for tol in ['1.0','2.0','3.0','inf']:
        s=sensitivity[(sensitivity.site.eq('macro'))&sensitivity.tolerance.eq(tol)].set_index('method')
        c=sc[(sc.site.eq('macro'))&sc.tolerance.eq(tol)&sc.comparison.eq('full_agent_minus_random_source_robust')].iloc[0]
        rows.append(['Conventional' if tol=='inf' else tol+' years']+[f(s.loc[k,'auc']) for k in ['full_agent','random_source_robust','evolutionary_robust','taxonomy_diverse_robust']]+[ci(c.difference,c.ci_low,c.ci_high)])
    table('sensitivity_table',['Metric','Agent','Random','Evolutionary','Taxonomy','Agent $-$ random [95\\% CI]'],rows,'lrrrrr',
        'Secondary metric sensitivity on the original full cohort at B=10. Interfaces, source fits and selections are fixed; only the evaluation metric changes. Conventional AUROC does not condition on age.','tab:sensitivity',True)
    excluded=pd.read_csv(OUT/'overlap_exclusion_sensitivity.csv')
    rows=[]
    for site in ['S0001','macro_three_retained_sites','macro_I0002_I0006_only']:
        s=excluded[excluded.site.eq(site)].set_index('method')
        label={'S0001':'S0001 retained (7 people, 4 pairs)','macro_three_retained_sites':'Macro, three retained sites','macro_I0002_I0006_only':'Macro, I0002 and I0006 only'}[site]
        rows.append([label]+[f(s.loc[k,'auc']) for k in ['full_agent','random_source_robust','evolutionary_robust','taxonomy_diverse_robust']])
    table('exclusion_table',['Evaluation set','Agent','Random','Evolutionary','Taxonomy'],rows,'lrrrr',
        'Secondary B=10 sensitivity after excluding model-development matches from target evaluation, with original interfaces and source-fitted probes fixed. The seven-person S0001 remainder is too sparse to establish reliable independent validation. The two-site macro is a separate descriptive estimand.','tab:exclusion',True)
    order=['full_agent','random_source_robust','evolutionary_robust','taxonomy_diverse_robust']
    rows=[['Original three-site benchmark']+[f(means.loc[(k,10)]) for k in order]]
    for site,label in [('macro_I0002_I0006_only','I0002/I0006 descriptive summary'),('macro_three_retained_sites','Three-site development-exclusion sensitivity')]:
        s=excluded[excluded.site.eq(site)].set_index('method')
        rows.append([label]+[f(s.loc[k,'auc']) for k in order])
    table('evaluation_scope_table',['Evaluation scope','Agent','Random','Evolutionary','Taxonomy'],rows,'lrrrr',
        'B=10 scores under distinct evaluation scopes. The two-site summary is post hoc and retains all original source fits, including the other institution. The development-exclusion sensitivity retains seven S0001 participants and four eligible pairs; it is too sparse for reliable independent third-site validation. The rows are not interchangeable estimates.','tab:scope',True)
    catalog=pd.read_csv(PACKAGE/'data/heads.csv').set_index('head_id').taxonomy_category.to_dict()
    selected=m[m.method.eq('full_agent')&m.selector.eq('robust')].copy()
    selected['category_count']=selected.quartet_key.map(lambda q:len({catalog[h] for h in q.split('|')}))
    proposed=pd.read_csv(PACKAGE/'analysis/output/quartet_category_diversity.csv')
    convergence=pd.read_csv(PACKAGE/'analysis/output/selected_quartet_convergence.csv')
    rows=[]
    for b in [10,25,50]:
        pc=proposed[proposed.method.eq('full_agent')&proposed.budget.eq(b)].mean_unique_categories.iloc[0]
        sc=selected[selected.budget.eq(b)].category_count.mean()
        j=convergence[convergence.method.eq('full_agent')&convergence.budget.eq(b)].mean_jaccard_selected_heads.mean()
        rows.append([str(b),f'{pc:.2f}',f'{sc:.2f}',f'{j:.3f}'])
    table('structure_table',['Budget','Proposed','Selected','Head overlap'],rows,'rrrr',
        'Physiological category coverage and convergence. Proposed and selected columns give mean distinct categories per quartet. Head overlap is mean pairwise Jaccard similarity of selected heads across runs, averaged over institutions. Categories were visible to the controller; no diversity rule was enforced.','tab:structure')
    if (OUT/'feedback_ablation_summary.csv').exists():
        fb=pd.read_csv(OUT/'feedback_ablation_summary.csv');rows=[]
        for r in fb.itertuples():
            rows.append([tex(r.site),f(r.feedback_auc),f(r.no_feedback_auc),ci(r.difference,r.participant_ci_low,r.participant_ci_high),f'[{r.run_ci_low:+.3f}, {r.run_ci_high:+.3f}]'])
        table('feedback_table',['Site','Feedback','No feedback','$\\Delta$ [participant 95\\% CI]','Run 95\\% CI'],rows,'lrrrr',
            'Post hoc paired ten-query feedback ablation. Each arm shares the same archived five-candidate opening; both continuations use the same current model ID and source selector. Ten paired continuations per institution. Intervals describe separate conditional uncertainty sources.','tab:feedback',True)
        macro=fb[fb.site.eq('macro')].iloc[0]
        macros={'FeedbackAUC':f(macro.feedback_auc),'NoFeedbackAUC':f(macro.no_feedback_auc),
             'FeedbackContrast':ci(macro.difference,macro.participant_ci_low,macro.participant_ci_high),
             'FeedbackRunCI':f'[{macro.run_ci_low:+.3f}, {macro.run_ci_high:+.3f}]'}
        (PAPER/'results_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
        paired=pd.read_csv(OUT/'paired_target_results.csv')
        fig,axes=plt.subplots(1,3,figsize=(6.6,2.6),sharey=True)
        for ax,site in zip(axes,['I0002','I0006','S0001']):
            p=paired[paired.site.eq(site)].pivot(index='seed',columns='arm',values='target_auc')
            for r in p.itertuples():ax.plot([0,1],[r.no_feedback,r.feedback],color='#808080',alpha=.55,lw=.8)
            ax.scatter(np.zeros(len(p)),p.no_feedback,s=12,c='#b2182b',label='No feedback')
            ax.scatter(np.ones(len(p)),p.feedback,s=12,c='#2166ac',label='Feedback')
            ax.set_xticks([0,1],['No feedback','Feedback'],rotation=20);ax.set_title(site);ax.set_ylim(0,1)
        axes[0].set_ylabel('Held-site age-conditioned AUROC');fig.tight_layout();fig.savefig(PAPER/'feedback.pdf');plt.close(fig)
    prompt=(PACKAGE/'configs/prompts/full_agent.txt').read_text()
    (PAPER/'controller_prompt.tex').write_text('\\begin{quote}\\small\\raggedright\\sloppy\n'+tex(prompt).replace(',',r',\allowbreak{} ').replace('\n\n','\n\n')+'\n\\end{quote}\n')
    # Keep supplementary tables beside their discussion in the standalone article.
    for name in ['feedback','overlap','exclusion','contrast','sensitivity']:
        source=(PAPER/(name+'_table.tex')).read_text()
        source=source.replace(r'\begin{table*}[t]',r'\begin{table}[H]').replace(r'\begin{table}[t]',r'\begin{table}[H]').replace(r'\end{table*}',r'\end{table}')
        (PAPER/('supplement_'+name+'_table.tex')).write_text(source)
    print('Generated tables and figures')

if __name__=='__main__':main()
