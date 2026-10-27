"""Post hoc ranks, positive control, and standardized cognitive associations."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from scipy.stats import rankdata,pearsonr
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from stable4_extension import load_inputs,CENTERS,TASKS,OUT
from analyze import fit_outer,config,covariate_design


def beta(frame,bhs,weights=None,clinical=False):
    if weights is None:ix=np.arange(len(frame))
    else:ix=np.repeat(np.arange(len(frame)),weights)
    g=frame.iloc[ix];y=g.pcet_accuracy2.to_numpy(float);score=bhs[ix];score=(score-score.mean())/score.std(ddof=0);y=(y-y.mean())/y.std(ddof=0)
    a,_=covariate_design(g,g,clinical=clinical);center=pd.get_dummies(g.center,dtype=float).iloc[:,1:].to_numpy();x=np.c_[np.ones(len(g)),score,a,center];coef=np.linalg.lstsq(x,y,rcond=None)[0]
    return float(coef[1])


def main():
    OUT.mkdir(parents=True,exist_ok=True);frame,sem,latent,bhs,names=load_inputs();nmap={n:i for i,n in enumerate(names)};rank_rows=[];weight_rows=[]
    for task,folder,column in TASKS:
        root=Path(__file__).parent/'results/provisional_assumed_uv'/folder;f=pd.read_csv(root/'analysis_subjects.csv');c=f.center.to_numpy();selection=pd.read_csv(root/'search_budget_selections.csv')
        for center in CENTERS:
            z=np.load(root/f'{center}.npz');target=z['target_mae'];ranks=rankdata(target,method='average');percentiles=100*(500-ranks)/499
            for budget in [1,10,50,100,500]:
                s=selection[(selection.outer==center)&(selection.budget==budget)];chosen=s.draw.to_numpy(int)
                rank_rows.append(dict(task=task,center=center,budget=budget,mean_rank=float(np.mean(ranks[chosen])),mean_percentile=float(np.mean(percentiles[chosen])),min_percentile=float(np.min(percentiles[chosen])),max_percentile=float(np.max(percentiles[chosen])),n_orderings=len(s)))
        for budget,g in selection.groupby('budget'):
            for order,h in g.groupby('permutation'):
                counts=np.array([(f.center==v).sum() for v in h.outer]);weight_rows.append(dict(task=task,budget=int(budget),order=int(order),patient_target_mae=float(np.average(h.target_mae,weights=counts)),equal_target_mae=float(h.target_mae.mean()),equal_source_mae=float(h.source_mae.mean())))
    pd.DataFrame(rank_rows).to_csv(OUT/'held_center_ranks.csv',index=False);pd.DataFrame(weight_rows).to_csv(OUT/'search_weighting_sensitivity.csv',index=False)
    rng=np.random.default_rng(2026102709);centers=frame.center.to_numpy();betas=[];observed={'demographics_center':beta(frame,bhs),'clinical_center':beta(frame,bhs,clinical=True)}
    for i in range(2000):
        w=np.zeros(len(frame),int)
        for center in CENTERS:
            ix=np.flatnonzero(centers==center);w[ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)))
        betas.append([beta(frame,bhs,w),beta(frame,bhs,w,clinical=True)])
    rows=[]
    for i,(name,value) in enumerate(observed.items()):
        low,high=np.quantile(np.array(betas)[:,i],[.025,.975]);rows.append(dict(adjustment=name,standardized_beta=value,ci_low=float(low),ci_high=float(high),n=590,bootstrap=2000))
    (OUT/'bhs_standardized_beta.json').write_text(json.dumps(rows,indent=2)+'\n')
    # A physiological positive control: the model AHI output is compared with observed
    # AHI and tested for added prediction beyond demographics. It is never included in the cognitive outcome.
    use=frame.ahi.notna().to_numpy();f=frame[use].reset_index(drop=True);pred=sem[use,nmap['pred_regression_f_ahi']][:,None];cfg=config();cfg['target_column']='ahi'
    control=[]
    with threadpool_limits(limits=1):
        for center in CENTERS:
            baseline=fit_outer(f,np.empty((len(f),0)),center,cfg,clinical=False);augmented=fit_outer(f,pred,center,cfg,clinical=False)
            control.append(dict(center=center,n=baseline['target_n'],demographic_mae=baseline['target_mae'],ahi_output_mae=augmented['target_mae'],delta_mae=augmented['target_mae']-baseline['target_mae']))
    pd.DataFrame(control).to_csv(OUT/'ahi_positive_control.csv',index=False)
    # Correlation is descriptive; predictive errors are held-center estimates.
    corr=pearsonr(f.ahi.to_numpy(float),pred[:,0])
    rng=np.random.default_rng(2026102710);c=f.center.to_numpy();vals=[]
    for _ in range(2000):
        ix=np.concatenate([rng.choice(np.flatnonzero(c==v),size=np.sum(c==v),replace=True) for v in CENTERS])
        vals.append(pearsonr(f.ahi.to_numpy(float)[ix],pred[ix,0]).statistic)
    lo,hi=np.quantile(vals,[.025,.975]);summary=dict(n=len(f),pearson_r=float(corr.statistic),ci_low=float(lo),ci_high=float(hi),bootstrap=2000,equal_demographic_mae=float(np.mean([x['demographic_mae'] for x in control])),equal_augmented_mae=float(np.mean([x['ahi_output_mae'] for x in control])),equal_mae_difference=float(np.mean([x['delta_mae'] for x in control])),by_center_pearson={v:float(pearsonr(f.loc[c==v,'ahi'],pred[c==v,0]).statistic) for v in CENTERS},interpretation='Respiratory signal association recovered; this did not improve equal-center held-center MAE over demographics.')
    (OUT/'ahi_positive_control_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('BHS beta',rows,flush=True);print('AHI positive control',control,'correlation',float(corr.statistic),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
