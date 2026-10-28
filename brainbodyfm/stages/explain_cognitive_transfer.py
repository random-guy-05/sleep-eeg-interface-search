"""Reproduce the descriptive why audit without new outcome/feature selection."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
import torch
from threadpoolctl import threadpool_limits
import stable4_extension as data
from analyze import STABLE4
from age_conditioned_transfer import outcome_z


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,required=True);ap.add_argument('--source-work',type=Path,required=True);ap.add_argument('--source-prefix-n',type=int,default=120);a=ap.parse_args()
    root=a.project_root.resolve();work=a.source_work.resolve();data.BASE=root/'brainbodyfm/stages';data.OUT=data.BASE/'results/stable4_extension'
    f,sem,_,bhs,names=data.load_inputs();oldp=np.load(data.BASE/'results/cinc_method_replication/composite_private_predictions.npz')
    ix=pd.Index(f.subject_id).get_indexer(oldp['ids']);assert (ix>=0).all();f=f.iloc[ix].reset_index(drop=True);sem=sem[ix];bhs=bhs[ix]
    risk=oldp['scores'][:,0];adjusted=outcome_z(f,['pcet_accuracy2','face_memory_correct'])
    heads=pd.read_csv(root/'experiments/philosophers_stone/pred_heads_full_338.csv',dtype={'subject_id':str})
    report={'purpose':'Describe endpoint, age and population shifts; not causal attribution or new predictive endpoint','n_stages':len(f),'n_hsp_development':len(heads),'correlations':{},'age_profiles':{},'head_distribution_shift':{}}
    for name,x in [('stable4',risk),('negated_brain_health',-bhs)]:
        report['correlations'][name]={}
        for target,y in [('age',f.age.to_numpy(float)),('raw_PCET',f.pcet_accuracy2.to_numpy(float)),('raw_face_memory',f.face_memory_correct.to_numpy(float)),('original_adjusted_composite',adjusted)]:
            xc=x-pd.Series(x).groupby(f.center).transform('mean').to_numpy();yc=y-pd.Series(y).groupby(f.center).transform('mean').to_numpy()
            report['correlations'][name][target]={'Pearson':float(pearsonr(x,y).statistic),'Spearman':float(spearmanr(x,y).statistic),'within_center_Pearson':float(pearsonr(xc,yc).statistic)}
    for name,ages in [('STAGES',f.age.to_numpy(float)),('HSP_development',heads.age.to_numpy(float)),('HSP_development_cases',heads.loc[heads.target.eq(1),'age'].to_numpy(float)),('HSP_development_noncases',heads.loc[heads.target.eq(0),'age'].to_numpy(float))]:
        report['age_profiles'][name]={'n':len(ages),'mean':float(ages.mean()),'sd':float(ages.std()),'median':float(np.median(ages)),'min':float(ages.min()),'max':float(ages.max()),'share_age60plus':float(np.mean(ages>=60))}
    for h in STABLE4:
        source=heads[h].to_numpy(float);target=sem[:,names.index(h)]
        report['head_distribution_shift'][h]={'HSP_development_mean':float(source.mean()),'HSP_development_sd':float(source.std()),'STAGES_mean':float(target.mean()),'STAGES_sd':float(target.std()),'STAGES_to_HSP_sd_ratio':float(target.std()/source.std()),'mean_shift_in_HSP_sd':float((target.mean()-source.mean())/source.std())}
    report['historical_source_scales_are_descriptive_not_original_organizer_constants']=True
    state=json.loads((work/'state_private.json').read_text());completed=state['completed'][:a.source_prefix_n]
    if len(completed)!=a.source_prefix_n:raise ValueError('Requested source prefix not complete')
    old=pd.read_csv(root/'brainbodyfm/cache/raw_latent_1024_338.csv',dtype={'subject_id':str}).set_index('subject_id');heads=heads.set_index('subject_id')
    matches=[]
    for row in completed:
        match=re.match(r'^sub-([IS]\d{4})(\d+)_ses-\d+$',row['key'])
        if match and match.group(2) in old.index and match.group(2) in heads.index and match.group(1)==heads.loc[match.group(2),'site_id']:matches.append((row,match.group(2)))
    comparison={'n_current_source_records':len(completed),'matched_original_HSP_development_records':len(matches),'mapping':'exact site plus numeric subject identifier extracted from BidsFolder; matching session 1'}
    if matches:
        keys=[sid for row,sid in matches];x=np.vstack([np.load(work/'latents'/row['latent_file']) for row,sid in matches]);y=old.loc[keys].to_numpy(float);rel=np.linalg.norm(x-y,axis=1)/np.linalg.norm(y,axis=1)
        comparison['latent_relative_L2']={'median':float(np.median(rel)),'maximum':float(rel.max()),'minimum':float(rel.min())}
        queue=json.loads((work/'queue_private.json').read_text());queue={v['record']['BidsFolder']+'_ses-'+str(v['record']['SessionID']):v for v in queue}
        comparison['metadata']={'age_max_abs_difference':float(max(abs(float(queue[row['key']]['age'])-heads.loc[sid,'age']) for row,sid in matches)),'sex_disagreements':sum(float(queue[row['key']]['sex'])!=float(heads.loc[sid,'sex']) for row,sid in matches)}
        state_dict=torch.load(work.parent/'SleepPhilosophersStone.ckpt',map_location='cpu',weights_only=False,mmap=True)['state_dict'];projected=[]
        for name in STABLE4:
            suffix=name.removeprefix('pred_regression_');w=state_dict['heads.final_head_'+suffix+'.weight'].numpy().astype(float);b=state_dict['heads.final_head_'+suffix+'.bias'].numpy().astype(float);projected.append((x@w.T+b).reshape(-1))
        projected=np.column_stack(projected);ref=heads.loc[keys,STABLE4].to_numpy(float);sd=heads[STABLE4].std(ddof=0).to_numpy()
        comparison['head_comparison']={'names':STABLE4,'median_abs_difference_in_old_source_sd':np.median(np.abs(projected-ref),axis=0).tolist(),'max_abs_difference_in_old_source_sd':np.max(np.abs(projected-ref),axis=0).tolist(),'new_old_correlations':[float(np.corrcoef(projected[:,j],ref[:,j])[0,1]) for j in range(4)]}
    report['current_vs_historical_HSP_inference']=comparison
    out=data.BASE/'results/general_cognitive_prediction/explanatory_audit.json'
    previous=json.loads(out.read_text())
    # The descriptive audit was executed before this script was packaged.
    # Require exact reproduction of its data rather than overwriting a change.
    assert json.dumps(report,sort_keys=True)==json.dumps(previous,sort_keys=True)
    print('Descriptive why audit reproduced exactly; matched source participants:',len(matches))


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
