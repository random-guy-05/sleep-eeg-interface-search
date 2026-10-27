"""Apply the explicitly amended reconstructed normalizer to unchanged STAGES tests."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import stable4_extension
from stable4_extension import load_inputs
from age_conditioned_transfer import (STABLE4, SIGNED, evaluate, bootstrap_one,
    permutation_p, rank_auc, freeze_receipt, clean_json, BOOTSTRAP_SEED, PERMUTATION_SEED)
from reconstruct_challenge_normalization import frozen_receipt, digest, ENTRY, META_SHA


def validate_reconstruction(obj):
    if obj.get('reconstructed') is not True or obj.get('original_organizer_artifact') is not False:
        raise ValueError('Must explicitly identify a reconstructed, non-original artifact')
    if obj['entry_commit'] != ENTRY or obj['dataset_version'] != 2 or obj['metadata_sha256'] != META_SHA:
        raise ValueError('Wrong reconstructed source or entry')
    normalizer = obj['normalizer']
    if normalizer['columns'] != list(STABLE4):
        raise ValueError('Changed head identity/order')
    np.testing.assert_array_equal(normalizer['direction'], np.sign(SIGNED))
    np.testing.assert_allclose(normalizer['weights'], [*np.abs(SIGNED), 0.], rtol=0, atol=1e-12)
    scale = np.asarray(normalizer['feat_scale'])
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Nonfinite/nonpositive reconstructed scale')
    return normalizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--normalization-json', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    amendment = frozen_receipt()
    initial_code = freeze_receipt()
    source = json.loads(args.normalization_json.read_text())
    normalizer = validate_reconstruction(source)
    root = args.data_root.resolve()
    out = root / 'results/age_conditioned_transfer'
    out.mkdir(exist_ok=True)
    stable4_extension.BASE = root
    stable4_extension.OUT = root / 'results/stable4_extension'
    frame, semantic, latent, bhs, heads_names = load_inputs()
    heads = semantic[:, [heads_names.index(x) for x in STABLE4]]
    summary = dict(normalization='reconstructed_public_large_v2_runtime_limited_source',
                   original_organizer_artifact=False, source_n=source['source_n'],
                   source_center_counts=source['source_center_counts'],
                   normalizer_sha256=digest(args.normalization_json),
                   source_freeze=source['freeze'], amendment_freeze=amendment,
                   unchanged_analysis_code=initial_code,
                   power_deviation='realized-count rerun not performed before available rank metrics; supplied planning percentages unverified',
                   analyses={})
    tasks = [('composite', ('pcet_accuracy2','face_memory_correct'), 'secondary_face_memory_correct'),
             ('executive_function', ('pcet_accuracy2',), 'nested_transport'),
             ('face_memory', ('face_memory_correct',), 'secondary_face_memory_correct'),
             ('vigilance', ('pvt_mean_reciprocal_rt',), 'secondary_pvt_mean_reciprocal_rt'),
             ('ahi_pipeline', (), None)]
    for ti,(task,columns,folder) in enumerate(tasks):
        if folder:
            ids = pd.read_csv(root/'results/provisional_assumed_uv'/folder/'analysis_subjects.csv').subject_id
            keep = frame.subject_id.isin(ids).to_numpy()
        else:
            keep = np.isfinite(frame.ahi.to_numpy(float))
        f,h,b = frame.loc[keep].reset_index(drop=True), heads[keep], bhs[keep]
        ahi = semantic[keep,heads_names.index('pred_regression_f_ahi')] if task=='ahi_pipeline' else None
        for ki,threshold in enumerate([-1.,-1.5,-.5] if task=='composite' else [-1.]):
            key=task+(f'_threshold_{threshold:g}' if task=='composite' else '')
            with threadpool_limits(limits=1):
                point,counts,labels,pairs,names,scores=evaluate(f,h,b,normalizer,columns,threshold,ahi)
            seeds=np.random.SeedSequence(BOOTSTRAP_SEED+100*ti+ki).generate_state(2000)
            jobs=[(f,h,b,normalizer,columns,threshold,ahi,int(seed)) for seed in seeds]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                boot=list(pool.map(bootstrap_one,jobs,chunksize=16))
            intervals={}
            for metric in point:
                draws=np.array([x[metric] for x in boot]);draws=draws[np.isfinite(draws)]
                intervals[metric]=dict(ci_low=float(np.quantile(draws,.025)) if len(draws) else None,
                                      ci_high=float(np.quantile(draws,.975)) if len(draws) else None,
                                      valid_replicates=len(draws),requested_replicates=2000)
            ps=permutation_p(labels,scores,pairs,f.center.to_numpy(),10000,PERMUTATION_SEED+100*ti+ki)
            result=dict(n=len(f),counts=counts,estimates=point,intervals=intervals,
                        permutation_p_one_sided=dict(zip(names,map(float,ps))),permutation_draws=10000)
            if task=='composite' and threshold==-1.:
                assignment=json.loads((Path(__file__).parent/'data/age_conditioned_random_weights.json').read_text())
                source_names=source['all_head_names'];means=np.asarray(source['all_head_mean']);scales=np.asarray(source['all_head_scale'])
                target_semantic=semantic[keep]
                random_scores=[]
                for candidate in assignment['candidates']:
                    target_ix=[heads_names.index(x) for x in candidate['heads']]
                    source_ix=[source_names.index(x) for x in candidate['heads']]
                    random_scores.append(((target_semantic[:,target_ix]-means[source_ix])/scales[source_ix])@np.asarray(candidate['signed_weights']))
                values,_=rank_auc(labels,np.column_stack(random_scores),pairs)
                score=point['stable4_frozen_source']
                result['random_score_specificity']=dict(candidate_count=500,
                    percentile=float(100*np.mean((values<score)+.5*(values==score))),
                    share_at_or_above=float(np.mean(values>=score)),candidate_aurocs=values.tolist())
                ci=intervals['stable4_frozen_source']
                if ci['ci_low']>.5:interpretation=1
                elif score>.5:interpretation=2
                else:interpretation=3
                result['prespecified_interpretation_category']=interpretation
                result['comparison_to_test_point']=dict(stages_upper_ci=ci['ci_high'],published_test_point=.622,
                    upper_ci_below_test_point=bool(ci['ci_high']<.622),formal_between_population_test=False)
            summary['analyses'][key]=result
            np.savez_compressed(out/(key+'_reconstructed_private_bootstrap.npz'),**{m:np.array([x[m] for x in boot]) for m in point})
            (out/'reconstructed_source_summary.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False)+'\n')
            print(key,json.dumps({n:[point[n],intervals[n]] for n in names}),flush=True)
    summary['complete']=True
    (out/'reconstructed_source_summary.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
