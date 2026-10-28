"""Bounded, explicitly post hoc STAGES replication of the published CinC method."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
import stable4_extension
from stable4_extension import CENTERS, load_inputs
from analyze import STABLE4, covariate_design
from age_conditioned_transfer import outcome_z, eligible_pairs, rank_auc, clean_json

BASE = Path(__file__).resolve().parent
SIGNED = np.array([.36, .27, -.235, -.135])
FIXED_NAMES = ['stable4', 'equal_weights'] + [f'without_{i+1}' for i in range(4)] + [f'component_{i+1}' for i in range(4)] + ['negated_brain_health']
MODELS = FIXED_NAMES + ['age', 'demographics', 'clinical', 'demographics_plus_four', 'clinical_plus_four']
CONTRASTS = [('stable4', 'demographics'), ('demographics_plus_four', 'demographics'), ('clinical_plus_four', 'clinical')]
PLAN_FILES = ['CINC_METHOD_REPLICATION_PLAN.md', 'cinc_method_replication.py', 'test_cinc_method_replication.py']
ENTRY = None


def load_entry(folder):
    global ENTRY
    manifest = json.loads((BASE/'data/reconstruction_source_manifest.json').read_text())['entry']
    for name, expected in manifest['files'].items():
        if hashlib.sha256((Path(folder)/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Entry source hash mismatch: '+name)
    sys.path.insert(0, str(folder))
    ENTRY = importlib.import_module('phi_challenge.downstream')
    if list(ENTRY.STABLE4_COLUMNS) != STABLE4:
        raise ValueError('Entry and STAGES head order differ')
    return manifest


def frozen_receipt():
    repo = Path(subprocess.check_output(['git','-C',str(BASE),'rev-parse','--show-toplevel'], text=True).strip())
    commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
    hashes = {}
    for name in PLAN_FILES:
        p = BASE/name
        b = subprocess.check_output(['git','-C',str(repo),'show',commit+':'+str(p.relative_to(repo))])
        if p.read_bytes() != b:
            raise ValueError('Commit plan and code before execution: '+name)
        hashes[name] = hashlib.sha256(b).hexdigest()
    return dict(commit=commit, files=hashes, execution_started_utc=datetime.now(timezone.utc).isoformat())


def fixed_fold_scores(train_heads, test_heads, train_sex, test_sex):
    """Exact submitted artifact for Stable4, matched source scales for diagnostics."""
    artifact = ENTRY._fit_stable4_artifact(train_heads, train_sex)
    oriented = test_heads*np.sign(SIGNED)
    z = (oriented-np.asarray(artifact.feat_mean)[:4])/np.asarray(artifact.feat_scale)[:4]
    score = artifact.score_matrix(test_heads, test_sex)
    np.testing.assert_allclose(score, z@np.abs(SIGNED), atol=1e-12)
    values = [score, z.sum(axis=1)]
    for i in range(4):
        weights = np.abs(SIGNED).copy(); weights[i] = 0.
        values.append(z@weights)
    values.extend(z[:,i] for i in range(4))
    return np.column_stack(values), artifact


def classifier_predict(x, y, ages, target):
    weights, intercept, _, _ = ENTRY.fit_linear(x, y.astype(float), ages,
        age_window=2., rank_weight=0., bce_weight=1., l2=1.)
    return target@weights+intercept


def held_center_scores(frame, heads, bhs, labels):
    predictions = np.empty((len(frame), len(MODELS)))
    center = frame.center.to_numpy()
    sex = frame.sex.to_numpy(float)
    details = {}
    for c in CENTERS:
        test = center == c; train = ~test
        fixed, artifact = fixed_fold_scores(heads[train], heads[test], sex[train], sex[test])
        predictions[test,:10] = fixed
        predictions[test,10] = -bhs[test]
        predictions[test,11] = frame.age.to_numpy(float)[test]
        means, scales = heads[train].mean(axis=0), heads[train].std(axis=0)
        scales = np.where(scales < 1e-8, 1., scales)
        za, zb = (heads[train]-means)/scales, (heads[test]-means)/scales
        da, db = covariate_design(frame.loc[train], frame.loc[test], clinical=False)
        ca, cb = covariate_design(frame.loc[train], frame.loc[test], clinical=True)
        for name, a, b in [('demographics', da, db), ('clinical', ca, cb),
                            ('demographics_plus_four', np.c_[da,za], np.c_[db,zb]),
                            ('clinical_plus_four', np.c_[ca,za], np.c_[cb,zb])]:
            predictions[test,MODELS.index(name)] = classifier_predict(a,labels[train],frame.age.to_numpy(float)[train],b)
        details[c] = dict(source_n=int(train.sum()), held_n=int(test.sum()),
            oriented_source_mean=artifact.feat_mean[:4], source_sd=artifact.feat_scale[:4])
    return predictions, details


def evaluate(frame, heads, bhs, columns, keep_predictions=False):
    z = outcome_z(frame, columns)
    labels = z <= -1.
    scores, details = held_center_scores(frame, heads, bhs, labels)
    center = frame.center.to_numpy()
    pairs = eligible_pairs(frame.age.to_numpy(float), center)
    point, n_pairs = rank_auc(labels, scores, pairs)
    a, b = np.triu_indices(len(frame), k=1)
    ordinary_pairs = (a[center[a]==center[b]], b[center[a]==center[b]])
    ordinary, ordinary_n = rank_auc(labels,scores,ordinary_pairs)
    center_metrics, counts = {}, {}
    for c in CENTERS:
        ix = center == c
        a, b = pairs; use = ix[a]
        v, n = rank_auc(labels,scores,(a[use],b[use]))
        center_metrics[c] = dict(zip(MODELS,map(float,v)))
        counts[c] = dict(n=int(ix.sum()),cases=int(labels[ix].sum()),noncases=int((~labels[ix]).sum()),eligible_pairs=n)
    macro = np.array([[center_metrics[c][m] for m in MODELS] for c in CENTERS]).mean(axis=0)
    result = dict(age_conditioned=dict(zip(MODELS,map(float,point))),
                  equal_center=dict(zip(MODELS,map(float,macro))),
                  ordinary_within_center=dict(zip(MODELS,map(float,ordinary))))
    for metric in ['pooled_ordinary_auc','pooled_average_precision']:
        values = [roc_auc_score(labels,scores[:,j]) if metric.endswith('auc') else average_precision_score(labels,scores[:,j]) for j in range(len(MODELS))]
        result[metric] = dict(zip(MODELS,map(float,values)))
    result['paired_differences'] = {m+'__minus__'+b:float(point[MODELS.index(m)]-point[MODELS.index(b)]) for m,b in CONTRASTS}
    if keep_predictions:
        result.update(n=len(frame), cases=int(labels.sum()), eligible_pairs=n_pairs,
            ordinary_within_center_pairs=ordinary_n, counts=counts, by_center=center_metrics,
            source_normalization=details, prevalence=float(labels.mean()))
        return result, labels, pairs, scores
    return result


def worker_init(entry_folder):
    load_entry(entry_folder)


def bootstrap_one(job):
    frame, heads, bhs, columns, seed = job
    rng = np.random.default_rng(seed)
    center = frame.center.to_numpy()
    ix = np.concatenate([rng.choice(np.flatnonzero(center==c), size=int((center==c).sum()), replace=True) for c in CENTERS])
    with threadpool_limits(limits=1):
        return evaluate(frame.iloc[ix].reset_index(drop=True),heads[ix],bhs[ix],columns)


def intervals(draws, family_size=1):
    a = np.asarray(draws,float); a = a[np.isfinite(a)]
    q = .025/family_size
    return dict(ci_low=float(np.quantile(a,q)) if len(a) else None,
                ci_high=float(np.quantile(a,1-q)) if len(a) else None,
                valid_replicates=len(a), requested_replicates=len(draws))


def permutation_fixed(labels, scores, pairs, centers, draws, seed):
    scores = scores[:,:len(FIXED_NAMES)]
    point, _ = rank_auc(labels,scores,pairs)
    rng = np.random.default_rng(seed); exceed = np.zeros(len(FIXED_NAMES),int)
    groups = [np.flatnonzero(centers==c) for c in CENTERS]
    for _ in range(draws):
        shuffled = labels.copy()
        for g in groups: shuffled[g] = rng.permutation(labels[g])
        v, _ = rank_auc(shuffled,scores,pairs)
        exceed += v >= point
    return (1+exceed)/(1+draws)


def holm(values):
    values = np.asarray(values,float)
    order = np.argsort(values,kind='stable')
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1.,np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    return adjusted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--entry-source',type=Path,required=True)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=2)
    args = ap.parse_args()
    receipt = frozen_receipt(); entry = load_entry(args.entry_source)
    root = args.data_root.resolve(); out = root/'results/cinc_method_replication';out.mkdir(exist_ok=True)
    stable4_extension.BASE=root;stable4_extension.OUT=root/'results/stable4_extension'
    frame, semantic, _, bhs, names = load_inputs()
    heads = semantic[:,[names.index(h) for h in STABLE4]]
    tasks = [('composite',('pcet_accuracy2','face_memory_correct'),'secondary_face_memory_correct'),
             ('executive_function',('pcet_accuracy2',),'nested_transport'),
             ('face_memory',('face_memory_correct',),'secondary_face_memory_correct')]
    summary = dict(status='running',freeze=receipt,entry_commit=entry['commit'],
        method='published_Stable4_with_STAGES_source_center_normalization',
        original_HSP_normalizer=False,bootstrap_replicates=2000,permutation_replicates=10000,
        primary_of_extension='composite',prior_results_known=True,models=MODELS,analyses={})
    permutation_keys, permutation_values = [], []
    (out/'execution_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for ti,(task,columns,folder) in enumerate(tasks):
        ids = pd.read_csv(root/'results/provisional_assumed_uv'/folder/'analysis_subjects.csv').subject_id
        use = frame.subject_id.isin(ids).to_numpy()
        f,h,b = frame.loc[use].reset_index(drop=True),heads[use],bhs[use]
        with threadpool_limits(limits=1): result,labels,pairs,scores=evaluate(f,h,b,columns,True)
        print(task,'point',json.dumps({m:result['age_conditioned'][m] for m in ['stable4','demographics','demographics_plus_four','clinical','clinical_plus_four']}),flush=True)
        np.savez_compressed(out/(task+'_private_predictions.npz'),ids=f.subject_id.to_numpy(str),labels=labels,scores=scores,names=np.array(MODELS))
        seeds = np.random.SeedSequence(2026102731+ti).generate_state(2000)
        jobs = [(f,h,b,columns,int(seed)) for seed in seeds]
        started = time.time();boot=[]
        with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init,initargs=(str(args.entry_source),)) as pool:
            for sample in pool.map(bootstrap_one,jobs,chunksize=8):
                boot.append(sample)
                if len(boot)%100==0: print(task,'bootstrap',len(boot),'seconds',round(time.time()-started,1),flush=True)
        result['intervals']={}
        private_draws={}
        for metric in ['age_conditioned','equal_center','ordinary_within_center','pooled_ordinary_auc','pooled_average_precision','paired_differences']:
            result['intervals'][metric]={}
            for m in result[metric]:
                draws = [s[metric][m] for s in boot]
                ci = intervals(draws)
                if metric=='paired_differences':ci['familywise_95_bonferroni']=intervals(draws,9)
                result['intervals'][metric][m] = ci
                private_draws[metric+'__'+m] = np.asarray(draws)
        np.savez_compressed(out/(task+'_private_bootstrap.npz'),**private_draws)
        with threadpool_limits(limits=1):
            ps = permutation_fixed(labels,scores,pairs,f.center.to_numpy(),10000,2026102741+ti)
        result['permutation_p_one_sided'] = dict(zip(FIXED_NAMES,map(float,ps)))
        for m,p in zip(FIXED_NAMES,ps):permutation_keys.append((task,m));permutation_values.append(float(p))
        summary['analyses'][task]=result
        (out/'summary.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False)+'\n')
        print(task,'complete',flush=True)
    for (task,m),p in zip(permutation_keys,holm(permutation_values)):
        summary['analyses'][task].setdefault('permutation_p_holm_33',{})[m]=float(p)
    summary['status']='complete';summary['completed_utc']=datetime.now(timezone.utc).isoformat()
    (out/'summary.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False)+'\n')
    rows=[]
    for task,r in summary['analyses'].items():
        for m in MODELS:
            ci=r['intervals']['age_conditioned'][m]
            rows.append(dict(outcome=task,model=m,n=r['n'],cases=r['cases'],eligible_pairs=r['eligible_pairs'],
                age_conditioned_auc=r['age_conditioned'][m],**ci,
                equal_center_auc=r['equal_center'][m],ordinary_within_center_auc=r['ordinary_within_center'][m],
                permutation_p=r['permutation_p_one_sided'].get(m),holm_p=r.get('permutation_p_holm_33',{}).get(m)))
    pd.DataFrame(rows).to_csv(out/'aggregate_model_results.csv',index=False)
    print('COMPLETE',out/'summary.json',flush=True)


if __name__=='__main__':main()
