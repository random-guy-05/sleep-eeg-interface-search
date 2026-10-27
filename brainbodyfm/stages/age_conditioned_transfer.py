"""Frozen post-continuous-analysis rank evaluation. Restricted inputs stay local.

Default execution requires the ORIGINAL trained Challenge normalizer. An explicit
--available-only run reports the primary endpoint as unavailable and evaluates
only the prespecified comparisons that do not require that missing artifact.
No target-derived scale is ever substituted for the primary source scale.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from analyze import covariate_design, STABLE4
from stable4_extension import load_inputs, CENTERS
import stable4_extension

BASE = Path(__file__).resolve().parent
SIGNED = np.array([.360, .270, -.235, -.135])
RANDOM_SEED = 2026102721
BOOTSTRAP_SEED = 2026102722
PERMUTATION_SEED = 2026102723
PLAN_FILES = ['age_conditioned_transfer.py', 'test_age_conditioned_transfer.py',
              'AGE_CONDITIONED_TRANSFER_PLAN.md', 'amendment_05_age_conditioned.json',
              'data/age_conditioned_random_weights.json']


def standardize(x):
    x = np.asarray(x, float)
    sd = np.std(x, axis=0, ddof=0)
    return (x - np.mean(x, axis=0)) / np.where(sd < 1e-8, 1., sd)


def outcome_z(frame, columns):
    """Outcome-only OLS; repeat the complete spline/adjustment in each bootstrap."""
    design, _ = covariate_design(frame, frame, clinical=False)
    centers = np.column_stack([frame.center.eq(c).to_numpy(float) for c in CENTERS[1:]])
    x = np.column_stack([np.ones(len(frame)), design, centers])
    y = frame[list(columns)].to_numpy(float)
    residuals = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    return standardize(np.mean(standardize(residuals), axis=1))


def eligible_pairs(ages, centers):
    """Unordered distinct within-center pairs with inclusive two-year age gap."""
    a, b = np.triu_indices(len(ages), k=1)
    keep = (centers[a] == centers[b]) & (np.abs(ages[a] - ages[b]) <= 2.)
    return a[keep], b[keep]


def rank_auc(labels, scores, pairs):
    """Same win/tie accounting as official Challenge metric, one center or pooled."""
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, float)
    if scores.ndim == 1:
        scores = scores[:, None]
    a, b = pairs
    keep = labels[a] != labels[b]
    a, b = a[keep], b[keep]
    if not len(a):
        return np.full(scores.shape[1], np.nan), 0
    delta = (scores[a] - scores[b]) * np.where(labels[a], 1., -1.)[:, None]
    return np.mean((delta > 0) + .5 * (delta == 0), axis=0), len(a)


def continuous_concordance(z, scores, pairs):
    a, b = pairs
    keep = np.abs(z[a] - z[b]) >= .25
    a, b = a[keep], b[keep]
    if not len(a):
        return np.nan
    delta = (scores[a] - scores[b]) * np.sign(z[b] - z[a])
    return float(np.mean((delta > 0) + .5 * (delta == 0)))


def load_normalization(path):
    """Accept a human-auditable export of the original artifact; never unpickle."""
    if path is None:
        return None
    obj = json.loads(Path(path).read_text())
    required = ['artifact_sha256', 'submission_id', 'source', 'selected_deployment_recipe',
                'columns', 'direction', 'weights', 'feat_mean', 'feat_scale']
    if any(k not in obj for k in required):
        raise ValueError('Original-artifact export lacks required provenance or arrays')
    if obj['submission_id'] != 2593 or obj['columns'] != STABLE4:
        raise ValueError('Wrong submission or Stable4 head ordering')
    if len(obj['artifact_sha256']) != 64:
        raise ValueError('Original trained artifact SHA256 is required')
    np.testing.assert_array_equal(np.asarray(obj['direction'])[:4], np.sign(SIGNED))
    np.testing.assert_allclose(np.asarray(obj['weights'])[:4], np.abs(SIGNED), rtol=0, atol=1e-12)
    mean, scale = np.asarray(obj['feat_mean'], float)[:4], np.asarray(obj['feat_scale'], float)[:4]
    if mean.shape != (4,) or scale.shape != (4,) or not np.isfinite([mean, scale]).all() or np.any(scale <= 0):
        raise ValueError('Normalizer must contain four finite source moments with positive scales')
    return obj


def fixed_score(heads, normalizer):
    # Exported means are for direction-oriented heads, exactly as in entry code.
    return ((heads * np.sign(SIGNED) - np.asarray(normalizer['feat_mean'])[:4]) /
            np.asarray(normalizer['feat_scale'])[:4]) @ np.abs(SIGNED)


def center_normalized_score(heads, centers):
    out = np.empty(len(heads))
    for center in CENTERS:
        ix = centers == center
        if np.any(ix):
            out[ix] = standardize(heads[ix]) @ SIGNED
    return out


def evaluate(frame, heads, bhs, normalizer, columns, threshold=-1., ahi_head=None):
    centers = frame.center.to_numpy()
    pairs = eligible_pairs(frame.age.to_numpy(float), centers)
    z = None if ahi_head is not None else outcome_z(frame, columns)
    labels = frame.ahi.to_numpy(float) >= 15 if ahi_head is not None else z <= threshold
    if ahi_head is not None:
        names, scores = ['model_ahi'], np.asarray(ahi_head)[:, None]
    else:
        names = ['negated_brain_health', 'stable4_center_normalization_sensitivity']
        scores = np.column_stack([-bhs, center_normalized_score(heads, centers)])
        if normalizer is not None:
            names.insert(0, 'stable4_frozen_source')
            scores = np.column_stack([fixed_score(heads, normalizer), scores])
    pooled, count = rank_auc(labels, scores, pairs)
    metrics = dict(zip(names, map(float, pooled)))
    counts, by_center = {}, {}
    for center in CENTERS:
        ix = centers == center
        a, b = pairs
        cpairs = (a[ix[a]], b[ix[a]])
        ca, cn = rank_auc(labels, scores, cpairs)
        by_center[center] = dict(zip(names, map(float, ca)))
        counts[center] = dict(n=int(ix.sum()), cases=int(labels[ix].sum()),
                              noncases=int((~labels[ix]).sum()), eligible_pairs=cn)
        for name, value in zip(names, ca):
            metrics[name + '__' + center] = float(value)
    for name in names:
        values = [by_center[c][name] for c in CENTERS]
        metrics[name + '__equal_center'] = float(np.mean(values)) if np.isfinite(values).all() else np.nan
    if z is not None:
        for j, name in enumerate(names):
            metrics[name + '__continuous_concordance'] = continuous_concordance(z, scores[:, j], pairs)
        if 'stable4_frozen_source' in names:
            metrics['stable4_minus_brain_health'] = metrics['stable4_frozen_source'] - metrics['negated_brain_health']
    return metrics, counts, labels, pairs, names, scores


def bootstrap_one(args):
    frame, heads, bhs, normalizer, columns, threshold, ahi, seed = args
    rng = np.random.default_rng(seed)
    ix = np.concatenate([rng.choice(np.flatnonzero(frame.center.eq(c)), size=int(frame.center.eq(c).sum()), replace=True) for c in CENTERS])
    with threadpool_limits(limits=1):
        return evaluate(frame.iloc[ix].reset_index(drop=True), heads[ix], bhs[ix], normalizer,
                        columns, threshold, None if ahi is None else ahi[ix])[0]


def permutation_p(labels, scores, pairs, centers, draws, seed):
    point, _ = rank_auc(labels, scores, pairs)
    rng = np.random.default_rng(seed)
    exceed = np.zeros(scores.shape[1], int)
    groups = [np.flatnonzero(centers == c) for c in CENTERS]
    for _ in range(draws):
        permuted = labels.copy()
        for ix in groups:
            permuted[ix] = rng.permutation(labels[ix])
        null, _ = rank_auc(permuted, scores, pairs)
        exceed += null >= point
    return (1 + exceed) / (draws + 1)


def freeze_receipt():
    repo = Path(subprocess.check_output(['git', '-C', str(BASE), 'rev-parse', '--show-toplevel'], text=True).strip())
    sha = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    files = {}
    for name in PLAN_FILES:
        p = BASE / name
        relative = str(p.relative_to(repo))
        committed = subprocess.check_output(['git', '-C', str(repo), 'show', sha + ':' + relative])
        if p.read_bytes() != committed:
            raise RuntimeError('Plan/code differs from frozen Git commit: ' + relative)
        files[name] = hashlib.sha256(committed).hexdigest()
    return dict(commit=sha, files=files, execution_started_utc=datetime.now(timezone.utc).isoformat())


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--normalization-json', type=Path)
    ap.add_argument('--data-root', type=Path, default=BASE,
                    help='Authorized local STAGES analysis folder; never committed')
    ap.add_argument('--available-only', action='store_true')
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--permutations', type=int, default=10000)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    normalizer = load_normalization(args.normalization_json)
    if normalizer is None and not args.available_only:
        raise SystemExit('PRIMARY NOT COMPUTED: original submission 2593 trained normalization is required. Use --available-only only for explicitly labeled secondary results.')
    if normalizer is not None:
        raise SystemExit('Original normalization recovered: verify deployment provenance and run the prespecified realized-design power simulation before unlocking primary computation. Do not bypass this gate.')
    receipt = freeze_receipt()
    data_root = args.data_root.resolve()
    out = data_root / 'results/age_conditioned_transfer'
    out.mkdir(parents=True, exist_ok=True)
    stable4_extension.BASE = data_root
    stable4_extension.OUT = data_root / 'results/stable4_extension'
    frame, semantic, latent, bhs, head_names = load_inputs()
    (out / 'execution_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    heads = semantic[:, [head_names.index(x) for x in STABLE4]]
    tasks = [
        ('composite', ('pcet_accuracy2', 'face_memory_correct'), 'secondary_face_memory_correct'),
        ('executive_function', ('pcet_accuracy2',), 'nested_transport'),
        ('face_memory', ('face_memory_correct',), 'secondary_face_memory_correct'),
        ('vigilance', ('pvt_mean_reciprocal_rt',), 'secondary_pvt_mean_reciprocal_rt'),
        ('ahi_pipeline', (), None),
    ]
    summary = dict(primary_status='not_computed_missing_original_source_normalizer',
                   random_specificity_status='not_computed_missing_source_scales_for_full_candidate_bank',
                   source_artifact=None, freeze=receipt, analyses={})
    for ti, (task, columns, folder) in enumerate(tasks):
        if folder:
            manifest = pd.read_csv(data_root / 'results/provisional_assumed_uv' / folder / 'analysis_subjects.csv')
            keep = frame.subject_id.isin(manifest.subject_id).to_numpy()
        else:
            keep = np.isfinite(frame.ahi.to_numpy(float))
        f, h, b = frame.loc[keep].reset_index(drop=True), heads[keep], bhs[keep]
        ahi = semantic[keep, head_names.index('pred_regression_f_ahi')] if task == 'ahi_pipeline' else None
        thresholds = [-1., -1.5, -.5] if task == 'composite' else [-1.]
        for ki, threshold in enumerate(thresholds):
            key = task + (f'_threshold_{threshold:g}' if task == 'composite' else '')
            with threadpool_limits(limits=1):
                point, counts, labels, pairs, names, scores = evaluate(f, h, b, normalizer, columns, threshold, ahi)
            seeds = np.random.SeedSequence(BOOTSTRAP_SEED + 100 * ti + ki).generate_state(args.bootstrap)
            jobs = [(f, h, b, normalizer, columns, threshold, ahi, int(seed)) for seed in seeds]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                boot = list(pool.map(bootstrap_one, jobs, chunksize=16))
            intervals = {}
            for metric in point:
                draws = np.array([x[metric] for x in boot]); draws = draws[np.isfinite(draws)]
                intervals[metric] = dict(ci_low=float(np.quantile(draws, .025)) if len(draws) else None,
                                         ci_high=float(np.quantile(draws, .975)) if len(draws) else None,
                                         valid_replicates=len(draws), requested_replicates=args.bootstrap)
            ps = permutation_p(labels, scores, pairs, f.center.to_numpy(), args.permutations, PERMUTATION_SEED + 100 * ti + ki)
            result = dict(n=len(f), counts=counts, estimates=point, intervals=intervals,
                          permutation_p_one_sided=dict(zip(names, map(float, ps))), permutation_draws=args.permutations)
            summary['analyses'][key] = result
            np.savez_compressed(out / (key + '_private_bootstrap.npz'), **{m: np.array([x[m] for x in boot]) for m in point})
            (out / 'available_summary.json').write_text(json.dumps(clean_json(summary), indent=2, allow_nan=False) + '\n')
            print(key, json.dumps({n: [point[n], intervals[n]] for n in names}), flush=True)
    print('Available analyses complete. The primary pure-transfer endpoint remains uncomputed.', flush=True)


if __name__ == '__main__':
    main()
