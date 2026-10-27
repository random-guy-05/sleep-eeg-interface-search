"""Summarize reporting fields without exporting participant-level records.

Run against the original authorized inputs and frozen analysis manifests.
Only aggregate counts, supplied-date ranges, and input hashes are written.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OUT = BASE / 'results/reporting_completeness'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cohort_path = BASE / 'data/cohort_before_signal_qc.csv'
    ids_path = BASE / 'results/provisional_assumed_uv/nested_transport/analysis_subjects.csv'
    cohort = pd.read_csv(cohort_path)
    ids = pd.read_csv(ids_path)
    frame = ids.merge(cohort, on=['subject_id', 'center'],
                      suffixes=('', '_source'), validate='one_to_one')
    assert len(frame) == 590 and frame.subject_id.is_unique
    rows, dates, outcomes = [], [], []
    groups = [('Overall', frame)] + list(frame.groupby('center', sort=True))
    for center, group in groups:
        for column in ['age', 'sex', 'education', 'bmi', 'ahi', 'ess', 'phq9']:
            value = group[column].replace('', np.nan)
            missing = int(value.isna().sum())
            unknown = int(value.eq('not_known').sum()) if column == 'education' else 0
            rows.append(dict(center=center, variable=column, n=len(group),
                             missing_n=missing, unknown_category_n=unknown))
        for column in ['psg_date', 'cnb_date']:
            value = pd.to_datetime(group[column], errors='raise')
            dates.append(dict(dataset='STAGES', group=center, field=column,
                              n=len(group), missing_n=int(value.isna().sum()),
                              supplied_date_min=value.min().date().isoformat(),
                              supplied_date_max=value.max().date().isoformat()))
        masks = {
            'executive_function': group.pcet_accuracy2.notna(),
            'face_memory': group.face_memory_correct.notna()
                           & group.face_memory_status.eq('C1')
                           & group.face_memory_validation.eq('V'),
            'vigilance': group.pvt_mean_reciprocal_rt.notna()
                         & group.pvt_status.eq('C1'),
        }
        columns = dict(executive_function='pcet_accuracy2',
                       face_memory='face_memory_correct',
                       vigilance='pvt_mean_reciprocal_rt')
        for name, valid in masks.items():
            outcomes.append(dict(center=center, outcome=name, n=len(group),
                                 missing_value_n=int(group[columns[name]].isna().sum()),
                                 unavailable_or_ineligible_n=int((~valid).sum()),
                                 analyzed_n=int(valid.sum())))

    arrays = dict(semantic=841, latent=1024, brain_health_score=1)
    representation = {name: dict(n_participants=590, n_features=width,
                                nonfinite_values=0, participants_with_nonfinite=0)
                      for name, width in arrays.items()}
    def inspect_cache(sid):
        result = {}
        with np.load(BASE / 'cache/assumed_uv/subjects' / f'{sid}.npz',
                     allow_pickle=False) as payload:
            for name, width in arrays.items():
                value = np.asarray(payload[name]).reshape(-1)
                assert value.size == width
                nonfinite = int((~np.isfinite(value)).sum())
                result[name] = nonfinite
        return result

    with ThreadPoolExecutor(max_workers=8) as workers:
        for i, result in enumerate(workers.map(inspect_cache, frame.subject_id), 1):
            for name, nonfinite in result.items():
                representation[name]['nonfinite_values'] += nonfinite
                representation[name]['participants_with_nonfinite'] += int(nonfinite > 0)
            if i % 100 == 0 or i == len(frame):
                print(f'Validated frozen representation records: {i}/{len(frame)}', flush=True)
    assert all(x['nonfinite_values'] == 0 for x in representation.values())

    manifest_path = ROOT / 'brainbodyfm/cache/common_interface_manifest.csv'
    raw_path = ROOT / 'demographics.csv'
    manifest = pd.read_csv(manifest_path)
    raw = pd.read_csv(raw_path, dtype={'BDSPPatientID': str})
    raw['record_id'] = raw.BidsFolder + '_ses-' + raw.SessionID.astype(int).astype(str)
    hsp = manifest.merge(raw, on='record_id', validate='one_to_one')
    assert len(hsp) == 338
    for column in ['CreationTime', 'Last_Known_Visit_Date']:
        value = pd.to_datetime(hsp[column], errors='raise')
        dates.append(dict(dataset='Human Sleep Project benchmark', group='Overall',
                          field=column, n=338, missing_n=int(value.isna().sum()),
                          supplied_date_min=value.min().date().isoformat(),
                          supplied_date_max=value.max().date().isoformat()))
    qc_path = ROOT / 'brainbodyfm/cache/representation_qc.json'
    qc = json.loads(qc_path.read_text())
    hsp_representation = {}
    for name, item in qc.items():
        finite = item['finite_fraction_by_feature']
        assert item['n_subjects'] == 338 and min(finite) == 1
        hsp_representation[name] = dict(n_participants=item['n_subjects'],
                                        n_features=item['n_features'],
                                        minimum_finite_fraction=min(finite),
                                        evidence='saved per-feature representation QC')
    followup = []
    for label, group in hsp.groupby('outcome'):
        for column in ['Time_to_Event', 'Time_to_Last_Visit']:
            value = pd.to_numeric(group[column], errors='coerce')
            followup.append(dict(outcome=int(label), variable=column, n=len(group),
                                 observed_n=int(value.notna().sum()),
                                 minimum=None if value.notna().sum() == 0 else float(value.min()),
                                 median=None if value.notna().sum() == 0 else float(value.median()),
                                 maximum=None if value.notna().sum() == 0 else float(value.max())))
    summary = dict(
        stages_n=590,
        stages_cognitive_minus_psg_days={str(int(k)): int(v) for k, v in
                                       frame.cnb_psg_days.value_counts().sort_index().items()},
        stages_representation=representation,
        benchmark_n=338,
        benchmark_covariate_missing=hsp[['age', 'sex', 'outcome']].isna().sum().astype(int).to_dict(),
        benchmark_representation=hsp_representation,
        benchmark_followup_days=followup,
        date_interpretation='Ranges are supplied deidentified dates. CreationTime is database record creation, not a verified PSG acquisition date. Unshifted recruitment dates cannot be reconstructed.',
        sample_size_basis='STAGES: all available eligible successfully extracted records under fixed criteria; historical benchmark: frozen successfully extracted selected records. No power or precision target was specified in the saved analysis protocols.',
        input_sha256={str(f.relative_to(ROOT)): digest(f) for f in
                      [cohort_path, ids_path, manifest_path, raw_path, qc_path]},
    )
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / 'predictor_missingness.csv', index=False)
    pd.DataFrame(outcomes).to_csv(OUT / 'outcome_availability.csv', index=False)
    pd.DataFrame(dates).to_csv(OUT / 'supplied_date_ranges.csv', index=False)
    (OUT / 'reporting_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ['stages_n', 'stages_cognitive_minus_psg_days',
                                            'stages_representation', 'benchmark_n']}, indent=2))


if __name__ == '__main__':
    main()
