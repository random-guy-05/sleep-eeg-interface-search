# STAGES sleep EEG probe search and center transport

This folder contains the analysis code and aggregate results for the STAGES extension of the sleep-EEG probe-search manuscript. The existing repository also contains code for the original cognitive-impairment benchmark and the related IEEE Access analyses; these projects are distinct and must not be treated as identical submissions or results.

The STAGES analysis held out each of four center families, selected among 1, 10, 50, 100, or 500 fixed four-feature candidates using source-only validation, and evaluated executive function, face memory and psychomotor vigilance. Its local dated protocol was created before STAGES EEG–cognition analysis, but was not registered. See `results/aggregate/` for macro budget means, center-level executive-function means, baseline means, and conditional interval summaries. Those files do not contain patient-level records or predictions. The candidate bank is `data/candidate_bank.json`.

## Code map

- `prepare.py`: joins authorized STAGES inputs and writes a private cohort manifest.
- `signal_io.py`, `wavelet.py`, `extract.py`, `verify_preprocessing.py`: channel resolution, signal preprocessing, frozen-model extraction, and preprocessing validation.
- `analyze.py`, `test_analysis.py`: nested center-held-out fitting, source-only tuning, and synthetic leakage/ridge tests.
- `run_study.py`, `report.py`, `audit_headers.py`: execution, reporting, and header audit. These create participant-level files and logs on local storage; do not publish those files.
- `amendment_*.json`: contemporaneous protocol amendments and the subsequent user confirmation of EEG units.
- `requirements-recorded.txt`: exact packages observed during the original run, including the upstream model source revision. Some versions are platform specific. This is a provenance record, not a cross-platform installation guarantee.

## Data access and reproduction boundary

Data are available to approved researchers from the [National Sleep Research Resource](https://sleepdata.org/datasets/stages). This repository does not contain restricted STAGES records, outcomes, participant IDs, participant-level model outputs or predictions, the checkpoint, or private Drive inventories. The previously run pipeline expects the source files and inventories described in `prepare.py`, a verified checkpoint and preprocessing-gate files, and caches with hashes. Its original machine paths and private metadata are deliberately not distributed. Therefore, the aggregate results can be inspected here, but end-to-end reproduction requires authorized inputs and local reconstruction of the documented inventory and gates. A successful full raw-data rerun has not been performed from this public package.

`protocol_template.json` is a privacy-sanitized template of the original protocol: the duplicate-subject list was removed and must be restored from the published NSRR duplicate-EDF notice before use. Renaming this template to `protocol_v1.json` will not preserve the original run's protocol hash. No claim of byte-identical rerun is made. Use the historical source and exact input hashes available to the original investigators if recovering the original run.

The user confirmed the microvolt interpretation of blank EEG voltage dimensions on October 26, 2026; the executed code and file names retain `assumed_uv` as an audit trail. The STAGES checkpoint used in the completed run had SHA256 `b2a9b8dab3ae8543241d613a80cd6a85a6dfca15573f897a1096861b3915af87`; this does not identify the historical benchmark checkpoint. Source/held-center gap estimates also reflect different training sets and center composition. Bootstrap intervals condition on the fitted models and selection; they do not quantify uncertainty from refitting or sampling new centers.

## Post hoc clinical comparisons and cohort accounting

`paired_cognitive_comparisons.py` calculates paired differences in held-center mean absolute error using 20,000 patient resamples stratified within the four observed center families. It holds preprocessing, fitted models, tuning, and feature selections fixed; intervals do not account for refitting or sampling new centers. The comparisons were added after the original outcome analyses, are exploratory, and are not adjusted for multiplicity. Negative differences favor the sleep-derived model. Aggregate outputs are in `results/aggregate/paired_comparison_summary.*`.

`cohort_flow_aggregate.py` assigns nonexclusive exclusion flags to the first matching condition in a documented order. The starting 1,931 identifiers are the union of records in the contributor-selected questionnaire or cognitive-battery sources, not the total STAGES enrollment. Changing the ordering changes intermediate counts; the final 593 eligible and 590 analyzed counts are invariant to the ordering. `clinical_descriptives_aggregate.py` computes center-level and overall baseline summaries. These scripts read restricted participant-level data locally, which are not included here. Only aggregate tables are shared.

## Tests

With the recorded environment available, run from this folder:

```bash
python test_analysis.py
```

The tests verify the ridge implementation against scikit-learn and check that held-center labels and covariates cannot affect source-only selection. They use synthetic data; passing them is not evidence of a full reproduction of restricted-data results.

## Completed reporting tables

`reporting_completeness.py` summarizes the final analysis sample’s supplied date fields, clinical-predictor missingness and unknown education categories, outcome-validation eligibility, and finite frozen representations. It reads authorized local inputs and writes aggregate outputs only. The dates describe the deidentified release; `CreationTime` in the historical benchmark denotes database record creation rather than a verified PSG acquisition date. The original unshifted recruitment calendars cannot be reconstructed from these files. The saved analysis protocols specified no power- or precision-based sample-size target. Aggregate outputs are in `results/aggregate/predictor_missingness.csv`, `outcome_availability.csv`, `supplied_date_ranges.csv`, and `reporting_summary.json`.
