# IEEE Access analysis guide

Current release: `ieee-access-v1`, including the October 19 selectors and joint uncertainty analysis. Start with [README.md](README.md) for results and public-data checks. Earlier entries in [REVISION_REPORT.md](REVISION_REPORT.md) describe the manuscript at those stages; the current analysis report is [BASELINE_AND_UNCERTAINTY_REPORT.md](agentic_search/ieee_access_revision/BASELINE_AND_UNCERTAINTY_REPORT.md).

## Public checks

The root README gives commands for the ten unit tests, checking all joint contrast intervals against the saved aggregate draws, and regenerating the added tables and Figure 2. Those operations need no participant data or model calls.

The release includes the accepted one-shot requests/responses, classical-selector source choices and costs, aggregate target results, all eight methods' exclusion summaries, and the 50,000 macro bootstrap draws. The manuscript and supplement are attached to the GitHub release. The code repository excludes individual-level data and IEEE template assets.

## Inputs for authorized-data reruns

The evaluators use:

- `agentic_search/data/source_folds/<held_site>.parquet`: source institutions only, with subject ID, site, age, outcome and feature columns.
- `agentic_search/data/participants.parquet`: IDs, site, age and 539 model outputs, without outcome.
- `agentic_search/sealed/target_labels.parquet`: held-site IDs, site, age and outcome.
- `agentic_search/analysis/output/participant_predictions.parquet`: predictions from the archived searches.
- `agentic_search/ieee_access_revision/results/paired_predictions_private.parquet`: paired feedback/no-feedback predictions.
- `agentic_search/ieee_access_revision/results/feature_selection_extension/predictions_private.parquet`: added-selector predictions.
- `agentic_search/ieee_access_revision/results/pretraining_membership_private.parquet`: participant-level development matches.

Cohort reconstruction also uses historical extraction artifacts under `experiments/philosophers_stone/` and `brainbodyfm/cache/`. These inputs are private and are not supplied by this release. Downloading raw EEG alone does not restore missing historical selection logs or the original checkpoint bytes.

## Added selectors

`feature_selection_extension.py` has separate `plan`, `propose`, `source` and `target` phases. The protocol and source choices were saved before target scoring. `propose` makes model calls; the other fitting/scoring phases need authorized private data. Read the archived protocol and implementation notes before rerunning. Use a new destination for a new experiment; do not replace the saved calls or source choices.

With the original private predictions available, these commands recompute the joint intervals and test-only exclusions:

```sh
python3 -m agentic_search.ieee_access_revision.scripts.joint_uncertainty
python3 -m agentic_search.ieee_access_revision.scripts.extension_exclusion_and_overlap
```

The joint bootstrap resamples runs and test participants together. Participant draws are shared across methods; one-shot quartet indices and random-bank ordering indices are shared across hospitals. Original ten-query runs and feedback continuations share opening-batch indices. The three added one-shot/classical contrasts use the same draws as the search contrasts. Models are not refitted and source cohorts are not resampled.

## Earlier analyses

The cohort/provenance audits, metric sensitivities and matched feedback experiment remain in `agentic_search/ieee_access_revision/scripts/`. Original participant-only and run-only intervals are retained as separate conditional analyses. The current main intervals combine their two empirical variation sources; they do not account for all sources of clinical or model uncertainty.

Both authors confirmed manuscript responsibility and authorized credentialed HSP use. The source dataset's IRB approval remains distinct from access authorization. No separate ethics review or exemption determination was obtained for this secondary analysis. The related accepted non-archival AgenticLS version is disclosed in the manuscript. No journal submission is performed by this repository release.
