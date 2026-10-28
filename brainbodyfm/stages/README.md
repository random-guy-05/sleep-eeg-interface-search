# Fixed sleep-EEG interface versus within-cohort feature search in STAGES

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

The user confirmed the microvolt interpretation of blank EEG voltage dimensions on October 26, 2026; the executed code and file names retain `assumed_uv` as an audit trail. The STAGES checkpoint used in the completed run had SHA256 `b2a9b8dab3ae8543241d613a80cd6a85a6dfca15573f897a1096861b3915af87`; this does not identify the historical benchmark checkpoint. Source/held-center gap estimates also reflect different training sets and center composition. The original primary gap interval conditions on fitted models and selection. The added paired intervals below refit models and preprocessing; neither procedure samples a population of new centers.

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


## Fixed-interface revision of October 27, 2026

The SLEEP study is titled *A fixed sleep-EEG interface versus within-cohort feature search for predicting cognition across sleep centers*. Its new analyses use 590 STAGES adults from four center families (588 for validated face memory), with Arnav Mana and Ivan Habib affiliated as Independent Researchers. HSP Challenge findings are prior provenance rather than a second new analysis; the original known-overlap benchmark curve is omitted from the revised manuscript.

**Fixed Stable4 means one scalar** with the four CinC signed weights +0.360, +0.270, −0.235 and −0.135. Head identity and weights remain fixed; standardization and a scalar task link are fitted only on source participants. Entering those heads as four separate freely refitted columns is labeled `stable4_four_refitted`, a different comparator.

- `stable4_extension.py` adds a source-mean intercept, the signed score, and 2,000 paired within-center patient bootstrap refits for every cognitive task. Each draw repeats clinical preprocessing, output normalization, inner penalty tuning and model fitting. The searched quartet is held fixed per original outer fold, so contrasts involving it are conditional on feature selection; no full 500-candidate search is repeated within these bootstrap draws. Equal-center and patient-weighted summaries use the same predictions. The original primary gap endpoint is unchanged.
- `stable4_supporting.py` computes candidate ranks, patient-weighted search summaries, standardized BHS–PCET associations and the AHI signal check.
- `audit_hidden_site_provenance.py` checks code-scoped matches in locally available released assets and verifies the official selected-entry results. It requires the referenced model assets and reconstructed membership files obtained separately. Zero hidden-code matches do not establish person-level disjointness, complete head-training membership, or organizer-side branch execution.
- `amendment_04_fixed_interface.json` records that these additions are post hoc. The earlier scalar/four-column ambiguity is not concealed by renaming the old estimates.
- `results/aggregate/stable4_extension/` contains permitted aggregate JSON/CSV outputs. It excludes private input caches, participant IDs, individual predictions and bootstrap working caches.

Executive-function equal-center MAE is 0.7635 for the intercept, 0.7492 for demographics, 0.7511 for clinical, 0.7495 for clinical plus BHS, 0.7573 for clinical plus fixed Stable4, and 0.7494 for clinical plus searched outputs. Fixed Stable4 minus demographics is +0.0081 (95% refitted interval −0.0219 to +0.0342); this does not establish an advantage for fixed Stable4. The freely refitted historical heads yield 0.7418, with an interval versus demographics that also crosses zero. Selected PCET probes lie at the 34th–71st held-center percentiles. The AHI head correlates with observed AHI but does not improve equal-center held-center AHI MAE over demographics under this procedure.

The [official Challenge results](https://moody-challenge.physionet.org/2026/results/) verify age-conditioned validation/test scores of 0.723/0.622 for the selected entry. The [accepted CinC abstract](https://cinc.org/2026/Program/accepted/581.html) identifies Stable4 and both authors. These prior diagnosis-task results are separate from the new concurrent cognitive-task analysis.

Run synthetic safeguards from this folder with `python -m unittest test_analysis.py test_fixed_interface.py`. With the original authorized cohort/manifests/caches restored, use `python stable4_extension.py --bootstrap 2000 --workers 4` and `python stable4_supporting.py`. Scripts write local working results to the ignored `results/stable4_extension/`; released aggregate copies live under `results/aggregate/stable4_extension/`. The original private inventory and gates remain necessary; these commands are not a full public raw-data rerun.

## Age-conditioned analysis added after continuous outcomes

`AGE_CONDITIONED_TRANSFER_PLAN.md` and `amendment_05_age_conditioned.json` freeze
the additional rank analysis before its STAGES results. The original primary
search-gap endpoint and continuous results remain intact. The signed four-head
score's primary rank endpoint requires the original organizer-trained source
normalizer for Challenge submission 2593. That artifact was not recovered in
local, Drive, or accessible GitHub searches; target moments or development-cache
moments are not substitutes. The entry repository's current deployment profile
also requires provenance verification before attribution to its historical
organizer score. There is no OSF registration at this stage.

`age_conditioned_transfer.py` implements the Challenge win/tie rule with
within-center age gaps <=2 years, outcome-only spline/sex/education/center
adjustment, composite cases, center-stratified bootstrap adjustment refits,
within-center label permutations, and the prespecified available comparator and
sensitivity analyses. It verifies that plan/code bytes match the current Git
commit before execution. Primary execution is locked pending recovery of the
original source artifact and the frozen realized-design power stage. An
explicit partial run never produces or substitutes a primary result:

```bash
python -m unittest test_age_conditioned_transfer.py
python age_conditioned_transfer.py --available-only \
  --data-root /path/to/authorized/stages --bootstrap 2000 --permutations 10000
```

The random weights in `data/age_conditioned_random_weights.json` were drawn
before the new metrics; primary-normalized random specificity also needs the
original source scales for every candidate head. Local participant working
files and bootstrap caches belong in ignored `results/age_conditioned_transfer/`.
Only reviewed aggregate summaries can be released. All available-only outputs
must identify the unavailable primary, and within-center normalization remains
an explicitly labeled sensitivity analysis.

## Reconstruction amendment after the sensitivity results

`RECONSTRUCTION_AMENDMENT_06.md` records the subsequent author-authorized
reconstruction from the public **large release, version 2**, with 6,600 records.
The exact submitted date gate is audited in `audit_challenge_date_gate.py` and
documented in `SUPPLEMENT_DATE_GATE_AND_RECONSTRUCTION.md`, including the
organizers' date-free evaluation statement. Missing `CreationTime` disables the
date branch; identifying the remaining source-selected streams is a separate
question because the original organizer-trained artifact remains unavailable.

`reconstruct_challenge_normalization.py` uses immutable snapshots of the exact
submission and pinned upstream source, checks the frozen dataset metadata and
checkpoint hashes, streams one raw record at a time, and preserves the original
68-hour extraction budget and queue rule. It is resumable and records the
completed source roster privately. These are reconstructed source moments, not
an exact recovery of the original runtime-dependent artifact. After extraction
finishes, `run_reconstructed_rank.py` applies the unchanged STAGES endpoint,
bootstrap, permutation and eight exploratory analysis families, including the
previously frozen 500 random scores. Restricted raw files and individual
latents/rosters are never committed. The amendment explicitly discloses that the
planned realized-count power rerun was skipped before the available rank
results; supplied planning percentages are not represented as verified power.
# October 27: full representations and cognitive domains

The completed exploratory follow-up is in
[`REPRESENTATION_AND_DOMAIN_FINDINGS_20261027.md`](REPRESENTATION_AND_DOMAIN_FINDINGS_20261027.md).
It reports all 12 readouts across four outcomes, random-patient versus
held-center evaluation, center classification, and direct paired cognitive-domain
contrasts in the same 561 patients. The
[`aggregate packet`](results/aggregate/representation_and_domain_followup/)
contains every model/metric estimate, interval, branch gate and three figures.
Fitted-readout intervals condition on trained models; the fixed-score domain
bootstrap repeats all data-fitted steps. The nominal attention-versus-executive
contrast does not survive its multiplicity correction, and threshold-free
concordance does not support domain specificity. These findings do not change
the original endpoints or transfer CinC validation to a new model.

[`TIMING14_FIXED_SCORE_PLAN.md`](TIMING14_FIXED_SCORE_PLAN.md) freezes a separate
additional-participant evaluation motivated by those findings. Its source-only
eligibility audit found 87 additional patients under the original 14-day timing
sensitivity. Their associations have not been computed. The private extraction
is queued after the existing 590-person CARR extraction; no participant arrays
or identifiers are included in the aggregate packet. Status snapshots are dated,
not live completion reports.

