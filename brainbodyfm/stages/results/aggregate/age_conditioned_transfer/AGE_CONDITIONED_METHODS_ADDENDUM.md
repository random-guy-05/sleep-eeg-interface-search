# Age conditioned evaluation and reconstruction amendment

Arnav Mana and Ivan Habib

October 27, 2026

## Analysis chronology

We added an age-conditioned evaluation after completing the continuous STAGES analyses. The plan and analysis code were committed before any STAGES rank metric was computed (b7a357d95d532fe6c01b2cd825f36be456866f64). The original search-gap endpoint and all continuous results were retained. We subsequently specified source-normalization reconstruction after examining the within-center-normalized sensitivity and comparator results, and before reconstructing source constants (amendment e7f3a5637c99d3dfed1415e2553c060959103974; file-transport repair cc9940e). No OSF registration was completed.

## Age conditioned cognitive evaluation

Low concurrent cognitive performance was defined using PCET accuracy2 and face-memory correct responses. We regressed each task on the same cubic age spline used in the continuous analysis, sex, education including its unknown category, and center. We standardized the residuals, averaged the two standardized tasks, and standardized the composite. A composite at or below -1.0 SD defined cases: 103 of 588 participants. The endpoint was the proportion of same-center case–noncase pairs aged within two years in which the case had the higher risk score, with half credit for ties. Pooling eligible pairs across centers yielded 2,193 comparisons. The two-year window reduces age differences but does not force an age-only score to have AUROC 0.5.

We used 2,000 participant bootstrap samples stratified by center, repeating outcome adjustment, residual and composite standardization, case classification, and pairing. A one-sided test used 10,000 within-center case-label permutations, with p calculated as (1 + exceedances)/(10,001). Paired score comparisons shared bootstrap draws. The within-center-normalization sensitivity also recalculated its label-free feature moments in each resample. Center-specific estimates without eligible case–noncase pairs were not estimable; equal-center means required all four centers. All exploratory analyses were retained and were not adjusted for multiplicity.

## Date branch in the submitted system

We inspected the exact final submission, commit 76a36e54251cfce85889a3b2390018ffc23329ee. The system included an optional 80% Stable4/20% date-rank branch. Its date stream read CreationTime and required at least 90% parseable dates and eight distinct timestamps, in addition to a source-validation gate and at least 200 target records. The code rejected the date branch when coverage < min_valid_fraction or unique < min_unique; missing dates therefore disabled it. On rejection, ordinary per-record predictions were retained. Five synthetic boundary checks of the submitted function verified these conditions.

The organizers confirmed that the posted evaluation excluded sleep-study dates.[1] The official selected-entry age-conditioned AUROCs were 0.723 for validation and 0.622 for test.[2] We therefore describe these as date-free submitted-system results. The date branch was present in the source but inactive in the date-free evaluation. Its removal does not independently verify which non-date source-selected stream was saved, because the entry also contained guarded fallbacks and optional adapters. The exact gate quotation and organizer statement are preserved in the supplementary provenance text.

## Reconstruction of source normalization

The receipt identified validation/large, and the submission variant was large. The training wrapper accepted the supplied data folder rather than choosing a release internally. We specified the public large release, version 2, dated June 9, 2026: 6,600 recordings from S0001 (5,139), I0002 (319), and I0006 (1,142).[3] The original organizer-trained model.sav was unavailable. The small local release and 338-person development cache were not substituted.

The reconstruction uses immutable copies of the submitted extraction and normalization functions, upstream revision 0b1b49a8665e11157a107e0d76b9eedddd3b04c7, and the recorded checkpoint. It follows the submitted site-by-label round-robin queue with seed 20260721 and a 68-hour active extraction budget with the submitted rolling-time guard. Records are streamed individually; completed-record membership, failures, site counts, input and code hashes, software versions, and active time are recorded. The first launch failed during file transport before any source extraction; the committed transport repair uses Kaggle’s current API and retains the same source release and analysis rules.

The source means and population standard deviations are deterministic conditional on the same extracted features and completed records. The original deadline makes the reached roster dependent on hardware and runtime. The new constants are consequently labeled reconstructed; they are not claimed to reproduce the unavailable organizer artifact exactly. The reconstructed-source analysis retains the four heads, signed weights, risk orientation, outcomes, cutoffs, adjustment, pair rules, pooling, bootstrap, permutations, and interpretation rules. The first extraction attempts did not complete a source record. We then froze a memory-bounded CWT implementation that retains the same time chunks, wavelet scales, one-second samples, dtype conversion, interpolation and padding. It computes groups of 16 frequency scales and omits the unused synchrosqueezed array. Both seeded numerical checks against the exact submitted function had maximum absolute error zero. These checks covered single-chunk and multiple-chunk synthetic inputs; a full-length reference comparison was not run. Reconstruction remains in progress; no reconstructed-source STAGES AUROC is reported here.

## Planning assumptions and reporting deviation

The planned simulation using realized case and pair counts was not performed before the available STAGES rank results were examined. This timing deviation cannot be repaired retrospectively. The supplied planning table is retained below as unverified planning material. These percentages are not a reproduced simulation or an estimate of achieved study power.

| Assumed AUROC | Supplied power at -1.0 SD | Supplied power at -1.5 SD |
|---|---:|---:|
| 0.55 | 28% | 12% |
| 0.58 | 53% | 30% |
| 0.60 | 76% | 38% |
| 0.62 | 88% | 60% |
| 0.65 | 97% | 74% |

Realized counts were 103 cases and 2,193 eligible pairs at -1.0 SD, and 42 cases and 1,009 pairs at -1.5 SD. The successful AHI check demonstrates detectable respiratory signal under this procedure. Cognitive precision is assessed from the cognitive confidence intervals; a crossing of 0.5 does not establish absence of a smaller effect. Comparing a STAGES upper confidence limit with the hidden-test point estimate is descriptive, rather than a formal test between settings with different outcomes.

## Available results

| Evaluation | AUROC | 95% CI |
|---|---:|---:|
| Composite, Stable4 within-center normalization sensitivity | 0.534 | 0.449–0.615 |
| Composite, negated brain-health score | 0.519 | 0.433–0.606 |
| AHI output for observed AHI at least 15 events per hour | 0.761 | 0.704–0.812 |

The Stable4 sensitivity permutation p was 0.187 and the brain-health comparator p was 0.314. The AHI check included 572 participants and 2,940 eligible pairs, with p < 0.001. The complete available-results record retains both alternative composite thresholds, all domain outcomes, continuous concordance, center-specific estimates, equal-center weighting, and estimable bootstrap counts. The original-source and reconstructed-source endpoints remain distinct from this completed sensitivity analysis.

## Sources

1. PhysioNet Challenge. PhysioNet Challenge 2026 Scores released. September 29, 2026. https://groups.google.com/g/physionet-challenges/c/WyxLESJjrnY

2. PhysioNet Challenge 2026. Official results table. Selected submission 2593, Arnav & Ivan. https://moody-challenge.physionet.org/2026/results/challenge_2026_results.tsv

3. PhysioNet. PhysioNet Challenge 2026 Data Large Version. Version 2. June 9, 2026. https://www.kaggle.com/datasets/physionet/physionetchallenge2026datalargeversion/versions/2
