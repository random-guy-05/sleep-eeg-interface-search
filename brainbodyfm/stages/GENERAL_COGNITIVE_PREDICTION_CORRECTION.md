# Corrected STAGES general low-performance prediction analysis

October 27, 2026. This extension is specified after the original continuous,
residual-composite rank, and CinC-method replication results were known. All
earlier results and definitions are retained. This is a new post hoc predictive
evaluation, not a revision of the frozen primary endpoint or a preregistration
of the study. Commit this document and its implementation before execution.

## Reason and scientific scope

An audit found a problem with interpreting fitted demographic classifiers
against labels residualized on those same demographics over the entire cohort.
In a simulation with the STAGES covariate/center structure, mean demographic
age-conditioned AUROC was 0.4778 after full-cohort residual-case construction,
versus 0.4991 using independently defined residual labels (1,000 replicates).
Thus those new fitted-classifier contrasts are retained as audit diagnostics,
not used as evidence for or against general incremental cognitive prediction.
The fixed outcome-blind score calculation is a different estimand and remains.

The corrected question is whether the published compact EEG method predicts
general low cognitive performance at a new center, with demographics available
as a proper predictor baseline. This is the user's requested general-prediction
question. No new clinical diagnosis of MCI or dementia is inferred from STAGES.

## Source-only definition of the target

Use the same eligible participants and tasks, with the same original validity
flags and dates. For each held-out center, use only the other three centers to:

1. Calculate each task's raw-score mean and population SD.
2. Standardize PCET Accuracy2 and face-memory correct responses using those
   source values; average them.
3. Calculate the source composite's mean and population SD and restandardize.
4. Label a source or held-center participant as low performance if their
   composite is <= -1 SD relative to that source reference.

No age, sex, education, or center regression is used to construct these labels.
Those variables enter predictor baselines or the age-conditioned evaluation.
Held-center outcomes are used only to determine that participant's ground-truth
label with the source-derived rule. They never determine reference means, SDs,
predictor coefficients, or score normalization. The reference rule may differ
between outer folds; report the resulting per-center case counts explicitly.

The primary outcome of this extension is the two-domain composite. Executive
function and face memory alone use the identical source-reference procedure
and are the complete secondary outcome set. Keep the -1 SD threshold fixed.
No threshold/subgroup search or endpoint expansion follows the new results.

## Models and inference

Retain the complete 16-model set from CINC_METHOD_REPLICATION_PLAN.md, including
the unchanged signed Stable4 score, all four ablations and components, equal
weights, negative brain-health score, age, demographic and clinical baselines,
and the two separately identified supervised four-output adaptation arms.
The submitted implementation, directions, fixed weights and L2=1 remain unchanged.
Only the label construction for this new general-prediction task differs.

The main metric is pooled same-center age-conditioned AUROC with the two-year
window and half-credit ties. Report ordinary within-center AUROC, equal-center
mean, ordinary AUROC of concatenated held-center predictions, per-center
estimates, and all 16 models. Case counts and eligibility
denominators accompany all outcomes. Equal-center summaries are not estimable
when any center has no eligible case/noncase pairs; do not drop that center.

Use 2,000 center-stratified patient bootstrap samples, repeating source outcome
reference construction, labels, all source preprocessing, coefficient fits,
and held-center prediction. Report the same nine paired contrasts and nominal
95% and Bonferroni familywise 95% intervals as in the previous extension.
For the 11 outcome-blind scores, perform the same 10,000 within-center label
permutations and Holm adjustment across all 33 score/outcome tests. This remains
exploratory; the comparison family also forms part of the broader history of
post hoc analyses and does not establish study-wide error control.

Bootstrap seed 2026102761 plus task index; permutation seed 2026102771 plus task
index. The original HSP-normalization reconstruction and frozen rank endpoint
continue unchanged. Whatever the result, report this corrected evaluation
alongside its rationale and the earlier residual-phenotype results. An improved
estimate must be attributed to a different target definition, not correction
of the arithmetic of the 0.533 fixed-score estimate.

## Explanatory audit

Alongside the full results, report the fixed score's and brain-health score's
relationships with age, raw PCET, raw face memory, and the earlier adjusted
composite. Compare the HSP development sample's and STAGES sample's ages and
four-output means/SDs. These are descriptive checks of domain shift and removed
demographic signal, not a new feature search or causal decomposition. Display
the corrected ordinary and age-conditioned metrics together. The organizer's
test values are 0.749 ordinary and 0.622 age-conditioned AUROC; 0.667 is not an
empirically established minimum for a new cohort or cognitive endpoint.
