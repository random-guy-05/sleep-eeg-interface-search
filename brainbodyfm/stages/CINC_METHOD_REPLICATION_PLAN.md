# STAGES replication of the CinC cognitive-risk method

Dated October 27, 2026. This is an exploratory extension specified after the
continuous analyses and the within-center-normalization rank results were known.
It is not a replacement for either earlier endpoint, and is not retrospective
preregistration. Commit this plan, code, and checks before computing this
extension's results. Do not stop, add endpoints, change signs, or change the case
definition because a result is favorable or unfavorable.

## Question and population

Does the four-output method described in the CinC paper discriminate concurrent
low cognitive performance at held-out STAGES centers? Use the existing cohort,
quality/validity rules, four center families, and age-matched pairing definition.
The primary outcome for this extension is the existing age-, sex-, education-,
and center-adjusted executive-plus-face-memory composite <= -1 SD (588 people).
Executive function and face memory separately, with the same adjustment and
threshold, are the complete set of secondary outcomes. No new thresholds,
subgroups, clinical diagnoses, or feature-bank searches are introduced.

These are sample-relative low-performance labels, not adjudicated MCI or
dementia diagnoses. Outcome-only adjustment uses the full eligible sample, as
in the already-frozen rank plan. It is repeated in every bootstrap. A classifier
is therefore evaluated against this research phenotype, not an externally
normed diagnostic reference standard.

## Published method replication

Use the exact Stable4 normalization/scoring implementation in submission commit
76a36e54251cfce85889a3b2390018ffc23329ee. Verify its source-file hashes against
the existing reconstruction manifest. In each leave-one-center-out fold, fit
only the four means and population SDs on the other three centers. Apply the
unchanged output roster, directions, and weight magnitudes to the held center.
This follows the CinC development protocol. It is different from transferring
the original HSP normalizer, which remains the ongoing reconstruction analysis.
There are no STAGES-trained weights in the published-method arm.

For interpretation, repeat the CinC paper's fixed equal-weight score, all four
leave-one-component-out ablations, and all four signed single components. No
orientation changes follow inspection. Evaluate the negated brain-health score,
age alone, and training-center demographic/clinical classifiers as comparators.
All diagnostic/ablation results are reported, including discordant results.

## Supervised adaptation, distinguished from published-score validation

Two additional models test whether the same four outputs carry locally learnable
information: demographics plus the four outputs, and clinical covariates plus
the four outputs. Compare each with its corresponding covariate-only model.
All coefficients in these models are estimated using only the other three
centers. The four-output roster is frozen; there is no feature selection.
Use the submitted `fit_linear` binary cross-entropy solver with its fixed L2=1,
rank_weight=0 and bce_weight=1. Age uses the existing four-knot cubic spline;
other covariate handling is unchanged, fit on source centers only. These arms
test adaptation of the representation, not transport of the fixed Stable4 score.
No additional algorithm or tuning grid is tried in this extension.

## Endpoints, uncertainty, and complete reporting

The main metric is the age-conditioned AUROC pooled over same-center case/noncase
pairs aged within two years, with half credit for ties. Report center estimates,
counts, eligible pairs, equal-center mean, and ordinary within-center AUROC.
Also report the usual AUROC and average precision of the concatenated held-center
predictions, clearly distinguished from within-center metrics. Scores are ranks,
not calibrated absolute probabilities.

Use 2,000 center-stratified participant bootstrap samples. Repeat outcome
adjustment, case labeling, source-only normalization, classifier fitting, and
held-center evaluation. Keep each participant in their original center, and
use paired replicates for model differences. The three planned comparisons are
Stable4 minus demographics, demographics-plus-four minus demographics, and
clinical-plus-four minus clinical. Across three outcomes this is a family of nine
exploratory contrasts. Report nominal percentile 95% intervals and Bonferroni
familywise 95% intervals for all nine; do not select only a favorable contrast.
Equal-center means require an estimable AUROC in every center in that replicate;
report the count of valid replicates rather than silently dropping a center.

For the 11 outcome-blind EEG scores (Stable4, equal weights, four ablations, four
components, brain-health), use 10,000 within-center case-label permutations for
one-sided chance tests. No permutation p-value is reported for trained classifiers
without refitting them under permutation. Adjust all 33 fixed-score/outcome
p-values together by Holm's method. This does not turn this post hoc extension
into a confirmatory study.

Bootstrap seed: 2026102731, plus outcome index. Permutation seed: 2026102741,
plus outcome index. The source reconstruction continues unchanged. Save private
predictions and bootstrap draws locally, publish only aggregate results and code.

## Decision rule

Concordance with CinC requires the published Stable4 orientation to discriminate
low performance in the expected direction; report its interval and all centers.
A locally refitted model's success must be attributed to supervised adaptation,
not to the unchanged score. A useful novel finding requires demonstrable held-center
performance or a well-supported incremental gain, not a favorable training result,
a selected subgroup, or one nominally significant member of the analysis family.
If neither arm succeeds, report that result and retain the original analyses.
