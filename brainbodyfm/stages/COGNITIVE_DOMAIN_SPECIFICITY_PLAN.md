# Same-participant cognitive-domain specificity analysis

October 27, 2026. This exploratory follow-up is specified after inspection of
the trainable-transfer results, including brain-health age-conditioned AUROC
0.620 for attention slowing versus 0.489 for executive accuracy. The prior
results remain intact. Commit this plan and code before computing the paired
domain contrasts. These contrasts are not confirmatory or preregistered.

## Purpose

Test the domain-specificity hypothesis directly: an integrated EEG brain-health
score may rank low sustained-attention speed better than low executive accuracy,
face-memory accuracy, or their composite. This would make an accuracy null more
informative than a claim that the representation has no cognitive information.
Do not infer specificity by comparing separate significance thresholds.

Use only participants valid for all three tasks, with the same existing CNB
session and extraction. No age subgroup or further outcome/threshold search.
The working hypothesis was motivated by earlier results and must be labeled so.

## Fixed scores, outcomes and contrasts

Evaluate the negative published brain-health score and, as a comparator, the
unchanged signed Stable4 score with source-only normalization in each leave-
one-center-out fold. No new weights are fit. Brain-health is a distinct score
from Stable4; do not assign the CinC scores to the brain-health output.

Four outcomes: PCET Accuracy2; face-memory correct responses; the existing
equal-domain executive/face-memory composite; and negative log median correct-
target RT on the sustained-attention task. For every outcome and outer center,
source reference means/SDs and the -1 SD case threshold are estimated only at
the other centers, using the same general-performance rule. All four outcomes
are evaluated in the same complete-case population. Report all case counts.

For each of the two fixed scores, compare attention-slowing discrimination with
executive, memory, and composite discrimination: six paired contrasts in total.
Primary descriptive metric is same-center age-conditioned AUROC with <=2-year
age gap, half-credit ties, pooling eligible pairs. Also report ordinary AUROC,
ordinary within-center AUROC, and threshold-free age-conditioned concordance
for pairs separated by >=.25 source-outcome SD, oriented so worse cognition
corresponds to higher risk. Do not reinterpret an accuracy null as equivalence.

## Uncertainty and reporting

Draw 2,000 participant bootstrap samples stratified by center. Repeat source
outcome reference values, label construction, Stable4 source normalization,
pairing and every paired contrast. This is a full repetition of all data-fitted
steps for these frozen scores: there is no classifier to refit. Brain-health
weights remain frozen in its published artifact. Seed 2026102798.

Report nominal percentile 95% intervals and Bonferroni intervals over the six
age-conditioned contrasts. For the full exploratory ledger, additionally report
intervals adjusted over all six contrasts x four metrics = 24 contrasts.
Finite bootstrap draws make tail estimates imprecise; disclose counts and
avoid presenting these intervals as study-wide error control over prior analyses.
All four metric families and both scores are reported, regardless of direction.

Interpret a positive paired interval as an exploratory domain-performance
difference in this population, not proof of a physiological mechanism, test
reliability difference, incremental clinical value, or a general law.
The study has not independently confirmed this follow-up in a new cohort.

CinC's organizer-scored outcomes concern later impairment diagnoses from the
submitted system. They can be presented as complementary prior evidence about
the clinical task. They do not validate this new domain contrast or the distinct
brain-health output. Original deployment provenance remains separately pending.
