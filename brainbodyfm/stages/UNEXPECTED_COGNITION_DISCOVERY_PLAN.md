# STAGES cognitive-signature discovery and held-center confirmation

October 27, 2026. This is a post hoc discovery extension following the earlier
continuous and binary analyses. Their results remain unchanged. No positive
result, minimum AUROC, or publication conclusion is promised. Plan and code are
committed before the new cognitive associations are examined.

## Discovery and confirmation

Use the same 590 eligible, previously extracted participants. The actual source
roster has MedSleep 306, Stanford 200, St Luke's 50, and Geisinger 34. Discovery
uses the 284 participants at the three US center families. Confirmation uses
the 306 MedSleep participants. Endpoint validity reduces those denominators.
Earlier statements of 285/305 referred to an incorrect generalization of the
588-person valid-face-memory counts. Report actual endpoint counts throughout.

MedSleep is reserved only for associations with the newly analyzed endpoints.
It is not a wholly untouched dataset: previous accuracy, memory-correct, and
mean-vigilance outcomes from all centers have already been examined. The split
is geographic/center-based, not chosen by predictive performance. Only aggregate
availability, numeric ranges, and source metadata of the new endpoints have
been inspected before this freeze.

## Complete endpoint roster and orientation

Risk orientation is worse performance = higher value. PCET and face memory
require C1/V. Sustained attention requires SPCPTNL C1/V. Vigilance and short
working memory require C1 and their provided validation value 0; this value is
retained as supplied, not silently renamed V. Exclude skipped tasks. Preserve
the same selected CNB session. Endpoint-specific missingness is reported.

Sixteen newly analyzed endpoints:

1. PCET concept-level responses, negative count.
2. PCET perseverative errors, log1p(count).
3. PCET failures to maintain set, log1p(count).
4. PCET correct-response slowing, log(median correct RT), positive RT only.
5. Face-memory correct-response slowing, log(median correct RT).
6. Face-memory confident-error fraction: (definite false positives + definite
   false negatives) / all definite responses, denominator >0.
7. Face-memory discrimination: negative d-prime, loglinear correction
   (hits+.5)/(hits+misses+1), (false alarms+.5)/(false alarms+correct rejections+1).
8. Sustained-attention discrimination: negative d-prime, same correction.
9. Sustained-attention slowing: log(median true-positive RT), positive RT only.
10. Sustained-attention variability: SD of all RT / mean of all RT, mean >0.
11. Short working-memory accuracy: negative correct 1-/2-back target responses.
12. Short working-memory slowing: log(median correct 1-/2-back RT).
13. Working-memory load cost: log(correct 2-back RT / correct 1-back RT).
14. Vigilance lapses: supplied proportion of responses >=500 ms.
15. Vigilance variability: SD RT / mean RT, mean >0.
16. Vigilance slow tail: negative mean reciprocal RT of slowest 10% responses.

Previously inspected PCET Accuracy2, face-memory correct responses and mean
vigilance reciprocal RT are also shown as descriptive context. They cannot
qualify as newly confirmed discoveries. Verbal memory and the long working
memory version are excluded because only 30 and 29 valid participants exist.

## Fixed-score track with directly relevant CinC provenance

Calculate Stable4 with the unchanged four outputs, directions and weights,
normalization fitted once on the US discovery participants and carried to
MedSleep. This is source-only normalization, not recovered HSP normalization.
Evaluate negative brain-health score as a distinct comparator. The four signed
components are descriptive diagnostics, not substitute definitions of Stable4.
No new fixed-score weights, sign flips, feature choices, or thresholds are fitted.

For all 32 score/new-endpoint combinations, regress the rank-normalized endpoint
on the standardized score and nuisance covariates: the existing age spline,
sex, education, BMI, AHI, ESS, PHQ-9, recording duration, and discovery-center
indicators. Report unadjusted and demographic-adjusted associations separately
as context. Main selection uses the clinical-adjusted two-sided partial-
correlation t-test. Apply BH FDR over all 32 discovery tests.

Select the three smallest main discovery p-values, regardless of significance,
with distinct endpoints (ties: endpoint roster order, then score roster order).
Write the selected list, directions, all discovery tests, and hashes BEFORE any
MedSleep association is calculated. Evaluate those three in MedSleep with the
same adjustment, and Holm-adjust their two-sided replication p-values. Require
the same direction and Holm p<.05 for replication; discovery FDR is also reported.
Do not add another candidate because one fails. A replicated unexpected reverse
direction may be scientifically interesting but is not concordance with CinC
cognitive-risk discrimination in the expected direction.

## Separate physiological discovery track

Search the existing frozen 539 physiology-only outputs against the same 16
new endpoints in US centers only, using the same main adjustment. Include all
539 x 16 tests in a separate BH FDR family; report every result. Select the
three lowest p-values with different endpoints and different physiological
families, using frozen taxonomy and deterministic tie breaks. Freeze before
MedSleep results, then use same-direction, Holm-adjusted p<.05 for confirmation.
This track is hypothesis generation. The old CinC scores do NOT validate a new
head or combination selected here. It would need a new, frozen external test.

## Effect sizes, prediction diagnostics, and uncertainty

Report partial correlations, standardized regression coefficients, nominal
95% intervals, p-values, BH q-values, and all center-specific estimates.
For each of the six selected pairs, draw 2,000 patient bootstraps stratified by
US center and MedSleep, refitting main adjustment and source score normalization;
report effects and a frozen-source predictor comparison at MedSleep. Selection
is conditioned on the frozen candidate and is not repeated within bootstrap.

Low performance for a prediction diagnostic is an endpoint risk value at or
above the US 84.1344746th percentile, with ties retained and realized cases
reported. This diagnostic uses no MedSleep outcome-derived threshold.
Compare the source-fitted clinical predictor and clinical-plus-selected-marker
predictor by ordinary AUROC, within-two-year age-conditioned AUROC, and transformed
endpoint MAE at MedSleep. All coefficients are fit on the US participants only.
The adverse direction for the raw marker is frozen by the discovery association
for physiology; fixed Stable4 retains its CinC risk direction even if discordant.
Patient bootstrap intervals repeat source reference cutoffs and all fitting.

Use seed 2026102781 for bootstrap and 2026102782 for simulation checks. HC3
regression sensitivity is reported for selected associations to check influence
of heteroskedasticity. No subgroup, threshold, endpoint, or score search is added
after confirmation. This is one bounded discovery pass, not a guarantee of a
publishable surprise or study-wide confirmatory error control over prior work.

## What CinC can validate

The organizer-scored .723 validation/.622 test age-conditioned AUROCs are
prior results of submission 2593 for later diagnoses. The deployment branch
remains unresolved until the original artifact/logs or a qualified reconstruction
is available. If the exact fixed score is confirmed to have run, these scores
can corroborate its cognitive-risk discrimination in a different setting.
They cannot independently verify a newly discovered response-speed mechanism,
specific domain contrast, interaction, new component, or new combination without
testing that claim in the external data. Do not call the same existing .723/.622
an independent validation of every STAGES discovery.
