# Additional-participant check under the original 14-day timing sensitivity

October 27, 2026. The original protocol lists `cnb_window_14_days` as a sensitivity
analysis. An eligibility-only audit found 87 additional participants excluded
solely because cognitive testing was more than one day from the diagnostic PSG,
but no more than 14 days away. No EEG–cognition associations or new participants'
cognitive score distributions were examined in that audit. Eighty-four are from
Geisinger, two from MedSleep, and one from St Lukes. These participants are not a
new external cohort. Timing and center composition distinguish them from the
original 590. This is an exploratory follow-up motivated by known results.

Commit this plan and its implementation before any additional-participant score
association is computed. Preserve all earlier plans, results and exclusions.
Rebuild eligibility with the unchanged original builder and sources, changing
only the maximum absolute testing gap from 1 to 14 days in a separate workspace.
Retain the same earliest eligible-session rule, task validity flags, demographic
checks, diagnostic recording requirements and duplicate exclusions. Remove all
593 participants in the original pre-QC roster to obtain the additional queue;
the three original signal-QC exclusions do not reenter. Do not select records by
cognitive value. The extraction queue contains no outcomes.

Apply the same strict C4–M1 channel rules, confirmed voltage calibration, 2-hour
minimum, 11-hour maximum input and preexisting signal-QC thresholds. Use the
exact pinned checkpoint and submitted feature extractor. Save all 841 outputs,
the published brain-health score and ordinary/global/temporal/counterfactual
representations for every additional eligible record in a separate cache.
Retain extraction failures and do not replace them selectively. Ordinary latent
to head projection follows the original STAGES extractor. Check shapes, finite
values, input and cache hashes. Do not overwrite the original 590 caches.

The fixed primary descriptive endpoint of this added check is the negative
published brain-health score's within-center, age-conditioned AUROC for low
sustained-attention speed in the additional participants. Outcomes use negative
log median correct-target RT, C1/V/non-skipped, with low performance at <=-1 SD.
All reference means/SDs are fixed from valid original-cohort participants at
other centers. Stable4 uses its existing signed weights and original-cohort,
other-center normalization, with no new fitted coefficients. This is not the
still-pending HSP-frozen-normalizer primary analysis.

Report both fixed scores for attention, executive accuracy, face-memory accuracy
and their equal-domain composite: eight score–outcome estimates, regardless of
direction. Ordinary pooled AUROC, ordinary within-center AUROC, age-conditioned
AUROC (same center, <=2-year age difference, half-credit ties), and threshold-free
age-conditioned concordance (>=.25 SD performance difference) are all reported.
The composite requires both tasks. Report realized n, cases and eligible pairs
per center. Preserve non-estimability if no eligible case/non-case pairs exist.
No additional thresholds, subgroups, feature selection or sign changes.

For each original-plus-additional timing sensitivity, keep the original 590
participants and original sessions. Original participants retain the existing
other-center references. Additional participants also use these original-only
references. This describes the sensitivity of frozen predictions to including
the added patients; it is not a retraining analysis or completion of the full
original continuous search-gap timing sensitivity.

Draw 2,000 participant bootstrap samples stratified by center, independently
resampling original and additional participants. Repeat every data-estimated
reference and Stable4 normalization, labels and pair construction. Seed
2026102811 plus outcome index. Report nominal 95% percentile intervals and
Bonferroni intervals over the eight age-conditioned estimates. Finite bootstrap
tail resolution and prior exploratory selection are disclosed; these intervals
do not control the error rate across the whole research history. Do not claim
replication from a positive point estimate alone. AHI-output discrimination of
observed AHI >=15, where observed AHI exists, is a descriptive pipeline check.

Interpretations fixed before the new associations:

* Attention BHS lower Bonferroni bound >.5: the previously observed slowing
  association extends to additional participants under the wider testing window.
* Nominal lower bound >.5 but adjusted bound includes .5: exploratory support,
  insufficient for a multiplicity-adjusted replication claim.
* Interval includes .5: the additional sample does not establish replication;
  report precision and do not treat a wide interval as equivalence.
* Estimate <=.5: direction is inconsistent in this additional sample.

CinC's hidden-site scores validate the submitted system on later diagnoses.
They do not independently validate this BHS/attention hypothesis. Do not pool
these estimates with organizer scores or substitute them for the original
composite endpoint. The additional analysis may proceed automatically after
the currently running 590-person CARR extraction finishes, avoiding a third
simultaneous backbone process. This local queued computation sends no messages.
