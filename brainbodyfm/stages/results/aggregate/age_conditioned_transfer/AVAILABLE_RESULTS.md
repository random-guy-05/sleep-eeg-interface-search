# STAGES age-conditioned analyses: available results

Plan/code frozen and pushed before rank computation: `b7a357d95d532fe6c01b2cd825f36be456866f64`.

The specified primary pure-transfer endpoint remains uncomputed because the original organizer-trained source scales were not recovered. The table reports the prespecified within-center normalization sensitivity and comparator analyses. No score orientation, signed weight, outcome threshold, or age-pair rule was changed after these results. The earlier continuous results remain unchanged.

| Outcome | n | Cases | Eligible pairs | Stable4, within-center normalization (95% CI) | Negated brain-health score (95% CI) |
|---|---:|---:|---:|---|---|
| composite_threshold_-1 | 588 | 103 | 2193 | 0.534 (0.449–0.615) | 0.519 (0.433–0.606) |
| composite_threshold_-1.5 | 588 | 42 | 1009 | 0.588 (0.434–0.693) | 0.494 (0.369–0.622) |
| composite_threshold_-0.5 | 588 | 181 | 3136 | 0.506 (0.447–0.578) | 0.521 (0.443–0.582) |
| executive_function | 590 | 128 | 2660 | 0.492 (0.428–0.577) | 0.499 (0.402–0.565) |
| face_memory | 588 | 93 | 1852 | 0.544 (0.461–0.631) | 0.471 (0.415–0.594) |
| vigilance | 590 | 60 | 1372 | 0.575 (0.455–0.689) | 0.511 (0.380–0.600) |

The AHI pipeline check was 0.761 (0.704–0.812) among 572 participants, with 166 cases and 2940 eligible pairs; permutation p=0.00010. This confirms that this pair-scoring pipeline detects respiratory signal in these data. It does not imply a positive cognitive result.

All listed intervals use 2,000 center-stratified participant resamples. Cognitive outcome adjustment, residual scaling, composite/case definition and pairing are repeated in each resample. Within-center sensitivity normalization is also recomputed. Permutation tests use 10,000 within-center label shuffles and (1+exceedances)/(10,001). All secondary analyses are exploratory and unadjusted for multiplicity.

## Composite at -1.0 SD: center counts

| Center | n | Cases | Noncases | Eligible pairs |
|---|---:|---:|---:|---:|
| Geisinger | 34 | 8 | 26 | 15 |
| MedSleep | 305 | 60 | 245 | 1666 |
| St_Lukes | 50 | 9 | 41 | 28 |
| Stanford | 199 | 26 | 173 | 484 |

Pooled Stable4 sensitivity AUROC is 0.534 (0.449–0.615), one-sided permutation p=0.187. The BHS comparator is 0.519 (0.433–0.606), p=0.314.

## Continuous concordance and equal-center sensitivity

| Summary | Stable4, within-center normalization (95% CI) | Negated BHS (95% CI) |
|---|---|---|
| Continuous concordance | 0.509 (0.467–0.552) | 0.510 (0.466–0.550) |
| Equal-center AUROC | 0.570 (0.372–0.729) | 0.677 (0.537–0.776) |

Equal-center intervals have 1,976 estimable resamples out of 2,000 because Geisinger has no eligible case/control pairs in 24 draws. Individual center estimates and exact valid-replicate counts for every outcome are included in the CSV/JSON. Pooled estimates have 2,000 valid draws.

## Complete disposition of the eight exploratory families

1. Continuous concordance: available for the BHS comparator and target-normalized sensitivity; original-source Stable4 remains unavailable.
2. BHS comparator: completed; paired primary Stable4-minus-BHS interval unavailable without the original Stable4 score.
3. Random-score specificity: coefficient assignments frozen; source-normalized evaluation unavailable without original source scales for all candidate heads.
4. Threshold sensitivity: both specified cutoffs reported for available scores.
5. Domain-specific outcomes: PCET, face memory and vigilance reported for available scores.
6. Normalization sensitivity: completed and explicitly distinguished from the primary frozen-source endpoint.
7. Equal-center weighting: completed subject to the frozen four-center estimability rule.
8. AHI pipeline check: completed. The older MAE check is preserved as a prior result.

The realized-design power simulation has not been run; no proposed power percentages are treated as verified. The primary computation remains locked pending both the original artifact and that frozen design-dependent stage. No OSF registration was completed.

## Exact final Challenge source and receipt

The corresponding author identified `https://github.com/i-habib/phys-26-large-new` at commit `76a36e54251cfce85889a3b2390018ffc23329ee` as the final submission. That exact commit is present in the local Git checkout; the six audited source files match the committed bytes.

The supplied organizer receipt is **validation/large**, entry 6 of 10: reward -0.033; age-conditioned AUROC 0.723; age-weighted AUROC 0.776; ordinary AUROC 0.738; AUPRC 0.318; accuracy 0.332; F-measure 0.155. This receipt does not report final-test performance. Separately, the current official results table lists final submission 2593 for Arnav & Ivan with validation age-conditioned AUROC 0.723 and test age-conditioned AUROC 0.622; that table is saved as `official_challenge_selected_entry.json`. The provided email itself establishes only the validation result.

The committed entry names the trained file `model.sav`, serialized with joblib. Source feature moments live in the trained downstream artifact, not in the code. This file and `train_meta.json` were not found. The same exact commit permits a gated 80% Stable4 / 20% recording-date rank branch with target normalization; its actual selection during organizer training/inference cannot be recovered from the scalar receipt alone. The 0.723 point therefore describes the submitted system until the executed branch is verified. It must not be labeled unqualified proof that the standalone frozen-source four-head score alone achieved 0.723.

## Interpretation

The available rank analyses do not establish cross-population cognitive discrimination: the pooled composite sensitivity estimate is modest and imprecise, with its interval crossing 0.5. The positive AHI control shows that a measurable signal can be recovered by the evaluation procedure. Original-source Stable4 remains an unanswered endpoint, rather than a negative or a positive result.
