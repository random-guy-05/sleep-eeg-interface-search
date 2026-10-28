# What the expanded STAGES analyses establish

October 27, 2026. This report integrates the completed trainable-readout,
validation-strategy and same-participant domain analyses. All are exploratory
follow-ups to already known STAGES results. The original continuous primary
endpoint and all earlier fixed-score analyses remain unchanged.

## The research question worth carrying forward

**Does a sleep-EEG brain-health representation capture the same cognitive
phenotype across populations, or does its apparent value depend on the domain
and clinical setting?**

An informative null would establish a meaningful boundary on transfer while
showing that the same inputs and procedure retain other measurable information.
The completed results make that question more specific: the clearest cognitive
lead is marked sustained-attention slowing, whereas executive and memory
accuracy are less predictable. Strong center information is retained. However,
the direct domain difference is not established after multiplicity correction,
and threshold-free concordance does not reproduce the apparent specificity.
These qualifications change the scientific conclusion, not just its wording.

## Full-representation comparison

The analysis evaluated 12 readouts on four outcomes in leave-one-center-out
evaluation. The original backbone, 841 outputs and 1024-coordinate latent
representation were retained. Linear readouts added 23 cognition outputs, 539
physiology outputs, all 841 outputs, or the full latent representation to clinical
covariates. Penalties and representation selection used only inner source
centers. The submitted available-branch procedure and native latent network
used the exact submitted downstream implementation at commit
`76a36e54251cfce85889a3b2390018ffc23329ee`. The missing CARR inputs are being
extracted separately; the current result is not a complete replay of every
submitted branch, and the original organizer-trained artifact is unavailable.

For these general-performance outcomes, reference means and SDs were learned
only at source centers, without regressing the outcomes on demographic
covariates. Low performance means <=-1 source SD. Consequently these are distinct
from the previously specified full-cohort residual-composite case labels.

| Readout | Executive + memory AUROC, n=588 | Attention-slowing AUROC, n=563 |
|---|---:|---:|
| Age | 0.601 | 0.614 |
| Demographics | 0.607 | 0.566 |
| Clinical covariates | 0.592 | 0.592 |
| Negative published brain-health score | 0.625 | 0.654 |
| Fixed Stable4, source normalization | 0.529 | 0.547 |
| Clinical + 23 cognition outputs | 0.625 | 0.635 |
| Clinical + 539 physiology outputs | 0.527 | 0.631 |
| Clinical + 1024 latent coordinates | 0.536 | 0.670 |
| Clinical + all 841 outputs | 0.539 | 0.642 |
| Submitted available branches | 0.502 | 0.566 |
| Submitted native latent network | 0.541 | 0.528 |
| Source-selected linear representation | 0.625 | 0.635 |

Every model, outcome, metric and nominal interval appears in
`all_model_metrics.csv`, including executive accuracy and face-memory accuracy
separately. There were 104 composite cases, 139 executive cases, 106 memory
cases, and 77 attention-slowing cases. Counts differ from the earlier residual
outcome definition; neither version replaces the other.

The latent-plus-clinical attention estimate was 0.670 (95% CI 0.590–0.727),
compared with 0.592 (0.502–0.670) for clinical covariates. Its paired difference
was +0.079 (−0.017 to +0.155). The fixed brain-health risk score achieved 0.654
(0.578–0.718), with difference +0.062 (−0.048 to +0.174) versus clinical
covariates. These paired intervals do not establish incremental value. The
trainable-model intervals condition on saved trained models and selections;
they repeat source outcome-reference estimation and labels, but not fitting.
The unselected 0.670 estimate is a post hoc lead, not a validated winning model.

For the composite, the fixed Stable4 estimate was 0.529 (0.459–0.596), the
brain-health risk score 0.625 (0.556–0.680), and the cognition-output readout
0.625 (0.563–0.678). Source selection chose the cognition-output representation
in every outer fold. Neither newly fitted prespecified procedure met the
plan's trigger for additional full-refit composite inference. All four
available-system final source deployment gates failed; their predictions are
reported as evaluations of that procedure, not successful deployment gates.

The earlier general-performance evaluation, which repeated all fitted steps
for the fixed score, gives an ordinary composite Stable4 AUROC of 0.529 with
95% CI 0.459–0.595. This nominal interval excludes discrimination as large as
0.667 under that source-normalized evaluation. It supplies a quantitative limit
on the expected transfer rather than only a nonsignificant test. It does not
settle the HSP-normalized primary endpoint, and the brain-health composite
interval still includes 0.667. The 0.667 target is a reader-specified benchmark,
not a prespecified equivalence margin or an established clinical-utility cutoff.

## Direct test of cognitive-domain specificity

The same-participant analysis included 561 patients with valid executive,
face-memory and sustained-attention measurements. The fixed score comparison
repeated every data-fitted step in 2,000 center-stratified participant
bootstraps. No classifier coefficients were fitted. Age-conditioned AUROC uses
same-center case–non-case pairs with ages differing by no more than two years.

| Outcome | Brain-health risk age-conditioned AUROC (95% CI) | Stable4 age-conditioned AUROC (95% CI) |
|---|---:|---:|
| Executive accuracy | 0.494 (0.405–0.564) | 0.517 (0.433–0.582) |
| Face-memory accuracy | 0.534 (0.426–0.604) | 0.526 (0.445–0.618) |
| Executive + memory | 0.552 (0.464–0.642) | 0.527 (0.433–0.609) |
| Sustained-attention slowing | 0.619 (0.517–0.712) | 0.569 (0.473–0.674) |

The paired brain-health difference for attention versus executive accuracy was
+0.125 (nominal 95% CI +0.016 to +0.249). The six-contrast Bonferroni interval
was −0.029 to +0.287; the full 24-contrast interval was −0.051 to +0.298. Other
paired domain contrasts also crossed zero. This is suggestive, not a
multiplicity-adjusted demonstration of domain specificity. Adjustments cover
the stated local comparison families, not all prior exploratory analyses.
With 2,000 draws, the Bonferroni tail quantiles have limited Monte Carlo
precision.

Threshold-free age-conditioned concordance for brain-health risk was 0.528 for
executive accuracy, 0.519 for memory, 0.521 for the composite and 0.522 for
attention speed. The attention-minus-executive difference was −0.006 (95% CI
−0.066 to +0.050). Thus the case-threshold lead does not establish a general
continuous relationship specific to attention. Do not characterize it as a
mechanism, a general marker of arousal, or a validated cognitive subtype.

The earlier discovery/confirmation search is also retained: six selected
score–outcome pairs had the same direction at MedSleep, but none confirmed after
the planned correction. The current follow-up does not convert that search
into successful independent validation.

## Center information and validation strategy

Four-fold patient splits stratified by center did not rescue the composite
prediction. The brain-health score achieved 0.629 with random patient folds
versus 0.625 with center holdout. Clinical-plus-latent AUROC was 0.576 versus
0.536; submitted available-branch AUROC was 0.531 versus 0.502. The complete
12-model comparison is in `patient_vs_center_cv.csv`. Source outcome references
change between designs (105 versus 104 cases), so this is a diagnostic
comparison rather than a causal estimate of the effect of center shift.

Separately, center classification from the representation was strong. In
patient folds among the 588 composite-eligible participants, latent coordinates
identified the four centers with balanced
accuracy 0.769; physiology outputs achieved 0.754, cognition outputs 0.556,
clinical covariates 0.407 and demographics 0.343. Chance balanced accuracy is
0.250. These results demonstrate center-associated information. They do not
separate technical artifacts from center differences in patient case mix or
prove that center information caused weak cognitive prediction. Center
classification and cognitive AUROC are different tasks and metrics.

The existing AHI-output control remains 0.761 (95% CI 0.704–0.812). It supports
the procedure's ability to recover that physiological signal. It cannot establish
that the cognitive analysis has adequate power for every small effect or that
all aspects of cognitive outcome measurement are error-free.

## Relation to the CinC results

The organizer-scored age-conditioned AUROCs of 0.723 and 0.622 concern later
impairment diagnoses at hidden institutions. They are prior results of the
submitted system containing Stable4; date inputs were absent, while remaining
non-date branch provenance still needs the reconstructed deployment decisions
or original organizer artifact/logs.

The current STAGES outcomes concern concurrent low test performance. The new
0.670 latent-plus-clinical result is a different fitted model; the brain-health
score is a different published output. Neither inherits the CinC hidden-site
validation. A cross-setting paper can ask whether clinical risk discrimination
and concurrent task performance support the same interpretation of a sleep
biomarker, but cannot assert that the two settings validate the new attention
finding or pool their estimates.

## What is potentially new, and what is already known

Existing work establishes sleep-EEG associations with cognition, their
attenuation after demographic adjustment, and benefits of multivariate
prediction in older adults. The parent deep-learning paper reports cognitive
and disease prediction across multiple cohorts. A recent Framingham analysis
relates the brain-health score to long-term cognitive decline. A SLEEP
commentary already calls for demographic ablations and clearer evaluation of
sleep foundation models. Accordingly, 'sleep EEG relates to processing speed',
'age matters', and 'center information exists' are not defensible first claims.

The potentially useful contribution here is the **external, domain-resolved
clinical evaluation of the released scores and representations**, with paired
clinical baselines, a working physiological control, complete failed and
successful readouts, and the direct relationship to prior hidden-site scoring.
Whether the null supports a strong SLEEP manuscript depends on the remaining
full-branch evaluation and how precisely these findings bound transfer. The
current domain contrast alone is not enough to claim a new dissociation.

Verified source records, checked October 27, 2026:

1. Ganglberger et al. Brain Health from Sleep EEG: A Multicohort, Deep Learning
   Biomarker for Cognition, Disease, and Mortality. NEJM AI. 2026;3(3).
   [DOI:10.1056/AIoa2500487](https://doi.org/10.1056/AIoa2500487).
2. Ujma et al. Multivariate prediction of cognitive performance from the sleep
   electroencephalogram. NeuroImage. 2023;279:120319.
   [DOI:10.1016/j.neuroimage.2023.120319](https://doi.org/10.1016/j.neuroimage.2023.120319).
3. Djonlagic et al. Macro and micro sleep architecture and cognitive performance
   in older adults. Nature Human Behaviour. 2021;5:123–145.
   [DOI:10.1038/s41562-020-00964-y](https://doi.org/10.1038/s41562-020-00964-y).
4. Marino et al. Sleep electroencephalography-derived brain health and long-term
   cognitive decline in community-dwelling adults. SLEEP Advances.
   2026;7(3):zpag089. [DOI:10.1093/sleepadvances/zpag089](https://doi.org/10.1093/sleepadvances/zpag089).
   This article is in SLEEP Advances; earlier project wording calling it a
   SLEEP article should be corrected.
5. Helmy et al. Foundation models in sleep research: opportunities and
   limitations. SLEEP. Published online September 15, 2026.
   [DOI:10.1093/sleep/zsag225](https://doi.org/10.1093/sleep/zsag225).

## Remaining fixed analyses

1. Complete the ongoing 590-person CARR extraction and its already implemented
   held-center evaluation. The first records have exactly reproduced the
   original spectrogram hashes and latent vectors. No CARR performance claim
   is yet supported by an extraction progress count.
2. Continue HSP source-normalization reconstruction. This is a reconstructed
   artifact, not recovery of the original submitted organizer file.
3. Evaluate the original 14-day timing sensitivity separately. The eligibility
   audit found 87 new participants, 84 from Geisinger, two from MedSleep and one
   from St Lukes. Their EEG–cognition associations were not inspected. The
   committed plan evaluates frozen brain-health and Stable4 scores against all
   four domains, retains every estimate and uncertainty interval, and describes
   the additional participants separately from the original cohort. This is
   participant-level follow-up within STAGES, not a new external cohort.

## Reproducibility and display captions

Trainable evaluation freeze: `1a6c78db212507076ae76f434dc85d3f7e85a9d0`.
Paired domain freeze: `1d7b884ee89b6777f3a9e7dfad8a8cc1fdcbccb1`.
The aggregate provenance JSON records hashes of the input result files and
their plan/code receipts. Public exports contain aggregates, not patient IDs,
individual outputs or cognitive scores.

**Figure 1.** Held-center ordinary AUROC for executive-plus-memory low
performance and sustained-attention slowing. Error bars show nominal 95%
center-stratified participant-bootstrap intervals. For fitted readouts, models
and selections are held fixed while outcome references and labels are repeated.
Stable4 uses unchanged signed weights and source-only normalization. The
available submitted system excludes the not-yet-extracted CARR branch.

**Figure 2.** Fixed-score discrimination across cognitive domains in the same
561 participants. Left: age-conditioned AUROC for <=-1 SD cases. Right:
threshold-free concordance for same-center, age-matched pairs differing by at
least 0.25 source SD. All data-fitted steps are repeated for the fixed-score
intervals. Direct paired contrasts and local multiplicity corrections are in
the accompanying CSV; separate significance of points is not a specificity test.

**Figure 3.** Validation-strategy comparison for concurrent executive-plus-memory
low performance, and patient-fold center classification. The first panel uses
ordinary AUROC; the second uses balanced accuracy. The center diagnostic does
not isolate acquisition artifacts from patient case mix. Changing source
references changes case counts between validation designs.
