# Trainable CinC methods and full-representation cognitive prediction in STAGES

October 27, 2026. This exploratory extension follows all earlier analyses,
including the unsuccessful confirmation of newly examined cognitive signatures.
It is not a new preregistered study or an untouched external validation set.
Commit this plan and implementation before computing its new predictive results.
All previous findings, definitions, and unsuccessful searches remain in the record.

## Question and fixed outcomes

Does the trainable method in the final CinC code, or a supervised readout of the
full pretrained representation, predict cognition better than the compact fixed
Stable4 interface? This tests a learning procedure. It does not relabel a newly
trained STAGES model as the original Challenge model.

The primary outcome in this extension remains the existing source-reference
executive-plus-face-memory composite <= -1 SD. Secondary outcomes are executive
function alone and face memory alone with the same threshold. The newly analyzed
sustained-attention slowing endpoint is included as an explicitly post-discovery
secondary task: negative log correct-target median response time, low if <= -1
source SD. Its earlier US/MedSleep association result has already been seen.
No new threshold, subgroup, domain combination, or outcome is selected here.

Leave one of the four center families out. In every outer and inner split,
outcome reference means/SDs and labels are defined using the training centers
only, using the previously frozen general-performance rule. The held center
does not enter preprocessing, feature selection, normalization, hyperparameter
choice or fitting. Available case counts and all centers are reported.

## Complete model set

Baselines: chronological age, demographic classifier, clinical classifier,
negative published brain-health score, and signed Stable4 with source-only
normalization and its original four weights.

Six trainable alternatives:

1. **Submitted available-representation system.** Call `tune_and_fit` with
   `strategy='b_primary'`, `tune=True`, the full 841-output bank and 1024-dimensional
   latent, using the exact source at commit 76a36e54251cfce85889a3b2390018ffc23329ee.
   Preserve the B+REM feature selectors, source-site nested fusion, Stable4 gate,
   latent ranker and deployment gates. Log each outer fold's selected streams,
   gates, fusion weights and recipe. CARR and SleepFM are absent because those
   representations are not cached for STAGES. No date branch is used. Accordingly
   this is the submitted available-representation procedure, not a reproduction
   of every possible submitted-system branch.
2. **Submitted latent ranker alone.** Use the exact `fit_latent_ranker`, 60 epochs,
   hidden width 128, learning rate .0003, dropout .4, source-site balancing,
   age-pair loss, worst-site term and adversarial site term. Seed is the submitted
   final latent seed SEED+991. No seed or epoch search is performed.
3. Clinical covariates plus the frozen taxonomy's 23 cognition/diagnosis outputs.
4. Clinical covariates plus 539 physiology-only outputs.
5. Clinical covariates plus all 1024 latent coordinates, linear classifier.
6. Clinical covariates plus all 841 outputs.

For models 3–6 and demographic/clinical classifiers, use the exact submitted
balanced binary-cross-entropy solver, with its intercept unpenalized and fixed
L2 candidate grid [.01,.1,1.,10.]. Feature scaling and all covariate transforms
are training-only. Choose L2 by equal-center ordinary AUROC in inner center
holdouts; ties within 1e-6 choose stronger regularization. There is no tuning
against the outer center. The 23 cognition-related outputs are permitted here
because this extension explicitly predicts cognition; they were deliberately
excluded from the prior physiology-only search. They are pretrained predictions,
not STAGES clinical diagnoses.

An additional **source-selected linear procedure** chooses one of models 3–6
using only their inner held-center ordinary AUROC, with the same tie preference
and fixed model-list order. Report this procedure alongside every constituent;
do not choose the best outer-result model and call that independent validation.

## Endpoints and uncertainty

Primary metric: ordinary AUROC of concatenated center-held-out predictions for
the composite. Also report ordinary within-center AUROC, same-center age-window
AUROC (<=2 years), each center's estimate and case count, and average precision.
Report both primary-method paired differences from clinical (submitted available
system and source-selected linear procedure), as well as all other models and
all four outcomes. No score reversal follows inspection of held-out results.

For all models, draw 2,000 center-stratified patient bootstrap samples, redoing
the outcome reference/case-definition stage but conditioning on saved fitted
predictions and model selection. Label these intervals conditional; they do not
include training, feature selection, or tuning uncertainty. They are exploratory
screening intervals, not a substitute for a refitted interval or new data.
Use paired samples for each comparison and nominal 95% intervals, plus Bonferroni
familywise intervals over the two main procedures x four outcomes = eight
comparisons. Seed 2026102791 plus outcome index.

If a main procedure reaches ordinary composite AUROC >=.65, or shows a positive
nominal paired lower bound versus clinical, it warrants a dedicated full-refit
uncertainty stage before a manuscript claim. A linear procedure is feasible for
2,000 complete refits; a neural/nested-system signal first requires a timed
refit benchmark and an explicit compute plan. Do not describe a conditional
positive screen as a confirmed result while that required stage is unfinished.

## Relation to CinC validation

The organizer-scored .723 validation/.622 test age-conditioned scores and .738/
.749 ordinary scores validate the original submitted system on later diagnoses.
They do not automatically validate whichever newly trained STAGES model performs
best. The unchanged learning procedure may be evaluated across populations,
with its retraining and omitted branches stated. Standalone Stable4 claims still
require resolution of the original deployed branch and source normalization.

The ongoing HSP source reconstruction, its active frozen files, and its rank
endpoints are not modified by this extension. No restricted records, participant
predictions, or trained participant-derived artifacts are published.

## Why a null could be informative: fixed diagnostic comparisons

For the composite only, also run four-fold patient-level cross-validation,
stratified by center, with seed 2026102795. The algorithms and source-only outcome
reference rule remain the same, including internal source-site fitting. The
outer patient folds now include all center families in training. The comparison
asks whether access to the target center changes prediction. It does not isolate
a causal effect of center because source sample sizes, represented centers, and
the training-based reference values also differ. Report each protocol's case
counts rather than treating the operational label definitions as identical.
Show all models in both protocols; do not focus only on the largest gap.

Predict center identity in the same four patient folds using demographic,
clinical, cognition-output, physiology-output, and latent representations.
Use source-only scaling and multinomial logistic regression (C=1, max_iter=2000),
report ordinary and balanced accuracy and the complete confusion matrix.
Chance balanced accuracy is .25 for four center families. Strong center
prediction demonstrates center-associated information, not necessarily equipment
artifact or an explanation of weak cognition prediction. Patient/case-mix
differences may also contribute. These diagnostics are exploratory and specified
before this extension's predictive results are calculated.

Execution repair before any new held-center prediction: the submitted model
retains a compatibility `derived` family whose `brain_age_gap` is computed from
the age output and chronological age. The wrapper now constructs it with the
submitted `brain_age_gap_batch` instead of incorrectly seeking it among the
841 direct outputs. B-primary scoring ignores that compatibility family; no
method, target, parameter, training result, or selection rule was changed.
