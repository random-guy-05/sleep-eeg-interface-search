# Audit of the low STAGES cognitive-prediction estimate

The standalone Stable4 composite estimate, age-conditioned AUROC 0.532604
(95% CI 0.444616–0.616040), is reproducible. The implementation checks below
found no discrepancy that explains it as an indexing, head-export, voltage,
score-orientation, or AUROC-calculation error.

- All 590 participant-level input rows agree with their individual feature caches;
  all have the same checkpoint and extraction-source hashes.
- All 590 selected cognitive records were recovered uniquely from the original
  STAGES CNB workbook. PCET and face-memory values match exactly. The PCET formula
  agrees with the codebook, with only the original rounding error (at most 0.00005).
- Reprojecting all four heads from the saved latents using independent checkpoint
  tensor arithmetic gives errors below 0.0000009 output SD.
- A fresh EDF extraction for one outcome-blind selected participant at each of the
  four centers reproduces the spectrogram hash, latent, Stable4 outputs, and
  brain-health score exactly. Two records use native C4–M1 and two use explicit
  C4 minus M1. This is a four-record raw-signal audit, not re-extraction of 590 EDFs.
- The submitted scoring code reproduces the original HSP development macro AUROC
  0.754144 (site AUROCs 0.756098, 0.755102, 0.751232).
- The entry's independent pair-AUROC implementation reproduces all three new
  STAGES estimates exactly.

The matched target remains an important distinction. The Challenge target was a
later cognitive-impairment diagnosis. The current STAGES target is concurrent low
executive-plus-memory performance after adjustment for age, sex, education, and
center. The 0.533 result is for the unchanged four-output score with normalization
fitted on the other STAGES centers. The reconstructed HSP-normalization endpoint
is still pending, and the original organizer-trained artifact is unavailable.
The confidence interval includes both chance and effects around AUROC 0.60; the
result is inconclusive about small-to-moderate discrimination.

One full-system reproduction omission was found and repaired prospectively:
the source reconstruction discarded the global, temporal, and counterfactual
EEG taps required by the optional CARR branch. Amendment 07 now retains these
already-computed arrays. All 102 completed latent files were verified and reused
unchanged. The unfinished record is retried, and the original elapsed source
extraction time and 68-hour budget are preserved. Source reconstruction has
resumed successfully and new tap files are being retained. Earlier missing taps
require a separate backfill after the original normalization/endpoint run.

This capture repair does not change the standalone Stable4 result. The selected
original deployment branch and the full submitted-system STAGES comparison
remain unresolved. Even a source reconstruction must be identified as a
reconstruction when original logs, training roster, or artifact are absent.

Evidence: input_audit.json, raw_reextraction_audit.json, summary.json,
amendment07_migration.json, and the published amendment/code commits.

## Subsequent statistical-design correction

A 1,000-replicate simulation showed that residualizing the case definition on
demographics over the full cohort can depress a held-center demographic
classifier: mean age-conditioned AUROC 0.4778 versus 0.4991 with independently
defined residual labels. The newly fitted binary comparisons against those
residual labels are retained as design diagnostics and are not used as a test
of general incremental cognitive prediction. This does not change the
arithmetic of the outcome-blind Stable4 AUROC for its residual phenotype.

A separate corrected analysis was frozen at
6c973b463cc4baa2233ec064f059cf463a311ef6. Its references and labels are estimated
only on source centers. For its general executive-memory low-performance
composite, ordinary AUROC is 0.529 (95% CI 0.459–0.595) for Stable4 and 0.625
(0.551–0.682) for the negated brain-health score. The brain-health advantage
over demographics is +0.020 (paired 95% CI -0.057 to +0.118), an additional
post hoc diagnostic comparison. All original analyses are retained.
