# Complete the missing submitted EEG-tap representation

October 27, 2026. This engineering extension is specified after the available-
representation composite results were inspected. Those results are retained.
It does not change outcomes, score directions, thresholds, or existing analyses.

The original submission supports an additional CARR branch that uses a frozen
EEG-only global representation, temporal summaries and counterfactual responses
to changing the input age. The 1024-coordinate STAGES cache cannot reconstruct
those inputs. To assess that actual branch rather than infer its behavior from
the smaller cache, re-extract the same 590 previously eligible STAGES EDFs using
the exact submitted `infer_feature_taps` and pinned checkpoint. Retain global
(2048), temporal (32 x 32), counterfactual (8), and ordinary latent (1024) arrays.

This extraction is outcome-blind: its queue contains only IDs, center, age, sex,
EDF location, and prior extraction provenance. Process all 590 in center/ID
order; do not select recordings by cognitive scores or new model results.
Use the already verified bounded CWT, existing strict C4–M1/native-or-derived
route, and the user-confirmed microvolt calibration. Source checkpoint and
submitted code hashes must match their existing manifest. Existing latent and
semantic caches are never replaced.

For every record, compare the re-extracted spectrogram hash and ordinary latent
with its previous extraction. A spectrogram mismatch or latent relative L2 >1e-5
is quarantined for review rather than silently accepted. Save checksums, elapsed
time, shapes, code/plan freeze and any failures; all IDs and arrays remain private.
The previous four-record raw audit achieved exact equality on both routes.

After extraction is complete, the fixed next analysis is the same four outcomes
and center-held protocol in CINC_TRAINABLE_TRANSFER_PLAN.md, adding only the
submitted CARR-enabled procedure and standalone CARR diagnostic. The standalone
diagnostic uses the submitted 80-epoch full-source fit, source-fitted Stable4
baseline, original hyperparameters and seed SEED+8801, regardless of whether
its original deployment gate passes. This is distinct from the gated system.
The gated method must report all source-side branch decisions. CARR input
flattening uses the submitted `carr_feature_matrix` without alteration.

No CARR results are interpreted before the extraction/audit completes. The
ordinary original-organizer artifact is still unavailable; retraining this
procedure in STAGES is not recovery of its official hidden-site artifact.
The ongoing HSP reconstruction is not edited or interrupted. The extraction
uses one CPU thread and processes one recording at a time, without retaining
new spectrogram copies or EDF copies. Expected additional feature storage is
small; raw EDFs are read from the already authorized mounted dataset.
