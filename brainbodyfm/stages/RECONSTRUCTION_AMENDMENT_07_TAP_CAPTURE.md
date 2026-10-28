# Amendment 07: retain the full submitted extractor's feature taps

October 27, 2026 (local date); recorded after 102 source extractions and before
any reconstructed-source STAGES endpoint. The exploratory STAGES source-center
normalization results (including composite AUROC 0.533) were already known.

## Reason

An implementation audit found that reconstruction retained the submitted
extractor's 1024-coordinate latent but discarded its global, temporal, and
counterfactual taps. Those taps are required to replay the optional CARR gate
in the final submission. Their absence does not invalidate the Stable4 score
calculation, but prevents attribution of the organizer's system-level scores
to a particular non-date branch. CARR may override Stable4 even if Stable4
passes its own gate; the four-output gate alone does not establish deployment.

## Change and preserved analysis

Retain those already-computed taps in a separate private NPZ alongside every
future successful source extraction. Save the same latent file as before; no
extraction, normalization, score weight, direction, target, threshold, endpoint,
bootstrap, permutation, or rank computation changes. A tap-cache write failure
is logged without changing the latent or falling back to another inference
path. A genuine tap-extraction failure retains the existing ordinary fallback.
Tap files are additional source data for a later deployment-branch replay, not
predictors added to the fixed Stable4 analysis.

The current process was interrupted during an unfinished record. Its state and
all frozen source files were archived. The 102 completed latent files are reused
byte for byte; the interrupted unfinished record is retried. Elapsed active
time remains charged against the original 68-hour extraction budget. Additional
disk writes may modestly affect a runtime-dependent reached roster, which has
always been described as a reconstruction rather than the original artifact.

State before migration: SHA256
`fd59cf9a08652ae3989642c0147011de958d0bb3366475facb839c5d7c88bacb`;
active extraction time 3771.276018540986 seconds; zero failed source records.
Previous execution freeze:
`447bb6fb7a2792e43a8d95a12a50f81f2c770258`.
The archived state contains restricted record identifiers and remains private.

Backfilling taps for the first 102 records must occur separately, after the
original normalization/endpoint run, using the same exact submitted extractor.
Backfill must not replace the latent files or change their source roster. All
backfill differences and failures must be logged. The source-only branch replay
must reproduce the submitted all-or-none CARR availability rule and identify
which settings were reconstructed. It cannot prove which original organizer
artifact ran when the original training roster, execution environment, and
training logs remain unavailable.

The unchanged rank analysis remains the next stage when normalization finishes.
Its conclusion will not be used to decide whether to backfill or which branch
to report. The original uncomputed frozen-organizer-normalization endpoint is
still distinct from the amended reconstructed-source endpoint.
