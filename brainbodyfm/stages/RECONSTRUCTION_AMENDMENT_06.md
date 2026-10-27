# Amendment 06: submitted date gate and reconstructed source normalization

Dated October 27, 2026. This amendment is made after the available STAGES
within-center-normalization sensitivity and comparator rank results were seen,
and before computing reconstruction constants or a STAGES endpoint using them.
The original continuous primary search-gap endpoint and all previous results
are unchanged. The initial added rank plan remains in its original commit.
This amendment is not a retrospective claim of preregistration.

Decisions frozen before reconstruction:

* Use the public **large training release, version 2**, identified below.
* Label all new source moments **reconstructed**, never the original saved file.
* Apply them to the unchanged rank endpoint and analyses; disclose the power
  timing deviation and the runtime-dependent source roster.

## Date branch

The final submission source is `i-habib/phys-26-large-new` at
`76a36e54251cfce85889a3b2390018ffc23329ee`, identified by the corresponding author
and present in an existing local Git checkout. Its four signed Stable4 weights
are +0.360, +0.270, -0.235, -0.135 in the original frozen head order.

The submitted system also contains an optional 80/20 Stable4/date rank blend.
`phi_challenge/cohort.py::_recording_date_features` reads only the metadata field
`CreationTime` for this stream. `recording_date_rank` raises `AdaptationDisabled`
when `coverage < float(min_valid_fraction)` or
`unique < max(1, int(min_unique))`. The caller supplies a default valid-date
fraction of 0.90 and `min_unique=8`. The target prepass also requires at least
200 records and the saved source-validation gate. Missing dates therefore
disable this branch. `team_code.py::_ensure_cohort_predictions` retains ordinary
per-record predictions if adaptation is disabled; source-only Stable4 also
bypasses the cohort prepass when it is the sole saved branch.

The organizers' September 29, 2026 announcement confirms that the posted
evaluation excludes sleep-study date information. The exact short quotation is
preserved in the supplementary audit text.

Source: https://groups.google.com/g/physionet-challenges/c/WyxLESJjrnY

Thus the reported organizer scores are date-free; the optional date branch must
be described as present in the submitted system and inactive in the date-free
evaluation. This establishes absence of date contribution. It does not by
itself identify the source-selected non-date deployment streams: the source
also contains guarded B/REM and optional CARR fallbacks/adapters. Attribution
to standalone Stable4 is conditional on its saved source-selection gate, whose
organizer artifact is unavailable. Report the official points as date-free
submitted-system performance, with the Stable4 component's provenance stated.

## Training release and reconstruction

The organizer receipt says `validation/large`; `SUBMISSION_VARIANT.txt` says
`large`. The training wrapper accepts the organizer-supplied `data_folder` and
does not select small versus large internally. The confirmed release category
is therefore large, not the locally stored small release.

Use the publicly released large training dataset, Kaggle
`physionet/physionetchallenge2026datalargeversion`, version **2**, last updated
June 9, 2026, with 6,600 metadata records (S0001 5,139; I0002 319; I0006 1,142).
The metadata SHA256 frozen before reconstruction is
`b9009598589846e84462636bfcaa290ed1d8666c89b1bc1ad1a9eda26b43ff7f`.
The exact internal organizer snapshot and completed extraction roster cannot be
verified. No hidden validation/test records enter reconstruction. The 338-person
development cache and the small release are not substituted.

Run the exact submitted record queue (site-by-label round robin, seed 20260721),
channel resolution, preprocessing, latent extraction, head projection, and
`_fit_stable4_artifact` normalization on the public large release. Pin the
upstream code to the submitted Docker revision
`0b1b49a8665e11157a107e0d76b9eedddd3b04c7` and record the current checkpoint
SHA256 `b2a9b8dab3ae8543241d613a80cd6a85a6dfca15573f897a1096861b3915af87`.
The old entry checked checkpoint byte count, not this historical digest; a
byte-identical organizer checkpoint cannot be established retrospectively.

Preserve the submission's default 70-hour total budget and two-hour reserve:
the reconstruction extraction stage has a 68-hour active-time budget, a rolling
1.5-times-recent-record guard, and the same failure-skipping policy. If the
entire release is extracted sooner, use all completed records. A resumed run
counts active processing/download time cumulatively and excludes the idle
interval between invocations; record both active and elapsed wall-clock time.
This continuation rule is a practical reconstruction rule, not a claim to
reproduce an interrupted organizer process. Do not choose a prefix or stop
early based on any STAGES result. Per-record files are streamed into a bounded
cache and removed after extraction. Network and CPU runtime affect the reached
prefix, as did organizer hardware/runtime. Record the exact private completed
roster, per-site counts, source failures, package versions, checkpoint and code
hashes, and input checksums. No STAGES outcome or rank metric is consulted.

**These are reconstructed constants, not the original `model.sav`.** The
population mean/SD transformation is deterministic conditional on identical
features and the same completed records; the original training loop's
wall-clock cutoff means identical constants are not guaranteed by identical
source code and release alone. The reconstruction uncertainty will be reported.

After the source run finishes, freeze its source-only normalizer export and
full-head source moments by SHA256, then compute the previously specified rank
endpoint and all dependent exploratory analyses with no change to score
orientation, signed weights, outcomes, cutoffs, covariate adjustment, age window,
within-center pairing, pooling, bootstrap, permutation, or interpretation rules.
Label this result the **reconstructed-source-normalized analysis**. It is not
represented as an exact application of an unavailable organizer artifact.

## Power reporting deviation

The planned rerun using realized case and pair counts was **not performed before
any STAGES rank endpoint**. The sensitivity/comparator results were examined
before this omission was identified. The reconstructed-source result remains
uncomputed at this amendment, but running power now could not repair the earlier
timing deviation. The preceding CLI lock is superseded for this explicitly
amended reconstructed-source analysis; it is not silently bypassed.

Retain the originally supplied planning table below, clearly identified as
**unverified planning approximations**, not as a reproduced simulation or as
achieved study power. No retrospective observed-power calculation will be used
to interpret the results. Realized composite counts are 103/588 cases and 2,193
eligible pairs at -1.0 SD; the -1.5 threshold has 42 cases and 1,009 pairs.

| Assumed true AUROC | Supplied planning power, -1.0 SD | Supplied planning power, -1.5 SD |
|---|---:|---:|
| 0.55 | 28% | 12% |
| 0.58 | 53% | 30% |
| 0.60 | 76% | 38% |
| 0.62 | 88% | 60% |
| 0.65 | 97% | 74% |

## Interpretation

The AHI result validates detection of respiratory signal by this procedure; it
does not establish adequate power for smaller cognitive effects. Interpret a
cognitive interval crossing 0.5 as failure to establish discrimination, not
proof of exactly no association. Comparing the STAGES upper confidence limit
with a hidden-site point estimate is descriptive; it is not a confidence
interval for a difference between the settings, which use different outcomes.

No OSF registration was completed. Git records the original freeze, the later
available results, and this post-sensitivity reconstruction amendment separately.

## Transport repair before the first source extraction

The first launch stopped at the legacy Kaggle download URL with HTTP 404;
zero source records were extracted and no source constants were calculated.
The downloader was changed to the current official Kaggle SDK's
`datasets.DatasetApiService/DownloadDataset` RPC with an explicit version-2
request. This repairs file transport only. The old failed initialization state
is archived; the extraction clock begins with the repaired source-only run.
Scientific rules and all STAGES code remain unchanged.
