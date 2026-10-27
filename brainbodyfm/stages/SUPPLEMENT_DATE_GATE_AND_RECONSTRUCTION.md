# Supplementary methods: date-free Challenge evaluation and source reconstruction

The submitted system at commit `76a36e54251cfce85889a3b2390018ffc23329ee`
included an optional 80% Stable4/20% date-rank branch. This branch read the
`CreationTime` metadata field and required a saved source-validation gate,
at least 200 target records, at least 90% parseable dates, and at least eight
distinct timestamps. The submitted `recording_date_rank` function enforced:

```python
if coverage < float(min_valid_fraction):
    raise AdaptationDisabled(...)
if unique < max(1, int(min_unique)):
    raise AdaptationDisabled(...)
```

The caller supplied `min_valid_fraction=0.90` and `min_unique=8`. With dates
absent, the date stream failed this gate. The target-cohort prepass caught
`AdaptationDisabled` and retained the ordinary per-record predictions. Synthetic
checks of the exact committed function verified rejection of absent dates,
coverage below 90%, and fewer than eight distinct dates, and acceptance of the
boundary values when the other requirements were met.

In their September 29, 2026 results announcement, the organizers stated:

> The scores in this table reflect the performance of the models without sleep study dates.

[PhysioNet Challenge 2026: Scores released](https://groups.google.com/g/physionet-challenges/c/WyxLESJjrnY).

The official validation and test age-conditioned AUROCs of 0.723 and 0.622 are
therefore reported as date-free submitted-system results. The date branch was
present in the source but inactive in that evaluation. Whether the ordinary
source-selected stream was standalone Stable4 or another guarded non-date
fallback remains conditional on the unavailable organizer-trained artifact;
date removal alone does not resolve that separate selection question.

The organizer receipt identified the large training release. Because the
organizer-trained `model.sav` was unavailable, source normalization was scheduled
for reconstruction using the public large release, version 2 (6,600 records),
the exact submitted extraction/normalization functions, and the pinned upstream
revision. The submitted record order was deterministic, but extraction stopped
at a wall-clock deadline. Reconstructed constants are therefore conditional on
the records reached by the new run; they are not claimed to be byte-identical
to the organizer's constants. The completed source roster and its checksum,
record counts by site, failures, source code, checkpoint, software versions,
and active extraction time will accompany the reconstruction result.

Amendment 06 was recorded after the within-center-normalized STAGES sensitivity
results were examined and before reconstructing source constants. Head identity,
score direction, weights, cognitive outcomes, thresholds, adjustment, pairing,
pooling, resampling, and interpretation rules were retained. The earlier
continuous analyses and their primary endpoint were retained unchanged.

The planned simulation using realized case and pair counts was not performed
before the available STAGES rank results. The planning table is retained in
the dated amendment as unverified planning material, rather than a reproduced
estimate of achieved study power. This timing deviation cannot be repaired
retrospectively. A successful respiratory pipeline check demonstrates detectable
respiratory signal under this procedure; cognitive precision is judged from the
reported cognitive intervals. A STAGES upper confidence limit below a hidden-site
point estimate is descriptive and is not a formal between-population comparison.
