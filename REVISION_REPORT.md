# Revision outcome

The requested work has been implemented as a separate journal revision, not written over the original paper or frozen searches. The results do **not** support the original causal claim that feedback explains the held-site gain, or a claim that all three evaluation sites are independent of foundation-model development.

| Requested item | Completed work | Material limit |
|---|---|---|
| Matched feedback ablation | 30 pairs, 60 valid continuation calls; shared opening five; ten source evaluations per arm; common robust selector; frozen all-arms target gate | Post hoc, archived openings, same reported model ID without immutable-build guarantees |
| Cohort flow | Site-specific initial cache, positive expansions and negative extension; exact 338-record union; available frame denominator 1,103 | Outcome-aware extraction; complete HSP denominator, initial selection log and two missing-extraction reasons are not recovered |
| Clinical outcome | All cached labels match the supplied field; positive times 390–2,175 days, negative follow-up at least 2,192 days; official diagnosis/follow-up rule stated | Two diagnosis episodes separated by a week cannot be independently re-adjudicated |
| Provenance/leakage | Published model manifests downloaded, checksums verified, subject and recording matches, exclusion sensitivity | 87 S0001 training-subject and 121 development-subject matches; historical extraction checkpoint identity remains unavailable |
| Strong comparator | Taxonomy comparison and site differences presented directly | Original macro taxonomy superiority unresolved; near tie at B=50 |
| Statistics | Conditional participant and paired-run intervals separated; zero-pair draws accounted for; all budgets retained; post hoc contrast status explicit | Neither bootstrap creates new clinical cohorts or removes sampling/development exposure |
| Sensitivity | One-year, two-year, three-year and conventional AUROC on fixed selections; conservative-development target exclusions | Metric robustness does not establish independent external validation |
| Reproducibility | Code, prompts, query histories, aggregate tables, checksums, environment lock, build scripts, tests and privacy-reviewed release | Protected private inputs and missing legacy extraction evidence are not redistributable/recoverable |
| Journal conversion | Official IEEE Access class; authors/affiliation/correspondence/biographies; no funding or conflicts; AI disclosure | Ethics and prior-publication statements remain placeholders at the user's request |

## New paired experiment

| Institution | Feedback | No feedback | Difference | Participant 95% CI | Paired-run 95% CI |
|---|---:|---:|---:|---|---|
| I0002 | 0.678049 | 0.697561 | -0.019512 | [-0.070000, +0.030005] | [-0.073171, +0.026829] |
| I0006 | 0.609751 | 0.617914 | -0.008163 | [-0.044858, +0.020170] | [-0.036735, +0.020181] |
| S0001 | 0.548768 | 0.541010 | +0.007759 | [-0.020951, +0.037127] | [-0.026601, +0.042118] |
| Macro | 0.612189 | 0.618828 | -0.006639 | [-0.029093, +0.014980] | [-0.029306, +0.014971] |

All 60 continuations were valid on the first call. There were no model repair calls, random fallbacks or invalid participant-bootstrap draws in this comparison. The selected minimum source AUROC rose by 0.017308 with feedback. That source-side gain did not yield an established held-site gain.

## Provenance changes the central interpretation

The model developers' reconstructed training list matches 87 S0001 subjects, with 74 exact evaluated-record matches. The broader development list matches 121 subjects, including 53 positives and 68 negatives. Neither list contains matched I0002/I0006 identifiers; this is not a cross-hospital identity audit.

After conservative target exclusions, S0001 has three positives and four negatives, supplying four eligible pairs. The original agent's score is 0.400, random robust 0.630, evolutionary 0.675 and taxonomy-diverse 0.5375. The retained three-site macro is 0.581826 for the agent and 0.593816 for random robust. The tiny S0001 remainder cannot support a reliable external-validation conclusion. Its bootstrap rejected 3,715 zero-pair resamples before retaining 50,000 draws.

The separate two-site I0002/I0006 descriptive macro remains favorable (agent 0.672739, random 0.575724, evolutionary 0.546220, taxonomy 0.629133). It changes the evaluation target and retains the original source fits; it is not substituted for the original three-site estimate.

## Editorial corrections

- Retitled the journal version around query-efficient search rather than an inferentially unresolved claim that ten queries beat 500.
- Identified future cognitive impairment in the abstract and defined the supplied clinical labels in Methods.
- Distinguished source-controller target blindness, historical target inspection, outcome-aware cohort construction and representation-development overlap.
- Replaced causal feedback claims with the actual matched result.
- Reported category labels as prior biological information while distinguishing the absence of a forced diversity objective.
- Corrected the statement that I0006 has the most eligible pairs: S0001 has 812, I0006 441.
- Separated Luna's model-choice values (0.6004 at B=10) from its fixed-robust value (0.6021), and explained the 29/30 valid final choices at larger budgets.
- Stated that the original hypothesis contrasts and focal B=10 presentation are exploratory; no retroactive preregistration claim.
- Preserved the favorable 500-bank point estimate together with its zero-spanning participant interval.
- Kept author-required placeholders explicit and made no claim of completed journal submission or guaranteed acceptance.

## Sources verified on October 18, 2026

- IEEE Access author requirements and official template: https://ieeeaccess.ieee.org/authors/submission-guidelines/
- Official cognitive-impairment construction: https://moody-challenge.physionet.org/2026/
- Model training/development manifests and caveats: https://github.com/bdsp-core/philosophers-stone/blob/master/paper/provenance/README.md
- Model paper identity: https://pubmed.ncbi.nlm.nih.gov/41953210/
- Human Sleep Project paper identity: https://pubmed.ncbi.nlm.nih.gov/42555111/

Local primary evidence is in `results/`, with private individual-level material explicitly labeled `*_private.parquet`. Public-release materials must not include those files.
