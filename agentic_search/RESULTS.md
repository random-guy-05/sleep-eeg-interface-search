# Final Report: Prospective Four-Head Search Across Institutions

**Final review date:** 26 September 2026  
**Status:** Search and target scoring are complete. The current integrity audit passes. This report reconciles the frozen protocol, source-only run records, manifest, sealed results, post-target analyses, and figures. It replaces the earlier draft, which contained material errors in the feature-space description and statistical interpretation.

## Executive summary

The project evaluated whether one source-guided GPT-6 Sol search could find four-physiology-head representations that transfer to a held-out institution better than random or classical search. The final experiment used **539 heads in 11 taxonomy categories**, not the 100 heads and 12 categories stated in the earlier draft. GPT-6 Sol at medium reasoning effort was the only language model in the main search.

The primary result is favorable against the random and evolutionary baselines on these three held institutions. With the fixed robust source selector, the full agent’s equal-weight macro age-conditioned AUROC was **0.6302 at B=10**, **0.6115 at B=25**, and **0.6155 at B=50**. The corresponding random robust means were 0.5644, 0.5679, and 0.5703; evolutionary means were 0.5503, 0.5487, and 0.5586. Corrected, exploratory, run-level bootstrap intervals for the macro differences from random robust and evolutionary search are above zero at all three budgets.

The comparison with strict taxonomy diversity is different: its means were 0.6073, 0.6047, and 0.6154. The full-agent advantage is small at B=10, nearly zero at B=25, and effectively zero at B=50; paired run-level intervals include zero at every budget. The B=25 ablations also do not establish a clear benefit from iterative feedback or semantic feature names. Their point estimates are modestly lower than the full agent, with wide paired intervals that include zero.

The best mean for the full agent occurred at **B=10**. Increasing the budget did not improve the mean selected target score, while the average source-to-target optimism gap rose from 0.0894 at B=10 to 0.1297 at B=25 and 0.1334 at B=50. This is consistent with stronger source-score selection pressure, but the separate runs do not establish that a larger budget caused worse target performance.

The result is evidence of better transfer performance than random and evolutionary selection on these particular folds. It is not evidence of reliable performance at new institutions in general. The evaluation covers only three fixed institutions and 338 participants, with substantial site imbalance and only seven positive participants at I0002.

## 1. Study question and data

The search space consists of **539 named physiological heads**, each represented as one feature, grouped into 11 taxonomy categories. A candidate is a quartet of four distinct heads. Previously target-inspected historical quartets, including Stable4, were excluded from new candidate generation.

| Taxonomy category | Heads |
|---|---:|
| NREM spectral | 166 |
| REM spectral | 53 |
| Arousal continuity | 1 |
| Delta slow wave | 63 |
| Other sleep physiology | 135 |
| Respiratory | 1 |
| Sigma dynamics | 44 |
| Signal morphology | 15 |
| Sleep architecture | 16 |
| Spindle morphology | 1 |
| Theta | 44 |
| **Total** | **539** |

The evaluation includes three institutions. The held-out cohort sizes and positive counts were:

| Held institution | Participants | Positive outcomes |
|---|---:|---:|
| I0002 | 43 | 7 |
| I0006 | 167 | 20 |
| S0001 | 128 | 56 |
| **Total** | **338** | **83** |

The source-only participant feature table contains no outcome column. For each held-site fold, the source-fold file contains outcomes only for the other two institutions. The sealed target-label file is joined to held-site features by subject, site, and age only inside the target scorer.

## 2. Metric and evaluation procedure

### Age-conditioned AUROC

The primary metric is the age-window micro-AUROC with a two-year window. At a given institution, the scorer considers every eligible positive-negative participant pair whose absolute age difference is at most two years. A correctly ordered pair contributes 1, a tie contributes 0.5, and an incorrectly ordered pair contributes 0. Every eligible pair has equal weight. This is not the mean of per-positive AUROCs and is not ordinary, age-unconditioned AUROC.

The primary macro score is the **unweighted mean of the three held-site AUROCs**. Each institution therefore has equal weight in the macro result, regardless of participant count. Site-specific scores are also retained so the macro cannot conceal fold differences.

### Source-only C selection and held-site scoring

For each held institution, the two remaining institutions are the visible source pair. A four-feature L2 logistic-regression probe is evaluated in both directions, A-to-B and B-to-A. In each direction, median imputation and standardization are fit using only the source institution, then applied to the other institution. PCA is disabled for this four-head probe. No held-site rows are included in preprocessing fits.

The regularization parameter C is chosen from 17 logarithmically spaced values from 10⁻⁴ through 10⁴. For each C, the two directional age-conditioned AUROCs are averaged. The highest mean is selected; values within 0.0001 of the maximum are treated as tied, and the smallest C is used. The final probe is refit on both source institutions using that C and source-only preprocessing, then scored once on the held institution.

For the primary result, each run selects the candidate with the largest **minimum of its two directional source AUROCs**. Lexicographic quartet order breaks an exact tie. This robust selector was fixed in the protocol. The agent’s own final source-informed pick is a secondary analysis.

### What the model could see

The sole model was GPT-6 Sol through Codex CLI, with medium reasoning effort. The controller supplied the fixed feature catalog and prior source-only scores, validated each batch of five proposed quartets, and ran the local source evaluator. The model did not receive shell, browser, evaluator, or target-label access. It was told not to request or use target information. On its final round, it received the full source-only history and selected one evaluated quartet.

The opaque ablation substituted opaque IDs for feature names and taxonomy categories. The batch-semantic ablation retained names and categories but produced its 25 candidates in one batch without intermediate feedback. The model’s temperature was requested as 0.2, but Codex CLI did not expose a temperature control; actual sampling was provider-default and uncontrolled. Seeds were recorded and passed to calls, but they do not make provider-default sampling fully reproducible.

## 3. Search conditions and completed run counts

| Condition | Replication design | Source-evaluated candidates |
|---|---|---:|
| Full GPT-6 Sol iterative search | 10 seeds × 3 sites × budgets 10, 25, and 50; batches of five | 2,550 |
| Opaque iterative ablation | 10 seeds × 3 sites × B=25 | 750 |
| Batch semantic ablation | 10 seeds × 3 sites × B=25 | 750 |
| Evolutionary search | 20 seeds × 3 sites × budgets 10, 25, and 50 | 5,100 |
| Taxonomy-diverse search | 20 seeds × 3 sites × budgets 10, 25, and 50 | 5,100 |
| Prospective random bank | 500 new quartets, each evaluated for 3 held-site folds | 1,500 |

The full-agent main grid therefore contains **90 trajectories**. The two ablations each contain 30 trajectories. Each classical baseline contains 180 trajectories. The random baseline uses one fixed bank of 500 quartets and 100 randomized orderings, with nested prefixes at B=1, 10, 25, 50, 100, and 500. Random quartets are selected by either mean source AUROC or minimum directional source AUROC. At B=1 those two selectors necessarily coincide.

The evolutionary baseline starts from a batch of five, keeps three elites, and mutates one head uniformly per proposal. The taxonomy-diverse baseline requires four different taxonomy categories within every quartet. Both have 20 repetitions per site and budget. The formal GPT-6 Sol pilot consists of five B=25 trajectories at I0002, seeds 3000–3004; it is separate from and excluded from the primary target manifest and results.

Across all conditions, the saved candidate-query table contains 15,750 evaluation rows. Repeated fold-quartet queries across runs are expected. The manifest deduplicates them to the **15,128 unique held-site/quartet pairs** that were scored against the target labels.

## 4. Target-label gate, historical exception, and integrity audit

The prospective candidate manifest was frozen before new target scoring. Its canonical payload hash is:

**83c1380ccaf2bd4f66e0a8036ddfe7688248a7e1a4331db18715a34ff6b4d34e**

The target result table has exactly 15,128 rows and 15,128 unique held-site/quartet keys. All scores are finite. Every prospective result records zero target-label use for model fitting or C selection. The target scorer checks manifest integrity, candidate membership, and the source-selected C before scoring. Target labels are used to calculate held-site AUROC after the manifest is frozen; those scores are then used for this report and other post-target analyses.

**Historical-regression exception:** the project also ran a separate regression check on previously target-inspected historical candidates. The historical path explicitly loads target labels for those old candidates before the new prospective search. Those historical scores were not supplied to GPT-6 Sol, and historical quartets were excluded from the new candidate manifest. Therefore the defensible claim is that prospective candidate generation and source-based selection were target-blind—not that the target-label file was never read anywhere in the project. The project README now records this exception.

The fresh audit is saved in runs/final_audit.json and passes. It verifies frozen protocol, prompt, execution-code and source-input hashes; checks source-fold separation and candidate validity; verifies the manifest and target-release alignment; confirms all prospective target-use flags are zero; and scans five source-path code files and 240 source-only JSON, JSONL, and text artifacts for selected target-result strings. The audit’s string scan is a bounded integrity check, not a system-level trace of every file read. The audit reports 120 primary-agent JSON trajectory states because it counts full-agent and opaque iterative states; batch-semantic runs are stored separately in parquet.

## 5. Primary results

The trajectory percentile column below is the empirical 2.5th–97.5th percentile range across completed macro trajectories or random orderings. These are **not bootstrap confidence intervals** and do not measure participant-sampling uncertainty. The optimism gap is the selected candidate’s mean source-direction AUROC minus its target AUROC, averaged over site-run selections. Target regret is the post-hoc best target AUROC among candidates queried in the same run minus the selected candidate’s target AUROC. Regret is an oracle diagnostic, not a deployable selection rule.

| Method | B | Macro mean | Median | Trajectory 2.5th–97.5th percentile | Mean optimism gap | Mean target regret | Site means: I0002 / I0006 / S0001 |
|---|---:|---:|---:|---:|---:|---:|---|
| Random + source robust | 1 | 0.5550 | 0.5668 | 0.4016–0.6888 | 0.0181 | 0.0000 | 0.6190 / 0.5101 / 0.5360 |
| Random + source robust | 10 | 0.5644 | 0.5697 | 0.4555–0.6717 | 0.1146 | 0.1365 | 0.6441 / 0.5073 / 0.5418 |
| Evolutionary robust | 10 | 0.5503 | 0.5590 | 0.4489–0.6518 | 0.1186 | 0.1371 | 0.6134 / 0.4790 / 0.5584 |
| Taxonomy-diverse robust | 10 | 0.6073 | 0.6015 | 0.5155–0.7145 | 0.0897 | 0.1318 | 0.7305 / 0.5278 / 0.5636 |
| **Full agent, GPT-6 Sol** | **10** | **0.6302** | **0.6386** | **0.5870–0.6642** | **0.0894** | **0.1005** | **0.7244 / 0.6211 / 0.5451** |
| Random + source robust | 25 | 0.5679 | 0.5796 | 0.4762–0.6410 | 0.1302 | 0.1735 | 0.6544 / 0.5138 / 0.5357 |
| Evolutionary robust | 25 | 0.5487 | 0.5481 | 0.4550–0.6392 | 0.1618 | 0.1849 | 0.6232 / 0.4872 / 0.5357 |
| Taxonomy-diverse robust | 25 | 0.6047 | 0.6130 | 0.5079–0.6833 | 0.0957 | 0.1473 | 0.7110 / 0.5435 / 0.5595 |
| Opaque iterative | 25 | 0.6039 | 0.5981 | 0.5511–0.6527 | 0.1078 | 0.1033 | 0.7195 / 0.5045 / 0.5877 |
| Batch semantic | 25 | 0.5984 | 0.5952 | 0.5325–0.6567 | 0.1183 | 0.1564 | 0.7024 / 0.5451 / 0.5478 |
| **Full agent, GPT-6 Sol** | **25** | **0.6115** | **0.6133** | **0.5678–0.6559** | **0.1297** | **0.1163** | **0.6756 / 0.5957 / 0.5633** |
| Random + source robust | 50 | 0.5703 | 0.5819 | 0.4619–0.6390 | 0.1393 | 0.1897 | 0.6583 / 0.5155 / 0.5372 |
| Evolutionary robust | 50 | 0.5586 | 0.5652 | 0.4473–0.6100 | 0.1687 | 0.1750 | 0.6646 / 0.4811 / 0.5300 |
| Taxonomy-diverse robust | 50 | 0.6154 | 0.6127 | 0.5321–0.6827 | 0.0949 | 0.1506 | 0.7561 / 0.5302 / 0.5600 |
| **Full agent, GPT-6 Sol** | **50** | **0.6155** | **0.6156** | **0.5962–0.6331** | **0.1334** | **0.1278** | **0.7000 / 0.5980 / 0.5486** |

### Extended random source-mean selector

The random source-mean selector chooses the highest mean of the two source directions rather than the highest minimum. B=1 is identical to the robust selector. Results beyond B=50 are useful reference points, but the full agent did not run at those budgets.

| B | Macro mean | Median | Trajectory 2.5th–97.5th percentile | Site means: I0002 / I0006 / S0001 |
|---:|---:|---:|---:|---|
| 10 | 0.5717 | 0.5690 | 0.4696–0.6627 | 0.6639 / 0.5105 / 0.5408 |
| 25 | 0.5682 | 0.5731 | 0.4755–0.6376 | 0.6529 / 0.5122 / 0.5394 |
| 50 | 0.5818 | 0.5878 | 0.4822–0.6555 | 0.6705 / 0.5237 / 0.5511 |
| 100 | 0.5834 | 0.5881 | 0.4862–0.6402 | 0.6712 / 0.5374 / 0.5416 |
| 500 | 0.5901 | 0.5901 | 0.5901–0.5901 | 0.7317 / 0.5374 / 0.5012 |

At B=100, random robust scored 0.5642, with site means 0.6556 / 0.5109 / 0.5262. At B=500, random robust scored 0.5612, with site means 0.7317 / 0.4422 / 0.5099. At B=500 each ordering contains the same entire bank, so the selected result is fixed and the across-ordering percentile range collapses to a point.

### Corrected run-level comparisons

The initial claim-statistics implementation treated the full Cartesian set of pairwise run differences as independent observations. That pseudoreplicated the same runs and produced intervals that were too narrow. It has been replaced by a run-level bootstrap. The final comparison file uses 50,000 percentile-bootstrap draws with random seed 20260926. Full-agent versus random comparisons independently resample macro trajectories and random orderings; comparisons with evolutionary, taxonomy-diverse, and B=25 ablations preserve shared seeds and resample paired macro differences.

These are post-hoc, nominal 95% intervals for **run variability conditional on these sites and the fixed random bank**. They are not participant-level intervals or p-values. No multiplicity correction was applied. The paired win fraction is the observed fraction of the ten shared-seed macro comparisons in which the full agent scored higher.

| Comparison: full agent minus comparator | B | Mean macro difference | 95% run-bootstrap interval | Design | Paired seed win fraction |
|---|---:|---:|---:|---|---:|
| Random + source robust | 10 | +0.0658 | +0.0444 to +0.0865 | Independent runs: 10 vs 100 | — |
| Random + source robust | 25 | +0.0436 | +0.0243 to +0.0629 | Independent runs: 10 vs 100 | — |
| Random + source robust | 50 | +0.0452 | +0.0328 to +0.0578 | Independent runs: 10 vs 100 | — |
| Random + source mean | 10 | +0.0584 | +0.0370 to +0.0791 | Independent runs: 10 vs 100 | — |
| Random + source mean | 25 | +0.0434 | +0.0245 to +0.0623 | Independent runs: 10 vs 100 | — |
| Random + source mean | 50 | +0.0338 | +0.0227 to +0.0450 | Independent runs: 10 vs 100 | — |
| Evolutionary robust | 10 | +0.0862 | +0.0545 to +0.1169 | Paired shared seeds, n=10 | 100% |
| Evolutionary robust | 25 | +0.0397 | +0.0147 to +0.0638 | Paired shared seeds, n=10 | 80% |
| Evolutionary robust | 50 | +0.0455 | +0.0280 to +0.0609 | Paired shared seeds, n=10 | 90% |
| Taxonomy-diverse robust | 10 | +0.0160 | −0.0270 to +0.0590 | Paired shared seeds, n=10 | 50% |
| Taxonomy-diverse robust | 25 | +0.0034 | −0.0436 to +0.0513 | Paired shared seeds, n=10 | 50% |
| Taxonomy-diverse robust | 50 | −0.0009 | −0.0224 to +0.0220 | Paired shared seeds, n=10 | 40% |
| Batch semantic | 25 | +0.0131 | −0.0098 to +0.0373 | Paired shared seeds, n=10 | 60% |
| Opaque iterative | 25 | +0.0076 | −0.0200 to +0.0349 | Paired shared seeds, n=10 | 50% |

The random comparison uses 100 orderings of one fixed bank, not 100 independently sampled banks. The evolutionary and taxonomy conditions each have 20 replications, but the paired comparison uses the ten seeds shared with the full agent. Site-level contrast estimates and intervals are in analysis/output/claim_statistics.csv.

**Evolutionary-search estimand clarification.** The reported evolutionary differences +0.0862, +0.0397, and +0.0455 are *paired macro run-level mean differences across the ten seed-matched comparisons*; they are not obtained by subtracting the all-run Table 1 means. Subtracting the Table 1 macro means instead gives +0.0799, +0.0628, and +0.0570 at B=10, 25, and 50, respectively. The first set uses a paired subset of ten shared seeds (the evolutionary condition has 20 repetitions per site); the second compares the method-level means using all available runs.

### Random robust search at B=500

The max-min source selector was rerun over all 500 frozen bank quartets at each held site, using only source-direction scores and the frozen bank-index tie rule. It selected the same winners as the stored prospective analysis at all three sites (audit: analysis/output/random_b500_selector_rerun.csv). The winners' target scores are already in the released candidate results. Against the ten-candidate full-agent condition at B=10, the 500-candidate random robust selector scores 0.5612 macro AUROC (site means 0.7317 / 0.4422 / 0.5099), versus 0.6302 for the agent (0.7244 / 0.6211 / 0.5451): a point difference of +0.0689 for the agent. A participant-level bootstrap interval for this contrast is −0.0136 to +0.1509, so the point estimate favors the agent but does not establish an advantage at the participant-sampling level. Site differences (agent minus random-500) are −0.0073 / +0.1789 / +0.0352, showing that the macro difference is driven mainly by I0006.

### Participant-level bootstrap and age-matched pair counts

To quantify participant-sampling uncertainty while keeping the search selections fixed, we resampled participants within each held site and within outcome class (sampling with replacement while preserving the observed positive and negative counts). The same site-specific participant draws were applied to all methods. Each draw recomputed the ±2-year age-conditioned micro-AUROC for every fixed selected quartet, averaged selections within site, and then equally averaged the three site means. We used 50,000 valid replicates (seed 20260927); all requested replicates were valid. These intervals are conditional on the selected quartets and their source-fitted scores; they do not include search-run variability.

| Held site | Participants | Positive / negative | Eligible age-matched positive-negative pairs |
|---|---:|---:|---:|
| I0002 | 43 | 7 / 36 | 41 |
| I0006 | 167 | 20 / 147 | 441 |
| S0001 | 128 | 56 / 72 | 812 |

| Full agent minus comparator | B | Difference in macro target AUROC | 95% participant-bootstrap interval |
|---|---:|---:|---:|
| Random + source robust | 10 | +0.0658 | +0.0051 to +0.1309 |
| Random + source robust | 25 | +0.0436 | −0.0150 to +0.1053 |
| Random + source robust | 50 | +0.0452 | −0.0176 to +0.1124 |
| Evolutionary robust | 10 | +0.0799 | +0.0120 to +0.1513 |
| Evolutionary robust | 25 | +0.0628 | +0.0006 to +0.1259 |
| Evolutionary robust | 50 | +0.0570 | −0.0093 to +0.1243 |
| Taxonomy-diverse robust | 10 | +0.0229 | −0.0337 to +0.0761 |
| Taxonomy-diverse robust | 25 | +0.0068 | −0.0648 to +0.0759 |
| Taxonomy-diverse robust | 50 | +0.0001 | −0.0514 to +0.0493 |
| Random + source robust, B=500 | 10 vs 500 | +0.0689 | −0.0136 to +0.1509 |

The bootstrap estimand is an equally weighted macro difference across held sites, with each method represented by its fixed selections at that site and budget. Random robust B=500 has only one distinct winning quartet per site (repeated across the 100 orderings), so those repetitions do not represent independent candidate banks.

### Formal agent versus taxonomy-diverse tests

We used two-sided participant-bootstrap tail tests for full agent versus taxonomy-diverse robust selection at each budget, followed by Holm correction across the three budget-specific tests. The null-centered, two-sided bootstrap p-values were 0.4234 (B=10), 0.8549 (B=25), and 0.9963 (B=50); Holm-adjusted p-values across the three budgets were 1.0000. The 95% participant-bootstrap intervals above all include zero. These tests find no evidence of a full-agent advantage over taxonomy-diverse search; they do not establish equivalence.

## 6. Findings and interpretation

### Full search against random and evolutionary search

At each matched budget, the full agent’s macro point estimate exceeds both random selectors and the evolutionary baseline. The corrected run-level intervals for those macro differences are above zero. Relative to random robust, full-agent differences are +0.0658, +0.0436, and +0.0452 at B=10, 25, and 50. Relative to evolutionary robust, paired macro run-level means across the ten seed-matched trajectories are +0.0862, +0.0397, and +0.0455; subtracting the all-run method means in Table 1 instead gives +0.0799, +0.0628, and +0.0570.

The clearest and most consistent site contribution is I0006. Full-agent minus random-robust site means are +0.1138 at B=10, +0.0819 at B=25, and +0.0825 at B=50. At S0001 the corresponding differences are only +0.0032, +0.0276, and +0.0115; the B=10 and B=50 site intervals include zero. The macro advantage is therefore not uniform across institutions.

### Taxonomy-diverse search

The taxonomy baseline is a strong comparator. It is within 0.023 macro AUROC of the full agent at B=10, within 0.007 at B=25, and essentially tied at B=50. Its paired intervals include zero at all budgets. At B=50 it scores 0.7561 on I0002 compared with 0.7000 for the full agent, while the full agent scores 0.5980 on I0006 compared with 0.5302 for taxonomy-diverse search. The averages conceal this site dependence.

The evidence supports a clear advantage over random and evolutionary selection under this setup, but not a clear advantage over an explicit four-category diversity rule.

### Ablations and candidate-set coverage

At B=25, full-agent source-robust selection scores 0.6115, compared with 0.5984 for batch semantic and 0.6039 for opaque iterative search. Their paired intervals are wide and include zero. The batch-semantic, opaque, and full conditions average 3.50, 2.94, and 3.34 unique taxonomy categories per candidate, respectively. These descriptive differences do not identify a causal contribution from feedback or semantic names.

The post-hoc best target candidate in the queried set averages 0.7549 for batch semantic and 0.7278 for full agent, even though the batch-semantic robust-selected score is lower. This target oracle is not available to a deployable procedure. It shows that candidate coverage and source-only selection are separate stages; it does not establish that batch semantic search is better.

### Budget behavior

Among the tested full-agent budgets, B=10 has the highest mean selected target score. B=25 and B=50 do not improve on it. The full-agent optimism gap rises with budget, and target regret within the queried set also rises from 0.1005 at B=10 to 0.1163 at B=25 and 0.1278 at B=50. With ten macro runs per budget, these patterns are descriptive; they are not a causal estimate of the effect of search budget.

### The agent’s own final choice

All 90 full-agent trajectories contain a valid final model choice among the candidates evaluated in that trajectory. A final review found that the frozen aggregator read the wrong JSON field name for this secondary choice, so the earlier aggregate output omitted those rows. The primary robust-selector table was unaffected. To preserve the frozen search and scoring code, the choices were recovered in a separate post-hoc analysis from the saved source-only states and the frozen target release. The per-trajectory detail and summary are saved in analysis/output/agent_choice_secondary_per_trajectory.csv and analysis/output/agent_choice_secondary_summary.csv.

| B | Agent-choice macro mean | Median | Trajectory 2.5th–97.5th percentile | Site means: I0002 / I0006 / S0001 | Paired difference vs robust selector | 95% paired run-bootstrap interval | Seed win fraction |
|---:|---:|---:|---:|---|---:|---:|---:|
| 10 | 0.6344 | 0.6370 | 0.5866–0.6738 | 0.7098 / 0.6492 / 0.5442 | +0.0042 | −0.0078 to +0.0134 | 60% |
| 25 | 0.6141 | 0.6149 | 0.5656–0.6564 | 0.6756 / 0.6020 / 0.5648 | +0.0026 | −0.0012 to +0.0084 | 30% |
| 50 | 0.6171 | 0.6149 | 0.5962–0.6461 | 0.7000 / 0.5975 / 0.5538 | +0.0016 | −0.0005 to +0.0052 | 10% |

The agent-choice versus robust differences are small and each interval includes zero. Across site-run selections, the agent’s own choice differs from the deterministic robust choice in 36.7% of B=10 trajectories, 26.7% at B=25, and 16.7% at B=50. This secondary outcome is post-hoc and exploratory.

### Descriptive trajectory review

The saved worst, median, and best examples were selected after target release by ranking full-agent robust-selected target AUROCs. They are illustrative case checks, not predeclared qualitative cases or independent evidence.

- The selected candidate in S0001, B=10, seed 1001 scored 0.4409 on the target despite a minimum source AUROC of 0.6916 and mean source AUROC of 0.7117. An unselected first-round candidate in that same run later scored 0.6810 on the target. The source-only selector could not use that target result.
- The median selected example, I0006 at B=50, seed 1006, scored 0.6100.
- The best selected example, I0002 at B=10, seed 1008, scored 0.8780. Its two source directions were 0.6256 and 0.6213. I0002 has only seven positive participants, so this high single-run score should not be treated as a stable estimate.

No causal biological interpretation is warranted from these three outcome-selected examples.

## 7. Historical evaluator regression

This diagnostic is separate from the prospective results and uses previously target-inspected historical candidates. The saved regression check passes an absolute tolerance of 5×10⁻⁹.

| Historical analysis | Source macro AUROC | Target macro AUROC |
|---|---:|---:|
| B=1 historical result | 0.566759 | 0.555830 |
| B=500 source-selected winners, freshly refit | 0.780061 | 0.578594 |

The B=500 held-site target scores were I0002=0.780488, I0006=0.532880, and S0001=0.422414. These historical scores are not part of the prospective search comparison. Their target-label read is the exception described in Section 4.

## 8. Reporting corrections and statistical status

The earlier summary and report draft overstated what the artifacts establish. This final report corrects the following:

1. The feature catalog is 539 heads in 11 categories, not 100 heads in 12 categories.
2. The formal pilot is five I0002 B=25 trajectories, not 30 main-agent trajectories. The primary full-agent grid is 90 trajectories.
3. The main table’s 2.5th and 97.5th columns are empirical percentiles across run outcomes, not bootstrap confidence intervals.
4. The experiment freeze covers the configured model, prompt hashes, evaluation code and source inputs. It does not document a preregistered set of statistical hypotheses. The hypothesis comparisons and run-level bootstrap in this report are post-hoc; the bootstrap has no multiple-comparison adjustment.
5. The historical evaluator regression reads target labels for previously inspected quartets. New prospective candidate generation and source-based selection remain target-blind, but the whole project did not have zero target-label reads.
6. Batch-semantic and opaque-iterative win fractions against full agent are 60% and 50% on the paired macro seeds, not the higher fractions in the earlier draft.
7. The trajectory examples were selected from observed target outcomes after release. They are not predeclared case studies.
8. The agent’s valid final choices were missing from an earlier secondary aggregate because of a field-name mismatch. They are now reported separately without altering the primary results or frozen evaluator.

The statistical intervals describe variability among repeated search trajectories or random orderings on the same fixed institutions and participants. They do not quantify uncertainty from sampling new participants or new institutions. Bootstrap fractions above zero are properties of the resampled run differences, not probabilities that a scientific effect is true.

The corrected bootstrap analysis and final-choice recovery are post-target analysis scripts. They are deliberately separate from the frozen candidate-generation and scoring code; the protocol’s original execution hashes remain unchanged.

## 9. Limitations

- There are only three held institutions. No independent institution-level test set remains after inspecting the reported target outcomes.
- The 338 participants are highly imbalanced by institution and outcome. I0002 contributes 43 participants and seven positives. Age conditioning further restricts the eligible positive-negative pairs.
- The ten full-agent repetitions per site and budget estimate search variability on a fixed dataset. They do not create ten new cohorts or ten new institutions.
- The random reference reorders one fixed bank; its 100 orderings do not represent 100 independently generated candidate banks.
- Requested sampling temperature was not controllable through Codex CLI. Exact regeneration of model proposals is therefore not assured by the recorded seeds alone.
- Historical regression outputs used previously inspected target labels. Although isolated from the GPT-6 Sol prompt and excluded from the prospective manifest, this limits any claim that the project’s entire analysis team was blind to all target information.
- The inferential contrasts were designed after outcomes were available, no multiplicity adjustment was made, and the bootstrap unit is a trajectory rather than a participant. They should be treated as exploratory support for the observed comparison, not confirmatory hypothesis tests.
- No clinical decision threshold, calibration analysis, or deployment study is reported here. AUROC is a ranking measure.

## Supplemental controller generalizability attempt (incomplete)

A preregistered plan was committed before the controller runs. It specified GPT-6 Luna and Claude Opus 5.5 at medium effort, Sol’s exact prompt/catalog/batch/retry/validator/exclusion/final-choice protocol, 90 adaptive runs and 30 zero-shot picks per model, a separate frozen manifest, participant bootstrap with a 15-test Holm family, and per-quartet category/overlap/cost reporting. The source-only attempt reused that protocol. Both two-run pilots passed. Luna completed 88/90 main runs with valid final choices and 30/30 zero-shot picks; Opus completed 0/90 main runs and 4/30 zero-shot picks because of the provider session limit. Free OpenRouter Qwen and Gemma fallback requests returned upstream HTTP 429.

A separate generalizability manifest was frozen before target scoring (SHA-256 `a33429f6a66bc568dfb47eaeb35368053f3ad8aac1982bd15c62d729821bf008`); it contains 2,790 unique held-site/candidate pairs. The original frozen manifest and result release were left unchanged. Luna final-choice target scores, category counts, exact matched-cell overlap with Sol, and zero-shot results are descriptive in [generalizability_status.md](analysis/output/generalizability_status.md) and its linked CSVs. The main planned participant bootstrap and Holm tests cannot be completed without Opus outcomes. No Luna/Opus rows are added to the paper’s Table 1 or Figure 1, and no cross-model generalizability claim is made.

Cost in tokens or dollars is unavailable because a complete usage ledger was not captured. Invalid-batch diagnostics exist in run state files, but rejected quartet contents were not consistently saved, so invalid-proposal rates cannot be reconstructed at quartet level. This controller extension remains incomplete and is not camera-ready.

## 10. Final conclusion

The completed study supports GPT-6 Sol’s source-guided search as a better way to choose four-head representations than the tested random and evolutionary procedures for these three held-site folds. It does not establish a reliable advantage over taxonomy-diverse search, and it does not isolate a clear causal contribution from semantic feature names or iterative feedback. B=10 is the most favorable full-agent operating point in the observed results; further validation should focus on additional independent institutions and a prospectively untouched evaluation set.

## 11. Reproducibility and artifacts

A local Git history was created for the preregistered generalizability plan only (commits 3c5b1f3 and aa6a40b); the full research project is not yet committed as a repository snapshot. Reproducibility provenance is recorded in the SHA-256 hashes in runs/protocol_freeze.json and the frozen manifest. The final manifest hash is listed in Section 4. The captured runtime is Python 3.14.6, NumPy 2.5.1, pandas 3.0.3, PyArrow 25.0.0, scikit-learn 1.9.0, SciPy 1.18.0, Matplotlib 3.11.1, PyYAML 6.0.3, and Codex CLI 0.155.1.

The all-method, all-budget per-site summary is in [appendix_per_site_results.md](analysis/output/appendix_per_site_results.md), with machine-readable values in [appendix_per_site_results.csv](analysis/output/appendix_per_site_results.csv). It covers every method/selector/budget currently in the released primary selection table; it does not imply completion of the newer supplemental ablation or controller scoring.

Key outputs:

- Main results table: analysis/output/main_results_table.csv
- Corrected run-level comparisons: analysis/output/claim_statistics.csv
- Corrected bootstrap analysis code: analysis/scripts/claim_statistics.py
- Paired macro trajectories: analysis/output/paired_macro_trials.csv
- Per-trajectory final model choices: analysis/output/agent_choice_secondary_per_trajectory.csv
- Final-choice secondary summary: analysis/output/agent_choice_secondary_summary.csv
- Final-choice recovery code: analysis/scripts/secondary_agent_choice.py
- Queried candidate result rows: analysis/output/queried_candidate_results.parquet
- Post-target trajectory review: analysis/output/manual_trajectory_review.json
- Fresh integrity audit: runs/final_audit.json
- Historical regression summary: runs/regression_baseline.json
- Target scoring release record: sealed/target_scoring_release.json
- Environment capture: runs/run_environment.json
- Frozen protocol, prompt, execution-code and source-input hashes: runs/protocol_freeze.json

Figures:

- [Figure 1: Search efficiency](analysis/output/figures/figure1_search_efficiency.png) ([PDF](analysis/output/figures/figure1_search_efficiency.pdf))
- [Figure 2: Semantics, feedback and search coverage](analysis/output/figures/figure2_semantics_feedback_ablation.png) ([PDF](analysis/output/figures/figure2_semantics_feedback_ablation.pdf))
- [Figure 3: Institution-specific results](analysis/output/figures/figure3_site_level.png) ([PDF](analysis/output/figures/figure3_site_level.pdf))
