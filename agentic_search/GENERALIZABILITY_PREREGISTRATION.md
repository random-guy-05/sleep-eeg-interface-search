# Controller generalizability study: preregistered execution plan

**Status:** committed protocol; execution is authorized tonight. Results will be added as supplemental artifacts after the source-only search grid and manifest freeze.

**Purpose.** Test whether the source-only quartet-search result generalizes beyond GPT-6 Sol, while holding the search task, information, and controller protocol constant. The primary comparisons are defined before any new held-site scoring. Existing target labels have already been examined in this project; therefore this is a preregistered analysis plan for a post hoc/exploratory extension, not a fresh confirmatory use of an untouched test set.

## 1. Models and fixed protocol

| Controller | Requested model identifier | Reasoning effort | Prompt |
|---|---|---|---|
| Reference | OpenAI GPT-6 Sol | medium | Existing frozen `configs/prompts/full_agent.txt` |
| Generalization A | OpenAI GPT-6 Luna | medium | Byte-identical Sol prompt |
| Generalization B | Anthropic Claude Opus 5.5 | medium | Byte-identical Sol prompt |

Use the literal requested model/version; verify the provider CLI reports the requested model before accepting the pilot. If a provider cannot serve that model at medium effort, stop and document the incompatibility; do not silently substitute a version or effort. Record requested and provider-reported model/version, CLI/API version, effort, temperature behavior, token usage, elapsed time, retries, and provider errors for every call. Request the frozen protocol temperature (0.2) wherever the provider exposes that control; otherwise record that sampling temperature was uncontrolled/provider-default, as with the reference Codex CLI.

Reuse Sol's frozen controller instructions, full semantic catalog (`data/heads.csv`, including the same IDs, names, and taxonomy labels), 5-quartet proposal batch, candidate validator, one repair attempt, historical exclusion list (`data/historical_quartets.csv`), source-evaluation tools, source metric feedback, stopping conditions, and final-choice prompt/schema/step exactly. Every trajectory gets the same one final choice from its complete source-only history after its last batch. The selected quartet for the primary cross-controller endpoint is that model's own valid `final_choice`; the pre-existing source-robust (`max_min_source_auc`) selector is also evaluated as a prespecified secondary endpoint so results remain directly comparable to the existing primary table. No held-site label, score, or outcome-derived statistic is exposed during search.

The search space remains quartets of four distinct catalog IDs. Preserve Sol's checks for unknown IDs, duplicate heads, duplicate quartets, historical target-inspected quartets, and repeated proposals within a trajectory. Apply the same single repair opportunity and fallback/exclusion rules as Sol; record every invalid candidate, invalid batch, repair, automatic fill, and failed call, including invalid proposals that are ultimately repaired. Do not edit the frozen prompt or primary protocol to accommodate another model.

## 2. Runs and pilot

First run a source-only, two-trajectory operational pilot for each generalization model: one trajectory at I0002/B10/seed 9900 and one at I0006/B50/seed 9901. The pilot exercises the small and largest budget paths, source-only tool delivery, the common validator/repair behavior, and the final-choice call. It is not used to tune prompts, choose methods, or make performance claims. Acceptance is operational only: the requested model/effort is verified, each trajectory either completes under the frozen retry rules or fails with a retained diagnostic, every accepted quartet is valid and source-evaluated, the final choice belongs to its evaluated history, and no target information is accessed. Any implementation repair must be documented and frozen before the main runs; rerun the pilot under that frozen implementation.

After pilot acceptance, run **90 trajectories per generalization model**: the full cross of 3 held sites (I0002, I0006, S0001) × 10 independent run seeds (1000–1009) × budgets B=10, 25, and 50. This is 180 main adaptive trajectories total, in addition to the four pilot trajectories. Preserve the same five-candidate batch size and exact budget accounting. Use matched site/budget/seed cells across Sol, Luna, and Opus for paired comparisons. Never replace an inconvenient result; resume only an incomplete trajectory under its unchanged state and frozen hashes, otherwise retain and report failure.

For a separate zero-shot baseline, request one quartet directly from each generalization model using the same full catalog and the frozen Sol prompt context, with no source metric history or tool feedback. Run 10 independent picks per site per model (30 picks/model), using seeds 1000–1009 and the same frozen validator, repair allowance, historical exclusions, and source-selected C rule. This is an explicitly separate one-shot condition, not counted in the 90 adaptive trajectories. Record its source metrics and final quartet; do not use it to tune adaptive prompts.

## 3. Freeze, scoring, endpoints, and statistical tests

Run all pilot, adaptive, zero-shot, and baseline source-only generation/evaluation before accessing held-site scores for these new candidates. Freeze a separate, append-only manifest containing every proposed and accepted quartet, canonical key, held site, model, condition, budget, seed, selected C from source-only tuning, proposal/repair status, and code/prompt/protocol hashes. Canonicalize and deduplicate only for target model fitting; retain every run-level occurrence for analysis. Hash and timestamp the manifest, verify the hash, then score the held sites with the existing target evaluator. Keep new outputs separate from the original frozen primary manifest/results. The already-viewed status of target labels must remain explicit in the manuscript and artifact metadata.

**Primary outcome:** age-conditioned AUROC using the existing ±2-year eligibility window, tie handling, target participants, source-fitted preprocessing/model, and source-selected regularization parameter. For each selected quartet, compute the target score once and retain participant-level prediction scores. For each site × budget × method, average over the fixed selected quartets/runs; the overall comparison is the equally weighted macro-average over the three held sites. Report each site separately. State that participant-level resampling conditions on the already selected quartets and does not measure variability from rerunning the search.

**Primary selector:** each controller's own final choice. **Secondary selectors:** Sol-compatible robust source selection; descriptive zero-shot results; taxonomy-diverse and random-source-robust reference rows from the existing frozen bank at the corresponding budget. Report pilot results only as operational diagnostics. Also report category coverage for each selected quartet (number of distinct pre-existing taxonomy categories, plus the category composition), exact-quartet overlap with Sol, shared-head count, and quartet Jaccard overlap. For overlap, match adaptive runs by held site/budget/seed; compare zero-shot picks with Sol's selected quartets at the same held site and report exact/any-head/Jaccard summaries separately.

For each budget, test five prespecified two-sided contrasts on the primary final-choice endpoint: Luna vs random-source-robust, Luna vs taxonomy-diverse, Opus vs random-source-robust, Opus vs taxonomy-diverse, and Luna vs Opus. This creates one family of 15 tests (5 contrasts × 3 budgets); use Holm step-down correction at familywise α=0.05. Estimate differences with a paired participant bootstrap: within each held site, resample participants with replacement stratified by outcome, preserving observed positive/negative counts; use the same sampled participants for every method and controller; keep all selected quartets and their scores fixed; recalculate age-conditioned AUROC for each quartet, average within site over runs, then equally average the three site means. Use 50,000 valid replicates and report percentile 95% CIs, null-centered two-sided bootstrap p-values, Holm-adjusted p-values, and valid/invalid replicate counts. If a draw has no eligible age-matched positive-negative pair at a site, redraw that site's sample and report the redraw fraction. The random and taxonomy-diverse comparator selections are fixed to the existing source-only outcomes. Report the number of eligible positive-negative age-matched participant pairs per site. Treat robust-selector and zero-shot comparisons as secondary/descriptive (or explicitly identify and Holm-correct any inferential family added before scoring); do not promote them after seeing outcomes.

## 4. Hypotheses

The study is designed to assess controller generalizability, not presume a winner.

1. **H1 (Luna vs search baselines):** at each budget, Luna's final-choice macro age-conditioned AUROC differs from random-source-robust and taxonomy-diverse search; a positive difference favors Luna.
2. **H2 (Opus vs search baselines):** at each budget, Opus 5.5's final-choice macro age-conditioned AUROC differs from random-source-robust and taxonomy-diverse search; a positive difference favors Opus.
3. **H3 (controller dependence):** at each budget, Luna and Opus 5.5 final-choice performance differ. A CI spanning zero is inconclusive; it is not evidence of equivalence.
4. **H4 (descriptive diversity):** category coverage and overlap with Sol may differ by controller and budget; these are descriptive outcomes, not success criteria.

No performance-based stopping, prompt editing, seed replacement, budget changes, or model/effort substitution is allowed after pilot acceptance. A failed run remains a failure in the run accounting; any replacement must be separately labeled and sensitivity-reported.

## 5. Cost, reporting, and camera-ready changes

Report calls, input/output tokens (and cached tokens when supplied), provider-reported spend when available, and estimated spend computed from the provider's published rates captured at run time. If usage or billing data are unavailable, label cost unavailable rather than zero. Report invalid quartet slots / attempted quartet slots and invalid batches / submitted batches, along with repairs, automatic fills, API failures, and completion rates, by model and condition.

Add model rows (and all three budgets) to Table 1 with site-macro target AUROC and participant-bootstrap intervals; distinguish model-final-choice and robust-selector rows. Add Luna and Opus lines with participant-bootstrap intervals to Figure 1 at B10/B25/B50, retaining the existing random and taxonomy-diverse references; show zero-shot as a clearly labeled separate reference if the figure remains legible. Add the exact search protocol, model identifiers/effort, pilot/main-run counts, manifest freeze/scoring order, bootstrap and Holm procedures, and cost/invalid accounting to Methods. Add a limitations note stating that (i) the target labels were examined before this extension, (ii) only three held sites and few positives at I0002 constrain uncertainty, (iii) participant bootstrap holds search choices fixed and omits search-run uncertainty, (iv) provider temperature/usage reporting may differ, and (v) model-provider comparisons cannot isolate architecture from provider implementation. Include full per-site results, per-quartet taxonomy coverage, overlap, invalid proposals, and cost in the appendix.

## 6. Frozen analysis constants and audit trail

- Catalog, prompt, protocol, validators, selection code, and target scoring: the byte-identical, hash-verified Sol artifacts named above; record SHA-256 values in the run manifest.
- Sites: `I0002`, `I0006`, `S0001`.
- Adaptive grid: batch size 5; B=10/25/50; seeds 1000–1009; 10 repeats per site and budget.
- Pilot: I0002/B10/9900 and I0006/B50/9901 per model; separate from the main 90.
- Zero-shot: 10 picks/site/model, seeds 1000–1009.
- Primary test family: the 15 contrasts listed in Section 3; two-sided, Holm-adjusted, α=0.05.
- Bootstrap: 50,000 paired, site-stratified, outcome-stratified participant resamples; RNG seed `20260927`; 95% percentile intervals.
- Do not overwrite original results. Save complete raw call/proposal logs, model metadata, source metrics, manifest/hash, held-site predictions, analysis tables, figure source/data, and report revisions as new supplemental artifacts.
