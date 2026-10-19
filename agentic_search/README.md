# Prospective target-blind quartet search

This package runs the new four-head search separately from the retrospective paper analysis. It keeps the original age-conditioned AUROC, preprocessing, bidirectional source evaluation, and 17-value L2 grid.

## Agent and evaluation boundary

The sole model is **GPT-6 Sol at medium reasoning**. Codex CLI reports the configured MCP server as enabled but does not expose its tools to model turns. The controller therefore sends the model only the fixed head catalog and the trajectory's prior source-only metrics, validates each five-quartet proposal, and invokes the local source evaluator. The model has no shell, browser, MCP, target evaluator, or target-label access. On the last round, it receives the final source results in a separate turn and selects one evaluated quartet.

The model's temperature is requested as 0.2, but Codex CLI does not expose a temperature setting. The run metadata records this as uncontrolled provider-default sampling; the selected reasoning effort is explicitly set to `medium`.

`data/source_folds/<held_site>.parquet` contains only the two visible source institutions. `data/participants.parquet` contains no outcome column. `sealed/target_labels.parquet` remains unopened during all candidate generation and source evaluation. Historical target-inspected quartets, including Stable4, are excluded from new searches.

**Historical regression exception:** the separate pre-search regression check does read the sealed labels for previously target-inspected historical quartets through the evaluator's explicit historical-regression path. Those legacy scores were not supplied to GPT-6 Sol and those quartets are excluded from the new manifest. Therefore the new search is blind to target labels, but it would be inaccurate to say target labels were never read anywhere in the project. For new candidates, target scoring remains locked until the frozen manifest is complete.

## Frozen design

The experiment uses a prospective random bank of 500 new quartets, 100 randomized orderings, random-prefix selectors, evolutionary and taxonomy-diverse baselines, one GPT-6 Sol adaptive search, a no-feedback semantic ablation, and an opaque-ID iterative ablation. Adaptive budgets are 10, 25, and 50 candidates in batches of five. The single-agent grid uses 10 replicates per held site and budget; the B=25 ablations use 10 replicates per held site.

The protocol, all three prompts, executable pipeline files, and source-only input data are SHA-256 hashed before the main search. The frozen manifest records every unique candidate/fold pair produced by the prospective bank, classical baselines, agent searches, and ablations, with the source-selected C. The target scorer verifies those hashes and refuses to start without the complete manifest.

## Run order

From the project root, use the project virtual environment:

1. Prepare the search space and regression-check the historical evaluator.
2. Generate and source-score the prospective random bank; run source-only classical baselines and random-prefix selectors.
3. Run the five B=25 GPT-6 Sol pilot trajectories. Pilot results are execution checks and are excluded from the target manifest.
4. Freeze prompts and hashes, then run the main agent grid and both B=25 ablations.
5. Run the pre-target audit, build and hash the frozen manifest, and run the leakage audit.
6. Run the target scorer once. It scores only manifest candidates and records that target labels were not used for model or C selection.
7. Aggregate results, render the figures, and run the final audit.

Every candidate evaluation consumes the trajectory's budget, even when its fit is cached. Runs are resumable from their source-only state files. Do not inspect newly scored target results until the frozen manifest and target run are complete.
