# Language-Model Feature Selection for a Sleep-EEG Foundation Model

Arnav Mana and Ivan Habib, Los Gatos High School.

Code, aggregate results and search histories for **“Language-Model Feature Selection for a Sleep-EEG Foundation Model: A Multihospital Comparison.”** The study compares four-feature predictors selected from 539 physiological outputs in 338 participants at three hospitals.

The current version includes the October 19 additions: one-shot language-model selection, L1, univariate and greedy selectors, joint run/participant bootstrap intervals, and development-exclusion results for every method. The original September searches and October 18 paired feedback experiment are retained.

**Version:** [`ieee-access-v1`](https://github.com/random-guy-05/sleep-eeg-interface-search/releases/tag/ieee-access-v1). Matching [main manuscript](https://github.com/random-guy-05/sleep-eeg-interface-search/releases/download/ieee-access-v1/IEEE_Access_Main.pdf), [supplement](https://github.com/random-guy-05/sleep-eeg-interface-search/releases/download/ieee-access-v1/IEEE_Access_Supplement.pdf), and code/results archive are attached to that release.

## STAGES probe-search extension

The separate STAGES center-held-out study and its aggregate results are documented in [`brainbodyfm/stages/`](brainbodyfm/stages/). This extension evaluates concurrent cognitive outcomes and does not reproduce the original future-diagnosis outcome. The restricted inputs and checkpoint are not included; the package explains the boundaries of a full rerun.

## Results

Every method selects four features and uses the same final L2 logistic regression probe. Features are selected and models fitted at two hospitals, then tested at the third. Scores are equal-hospital means of age-conditioned AUROC.

| Method | Macro AUROC | Feature-selection work per hospital |
|---|---:|---|
| One-shot language model | 0.619 | No performance evaluations for feature choice |
| L1 selection | 0.605 | 51–59 pooled-source path fits |
| Univariate top-four | 0.575 | 539 singleton evaluations |
| Greedy forward | 0.604 | 2,150 subset evaluations |
| Language-model search | 0.630 | 10 subset evaluations |
| Random search | 0.564 | 10 subset evaluations |
| Evolutionary search | 0.550 | 10 subset evaluations |
| Taxonomy-diverse search | 0.607 | 10 subset evaluations |

Final-probe tuning and fitting are additional costs. L1 selects features before refitting the common L2 model. Ten independent one-shot calls each supplied a quartet that was tested at all three hospitals; their mean is reported, without selecting the best call.

The main uncertainty analysis jointly resamples search runs and test participants, using the same participant draws across methods and preserving shared run indices. It includes empirical search-run and test-participant variation, conditional on the observed source cohort and fitted models. It does not resample source cohorts, refit models, or estimate future model-version changes.

| AUROC difference | Estimate | Joint 95% interval |
|---|---:|---|
| One-shot minus L1 | 0.0141 | [-0.1238, 0.1360] |
| One-shot minus greedy | 0.0153 | [-0.1224, 0.1366] |
| One-shot minus univariate | 0.0439 | [-0.1134, 0.1887] |
| Ten-evaluation search minus random | 0.0658 | [-0.0034, 0.1374] |
| Ten-evaluation search minus one-shot | 0.0111 | [-0.0885, 0.1283] |
| Feedback minus no feedback | -0.0066 | [-0.0494, 0.0334] |

These are exploratory pointwise intervals from 50,000 joint draws. Earlier participant-only and run-only intervals remain labeled as separate analyses. The feedback comparison shares the first five candidates within each pair; it tests one continuation step.

Method rankings vary by test hospital. The selected cohort is outcome-enriched. At S0001, 87/128 participants match reconstructed training records and 121/128 match the broader model-development list; the historical extraction checkpoint could not be verified. Excluding development matches leaves seven S0001 participants and four eligible pairs. All original source fits are retained in that sensitivity analysis. These results do not establish independent three-hospital validation or equivalence between selectors.

## Where to find the added analyses

All paths below are under [`agentic_search/ieee_access_revision/`](agentic_search/ieee_access_revision/).

| Contents | Path |
|---|---|
| L1, univariate, greedy and one-shot selection | [`scripts/feature_selection_extension.py`](agentic_search/ieee_access_revision/scripts/feature_selection_extension.py) |
| Joint run/participant bootstrap | [`scripts/joint_uncertainty.py`](agentic_search/ieee_access_revision/scripts/joint_uncertainty.py) |
| Development exclusions and quartet overlap | [`scripts/extension_exclusion_and_overlap.py`](agentic_search/ieee_access_revision/scripts/extension_exclusion_and_overlap.py) |
| Tables and Figure 2 | [`scripts/render_feature_selection_extension.py`](agentic_search/ieee_access_revision/scripts/render_feature_selection_extension.py) |
| One-shot prompts and ten accepted responses | [`results/feature_selection_extension/one_shot/`](agentic_search/ieee_access_revision/results/feature_selection_extension/one_shot/) |
| Source choices, costs and target summaries | [`results/feature_selection_extension/`](agentic_search/ieee_access_revision/results/feature_selection_extension/) |
| Joint contrasts | [`joint_contrasts.csv`](agentic_search/ieee_access_revision/results/feature_selection_extension/joint_contrasts.csv) |
| Saved aggregate joint draws | [`macro_bootstrap_draws.npz`](agentic_search/ieee_access_revision/results/feature_selection_extension/macro_bootstrap_draws.npz) |
| Analysis report | [`BASELINE_AND_UNCERTAINTY_REPORT.md`](agentic_search/ieee_access_revision/BASELINE_AND_UNCERTAINTY_REPORT.md) |

Ten one-shot calls produced nine distinct quartets. Across those calls, 38/40 feature positions occur in the 29-feature union of the original 30 search selections, versus 2.15 expected under uniform sampling from 539 features. This describes shared feature choices; it does not establish a mechanism for prediction performance.

## Check the released results

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.lock.txt
.venv/bin/python3 -m pytest agentic_search/ieee_access_revision/tests -q
.venv/bin/python3 agentic_search/ieee_access_revision/reproducibility/checks/verify_current_results.py
```

The recorded numerical environment is Python 3.14, NumPy 2.5.1, pandas 3.0.3, SciPy 1.18.0, scikit-learn 1.9.0 and PyArrow 25.0.0. The original model client was Codex CLI 0.155.1; the added comparisons used 0.158.0, requesting `gpt-6-sol` at medium effort. Sampling was not seed-controlled, and the provider did not expose immutable model weights. One-shot calls were made after the original searches.

To regenerate the added tables and Figure 2 using only released aggregates:

```sh
mkdir -p agentic_search/ieee_access_revision/paper
.venv/bin/python3 -m agentic_search.ieee_access_revision.scripts.render_feature_selection_extension
```

The older `build_paper.py` creates an earlier manuscript draft. Use the current renderer above for the added comparisons. The matching compiled PDFs are release assets.

## Participant data and full reruns

Participant-level outcomes, demographics, features, predictions, raw EEG, membership links, pretrained weights and credentials are excluded. Full fitting and resampling require authorized [Human Sleep Project/Challenge access](https://moody-challenge.physionet.org/2026/) and the corresponding private inputs. `PRIVATE_INPUT_HASHES.json` identifies the archived inputs without disclosing them. See [REVISION_README.md](REVISION_README.md) for the expected files and analysis commands.

The [pretrained-model repository](https://github.com/bdsp-core/philosophers-stone) describes extraction and its [training/development provenance](https://github.com/bdsp-core/philosophers-stone/tree/master/paper/provenance). Missing historical selection/extraction records and the unverified original checkpoint prevent complete reconstruction from raw downloads alone. Recomputing the saved-model intervals requires the corresponding private predictions; the public aggregate draws support checking the reported joint intervals.

The original searches preceded the added selectors and exclusion analyses. The focal contrasts were not preregistered. Do not overwrite archived proposals to replace an observed outcome; a new stochastic experiment needs a separate output directory.

## Attribution

The authors report no external funding or conflicts. Preserve notices in copied files and cite the model and dataset publications described in the manuscript. Third-party model, data and software terms remain separate; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No model weights or IEEE template assets are included in this code repository, and no blanket license is asserted for previously unlicensed research code.
