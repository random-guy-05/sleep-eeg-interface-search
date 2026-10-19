# Sleep-EEG interface search: code, aggregate results and query histories

Arnav Mana and Ivan Habib — Los Gatos High School.

This release supports a study of source-guided four-head interface search over 539 physiological outputs of a frozen sleep-EEG model. It includes the original September 2026 operationally frozen experiment and an explicitly post hoc October 18 matched feedback extension. See [REVISION_REPORT.md](REVISION_REPORT.md) for the interpretation and [REVISION_README.md](REVISION_README.md) for detailed local workflows.

**Main qualification:** this benchmark is outcome-enriched. The provenance audit finds 87/128 S0001 subjects in a reconstructed released-checkpoint training set and 121/128 in its conservative development list. The exact checkpoint bytes used in historical cache extraction are not independently verified. The original full-cohort macro must not be described as independent three-site validation.

**Feedback result:** the paired ten-query extension scores 0.612189 with feedback and 0.618828 without, difference -0.006639, participant 95% interval [-0.029093, +0.014980]. Source feedback improved source scores without establishing a held-site benefit. The strong expert taxonomy baseline remains competitive.

## Included

- Experimental driver, source-only evaluator, baselines, 539-head catalog, frozen protocol and candidate manifest.
- Source-only proposal histories and source aggregate metrics; original and extension prompts and controller responses.
- Original per-selection aggregate results; new bootstrap, cohort-flow, provenance and sensitivity summaries.
- Pinned Python environment; scripts for analyses, aggregate tables and scientific figures; leakage-boundary tests.
- Private-input checksums, with no patient-level data.

No individual labels, demographics, feature vectors, predictions, raw EEG, subject-membership links, pretrained weights, credentials or historical Git objects are included. Published model-development exclusion lists are referenced upstream rather than linked here to outcomes.

## Install and run safe tests

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.lock.txt
.venv/bin/python3 -m pytest agentic_search/ieee_access_revision/tests -q
```

The recorded numerical environment is Python 3.14, NumPy 2.5.1, pandas 3.0.3, SciPy 1.18.0, scikit-learn 1.9.0 and PyArrow 25.0.0. Model sampling is not seed-controlled; the original model runtime reports Codex CLI 0.155.1 and the new extension 0.158.0, using `gpt-6-sol` at medium effort. A matching reported model name does not guarantee immutable weights.

## Regenerate aggregate journal tables and figures

The main manuscript and supplement sources are delivered separately to the authors; it is not required for code use. To generate its `.tex` tables and PDF figures from released aggregate results:

```sh
mkdir -p agentic_search/ieee_access_revision/paper
.venv/bin/python3 -m agentic_search.ieee_access_revision.scripts.build_paper
```

These operations use aggregate files only. They do not rerun a patient-level analysis or model calls.

For the independent category/supplement check, extract the separately supplied manuscript source archive into `agentic_search/ieee_access_revision/paper/` and place the main and supplementary PDFs there as `paper.pdf` and `supplement.pdf` (including the compilation log). Run `python3 agentic_search/ieee_access_revision/reproducibility/checks/verify_category_and_supplement.py`. It reconstructs category counts and robust selections from the 90 released original run histories, cross-checks the tables, and inspects the supplied supplement. B=10 gives 1,032/300 = 3.44 categories among proposals and 109/30 = 3.633333 among source-selected quartets. This checks existing artifacts without new searches or clinical fitting.

## Authorized-data reproduction

Obtain the [Human Sleep Project/Challenge data](https://moody-challenge.physionet.org/2026/) under its access requirements. Source EEG and derived representations remain subject to their applicable data terms. The [model inference repository](https://github.com/bdsp-core/philosophers-stone) describes authorized extraction. Verify its [training/development provenance](https://github.com/bdsp-core/philosophers-stone/tree/master/paper/provenance) before calling a validation set independent.

The archived source/target evaluator expects:

- `agentic_search/data/source_folds/<held_site>.parquet`: only the two visible source institutions, columns `subject_id`, `site`, `age`, `outcome` and catalog head names.
- `agentic_search/data/participants.parquet`: participant identifiers, site, age and 539 heads, without outcome.
- `agentic_search/sealed/target_labels.parquet`: held-site identifiers, subject ID, site, age and supplied binary outcome.
- Historical cohort-audit inputs under `experiments/philosophers_stone/`, and the common-interface manifest in `brainbodyfm/cache/`.

`PRIVATE_INPUT_HASHES.json` identifies the exact archived main inputs without disclosing their contents. The complete selected-cohort identity cannot be recovered from this public package alone. An authorized investigator needs the corresponding private manifest and historical extraction artifacts; raw EEG download alone cannot recover missing historical choices or exact checkpoint bytes. This is a stated reproducibility boundary.

Do not run the old `prepare_search_space` entry point until the exact frozen cohort/cache inputs are available: it deliberately asserts the original 338-person cohort, not an arbitrary replacement. A new independent dataset requires a separately documented experiment rather than relabeling the old results.

## Distinct inference and chronology

The original operational freeze fixed search procedures; inferential contrasts and the B=10 focal presentation were not preregistered hypotheses. Later controller studies and the present feedback extension have separate dated plans/manifests. The feedback experiment is conditional on thirty archived opening batches. Participant bootstrap resamples people within site/outcome while holding source fits and selected interfaces fixed; paired-run bootstrap resamples continuations on fixed people. Neither combines all clinical and model-search uncertainty. Metric sensitivities change scoring only, without reselection or retuning. The overlap sensitivity removes development matches from target evaluation only, preserving original source fits.

## Attribution and use

Preserve existing notices in copied research files and cite the underlying model/data publications described in the manuscript. Third-party dependencies and model/data assets have their own terms. No model weights or IEEE template assets are redistributed here. No new blanket license is asserted for previously unlicensed research code; obtain author permission for uses requiring a license. The authors report no external funding or conflicts. This is research code, not a clinically validated decision tool.
