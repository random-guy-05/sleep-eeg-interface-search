# IEEE Access revision

This folder preserves the original September searches and manuscript unchanged and adds an explicitly post hoc October 18 extension.

The revised main paper is `paper/paper.tex`, with complete technical details in `paper/supplement.tex`. The narrative leads with query efficiency, physiological structure and the reusable evaluation framework; the main paper explicitly retains the feedback null, development overlap and retained-sample reversal. Generated tables and figures read the saved numerical results directly. The final source archive includes the official May 2026 IEEE Access class and fonts. The ethics/consent and data-access declarations use the official Challenge and HSP documentation; related accepted non-archival NeurIPS FMTS work and the added journal analyses are disclosed.

## What was completed

- Reconstructed the 338-person cohort by site from historical extraction artifacts and matched it to the 1,103-person locally available labeled frame. Verified all cached labels and summary event/follow-up timing. Diagnosis-count requirements cannot be re-adjudicated from these summary fields.
- Ran 30 matched pairs of ten-query searches. Both conditions share identical archived opening proposals and use contemporaneous `gpt-6-sol`, medium effort continuations; one receives source scores and the other does not. All 60 continuation calls were valid without repair or fallback. A separate all-arms manifest froze before new target scoring.
- Reported participant and paired-run uncertainty separately, with 50,000 draws, and audited the sampling/age-pair behavior.
- Evaluated fixed original selections at one-year and three-year age tolerances and conventional AUROC.
- Matched the model developers' published reconstructed training/development manifests to the cohort, verified their checksums, and ran a target-only exclusion sensitivity.
- Reframed taxonomy and 500-candidate comparisons, corrected the largest-pair site to S0001, and distinguished Luna's own final choices from its fixed-robust selections.
- Built a journal manuscript with author names, affiliations, correspondence, biographies, funding/conflict declarations and AI assistance disclosure.

## Main findings

At ten candidate evaluations, the original benchmark macro is 0.6302 for agent search, 0.5644 for matched random search and 0.5503 for evolutionary search. Repeated selections span physiological categories and return to overlapping heads; the taxonomy comparator remains competitive. Proposed B=10 quartets average 3.44 categories, while robust-selected quartets average 3.63. These are different denominators. The I0002/I0006 post hoc descriptive macro is 0.6727 versus 0.5757 for random search, retaining all original source fits.


The matched feedback arm scores 0.612189 and no-feedback scores 0.618828. The difference is -0.006639, participant 95% interval [-0.029093, +0.014980], paired-run interval [-0.029306, +0.014971]. Feedback improves the selected minimum source AUROC by 0.017308 but does not establish a held-site advantage.

The provenance audit identifies 87/128 S0001 participants in the reconstructed released-checkpoint training set, 74 exact evaluated-record matches, and 121/128 in the conservative development exclusion list. Only seven S0001 participants and four eligible pairs remain after the broader exclusion. Original full-cohort results cannot be presented as independent three-institution validation. The exact historical extraction checkpoint bytes are unavailable, so these are documented development-population matches rather than a complete historical execution ledger.

## Reproduction

Use a new virtual environment with `reproducibility/requirements.lock.txt`. The original project's `.venv` was not changed. Run from the KaggleData project root:

```sh
python3 -m agentic_search.ieee_access_revision.scripts.cohort_audit
python3 -m agentic_search.ieee_access_revision.scripts.provenance_audit
python3 -m agentic_search.ieee_access_revision.scripts.statistical_analysis sensitivity
python3 -m agentic_search.ieee_access_revision.scripts.statistical_analysis feedback
python3 -m agentic_search.ieee_access_revision.scripts.build_paper
python3 -m pytest agentic_search/ieee_access_revision/tests -q
```

The first four operations require the authorized private historical data/feature inputs described in the scripts. `build_paper` uses released aggregate outputs only. Compile both `paper.tex` and `supplement.tex` with **pdfLaTeX**, twice each, from `paper/`. The official IEEE Access class uses pdfTeX spot-color primitives and does not compile unmodified with Tectonic/XeTeX. A temporary TinyTeX runtime was used for local verification; it is not part of the release or a change to the user's global TeX installation.

To rerun controller generation, use a fresh output destination or an independent copy of this package. The archived extension refuses changed hashed inputs and resumes completed arms rather than silently rerolling them:

```sh
python3 -m agentic_search.ieee_access_revision.scripts.feedback_ablation freeze
python3 -m agentic_search.ieee_access_revision.scripts.feedback_ablation propose --workers 4
python3 -m agentic_search.ieee_access_revision.scripts.feedback_ablation source
python3 -m agentic_search.ieee_access_revision.scripts.feedback_ablation target
```

Do not overwrite an existing result directory to obtain a different result. A new stochastic experiment must have a new protocol/output directory. Model sampling and immutable model builds were not controlled; recorded seeds do not imply identical repeat output.

## Release boundary

Patient-level outcomes, demographics, feature matrices, predictions and membership matches remain private. Public query logs contain quartet proposals, rationales and source aggregate scores only. The upstream overlap lists are referenced by URL and checksums; we do not redistribute them with linked outcomes. The release includes aggregate results, code, permitted head metadata and source-only trajectories. It cannot recreate missing historical extraction logs, source-population exclusions or historical checkpoint bytes.

## Remaining author inputs

- Confirm that both authors reviewed the AI-assisted manuscript and accept responsibility. This completed action is not inferred from a request to edit the declaration.
- Confirm compliance with the applicable access/data-use terms. The stated BIDMC approval and consent waiver belong to the original source dataset; no separate author-specific approval or exemption is asserted.
- Confirm the exact accepted FMTS title and any other related accepted workshop versions for complete bibliographic disclosure; confirm any submission-system concurrent-review declaration separately.
- Review all revised text and findings, and supply ORCIDs for submission as applicable.
- The unresolved model-development independence issue requires adequate untouched validation to support a stronger external-validation claim; manuscript formatting cannot resolve it.

No journal submission has been made.

## Published code release

The screened code and aggregate-results repository was published and read back successfully on October 18, 2026: https://github.com/random-guy-05/sleep-eeg-interface-search . The journal manuscript was not posted there and no journal submission was made.
