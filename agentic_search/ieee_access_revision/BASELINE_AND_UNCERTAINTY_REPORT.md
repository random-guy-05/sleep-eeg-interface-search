# Added feature-selection comparisons and joint uncertainty

All methods use four features, the original source folds and the same final L2 logistic probe. These analyses were added after the original results.

| Selector | Macro AUROC | Selection work per hospital |
|---|---:|---|
| Language-model search | 0.630 | 10 subsets |
| L1 selection | 0.605 | 51–59 pooled-source path fits; one five-feature support truncated to four |
| Source-univariate top-four | 0.575 | 539 singleton evaluations |
| Greedy forward | 0.604 | 2,150 subset evaluations |
| One-shot language model | 0.619 | 0 performance evaluations for feature choice |

Final probe tuning/fitting is additional. One-shot and the L1/univariate selectors use 34 directional fits plus one pooled-source fit after feature choice. Greedy already tunes its final subset during selection and adds one pooled-source fit. L1 selects inputs and is followed by the common L2 classifier.

The original search minus random difference is 0.0658 with joint run/participant 95% interval [-0.0034, 0.1374]. The difference versus one-shot selection is 0.0111 [-0.0885, 0.1283]. The joint feedback-minus-no-feedback interval is [-0.0494, 0.0334].

The 50,000-draw bootstrap shares resampled participants across methods and runs, preserves paired opening-batch indices, and shares one-shot run indices and random-bank ordering indices across hospitals. It does not refit models, resample source cohorts, or estimate future model-version drift. The later feedback score of 0.612 shares opening batches with the original 0.630 score and is not an independent full rerun.

The main manuscript, figure and cover letter now use the joint primary interval. Earlier participant-only and run-only tables remain explicitly labeled in the supplement. All accepted one-shot requests and responses are saved; no call received source or target scores. A source-output path error overwrote one proposal log before target scoring; the log-loss replacement is documented in interruption_note.json.

No participant-level data or predictions are included in the deliverable/code archive. No remote repository update or journal submission was made.

## Hospital comparison and revised framing

The manuscript is now titled “Language-Model Feature Selection for a Sleep-EEG Foundation Model: A Multihospital Comparison.” It leads with one-shot selection: feature descriptions yielded macro AUROC 0.619 without performance evaluations to choose features. It does not claim equivalence or a demonstrated feedback advantage.

All eight methods now appear in the hospital/development-exclusion comparison. On the retained S0001 test set (seven participants, four eligible pairs), L1, univariate and greedy selection each score 0.750, and one-shot selection scores 0.575. Features and source fits are unchanged. The three-hospital means become 0.651, 0.618, 0.665 and 0.615, respectively. The joint exclusion bootstrap retained 50,000 draws per hospital; 3,675 S0001 draws lacked eligible pairs and were rejected. This is separate from the earlier participant-only exclusion bootstrap and its 3,715 rejections.

Ten one-shot calls produced nine distinct quartets. No quartet exactly matched an original selected quartet; 38 of 40 feature positions belonged to the union of features selected in the original 30 runs. This describes feature recurrence, not a demonstrated explanation for performance. The nine quartets are listed in Supplementary Table S8.

Figure 2 shows seven comparators and omits the B=100 random-only point, keeping plotted budgets aligned with the budget table. Feedback intervals now use four decimal places to distinguish participant-only [-0.0291, 0.0150] from run-only [-0.0293, 0.0150] rounding. Six relevant references were added; the bibliography has 29 entries.

## One-shot comparisons: minor revision

The headline comparisons now have their own joint run/participant intervals, using the same 50,000 draws as the search contrasts. All saved draws match the preceding version exactly; every previous contrast is unchanged.

| One-shot minus | AUROC difference | Joint 95% interval |
|---|---:|---|
| L1 selection | 0.0141 | [-0.1238, 0.1360] |
| Greedy forward | 0.0153 | [-0.1224, 0.1366] |
| Univariate top-four | 0.0439 | [-0.1134, 0.1887] |

Table S9 reports both joint and participant-only intervals. The conclusion describes comparable observed AUROCs with intervals spanning zero; it does not claim equivalence. Results and Discussion now lead with one-shot/classical selection. Retained macro scores are confined to Supplementary Table S3.

The original selected-feature union has 29 of 539 features. Forty uniform random selections would yield an expected 2.15 positions in this union, compared with 38 observed positions. This is a descriptive reference, not a category-adjusted significance test. H0191 appears in eight of nine distinct one-shot quartets (nine of ten calls) and four of 30 original selected quartets. S2 explicitly reports 3,675 rejected S0001 joint exclusion draws, separate from the earlier 3,715 participant-only rejections. Existing biographies are retained; no author photos were supplied.

## Final figure/discussion polish

Discussion IV.C now reports the 38/40 one-shot feature positions in the original 29-feature search union, with the 2.15 uniform-random reference and a pointer to Supplementary Table S8. Figure 2b now includes three one-shot-minus-classical intervals (diamonds) above the seven ten-evaluation-search intervals (circles). Every interval is read from the existing joint-contrast table; no estimates or analyses changed.

## Author-photo update

Both authors supplied headshots, now included on the final page in IEEE biography format. The biography text is unchanged. Figure 2 was reduced from 4.4 to 4.1 inches in height and final-column spacing was adjusted to keep the manuscript at eight pages. All numerical estimates and manuscript prose are unchanged. Ivan's supplied .jpg file contained WebP data; its PNG conversion was verified to preserve every decoded RGB pixel.
