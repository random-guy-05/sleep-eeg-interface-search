"""Post hoc four-feature baselines. Source selection precedes separate target scoring.

Run as a module from KaggleData. Phases: plan, propose, source, target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from agentic_search.evaluator.source_evaluator import C_GRID, _source_frame
from brainbodyfm.preprocess import SourcePreprocessor

PACKAGE = Path(__file__).resolve().parents[2]
OUT = PACKAGE / "ieee_access_revision/results/feature_selection_extension"
SITES = ("I0002", "I0006", "S0001")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def plan():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        return
    save(path, {
        "created_utc": now(), "designation": "post_hoc_after_original_results",
        "source_only_selection": True, "features": 4,
        "univariate": "Rank single-feature L2 probes by minimum directional source age-AUROC; C selected by directional mean, same 17-value grid; joint top four then retuned/refit with L2.",
        "greedy": "At each of four steps evaluate every unused-feature extension; select highest minimum directional source AUROC, tie by sorted IDs. Tune C by directional mean for every subset. 539+538+537+536=2150 subset evaluations per fold, 34 directional fits each; no restarts.",
        "l1": "Pooled-source standardized L1 logistic path, 161 ascending C values logspace(-4,4). Stop at first model with >=4 absolute coefficients above 1e-8. If support jumps past four, keep four largest absolute coefficients (ID ties). Selected quartet then retuned/refit with the common L2 probe. Report actual support size, C and path fit count. No test tuning.",
        "zero_evaluation_llm": "Ten independent tool-free gpt-6-sol medium requests, feature names/categories and task description only. Each request selects one quartet, applied to all three folds. Zero evaluated candidates used to choose the quartet; subsequent C tuning costs 34 source fits per fold plus final refit. Same run index shared across folds. One format-only repair, no random replacement.",
        "historical_quartet_policy": "Classical deterministic selectors use all 539 eligible features; no target scores or historical-bank membership in their selection. LLM receives no historical scores or exclusion list. Any coincidental historical overlap is recorded after choices, not used to alter choices.",
        "bootstrap": "50000 crossed run-and-participant draws. Resample runs within hospital for original independent searches, participants by hospital/outcome once per draw shared by every run/method; shared one-shot run index across hospitals; paired continuation/opening indices shared for feedback/original repeat comparisons. No refitting/reselection on resampled source data; fixed hospitals, original source data. Percentile exploratory intervals.",
        "hashes": {str(p.relative_to(PACKAGE)): sha(p) for p in [
            Path(__file__), PACKAGE / "data/heads.csv",
            *[PACKAGE / f"data/source_folds/{s}.parquet" for s in SITES]]},
    })


class SubsetEvaluator:
    """Columnwise preprocessing once per source direction; same L2 objective."""

    def __init__(self, site):
        self.site = site
        self.frame = _source_frame(site)
        self.catalog = pd.read_csv(PACKAGE / "data/heads.csv", dtype=str)
        self.ids = self.catalog.head_id.tolist()
        self.names = self.catalog.head_name.tolist()
        self.index = {key: i for i, key in enumerate(self.ids)}
        self.x = self.frame[self.names].to_numpy(float)
        self.y = self.frame.outcome.to_numpy(int)
        self.directions = []
        for source in sorted(self.frame.site.unique()):
            train = self.frame.site.eq(source).to_numpy()
            val = ~train
            prep = SourcePreprocessor()
            xt = prep.fit_transform(self.x[train])
            xv = prep.transform(self.x[val])
            yv = self.y[val]
            ages = self.frame.age.to_numpy(float)[val]
            pos, neg = np.flatnonzero(yv == 1), np.flatnonzero(yv == 0)
            pp, nn = np.where(np.abs(ages[pos, None] - ages[None, neg]) <= 2.)
            self.directions.append((xt, self.y[train], xv, pos[pp], neg[nn]))
        self.cache = {}
        self.fits = 0
        self.convergence_warnings = 0

    def evaluate(self, ids):
        ids = tuple(sorted(ids))
        if ids in self.cache:
            return self.cache[ids]
        ix = [self.index[key] for key in ids]
        aucs = []
        for c in C_GRID:
            scores = []
            for xt, yt, xv, pp, nn in self.directions:
                model = LogisticRegression(C=c, solver="lbfgs", penalty="l2",
                    max_iter=10000, tol=1e-5, random_state=2026090401)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(xt[:, ix], yt)
                self.convergence_warnings += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
                self.fits += 1
                pred = model.decision_function(xv[:, ix])
                delta = pred[pp] - pred[nn]
                scores.append(float(np.mean((delta > 0) + .5 * (delta == 0))))
            aucs.append(scores)
        aucs = np.asarray(aucs)
        mean = aucs.mean(axis=1)
        best = np.flatnonzero(mean >= mean.max() - 1e-4)[0]
        row = dict(held_site=self.site, quartet_key="|".join(ids),
                   selected_C=float(C_GRID[best]),
                   auc_source1_to_source2=float(aucs[best, 0]),
                   auc_source2_to_source1=float(aucs[best, 1]),
                   min_source_auc=float(aucs[best].min()), mean_source_auc=float(mean[best]))
        self.cache[ids] = row
        return row


def propose_one(run):
    from agentic_search.agents.model_cli import call_static
    path = OUT / "one_shot" / f"run_{run}.json"
    if path.exists():
        return json.loads(path.read_text())
    cat = pd.read_csv(PACKAGE / "data/heads.csv", dtype=str)
    system = ("Select physiological features for predicting future cognitive impairment diagnoses "
              "from a pretrained sleep EEG model. Choose exactly four distinct features based only "
              "on their names and categories. You have no participant data, performance scores or "
              "evaluation tools. Prefer predictive and complementary measurements. Return only "
              "the requested JSON object. Do not browse or use any tools.")
    user = ("Select one four-feature set for use at a hospital absent from model fitting. "
            "No candidate evaluations will be supplied before your selection. Catalog (ID, name, category):\n" +
            "\n".join(f"{x.head_id}\t{x.head_name}\t{x.taxonomy_category}" for x in cat.itertuples()))
    schema = {"type": "object", "additionalProperties": False, "required": ["head_ids"],
        "properties": {"head_ids": {"type": "array", "minItems": 4, "maxItems": 4,
                                   "items": {"type": "string"}}}}
    logdir = OUT / "one_shot"
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "system.txt").write_text(system)
    (logdir / "user.txt").write_text(user)
    calls = []
    for attempt in range(2):
        response = call_static("codex", "gpt-6-sol", system, user, schema,
            seed=2026101900 + run, reasoning_effort="medium", timeout_seconds=600)
        calls.append(response)
        ids = response["answer"].get("head_ids", [])
        if len(ids) == 4 and len(set(ids)) == 4 and set(ids).issubset(set(cat.head_id)):
            break
        user += "\nPrevious response had invalid identifiers or count. Return four distinct catalog IDs."
    else:
        save(logdir / f"invalid_run_{run}.json", calls)
        raise ValueError("Invalid one-shot response; no fallback substituted")
    result = dict(run=run, head_ids=sorted(ids), quartet_key="|".join(sorted(ids)),
                  completed_utc=now(), candidate_evaluations_for_selection=0, calls=calls)
    save(path, result)
    print("One-shot LLM completed", run, flush=True)
    return result


def propose():
    assert (OUT / "protocol.json").exists()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(propose_one, range(10)))


def source_one(site):
    destination = OUT / f"source_{site}.json"
    if destination.exists():
        return json.loads(destination.read_text())
    start = time.monotonic()
    with threadpool_limits(limits=1):
        ev = SubsetEvaluator(site)
        # Resume completed, source-only subset evaluations after an interrupted run.
        for cached_path in [OUT / f"univariate_ranking_{site}.csv",
                            *sorted(OUT.glob(f"greedy_{site}_step*.csv"))]:
            if cached_path.exists():
                for row in pd.read_csv(cached_path).to_dict("records"):
                    ev.cache[tuple(row["quartet_key"].split("|"))] = row
        # Check the accelerated evaluator against the original on three archived quartets.
        checks = []
        for seed in [1000, 1001, 1002]:
            old = json.loads((PACKAGE / f"runs/source_only/agents/full_agent/codex/{site}/B10_seed{seed}.json").read_text())["history"][0]
            row = ev.evaluate(old["head_ids"])
            for field in ["selected_C", "auc_source1_to_source2", "auc_source2_to_source1"]:
                assert np.isclose(row[field], old[field], atol=1e-10, rtol=1e-10), (site, field)
            checks.append(row["quartet_key"])
        rank = [ev.evaluate([key]) for key in ev.ids]
        rank.sort(key=lambda x: (-x["min_source_auc"], x["quartet_key"]))
        choices = [dict(method="univariate_top4", run=0,
            feature_selection_subset_evaluations=539, c_tuning_directional_fits=34,
            selector_directional_fits=539 * 34,
            **ev.evaluate([x["quartet_key"] for x in rank[:4]]))]
        pd.DataFrame(rank).to_csv(OUT / f"univariate_ranking_{site}.csv", index=False)
        print(site, "univariate selection complete", flush=True)

        prep = SourcePreprocessor()
        x = prep.fit_transform(ev.x)
        l1_path = []
        for c in np.logspace(-4, 4, 161):
            model = LogisticRegression(C=float(c), penalty="l1", solver="liblinear",
                max_iter=10000, tol=1e-5, random_state=2026090401)
            model.fit(x, ev.y)
            coeff = np.abs(model.coef_[0])
            active = np.flatnonzero(coeff > 1e-8)
            l1_path.append(dict(C=float(c), n_active=len(active)))
            if len(active) >= 4:
                ids = sorted([ev.ids[j] for j in sorted(active, key=lambda j: (-coeff[j], ev.ids[j]))[:4]])
                choices.append(dict(method="l1_top4", run=0,
                    feature_selection_subset_evaluations=0, selector_directional_fits=0,
                    selector_pooled_fits=len(l1_path), l1_path_C=float(c),
                    l1_support_at_stop=int(len(active)), l1_truncated=bool(len(active) > 4),
                    c_tuning_directional_fits=34, **ev.evaluate(ids)))
                break
        else:
            raise ValueError("L1 path never reached four features")
        save(OUT / f"l1_path_{site}.json", l1_path)

        selected = []
        greedy_steps = []
        for step in range(4):
            candidates = [ev.evaluate(selected + [key]) for key in ev.ids if key not in selected]
            chosen = sorted(candidates, key=lambda x: (-x["min_source_auc"], x["quartet_key"]))[0]
            selected = chosen["quartet_key"].split("|")
            greedy_steps.append(dict(step=step + 1, n_candidates=len(candidates), **chosen))
            pd.DataFrame(candidates).to_csv(OUT / f"greedy_{site}_step{step+1}.csv", index=False)
            print(site, "greedy step", step + 1, "complete", flush=True)
        choices.append(dict(method="greedy_forward4", run=0,
            feature_selection_subset_evaluations=2150, selector_directional_fits=2150 * 34,
            c_tuning_directional_fits=0, **chosen))
        for proposal_path in sorted((OUT / "one_shot").glob("run_*.json")):
            run = json.loads(proposal_path.read_text())
            choices.append(dict(method="llm_zero_evaluation", run=run["run"],
                feature_selection_subset_evaluations=0, selector_directional_fits=0,
                c_tuning_directional_fits=34, **ev.evaluate(run["head_ids"])))
        assert sum(c["method"] == "llm_zero_evaluation" for c in choices) == 10
        # Historical-bank overlap is audited after selecting; it never changes a choice.
        forbidden = set(pd.read_csv(PACKAGE / "data/historical_quartets.csv").quartet_key)
        for choice in choices:
            choice["historical_bank_match"] = choice["quartet_key"] in forbidden
        result = dict(site=site, completed_utc=now(), choices=choices,
            greedy_steps=greedy_steps, accelerated_evaluator_matches_archived=checks,
            unique_subsets_computed=len(ev.cache), actual_directional_fits=ev.fits,
            convergence_warnings=ev.convergence_warnings,
            seconds=time.monotonic() - start)
        save(destination, result)
    return result


def source():
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(source_one, SITES))
    choices = pd.DataFrame([row for result in results for row in result["choices"]])
    choices.to_csv(OUT / "source_choices.csv", index=False)
    save(OUT / "choices_before_target.json", dict(created_utc=now(),
        source_choice_sha256=sha(OUT / "source_choices.csv"),
        protocol_sha256=sha(OUT / "protocol.json"),
        n_choices=len(choices), target_scores_used_for_selection=False))


def target():
    gate = json.loads((OUT / "choices_before_target.json").read_text())
    assert sha(OUT / "source_choices.csv") == gate["source_choice_sha256"]
    from agentic_search.analysis.scripts.participant_bootstrap import _score_one, _target_frame
    choices = pd.read_csv(OUT / "source_choices.csv")
    results = []
    predictions = []
    with threadpool_limits(limits=1):
        for site in SITES:
            target_frame = _target_frame(site)
            for row in choices[choices.held_site.eq(site)].to_dict("records"):
                score = _score_one(site, row["quartet_key"], row["selected_C"], target_frame)
                results.append({**row, "target_auc": score["target_auc_recomputed"]})
                predictions.extend(dict(held_site=site, method=row["method"], run=row["run"],
                    subject_id=str(pid), score=float(value))
                    for pid, value in zip(target_frame.subject_id, score["scores"]))
    pd.DataFrame(results).to_csv(OUT / "target_results.csv", index=False)
    pd.DataFrame(predictions).to_parquet(OUT / "predictions_private.parquet", index=False)
    print(pd.DataFrame(results).groupby(["method", "held_site"]).target_auc.mean().unstack().to_string())
    save(OUT / "target_scoring.json", dict(completed_utc=now(),
        source_choice_sha256=gate["source_choice_sha256"], n_predictions=len(predictions)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["plan", "propose", "source", "target"])
    args = parser.parse_args()
    globals()[args.phase]()
