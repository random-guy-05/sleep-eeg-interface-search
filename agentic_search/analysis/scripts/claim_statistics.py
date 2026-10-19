"""Post-hoc run-level bootstrap contrasts for the completed search study.

The resampling unit is an independent search trajectory (or random ordering),
not a candidate row, patient, or all-pairs Cartesian product. Shared seeds are
paired for the evolutionary, taxonomy, and B=25 ablation comparisons. These
are exploratory run-to-run intervals, not patient-level confidence intervals
or preregistered hypothesis tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[2]
OUT = PACKAGE / "analysis/output"
DRAWS = 50_000
SEED = 20260926
SITES = ("I0002", "I0006", "S0001")


def _bootstrap_unpaired(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    """Bootstrap the difference in means by independently resampling runs."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Unpaired run bootstrap requires at least two runs per method")
    ai = rng.integers(0, len(a), size=(DRAWS, len(a)))
    bi = rng.integers(0, len(b), size=(DRAWS, len(b)))
    draws = a[ai].mean(axis=1) - b[bi].mean(axis=1)
    return {
        "design": "independent_run_bootstrap",
        "n_full": int(len(a)),
        "n_comparator": int(len(b)),
        "mean_difference": float(a.mean() - b.mean()),
        "bootstrap_ci_2_5": float(np.quantile(draws, 0.025)),
        "bootstrap_ci_97_5": float(np.quantile(draws, 0.975)),
        "bootstrap_fraction_positive": float(np.mean(draws > 0)),
        "observed_paired_win_fraction": np.nan,
    }


def _bootstrap_paired(a: pd.DataFrame, b: pd.DataFrame, value: str,
                      rng: np.random.Generator) -> dict:
    """Bootstrap shared-seed differences, preserving each method pair."""
    merged = a[["seed", value]].merge(b[["seed", value]], on="seed",
                                      suffixes=("_full", "_comparator"), validate="one_to_one")
    if len(merged) < 2:
        raise ValueError("Paired run bootstrap requires at least two shared seeds")
    diff = (merged[f"{value}_full"] - merged[f"{value}_comparator"]).to_numpy(float)
    ix = rng.integers(0, len(diff), size=(DRAWS, len(diff)))
    draws = diff[ix].mean(axis=1)
    return {
        "design": "paired_shared_seed_bootstrap",
        "n_full": int(len(diff)),
        "n_comparator": int(len(diff)),
        "mean_difference": float(diff.mean()),
        "bootstrap_ci_2_5": float(np.quantile(draws, 0.025)),
        "bootstrap_ci_97_5": float(np.quantile(draws, 0.975)),
        "bootstrap_fraction_positive": float(np.mean(draws > 0)),
        "observed_paired_win_fraction": float(np.mean(diff > 0)),
    }


def _pick(frame: pd.DataFrame, method: str, budget: int, selector: str,
          *, site: str | None = None) -> pd.DataFrame:
    rows = frame[(frame.method == method) & (frame.budget == budget) &
                 (frame.selector == selector)]
    if site is not None:
        rows = rows[rows.held_site.astype(str) == site]
    return rows.sort_values("seed")


def _append_contrast(rows: list[dict], *, budget: int, comparator: str,
                     comparator_selector: str, site: str | None,
                     selections: pd.DataFrame, macro: pd.DataFrame,
                     paired: bool, rng: np.random.Generator) -> None:
    scope = site or "MACRO"
    value = "target_auc" if site else "macro_target_auc"
    source = selections if site else macro
    a = _pick(source, "full_agent", budget, "robust", site=site)
    b = _pick(source, comparator, budget, comparator_selector, site=site)
    result = (_bootstrap_paired(a, b, value, rng) if paired
              else _bootstrap_unpaired(a[value].to_numpy(), b[value].to_numpy(), rng))
    rows.append({"scope": scope, "budget": budget, "comparison": f"full_agent_minus_{comparator}",
                 **result})


def main() -> pd.DataFrame:
    selections = pd.read_csv(OUT / "final_selection_metrics.csv")
    macro = pd.read_csv(OUT / "paired_macro_trials.csv")
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []

    comparisons = [
        ("random_source_robust", "source_robust", False),
        ("random_source_mean", "source_mean", False),
        ("evolutionary_robust", "robust", True),
        ("taxonomy_diverse_robust", "robust", True),
    ]
    for budget in (10, 25, 50):
        for comparator, selector, paired in comparisons:
            _append_contrast(rows, budget=budget, comparator=comparator,
                             comparator_selector=selector, site=None,
                             selections=selections, macro=macro, paired=paired, rng=rng)
            for site in SITES:
                _append_contrast(rows, budget=budget, comparator=comparator,
                                 comparator_selector=selector, site=site,
                                 selections=selections, macro=macro, paired=paired, rng=rng)

    for comparator in ("batch_semantic", "opaque_iterative"):
        _append_contrast(rows, budget=25, comparator=comparator,
                         comparator_selector="robust", site=None,
                         selections=selections, macro=macro, paired=True, rng=rng)
        for site in SITES:
            _append_contrast(rows, budget=25, comparator=comparator,
                             comparator_selector="robust", site=site,
                             selections=selections, macro=macro, paired=True, rng=rng)

    result = pd.DataFrame(rows)
    result.insert(0, "bootstrap_draws", DRAWS)
    result.insert(1, "bootstrap_seed", SEED)
    result.to_csv(OUT / "claim_statistics.csv", index=False)
    return result


if __name__ == "__main__":
    frame = main()
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
