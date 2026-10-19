"""Verify the historical random-4 benchmark and refit its B=500 winners."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_search.evaluator.source_evaluator import evaluate_source
from agentic_search.evaluator.target_evaluator import evaluate_target


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
EXPECTED = {
    "B1_source_macro": 0.5667585361403696,
    "B1_target_macro": 0.5558301182355009,
    "B500_source_macro": 0.7800614031240005,
    "B500_target_macro": 0.5785938055252003,
    "I0002_target": 0.7804878048780488,
    "I0006_target": 0.5328798185941043,
    "S0001_target": 0.4224137931034483,
}


def run() -> dict[str, object]:
    stored = pd.read_csv(ROOT / "brainbodyfm/results/probe_search/search_budget_summary.csv").set_index("budget")
    for budget, metric in ((1, "source_auc_mean"), (1, "target_auc_mean"),
                           (500, "source_auc_mean"), (500, "target_auc_mean")):
        key = f"B{budget}_{'source' if metric.startswith('source') else 'target'}_macro"
        value = float(stored.loc[budget, metric])
        if not np.isclose(value, EXPECTED[key], rtol=0, atol=5e-9):
            raise AssertionError(f"Stored historical {key} drifted: {value}")

    selected = pd.read_csv(ROOT / "brainbodyfm/results/probe_search/source_selected_random4.csv")
    expected_site = {site: EXPECTED[f"{site}_target"] for site in ("I0002", "I0006", "S0001")}
    source_rows, target_rows = [], []
    for row in selected.itertuples(index=False):
        names = json.loads(row.heads)
        catalog = pd.read_csv(PACKAGE / "data/heads.csv", dtype=str)
        name_to_id = dict(zip(catalog.head_name, catalog.head_id))
        quartet = sorted(name_to_id[name] for name in names)
        source = evaluate_source(row.outer_site, quartet)
        old_source = float(row.source_auc)
        if not np.isclose(source["mean_source_auc"], old_source, rtol=0, atol=5e-9):
            raise AssertionError(f"Source evaluator mismatch for {row.outer_site}: {source['mean_source_auc']} != {old_source}")
        target = evaluate_target(row.outer_site, quartet, float(row.chosen_C), historical_regression=True)
        if not np.isclose(target["target_auc"], expected_site[row.outer_site], rtol=0, atol=5e-9):
            raise AssertionError(f"Target evaluator mismatch for {row.outer_site}: {target['target_auc']}")
        source_rows.append(source)
        target_rows.append(target)
    source_macro = float(np.mean([r["mean_source_auc"] for r in source_rows]))
    target_macro = float(np.mean([r["target_auc"] for r in target_rows]))
    if not np.isclose(source_macro, EXPECTED["B500_source_macro"], rtol=0, atol=5e-9):
        raise AssertionError(f"Refit B=500 source macro mismatch: {source_macro}")
    if not np.isclose(target_macro, EXPECTED["B500_target_macro"], rtol=0, atol=5e-9):
        raise AssertionError(f"Refit B=500 target macro mismatch: {target_macro}")
    report = {
        "status": "passed",
        "historical_summary_recovered_from_frozen_500_candidate_outputs": True,
        "fresh_refits": "three source-selected historical B=500 candidates, one per held site",
        "B1_source_macro": float(stored.loc[1, "source_auc_mean"]),
        "B1_target_macro": float(stored.loc[1, "target_auc_mean"]),
        "B500_source_macro_refit": source_macro,
        "B500_target_macro_refit": target_macro,
        "site_target_scores": {r["held_site"]: r["target_auc"] for r in target_rows},
        "tolerance_absolute": 5e-9,
        "interpretation": "B=1 is recovered from the frozen 100-ordering historical analysis; B=500 winners and their site scores are independently refit.",
    }
    destination = PACKAGE / "runs/regression_baseline.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
