"""Budget curves from 100 frozen orderings of the prospective random bank."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PACKAGE = Path(__file__).resolve().parents[1]


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol = yaml.safe_load((PACKAGE / "configs/protocol.yaml").read_text())
    bank = pd.read_csv(PACKAGE / "data/prospective_random500.csv", dtype=str)
    scores = pd.read_parquet(PACKAGE / "runs/source_only/prospective_random500.parquet")
    order_rng = np.random.default_rng(int(protocol["prospective_random_bank"]["ordering_seed"]))
    orderings = np.asarray([order_rng.permutation(len(bank))
                            for _ in range(int(protocol["prospective_random_bank"]["ordering_repeats"]))])
    ordering_rows, selection_rows = [], []
    for repeat, permutation in enumerate(orderings):
        for position, bank_index in enumerate(permutation):
            ordering_rows.append({"repeat": repeat, "position": position, "bank_index": int(bank_index)})
        prefix = bank.iloc[permutation].copy()
        for budget in protocol["random_prefix_budgets"]:
            if budget > len(prefix):
                continue
            top = prefix.iloc[:budget]
            for held_site, site_scores in scores.groupby("held_site", sort=True):
                merged = top.merge(site_scores, on="quartet_key", validate="one_to_one")
                for selector, metric in (("source_mean", "mean_source_auc"), ("source_robust", "min_source_auc")):
                    ranked = merged.sort_values([metric, "bank_index"], ascending=[False, True], kind="mergesort")
                    winner = ranked.iloc[0]
                    selection_rows.append({
                        "method": f"random_{selector}", "selector": selector,
                        "repeat": int(repeat), "budget": int(budget), "held_site": str(held_site),
                        "bank_index": int(winner.bank_index), "quartet_key": winner.quartet_key,
                        "source_auc": float(winner.mean_source_auc),
                        "min_source_auc": float(winner.min_source_auc),
                        "direction_gap": float(winner.direction_gap), "selected_C": float(winner.selected_C),
                    })
    ordering_frame = pd.DataFrame(ordering_rows)
    detail = pd.DataFrame(selection_rows)
    out = PACKAGE / "runs/source_only"
    out.mkdir(parents=True, exist_ok=True)
    ordering_frame.to_csv(out / "random_bank_orderings.csv", index=False)
    detail.to_csv(out / "random_prefix_selections.csv", index=False)
    macro = detail.groupby(["method", "selector", "repeat", "budget"], as_index=False).agg(
        source_auc=("source_auc", "mean"), min_source_auc=("min_source_auc", "mean"),
        direction_gap=("direction_gap", "mean"))
    if not macro.groupby(["method", "repeat", "budget"]).size().eq(1).all():
        raise AssertionError("Random macro table has duplicate method/site groups")
    macro.to_csv(out / "random_prefix_macro_source_only.csv", index=False)
    return detail, macro


if __name__ == "__main__":
    detail, macro = run()
    print(f"Saved {len(detail)} site selections and {len(macro)} source-only macro rows")
