"""Build a site-stratified appendix from the released selection table."""
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]
OUT = PACKAGE / "analysis/output"
SITES = ("I0002", "I0006", "S0001")


def main() -> tuple[Path, Path]:
    frame = pd.read_csv(OUT / "final_selection_metrics.csv")
    keys = ["method", "method_label", "provider", "model", "budget", "selector", "held_site"]
    summary = frame.groupby(keys, dropna=False).agg(
        n_trajectory_rows=("seed", "size"),
        n_unique_quartets=("quartet_key", "nunique"),
        mean_target_auc=("target_auc", "mean"),
        median_target_auc=("target_auc", "median"),
        p2_5_target_auc=("target_auc", lambda x: x.quantile(.025)),
        p97_5_target_auc=("target_auc", lambda x: x.quantile(.975)),
    ).reset_index()
    # Add newly scored post-freeze B=10/B=50 ablation runs from their
    # separately frozen manifest. B=25 remains in the primary release.
    supplemental_dir = PACKAGE / "runs/supplemental/held_site"
    target_path = supplemental_dir / "ablation_target_results.parquet"
    manifest_path = supplemental_dir / "ablation_manifest.json"
    if target_path.exists() and manifest_path.exists():
        targets = pd.read_parquet(target_path).set_index(["held_site", "quartet_key"])
        extra = []
        for method in ("batch_semantic", "opaque_iterative"):
            for path in sorted((PACKAGE / "runs/supplemental" / method).glob("**/*.json")):
                try:
                    state = __import__("json").loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                budget, site = int(state.get("budget", 0)), str(state.get("held_site_internal", ""))
                if budget not in (10, 50) or len(state.get("history", [])) != budget:
                    continue
                selected_opaque = str(state.get("robust_selected_quartet_key", ""))
                selected_item = next((r for r in state.get("history", [])
                                      if str(r.get("quartet_key", "")) == selected_opaque), None)
                if selected_item is None:
                    raise ValueError(f"Robust selection absent from history in {path}")
                key = str(selected_item.get("real_quartet_key") or
                          "|".join(sorted(map(str, selected_item.get("real_head_ids") or selected_item["head_ids"]))))
                if (site, key) not in targets.index:
                    raise ValueError(f"Missing frozen supplemental score for {site}/{key}")
                extra.append({"method": method,
                              "method_label": "Batch semantic" if method == "batch_semantic" else "Opaque iterative",
                              "provider": str(state.get("provider", "codex")),
                              "model": str(state.get("model_requested", "gpt-6-sol")),
                              "budget": budget, "selector": "robust", "held_site": site,
                              "seed": int(state["seed"]), "quartet_key": key,
                              "target_auc": float(targets.loc[(site, key)].target_auc)})
        if extra:
            extra_summary = pd.DataFrame(extra).groupby(keys, dropna=False).agg(
                n_trajectory_rows=("seed", "size"), n_unique_quartets=("quartet_key", "nunique"),
                mean_target_auc=("target_auc", "mean"), median_target_auc=("target_auc", "median"),
                p2_5_target_auc=("target_auc", lambda x: x.quantile(.025)),
                p97_5_target_auc=("target_auc", lambda x: x.quantile(.975))).reset_index()
            summary = pd.concat([summary, extra_summary], ignore_index=True)
    # Add the separately scored Luna controller extension. Its model-final
    # choices are the primary endpoint; its source-robust selector and zero-shot
    # picks remain separate labeled conditions.
    luna_final_path = OUT / "luna_final_choice_scored.csv"
    luna_robust_path = OUT / "luna_generalizability_robust_choices.csv"
    luna_zero_path = OUT / "luna_zero_shot_scored.csv"
    if luna_final_path.exists() and luna_robust_path.exists() and luna_zero_path.exists():
        robust_target = pd.read_parquet(supplemental_dir / "generalizability_target_results.parquet")
        robust_target = robust_target.set_index(["held_site", "quartet_key"])
        luna_rows = []
        final = pd.read_csv(luna_final_path)
        for r in final.itertuples(index=False):
            luna_rows.append({"method": "luna_final_choice", "method_label": "GPT-6 Luna final choice",
                              "provider": "codex", "model": "gpt-6-luna", "budget": int(r.budget),
                              "selector": "final_choice", "held_site": str(r.held_site),
                              "seed": int(r.seed), "quartet_key": str(r.quartet_key),
                              "target_auc": float(r.target_auc)})
        robust = pd.read_csv(luna_robust_path)
        for r in robust.itertuples(index=False):
            target = robust_target.loc[(str(r.held_site), str(r.quartet_key))]
            luna_rows.append({"method": "luna_source_robust", "method_label": "GPT-6 Luna source robust",
                              "provider": "codex", "model": "gpt-6-luna", "budget": int(r.budget),
                              "selector": "source_robust", "held_site": str(r.held_site),
                              "seed": int(r.seed), "quartet_key": str(r.quartet_key),
                              "target_auc": float(target.target_auc)})
        zero = pd.read_csv(luna_zero_path)
        for index, r in zero.iterrows():
            luna_rows.append({"method": "luna_zero_shot", "method_label": "GPT-6 Luna zero-shot",
                              "provider": "codex", "model": "gpt-6-luna", "budget": 0,
                              "selector": "zero_shot", "held_site": str(r.held_site),
                              "seed": index, "quartet_key": str(r.quartet_key),
                              "target_auc": float(r.target_auc)})
        luna_summary = pd.DataFrame(luna_rows).groupby(keys, dropna=False).agg(
            n_trajectory_rows=("seed", "size"), n_unique_quartets=("quartet_key", "nunique"),
            mean_target_auc=("target_auc", "mean"), median_target_auc=("target_auc", "median"),
            p2_5_target_auc=("target_auc", lambda x: x.quantile(.025)),
            p97_5_target_auc=("target_auc", lambda x: x.quantile(.975))).reset_index()
        summary = pd.concat([summary, luna_summary], ignore_index=True)
    summary["held_site"] = pd.Categorical(summary.held_site, categories=SITES, ordered=True)
    summary = summary.sort_values(["method", "budget", "selector", "held_site"], kind="stable")
    summary["held_site"] = summary.held_site.astype(str)
    csv_path = OUT / "appendix_per_site_results.csv"
    summary.to_csv(csv_path, index=False, float_format="%.6f")

    md_path = OUT / "appendix_per_site_results.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Per-site results for every released method\n\n")
        f.write("Each row reports the mean and empirical 2.5th–97.5th percentiles over the saved selection rows for that method, budget, selector, and held site. These are descriptive run/order percentiles, not participant confidence intervals. For random search, repeated orderings reuse one fixed candidate bank; at B=500 the robust selector is a single fixed winner per site. `oracle` denotes post-hoc best-target selection and is not a deployable selector. Supplemental B=10/B=50 ablation rows use their separate append-only manifest; opaque-iterative B=50 has only eight complete runs. Full precision is in the accompanying CSV.\n\n")
        f.write("| Method | Budget | Selector | Site | Rows | Unique quartets | Mean target AUROC | 2.5th–97.5th percentile |\n")
        f.write("|---|---:|---|---|---:|---:|---:|---:|\n")
        for r in summary.itertuples(index=False):
            f.write(f"| {r.method_label} | {r.budget} | {r.selector} | {r.held_site} | {r.n_trajectory_rows} | {r.n_unique_quartets} | {r.mean_target_auc:.4f} | {r.p2_5_target_auc:.4f}–{r.p97_5_target_auc:.4f} |\n")
    return csv_path, md_path


if __name__ == "__main__":
    for path in main():
        print(path)
