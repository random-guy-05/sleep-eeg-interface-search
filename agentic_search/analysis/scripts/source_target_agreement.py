"""Post-hoc §31/§32 diagnostics: source-best vs target-best agreement per method."""
import pandas as pd

f = pd.read_parquet("agentic_search/analysis/output/queried_candidate_results.parquet")


f["traj_key"] = f.experiment_id.astype(str).where(
    f.experiment_id.astype(str).str.len() > 0,
    f.held_site.astype(str) + "|" + f.budget.astype(str) + "|" + f.seed.astype(str))


def stat(sub, label):
    hits, forg = [], []
    for _eid, g in sub.groupby("traj_key"):
        g = g.dropna(subset=["target_auc"])
        if len(g) < 2:
            continue
        src = g.loc[g.mean_source_auc.idxmax()]
        tgt = g.loc[g.target_auc.idxmax()]
        hits.append(float(src.target_auc == tgt.target_auc))
        forg.append(float(tgt.target_auc - src.target_auc))
    h, fg = pd.Series(hits), pd.Series(forg)
    print(f"{label:>24s}: traj={len(h):4d}  P(source-best==target-best)={h.mean():.4f}  "
          f"mean forgone AUROC={fg.mean():.4f}  median={fg.median():.4f}")


for m in ["full_agent", "opaque_iterative", "batch_semantic",
          "random_source_robust", "evolutionary_robust", "taxonomy_diverse_robust"]:
    sub = f[f.method == m]
    if m == "full_agent":
        sub = sub[sub.budget <= 50]
    stat(sub, m)

print()
print("=== per-site vs equal-weight macro (main_results_table) ===")
m = pd.read_csv("agentic_search/analysis/output/main_results_table.csv")
for _i, r in m.iterrows():
    if r["method"] in {"full_agent", "random_source_robust", "taxonomy_diverse_robust",
                       "opaque_iterative", "batch_semantic", "evolutionary_robust"}:
        print(f"{r['method']:>22s} B={int(r['budget']):3d}  macro={r.macro_mean:.4f}  "
              f"I0002={r.I0002_mean:.4f}  I0006={r.I0006_mean:.4f}  S0001={r.S0001_mean:.4f}")
