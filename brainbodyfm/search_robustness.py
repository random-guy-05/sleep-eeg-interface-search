"""Order-sensitivity summaries; no fitting or target-informed selection."""
import numpy as np
import pandas as pd

BUDGETS = (1, 10, 50, 100, 500)
SITES = ("I0002", "I0006", "S0001")
METRICS = ("source_auc", "target_auc", "gap")


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("All saved permutations must be retained and finite")
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                low95=float(np.quantile(values, .025)), high95=float(np.quantile(values, .975)))


def summarize_search(detail, n_permutations=100):
    required = {"outer_site", "permutation", "budget", "source_auc", "target_auc"}
    if not required.issubset(detail):
        raise ValueError("Missing search columns")
    keys = ["outer_site", "permutation", "budget"]
    expected = {(s, p, b) for s in SITES for p in range(n_permutations) for b in BUDGETS}
    if detail.duplicated(keys).any() or set(detail[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Complete site/permutation/budget grid required")
    if not np.isfinite(detail[["source_auc", "target_auc"]]).all().all():
        raise ValueError("Nonfinite score")
    if not detail[["source_auc", "target_auc"]].apply(lambda x: x.between(0, 1)).all().all():
        raise ValueError("Invalid AUROC")
    scores = detail[keys + ["source_auc", "target_auc"]].copy()
    macro = scores.groupby(["permutation", "budget"], as_index=False)[["source_auc", "target_auc"]].mean()
    macro["outer_site"] = "macro"
    scores = pd.concat([scores, macro], ignore_index=True)
    scores["gap"] = scores.source_auc-scores.target_auc
    baseline = scores.loc[scores.budget.eq(1), ["outer_site", "permutation", *METRICS]]
    paired = scores.merge(baseline, on=["outer_site", "permutation"], suffixes=("", "_baseline"), validate="many_to_one")
    for metric in METRICS:
        paired[f"delta_{metric}"] = paired[metric]-paired[f"{metric}_baseline"]
    rows, changes = [], []
    for (site, budget), group in paired.groupby(["outer_site", "budget"], sort=True):
        row = dict(outer_site=site, budget=int(budget), n_permutations=len(group))
        delta = row.copy()
        for metric in METRICS:
            row.update({f"{metric}_{k}": v for k, v in distribution(group[metric]).items()})
            values = group[f"delta_{metric}"]
            delta.update({f"delta_{metric}_{k}": v for k, v in distribution(values).items()})
            delta[f"fraction_{metric}_increase"] = float((values > 1e-12).mean())
        rows.append(row)
        changes.append(delta)
    return scores, pd.DataFrame(rows), paired, pd.DataFrame(changes)


def correlation_overview(correlations):
    if len(correlations) != 3 or set(correlations.outer_site) != set(SITES):
        raise ValueError("Three distinct within-site correlations required")
    rows = []
    for metric in ("spearman", "pearson"):
        values = correlations[metric].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Undefined within-site correlation")
        rows.append(dict(metric=metric, unweighted_mean=float(values.mean()),
                         site_min=float(values.min()), site_max=float(values.max()), n_sites=3,
                         interpretation="descriptive mean of within-site coefficients; not pooled raw-score correlation"))
    return pd.DataFrame(rows)
