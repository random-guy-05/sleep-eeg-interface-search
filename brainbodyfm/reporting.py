"""Shared construction of result tables, independent of report release status."""
from __future__ import annotations
import pandas as pd
from .controls import summarize_random_k_controls


def build_interface_table(core, references, pca, l1, uncertainty):
    pca_rows = pca.assign(representation=pca.k.map(lambda k: f"pca_{k}"), dimensionality=pca.k, evidence_section="properly_nested_interface")
    result = pd.concat([core, references, pca_rows, l1.assign(evidence_section="sensitivity")], ignore_index=True)
    result = result.merge(uncertainty, on="representation", how="left", validate="one_to_one")
    if result.ci_low.isna().any() or result.ci_high.isna().any():
        raise ValueError("Missing interval for an interface")
    if not result.n_replicates.eq(20000).all():
        raise ValueError("Interface intervals do not contain 20,000 requested replicates")
    return result[["representation", "evidence_section", "dimensionality", "macro_age_auc", "ci_low", "ci_high", "n_finite"]]


def build_stable4_tail_table(interfaces, random_draws):
    """Descriptive k=4 comparisons; not selection-adjusted significance tests."""
    null = random_draws.loc[random_draws.k.eq(4)]
    if len(null) != 500 or null.draw.nunique() != 500:
        raise ValueError("The k=4 null must contain all 500 unique draws")
    rows = []
    for row in interfaces.loc[interfaces.representation.str.contains("stable4", case=False)].itertuples():
        tail = summarize_random_k_controls(null, observed_auc=row.macro_age_auc)
        tail.insert(0, "representation", row.representation)
        rows.append(tail)
    return pd.concat(rows, ignore_index=True)
