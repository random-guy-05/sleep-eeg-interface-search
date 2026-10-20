"""Check released joint intervals, macro estimates and accepted one-shot choices."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[2]
OUT = REV / "results/feature_selection_extension"
summary = pd.read_csv(OUT / "joint_summary.csv")
points = summary[summary.site.eq("macro")].set_index("method").auc
contrasts = pd.read_csv(OUT / "joint_contrasts.csv")
with np.load(OUT / "macro_bootstrap_draws.npz") as draws:
    assert all(draws[k].shape == (50000,) for k in draws.files)
    for row in contrasts.itertuples():
        np.testing.assert_allclose(row.difference, points[row.left] - points[row.right], atol=1e-12, rtol=0)
        expected = np.quantile(draws[row.left] - draws[row.right], [.025, .975])
        np.testing.assert_allclose([row.ci_low, row.ci_high], expected, atol=1e-12, rtol=0)
    for method, rows in summary[~summary.site.eq("macro")].groupby("method"):
        np.testing.assert_allclose(points[method], rows.auc.mean(), atol=1e-12, rtol=0)
proposals = [json.loads(p.read_text()) for p in (OUT / "one_shot").glob("run_*.json")]
assert len(proposals) == 10
assert len({p["quartet_key"] for p in proposals}) == 9
assert len(contrasts[contrasts.left.eq("llm_zero_evaluation")]) == 3
print(f"PASS: {len(contrasts)} joint contrast intervals match 50,000 saved draws; macro estimates and ten one-shot calls verified.")
